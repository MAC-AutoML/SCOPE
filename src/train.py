"""SCOPE training: GRPO with span-level credit on a lagged policy ratio."""
import argparse
import copy
import json
import os

import torch
from accelerate import PartialState
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import nanmax, nanmin

from reward_client import RewardClient
from scope_credit import (
    clean_prompt,
    completion_scope_weights,
    format_reward,
    hybrid_advantages,
    stable_group_seed,
)

_state = None
_client = None
_metadata = {}
_trainer = None
NUM_GEN = 8
W_T2I = 1.0
W_FMT = 1.0


class SCOPETrainer(GRPOTrainer):
    """GRPO trainer whose token advantages are reweighted by SCOPE span coefficients.

    A frozen copy of the policy lags one optimizer step behind and fills
    old_per_token_logps. The per-token log ratio is log π_θ - log π_old.
    SCOPE pools that ratio inside each semantic span and multiplies the shared
    completion advantage by the detached coefficient.
    """

    def __init__(self, *args, span_beta=1.0, adv_clip=3.0, adv_threshold=1.0, adv_epsilon=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.span_beta = span_beta
        self.adv_clip = adv_clip
        self.adv_threshold = adv_threshold
        self.adv_epsilon = adv_epsilon
        self._old_policy_model = None
        self._old_policy_synced_step = -1
        self._scope_local_rewards = None
        if self.accelerator.is_main_process:
            print(
                f"[SCOPE] span_beta={self.span_beta} adv_clip={self.adv_clip} "
                f"adv_threshold={self.adv_threshold} group_size={self.num_generations}",
                flush=True,
            )

    def _ensure_old_policy_model(self):
        if self._old_policy_model is not None:
            return
        with torch.no_grad():
            source = self.accelerator.unwrap_model(self.model)
            self._old_policy_model = copy.deepcopy(source).to(self.accelerator.device).eval()
            for parameter in self._old_policy_model.parameters():
                parameter.requires_grad_(False)
        self._old_policy_synced_step = int(self.state.global_step)
        if self.accelerator.is_main_process:
            print(f"[SCOPE] lagged old policy initialized at step={self._old_policy_synced_step}", flush=True)

    def _sync_old_policy_model(self):
        self._ensure_old_policy_model()
        with torch.no_grad():
            source = self.accelerator.unwrap_model(self.model)
            for old_parameter, new_parameter in zip(self._old_policy_model.parameters(), source.parameters()):
                old_parameter.copy_(new_parameter)
            for old_buffer, new_buffer in zip(self._old_policy_model.buffers(), source.buffers()):
                old_buffer.copy_(new_buffer)
        self._old_policy_synced_step = int(self.state.global_step)

    def _apply_hybrid_advantages(self, output):
        local_rewards = self._scope_local_rewards
        if not local_rewards:
            raise RuntimeError("SCOPE advantage requested before rewards were recorded")
        device = output["advantages"].device
        local = torch.tensor(local_rewards, dtype=torch.float32, device=device)
        if local.numel() != output["advantages"].numel():
            raise RuntimeError("Reward rows do not match the completion batch")
        gathered = self.accelerator.gather(local)
        scores = hybrid_advantages(
            gathered.detach().float().cpu().tolist(),
            self.num_generations,
            self.adv_clip,
            self.adv_threshold,
            self.adv_epsilon,
        )
        start = self.accelerator.process_index * local.numel()
        output["advantages"] = torch.tensor(
            scores[start:start + local.numel()], dtype=output["advantages"].dtype, device=device
        )
        logged = list(self._logs["advantages"])
        if len(logged) >= len(scores):
            logged[-len(scores):] = scores
            self._logs["advantages"].clear()
            self._logs["advantages"].extend(logged)

    def _mask_baseline_rows(self, output):
        """The unedited group member is a reward reference and is not optimized."""
        mask = output["completion_mask"]
        count = mask.size(0)
        base = self.accelerator.process_index * count
        for index in range(count):
            if (base + index) % self.num_generations == self.num_generations - 1:
                mask[index] = 0

    def _old_per_token_logps(self, output):
        self._ensure_old_policy_model()
        completion_ids = output["completion_ids"]
        mode = "train" if self.model.training else "eval"
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        forward_inputs = {
            key: output[key]
            for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes", "token_type_ids", "num_images")
            if key in output
        }
        with torch.no_grad():
            old_logps, _ = self._get_per_token_logps_and_entropies(
                self._old_policy_model,
                torch.cat([output["prompt_ids"], completion_ids], dim=1),
                torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1),
                completion_ids.size(1),
                batch_size=batch_size,
                **forward_inputs,
            )
        return old_logps

    def _is_baseline_index(self, local_index, local_count):
        global_index = self.accelerator.process_index * local_count + local_index
        return global_index % self.num_generations == self.num_generations - 1

    @staticmethod
    def _user_text(prompt):
        if isinstance(prompt, list):
            for message in reversed(prompt):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
            return ""
        return str(prompt)

    def _encode_text(self, text, max_length):
        kwargs = {"add_special_tokens": False}
        if max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = max_length
        encoded = self.processing_class(text, **kwargs)
        ids = encoded["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def _baseline_completion_ids(self, prompt):
        text = self._user_text(prompt).strip() or " "
        ids = self._encode_text(text, self.max_completion_length)
        if ids:
            return ids
        eos = self.eos_token_id[0] if isinstance(self.eos_token_id, (list, tuple)) else self.eos_token_id
        return [eos]

    def _prompt_ids_for(self, prompt):
        text = maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"]
        return self._encode_text(text, self.max_prompt_length)

    def _overwrite_baseline_completions(self, prompts, completion_ids, logprobs):
        count = len(prompts)
        for index, prompt in enumerate(prompts):
            if not self._is_baseline_index(index, count):
                continue
            completion_ids[index] = self._baseline_completion_ids(prompt)
            if logprobs is not None:
                logprobs[index] = [0.0] * len(completion_ids[index])

    def _generate(self, prompts, images):
        """Sample N-1 rewrites and append the original prompt as the last group member."""
        count = len(prompts)
        baseline = [self._is_baseline_index(index, count) for index in range(count)]
        rewrite_index = [index for index, flag in enumerate(baseline) if not flag]
        if self.use_vllm:
            prompt_ids, completion_ids, total_tokens, logprobs, forward_kwargs = super()._generate(prompts, images)
            self._overwrite_baseline_completions(prompts, completion_ids, logprobs)
            return prompt_ids, completion_ids, total_tokens, logprobs, forward_kwargs

        rewrite_prompts = [prompts[index] for index in rewrite_index]
        rewrite_images = [images[index] for index in rewrite_index] if images is not None else None
        if rewrite_prompts:
            sampled_prompt_ids, sampled_completion_ids, sampled_logprobs, forward_kwargs = self._generate_single_turn(
                rewrite_prompts, rewrite_images
            )
        else:
            sampled_prompt_ids, sampled_completion_ids, sampled_logprobs, forward_kwargs = [], [], None, {}

        prompt_ids = [None] * count
        completion_ids = [None] * count
        logprobs = [None] * count if sampled_logprobs is not None else None
        for slot, index in enumerate(rewrite_index):
            prompt_ids[index] = sampled_prompt_ids[slot]
            completion_ids[index] = sampled_completion_ids[slot]
            if logprobs is not None:
                logprobs[index] = sampled_logprobs[slot]
        for index, prompt in enumerate(prompts):
            if not baseline[index]:
                continue
            prompt_ids[index] = self._prompt_ids_for(prompt)
            completion_ids[index] = self._baseline_completion_ids(prompt)
            if logprobs is not None:
                logprobs[index] = [0.0] * len(completion_ids[index])
        return self._record_generation(prompt_ids, completion_ids, logprobs, forward_kwargs)

    def _record_generation(self, prompt_ids, completion_ids, logprobs, forward_kwargs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        total_completion_tokens = self.accelerator.gather(completion_lengths).sum()
        if mode == "train":
            seen = self.accelerator.gather(prompt_lengths).sum() + total_completion_tokens
            self.state.num_input_tokens_seen += seen.item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        gathered_lengths = self.accelerator.gather(completion_lengths).float()
        self._metrics[mode]["completions/mean_length"].append(gathered_lengths.mean().item())
        self._metrics[mode]["completions/min_length"].append(gathered_lengths.min().item())
        self._metrics[mode]["completions/max_length"].append(gathered_lengths.max().item())
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        gathered_truncated = self.accelerator.gather(truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(gathered_truncated.float().mean().item())
        finished = gathered_lengths[~gathered_truncated]
        if finished.numel() == 0:
            finished = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(finished.mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(finished.min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(finished.max().item())
        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    def _generate_and_score_completions(self, inputs):
        self._scope_local_rewards = None
        output = super()._generate_and_score_completions(inputs)
        self._apply_hybrid_advantages(output)
        output["old_per_token_logps"] = self._old_per_token_logps(output)
        self._mask_baseline_rows(output)
        return output

    def _scope_weights(self, completion_ids, completion_mask, log_ratio):
        eos_token_id = self.eos_token_id[0] if isinstance(self.eos_token_id, (list, tuple)) else self.eos_token_id
        token_rows = completion_ids.detach().tolist()
        mask_rows = completion_mask.detach().tolist()
        shift_rows = log_ratio.detach().float().cpu().tolist()
        weights = []
        fallbacks = []
        for tokens, mask, shifts in zip(token_rows, mask_rows, shift_rows):
            valid_tokens = [token for token, bit in zip(tokens, mask) if bit]
            valid_shifts = [shift for shift, bit in zip(shifts, mask) if bit]
            if not valid_tokens:
                weights.append([0.0] * len(mask))
                continue
            local_weights, fallback = completion_scope_weights(
                valid_tokens,
                valid_shifts,
                self.processing_class,
                span_beta=self.span_beta,
                eos_token_id=eos_token_id,
            )
            if len(local_weights) != len(valid_tokens):
                local_weights = [1.0] * len(valid_tokens)
                fallback = True
            fallbacks.append(fallback)
            cursor = 0
            row = []
            for bit in mask:
                if bit:
                    row.append(local_weights[cursor])
                    cursor += 1
                else:
                    row.append(0.0)
            weights.append(row)
        tensor = torch.tensor(weights, dtype=log_ratio.dtype, device=log_ratio.device)
        return tensor, fallbacks

    def _compute_loss(self, model, inputs):
        step = int(self.state.global_step)
        if step > self._old_policy_synced_step and step - self._old_policy_synced_step >= 1:
            self._sync_old_policy_model()

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            torch.cat([prompt_ids, completion_ids], dim=1),
            torch.cat([prompt_mask, completion_mask], dim=1),
            completion_ids.size(1),
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps
        span_weights, fallbacks = self._scope_weights(completion_ids, completion_mask, log_ratio)
        token_advantages = inputs["advantages"].unsqueeze(1) * span_weights

        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        per_token_loss = -torch.min(coef_1 * token_advantages, coef_2 * token_advantages)
        if self.beta != 0.0 and "ref_per_token_logps" in inputs:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Sum token losses within each completion, then average only the N-1 rewrites.
        # Baseline rows have an empty completion mask and stay out of the mean.
        sequence_loss = (per_token_loss * completion_mask).sum(-1)
        active = completion_mask.sum(-1) > 0
        loss = sequence_loss[active].mean() if active.any() else sequence_loss.sum() * 0.0
        loss = loss / self.current_gradient_accumulation_steps

        mode = "train" if self.model.training else "eval"
        valid = completion_mask.bool()
        valid_weights = span_weights[valid]
        if valid_weights.numel() > 0:
            self._metrics[mode]["SCOPE/weight_mean"].append(self.accelerator.gather(valid_weights.mean()).mean().item())
            self._metrics[mode]["SCOPE/weight_max"].append(nanmax(self.accelerator.gather(valid_weights.max())).item())
            self._metrics[mode]["SCOPE/weight_min"].append(nanmin(self.accelerator.gather(valid_weights.min())).item())
        if fallbacks:
            fallback_rate = torch.tensor([sum(fallbacks) / len(fallbacks)], device=completion_mask.device)
            self._metrics[mode]["SCOPE/fallback_rate"].append(self.accelerator.gather(fallback_rate).mean().item())
        if entropies is not None:
            token_count = completion_mask.sum().clamp(min=1.0)
            mean_entropy = (entropies * completion_mask).sum() / token_count
            self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        return loss


def reward_fn(completions, ori_prompt, **kwargs):
    count = len(completions)
    rank = 0 if _state is None else _state.process_index
    render_prompts, originals, metadata, formats = [], [], [], []
    for index, (original, completion) in enumerate(zip(ori_prompt, completions)):
        content = completion[0]["content"] if completion else ""
        baseline = (rank * count + index) % NUM_GEN == NUM_GEN - 1
        if baseline:
            render_prompts.append((original or "").strip() or " ")
            formats.append(0.25)
        else:
            render_prompts.append(clean_prompt(content))
            formats.append(format_reward(content))
        originals.append(original)
        metadata.append(_metadata.get(original, {"prompt": original, "include": [], "exclude": []}))

    image_scores = [0.0] * count
    step = int(_trainer.state.global_step) if _trainer is not None else 0
    groups = {}
    for index in range(count):
        groups.setdefault((rank * count + index) // NUM_GEN, []).append(index)
    for group_id, indices in groups.items():
        result = _client.score(
            [render_prompts[index] for index in indices],
            [originals[index] for index in indices],
            [metadata[index] for index in indices],
            seed=stable_group_seed(group_id, step),
        )
        for offset, score in enumerate(result["scores"]):
            image_scores[indices[offset]] = float(score)
    mixed = [W_T2I * image_scores[index] + W_FMT * formats[index] for index in range(count)]
    if _trainer is not None:
        _trainer._scope_local_rewards = mixed
    return mixed


def main():
    global _state, _client, _metadata, _trainer, NUM_GEN, W_T2I, W_FMT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num_gen", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--no_deepspeed", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--span-beta", type=float, default=1.0)
    parser.add_argument("--adv-clip", type=float, default=3.0)
    parser.add_argument("--adv-threshold", type=float, default=1.0)
    parser.add_argument("--w-t2i", type=float, default=1.0)
    parser.add_argument("--w-fmt", type=float, default=1.0)
    args = parser.parse_args()
    if args.num_gen < 2:
        raise ValueError("--num_gen must be at least 2 so each group has a baseline")
    NUM_GEN = args.num_gen
    W_T2I = args.w_t2i
    W_FMT = args.w_fmt
    _state = PartialState()
    endpoints = [item.strip() for item in os.environ.get("REWARD_ENDPOINTS", "http://127.0.0.1:5000").split(",") if item.strip()]
    _client = RewardClient(endpoints, timeout=int(os.environ.get("REWARD_TIMEOUT", "1800")))
    with open(args.data, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if args.max_samples:
        records = records[:args.max_samples]
    _metadata = {record["prompt"]: record for record in records}
    system = """You are an expert prompt rewriter for text-to-image generation. Rewrite the user's request into a clear, concrete, visually grounded prompt that helps an image generator produce the intended image. Preserve the user's intent and all specified details. Do not add objects, people, text, styles, actions, or scene elements that the user did not request. Do not omit constraints or change relationships.
Make the visual description explicit where the request supports it. Prioritize:
• the main subject and the number of subjects or objects;
• appearance, colors, materials, clothing, and distinctive attributes;
• pose, action, and interaction between subjects and objects;
• spatial relationships, orientation, foreground/background, and composition;
• setting, lighting, camera view, and visual style, when specified or clearly implied.
Resolve vague wording only when the intended meaning is clear. Do not invent details to fill gaps. Keep the result concise and avoid repetition, commentary, explanations, and negative instructions that do not come from the user.
Divide the rewritten prompt into non-empty semantic phrases. Each phrase should describe one coherent visual unit or related group of details. Keep details that belong together in the same phrase; for example, do not separate an object's attribute from the object it describes.
Return only this format: <answer>phrase 1 | phrase 2 | phrase 3</answer>. Use | only as a boundary between adjacent non-empty phrases. Never place it at the beginning or end, and never use consecutive | characters. Do not include any text outside the <answer> tags.
The | characters are internal phrase boundaries, not part of the final image prompt."""
    dataset = Dataset.from_list([
        {"prompt": [{"role": "system", "content": system}, {"role": "user", "content": record["prompt"]}], "ori_prompt": record["prompt"]}
        for record in records
    ])
    config = GRPOConfig(
        output_dir=args.outdir, bf16=True, report_to=[], num_train_epochs=1,
        num_generations=args.num_gen, per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=1e-6,
        logging_steps=1, save_steps=50, save_total_limit=2, beta=0.01,
        max_completion_length=args.max_completion_length, max_steps=args.max_steps,
        deepspeed=None if args.no_deepspeed else os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "ds_zero2_bf16.json"),
    )
    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM")
    _trainer = SCOPETrainer(
        model=args.model, args=config, train_dataset=dataset, reward_funcs=[reward_fn], peft_config=peft_config,
        span_beta=args.span_beta, adv_clip=args.adv_clip, adv_threshold=args.adv_threshold,
    )
    if getattr(_trainer.processing_class, "chat_template", None):
        _trainer.processing_class.chat_template = _trainer.processing_class.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "enable_thinking is not defined or enable_thinking is false",
        )
    _trainer.train(resume_from_checkpoint=args.resume)
    _trainer.save_model(args.outdir)


if __name__ == "__main__":
    main()
