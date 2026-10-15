"""SCOPE credit assignment on top of a token log-ratio.

The lagged policy supplies log π_old. The current policy supplies log π_θ.
Their difference is the per-token log ratio. SCOPE averages that ratio inside
each phrase. The | delimiters stay outside those phrases and keep weight 1.
"""
import math
import re

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def clean_prompt(text):
    """Drop answer wrappers and | delimiters before the frozen renderer."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    match = _ANSWER.search(text)
    inner = match.group(1) if match else text
    parts = [part.strip() for part in inner.split("|")]
    parts = [part for part in parts if part]
    return " ".join(parts) if parts else " "


def span_char_ranges(text):
    """Character spans of the answer phrases, or None when the partition is unusable.

    The | characters are boundaries only and are not included in any span.
    """
    if not text:
        return None
    matches = list(_ANSWER.finditer(text))
    if len(matches) != 1:
        return None
    inner = matches[0].group(1)
    if re.search(r"</?answer\b", inner, re.I):
        return None
    stripped = inner.strip()
    if not stripped or stripped[0] == "|" or stripped[-1] == "|":
        return None
    parts = stripped.split("|")
    if any(part.strip() == "" for part in parts):
        return None
    base = matches[0].start(1)
    lead = len(inner) - len(inner.lstrip())
    cursor = lead
    ranges = []
    for index, part in enumerate(parts):
        left = len(part) - len(part.lstrip())
        right = len(part) - len(part.rstrip())
        start = base + cursor + left
        end = base + cursor + len(part) - right
        if end <= start:
            return None
        ranges.append((start, end))
        cursor += len(part)
        if index < len(parts) - 1:
            if cursor >= len(inner) or inner[cursor] != "|":
                return None
            cursor += 1
    return ranges


def format_reward(text):
    """Reward a parseable span partition and penalize a malformed one."""
    return 0.25 if span_char_ranges(text or "") is not None else -0.25


def stable_group_seed(group_id, step):
    """Same seed for every member of a group, independent of which rank holds it."""
    mixed = ((int(group_id) + 1) * 1_000_003) ^ ((int(step) + 1) * 9_176)
    return (mixed % 2_147_483_646) + 1


def hybrid_advantages(rewards, group_size, clip_value, threshold, epsilon):
    """One completion-level advantage per sample.

    The last member of each group is the unedited prompt. It is a reference
    only, so its advantage is zero. When that reference scores below the
    threshold, rewrites are compared with it. Otherwise they are centered and
    scaled inside the group.
    """
    count = len(rewards)
    if group_size < 2:
        return [0.0] * count
    advantages = []
    groups = count // group_size
    for group in range(groups):
        chunk = list(rewards[group * group_size:(group + 1) * group_size])
        baseline = chunk[-1]
        samples = chunk[:-1]
        mean = sum(samples) / len(samples)
        variance = sum((reward - mean) ** 2 for reward in samples) / len(samples)
        sigma = math.sqrt(variance)
        for reward in samples:
            if baseline < threshold:
                advantage = reward - baseline
            else:
                advantage = (reward - mean) / (sigma + epsilon)
            advantages.append(max(-clip_value, min(clip_value, advantage)))
        advantages.append(0.0)
    advantages.extend([0.0] * (count - len(advantages)))
    return advantages


def _exp_clip(value, low, high):
    if value > 40:
        unclipped = float("inf")
    elif value < -40:
        unclipped = 0.0
    else:
        unclipped = math.exp(value)
    return min(high, max(low, unclipped))


def _ids_and_offsets(encoded):
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
        offsets = offsets[0]
    return list(ids), [(int(start), int(end)) for start, end in offsets]


def _assign_spans(offsets, ranges, delimiters):
    assigned = []
    covered = set()
    for start, end in offsets:
        if end <= start:
            assigned.append(None)
            continue
        span_hit = None
        touches_delimiter = False
        for index in range(start, end):
            if index in delimiters:
                touches_delimiter = True
            for span_index, (span_start, span_end) in enumerate(ranges):
                if span_start <= index < span_end:
                    if span_hit is None:
                        span_hit = span_index
                    elif span_hit != span_index:
                        return None
        # A token that mixes phrase text with | cannot be assigned to one span.
        if span_hit is not None and touches_delimiter:
            return None
        if span_hit is not None:
            covered.add(span_hit)
        assigned.append(span_hit)
    if covered != set(range(len(ranges))):
        return None
    return assigned


def _span_weights(log_shifts, span_ids, beta, low, high):
    groups = {}
    for index, span_id in enumerate(span_ids):
        if span_id is not None:
            groups.setdefault(span_id, []).append(index)
    if not groups:
        return None
    raw = {}
    total_count = 0
    total_mass = 0.0
    for span_id, indices in groups.items():
        mean_shift = sum(log_shifts[index] for index in indices) / len(indices)
        value = _exp_clip(beta * mean_shift, low, high)
        raw[span_id] = value
        total_count += len(indices)
        total_mass += len(indices) * value
    if total_count == 0 or total_mass == 0.0:
        return None
    scale = total_mass / total_count
    return [1.0 if span_id is None else raw[span_id] / scale for span_id in span_ids]


def completion_scope_weights(token_ids, log_shifts, tokenizer, span_beta=1.0, v_min=0.8, v_max=1.2, eos_token_id=None):
    """Per-token SCOPE coefficients for one completion.

    Returns uniform weights and fallback=True when the span partition cannot
    be aligned. A single trailing token that is not part of the retokenized
    text, typically EOS, gets weight zero instead of discarding the spans.
    """
    length = len(token_ids)
    if length == 0:
        return [], False
    if len(log_shifts) != length:
        return [1.0] * length, True
    if eos_token_id is not None and token_ids[-1] != eos_token_id:
        return [1.0] * length, True
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        text_ids, offsets = _ids_and_offsets(encoded)
    except Exception:
        return [1.0] * length, True
    extra = length - len(text_ids)
    if extra not in (0, 1) or list(token_ids[:len(text_ids)]) != list(text_ids):
        return [1.0] * length, True
    ranges = span_char_ranges(text)
    if not ranges or len(offsets) != len(text_ids):
        return [1.0] * length, True
    span_ids = _assign_spans(offsets, ranges, {index for index, char in enumerate(text) if char == "|"})
    if span_ids is None:
        return [1.0] * length, True
    prefix = _span_weights(log_shifts[:len(text_ids)], span_ids, span_beta, v_min, v_max)
    if prefix is None:
        return [1.0] * length, True
    return prefix + [0.0] * extra, False
