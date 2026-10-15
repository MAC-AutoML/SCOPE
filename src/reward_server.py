# reward_server.py - 奖励协调服务
# 每个 prompt 生成 1 张图，GenEval 返回 0 或 1。同一组使用同一个 seed。

import os, re, json, base64, hashlib, time, threading
from typing import List, Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import requests
from service_auth import auth_headers, authorized, read_json_body
from urllib.parse import urlsplit, urlunsplit

IMG_URLS = os.environ.get("IMG_URLS", "http://127.0.0.1:8002/generate_b64_batch")
IMG_TIMEOUT = float(os.environ.get("IMG_TIMEOUT", "600"))
IMG_W = int(os.environ.get("IMG_W", "512"))
IMG_H = int(os.environ.get("IMG_H", "512"))
# Qwen-Image 默认生成参数：20 步、CFG=4.0、512×512
IMG_STEPS = int(os.environ.get("IMG_STEPS", "20"))
IMG_CFG = float(os.environ.get("IMG_CFG", "4.0"))
SAVE_DIR = os.environ.get("SAVE_DIR", "generated_images")
os.makedirs(SAVE_DIR, exist_ok=True)

GENEVAL_URLS = os.environ.get("GENEVAL_URL", "http://127.0.0.1:7000/score").split(",")
_session = requests.Session()
_session.trust_env = False
def image_url(value):
    parts = urlsplit(value.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("Invalid image endpoint")
    return urlunsplit(parts._replace(path=parts.path.rstrip("/") or "/generate_b64_batch"))


_img_urls = [image_url(u) for u in IMG_URLS.split(",") if u.strip()]
_geneval_lock, _geneval_ptr = threading.Lock(), 0

def next_geneval_url():
    global _geneval_ptr
    with _geneval_lock:
        url = GENEVAL_URLS[_geneval_ptr % len(GENEVAL_URLS)].strip(); _geneval_ptr += 1; return url

def extract_answer_content(text: str) -> str:
    if not text: return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if m: answer = m.group(1).strip()
    else:
        m2 = re.search(r"<answer>(.*)$", text, re.S | re.I)
        answer = m2.group(1).strip() if m2 else text.strip()
    for tag in ["think","desc","scene","person","street","car","environment","building","ground","light","group","clothes","face"]:
        answer = re.sub(rf"</?{tag}[^>]*>", "", answer, flags=re.I)
    answer = re.sub(r"<type>[^<]*</type>", "", answer, flags=re.I)
    return re.sub(r"\n\s*\n", "\n", answer).strip()

def next_img_url():
    import random; return random.choice(_img_urls)

def gen_single_image(prompt: str, seed: int) -> bytes:
    clean = extract_answer_content(prompt)
    url = next_img_url()
    payload = {"prompts": [clean], "width": IMG_W, "height": IMG_H,
               "steps": IMG_STEPS, "true_cfg_scale": IMG_CFG, "seed": seed}
    for att in range(3):
        try:
            r = _session.post(url, json=payload, timeout=IMG_TIMEOUT, headers=auth_headers()); r.raise_for_status()
            b64 = r.json().get("png_b64_list", [])
            if not b64 or not b64[0]: continue
            data = base64.b64decode(b64[0])
            if len(data) < 1000: continue
            return data
        except Exception as e:
            print(f"[GenError] seed={seed} att={att+1} {repr(e)}", flush=True); time.sleep(0.5)
    raise RuntimeError("Image generation failed after retries")

def gen_images_single_seed(prompts: List[str], seed: int) -> List[bytes]:
    """Generate one image per prompt. Every prompt in the call uses the same seed."""
    n = len(prompts)
    results = [b""] * n
    lock = threading.Lock()
    def call(idx):
        data = gen_single_image(prompts[idx], seed)
        with lock: results[idx] = data
    with ThreadPoolExecutor(max_workers=min(len(_img_urls) * 2, n)) as ex:
        list(ex.map(call, range(n)))
    return results

def call_geneval(img_b64_list: List[str], metadata_list: List[dict]) -> List[float]:
    n = len(img_b64_list)
    def _call(start, end):
        url = next_geneval_url()
        try:
            r = _session.post(url, json={"images_b64": img_b64_list[start:end], "metadata_list": metadata_list[start:end]}, timeout=600, headers=auth_headers())
            r.raise_for_status()
            scores = r.json()["scores"]
            if len(scores) != end - start or any(s not in (0, 1) for s in scores):
                raise ValueError("Invalid GenEval scores")
            return start, scores
        except Exception as e:
            raise RuntimeError(f"GenEval request failed: {start}-{end}") from e
    if len(GENEVAL_URLS) <= 1 or n <= 2:
        _, scores = _call(0, n); return scores
    chunk_size = max(1, (n + len(GENEVAL_URLS) - 1) // len(GENEVAL_URLS))
    scores = [0.0] * n
    with ThreadPoolExecutor(max_workers=len(GENEVAL_URLS)) as ex:
        futures = {ex.submit(_call, i, min(i+chunk_size, n)): i for i in range(0, n, chunk_size)}
        for fut in as_completed(futures):
            start, cs = fut.result()
            for j, s in enumerate(cs):
                if start + j < n: scores[start + j] = s
    return scores

def score_batch(
    prompts: List[str],
    ori_prompts: List[str],
    metadata_list: List[dict],
    seed: int,
    save_batch_id: Optional[str] = None,
) -> List[float]:
    """Score one image per prompt with the shared group seed. GenEval returns 0 or 1."""
    n = len(prompts)
    if len(ori_prompts) != n or len(metadata_list) != n:
        raise ValueError("prompts, ori_prompts and metadata_list must have equal lengths")
    if save_batch_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", save_batch_id):
        raise ValueError("Invalid save_batch_id")
    if n == 0:
        return []
    t_start = time.time()

    t0 = time.time()
    all_images = gen_images_single_seed(prompts, seed)
    t_gen = time.time() - t0

    # 转 base64 用于 GenEval 评分
    img_b64_list = []
    for img in all_images:
        img_b64_list.append(base64.b64encode(img).decode("utf-8") if img else "")

    # GenEval 二值评分
    t0 = time.time()
    ge_scores = call_geneval(img_b64_list, metadata_list)
    t_score = time.time() - t0

    # 处理无效图片
    for i in range(n):
        if not all_images[i]:
            ge_scores[i] = 0.0

    scores = list(ge_scores)

    # 标记无效
    for i in range(n):
        if not all_images[i]:
            scores[i] = -1.0

    # Save images before returning so a later process exit still keeps this batch.
    clean = [extract_answer_content(p) for p in prompts]
    batch_ts = datetime.now().strftime("%m%d_%H%M%S")
    groups = {}
    for i in range(n):
        if not all_images[i]:
            continue
        ori = ori_prompts[i]
        groups.setdefault(ori, []).append(i)
    for ori, indices in groups.items():
        digest = hashlib.md5(ori.encode()).hexdigest()[:8]
        folder = save_batch_id or batch_ts
        parent = os.path.join(SAVE_DIR, f"{folder}_{digest}")
        os.makedirs(parent, exist_ok=True)
        with open(os.path.join(parent, "overview.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"原始提示词: {ori}\n改写数量: {len(indices)}\nseed: {seed}\n\n")
            for i in indices:
                is_baseline = prompts[i].strip().lower() == ori.strip().lower()
                label = "baseline" if is_baseline else f"rewrite_{i:04d}"
                handle.write(f"[{label} idx={i}] {clean[i]}\n")
                handle.write(f"  GenEval={ge_scores[i]:.0f} Final={scores[i]:.3f}\n")
        for i in indices:
            is_baseline = prompts[i].strip().lower() == ori.strip().lower()
            sub = os.path.join(parent, f"{i:04d}_{'baseline' if is_baseline else 'rewrite'}")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, f"s{seed}_ge{ge_scores[i]:.0f}.png"), "wb") as handle:
                handle.write(all_images[i])
            with open(os.path.join(sub, "prompt.txt"), "w", encoding="utf-8") as handle:
                handle.write(f"原始: {ori}\n优化: {clean[i]}\n")
                handle.write(f"idx={i} seed={seed}: GenEval={ge_scores[i]:.0f} Final={scores[i]:.3f}\n")

    t_total = time.time() - t_start; ts = datetime.now().strftime("%H:%M:%S")
    ge_avg = sum(s for s in ge_scores if s >= 0) / max(sum(1 for s in ge_scores if s >= 0), 1)
    print(f"[{ts}][Batch n={n} seed={seed}] gen={t_gen:.1f}s score={t_score:.1f}s total={t_total:.1f}s", flush=True)
    print(f"[{ts}][Scores] geneval_avg={ge_avg:.3f} final={sum(s for s in scores if s>=0)/max(n,1):.3f}", flush=True)

    return scores

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not authorized(self.headers):
            self._json({"error": "Unauthorized"}, 401)
            return
        self._json({"ok": True} if self.path == "/health" else {"error": "Not found"},
                   200 if self.path == "/health" else 404)

    def _json(self, obj, status=200):
        out = json.dumps(obj).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        except (BrokenPipeError, ConnectionResetError):
            # Client may timeout/cancel while server is still finishing work.
            # Ignore to keep logs readable; request side already failed.
            return
    def do_POST(self):
        if not authorized(self.headers):
            self._json({"error": "Unauthorized"}, 401)
            return
        try:
            data = read_json_body(self)
            prompts = data.get("prompts",[]); ori_prompts = data.get("ori_prompts",[])
            metadata_list = data.get("metadata_list",[]); seed = data.get("seed",42)
            sb = data.get("save_batch_id")
            if not sb or not isinstance(sb, str):
                sb = None
            else:
                sb = sb.strip() or None
            scores = score_batch(
                prompts,
                ori_prompts,
                metadata_list,
                seed,
                save_batch_id=sb,
            )
            self._json({"scores": scores})
        except Exception as e:
            import traceback; print(f"[Error] {traceback.format_exc()}")
            self._json({"error":str(e),"scores":[]}, 500)
    def log_message(self, *a): pass

if __name__ == "__main__":
    port = int(os.environ.get("REWARD_PORT","5000"))
    print(f"[RewardServer] port={port} size={IMG_W}x{IMG_H} steps={IMG_STEPS} cfg={IMG_CFG}")
    print(f"[RewardServer] GenEval: {GENEVAL_URLS}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
