# image_server.py - Qwen-Image 本地图像生成服务
import base64
import io
import os
import threading
import time
from datetime import datetime
from collections import deque
from typing import List

import torch
from diffusers import QwenImagePipeline
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from model_paths import resolve_model
from service_auth import authorized

app = FastAPI()


@app.middleware("http")
async def authenticate(request: Request, call_next):
    if not authorized(request.headers):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)
torch.set_grad_enabled(False)

_timing_lock = threading.Lock()
_timing_stats = {"gen": deque(maxlen=1024)}

MODEL_NAME = resolve_model("MODEL_NAME")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or os.path.join(
    PROJECT_ROOT, ".cache", "huggingface")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

GEN_CONCURRENCY = int(os.environ.get("GEN_CONCURRENCY", "1"))
GEN_SEM = threading.Semaphore(GEN_CONCURRENCY)

print(
    f"[QwenImageServer] loading local model: {MODEL_NAME} | device={device} | "
    f"dtype={dtype} | concurrency={GEN_CONCURRENCY}"
)

pipe = QwenImagePipeline.from_pretrained(
    MODEL_NAME, torch_dtype=dtype, local_files_only=True, cache_dir=CACHE_DIR
).to(device)

try:
    pipe.safety_checker = None
except Exception:
    pass

for fn in ["enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"]:
    try:
        getattr(pipe, fn)()
        print(f"[QwenImageServer] {fn} enabled")
    except Exception:
        pass


class BatchReq(BaseModel):
    prompts: List[str] = Field(min_length=1, max_length=64)
    negative_prompt: str = ""
    width: int = Field(default=512, ge=32, le=2048, multiple_of=16)
    height: int = Field(default=512, ge=32, le=2048, multiple_of=16)
    steps: int = Field(default=20, ge=1, le=100)
    true_cfg_scale: float = 4.0
    seed: int = 42


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": device,
        "dtype": str(dtype),
        "model": MODEL_NAME,
        "backend": "qwen_image_local",
        "concurrency": GEN_CONCURRENCY,
    }


def _gen_one(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    true_cfg_scale: float,
    seed: int,
) -> str:
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    with torch.inference_mode():
        img = pipe(
            prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_inference_steps=steps,
            true_cfg_scale=float(true_cfg_scale),
            generator=gen,
        ).images[0]

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    t_gen = time.time() - t0
    with _timing_lock:
        _timing_stats["gen"].append(t_gen)

    ts = datetime.now().strftime("%H:%M:%S")
    port = os.environ.get("PORT", "8000")
    print(
        f"[{ts}][Img:{port}] gen={t_gen:.1f}s steps={steps} size={width}x{height} cfg={true_cfg_scale}"
    )

    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/generate_b64_batch")
def generate_batch(req: BatchReq):
    if not isinstance(req.prompts, list) or len(req.prompts) == 0:
        raise HTTPException(status_code=400, detail="prompts must be a non-empty list")

    with GEN_SEM:
        try:
            outs = []
            for i, p in enumerate(req.prompts):
                b64 = _gen_one(
                    prompt=p,
                    negative_prompt=req.negative_prompt,
                    width=req.width,
                    height=req.height,
                    steps=req.steps,
                    true_cfg_scale=req.true_cfg_scale,
                    seed=req.seed + i,
                )
                outs.append(b64)
            return {"png_b64_list": outs}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"generate_batch failed: {repr(e)}")
