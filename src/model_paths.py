"""Resolve local model files. Nothing is downloaded and no revision is picked automatically.

Place snapshots under MODEL_ROOT. The default is <repo>/models:

    Qwen/Qwen3-8B/config.json
    Qwen/Qwen-Image/model_index.json
    mmdet/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
    openai/clip-vit-large-patch14/ViT-L-14.pt

Qwen3-8B is the public Hugging Face model Qwen/Qwen3-8B.
Qwen-Image is the public Hugging Face model Qwen/Qwen-Image.
The detector file is the public MMDetection Mask2Former Swin-S COCO checkpoint used by GenEval.
The CLIP file is the public OpenAI CLIP ViT-L/14 weight, ViT-L-14.pt.

A directory model may sit in that folder or in exactly one snapshot subdirectory.
Set MODEL_PATH or MODEL_NAME to a model directory when more than one snapshot exists.
Set GENEVAL_MODEL_PATH or CLIP_LOCAL_PATH to the weight file itself.
"""
import json
import os
from pathlib import Path
import shlex


_REPO = Path(__file__).resolve().parent.parent

# Relative directory under MODEL_ROOT, and the file that proves the snapshot is complete.
MODELS = {
    "MODEL_PATH": ("Qwen/Qwen3-8B", "config.json"),
    "MODEL_NAME": ("Qwen/Qwen-Image", "model_index.json"),
    "GENEVAL_MODEL_PATH": (
        "mmdet/mask2former",
        "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth",
    ),
    "CLIP_LOCAL_PATH": ("openai/clip-vit-large-patch14", "ViT-L-14.pt"),
}
_FILE_MODELS = ("GENEVAL_MODEL_PATH", "CLIP_LOCAL_PATH")


def model_root():
    return Path(os.environ.get("MODEL_ROOT", str(_REPO / "models"))).expanduser()


def resolve_model(variable):
    relative, marker = MODELS[variable]
    file_model = variable in _FILE_MODELS
    if os.environ.get(variable):
        path = Path(os.environ[variable]).expanduser()
        target = path if file_model else path / marker
        if not target.is_file():
            raise FileNotFoundError(f"{variable}: required local file missing: {target}")
        return str(path)
    base = model_root() / relative
    candidates = [base / marker] if (base / marker).is_file() else sorted(base.glob("*/" + marker))
    if len(candidates) != 1:
        raise ValueError(
            f"{variable}: expected one snapshot under {base}, found {len(candidates)}; "
            f"set {variable} explicitly"
        )
    return str(candidates[0] if file_model else candidates[0].parent)


def detector_config(package_file):
    package = Path(package_file).parent
    relative = "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    for path in (package / ".mim" / relative, package.parent / relative):
        if path.is_file():
            return str(path)
    raise FileNotFoundError("Set GENEVAL_MODEL_CONFIG to the installed Mask2Former config")


if __name__ == "__main__":
    import sys
    paths = {name: resolve_model(name) for name in MODELS}
    if "--shell" in sys.argv:
        for name, path in paths.items():
            print(f"export {name}={shlex.quote(path)}")
    else:
        print(json.dumps(paths, indent=2))
