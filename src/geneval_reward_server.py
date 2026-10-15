# geneval_reward_server.py - Mask2Former + CLIP GenEval 评分服务
# 接收 base64 图片和 metadata（tag/include/exclude），返回 0 或 1。
# 使用 geneval 虚拟环境里的 mmdet 和 open_clip。每张卡一个进程。

import os
import sys
import json
import base64
import time
import threading
import tempfile
import shutil
from io import BytesIO
from typing import List, Dict, Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from model_paths import resolve_model, detector_config
from service_auth import authorized, read_json_body
from geneval_backend import configure_attention

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageOps
import torch
torch.set_grad_enabled(False)

import mmdet
from mmdet.apis import inference_detector, init_detector
import open_clip
from open_clip.openai import load_openai_model
from open_clip.transform import image_transform
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it

# ============================================================
# Configuration
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = resolve_model("GENEVAL_MODEL_PATH")
MODEL_CONFIG = os.environ.get("GENEVAL_MODEL_CONFIG", None)

THRESHOLD = float(os.environ.get("GENEVAL_THRESHOLD", "0.3"))
COUNTING_THRESHOLD = float(os.environ.get("GENEVAL_COUNTING_THRESHOLD", "0.9"))
MAX_OBJECTS = int(os.environ.get("GENEVAL_MAX_OBJECTS", "16"))
NMS_THRESHOLD = float(os.environ.get("GENEVAL_NMS_THRESHOLD", "1.0"))
POSITION_THRESHOLD = float(os.environ.get("GENEVAL_POSITION_THRESHOLD", "0.1"))
BGCOLOR = os.environ.get("GENEVAL_BGCOLOR", "#999")

CLIP_LOCAL_PATH = resolve_model("CLIP_LOCAL_PATH")
OBJECT_NAMES_PATH = os.environ.get("OBJECT_NAMES_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "geneval", "evaluation", "object_names.txt"))

# ============================================================
# Global models
# ============================================================
_object_detector = None
_clip_model = None
_clip_transform = None
_clip_tokenizer = None
_classnames = None
_lock = threading.Lock()

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}


def preload_models():
    global _object_detector, _clip_model, _clip_transform, _clip_tokenizer, _classnames
    configure_attention()

    # Auto-resolve config
    config_path = MODEL_CONFIG
    if config_path is None:
        config_path = detector_config(mmdet.__file__)

    print(f"[GenEval] Loading Mask2Former: {MODEL_PATH}", flush=True)
    print(f"[GenEval] Config: {config_path}", flush=True)
    t0 = time.time()
    _object_detector = init_detector(config_path, MODEL_PATH, device=DEVICE)
    print(f"[GenEval] Mask2Former loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"[GenEval] Loading CLIP ViT-L-14 from local: {CLIP_LOCAL_PATH}", flush=True)
    t0 = time.time()
    _clip_model = load_openai_model(CLIP_LOCAL_PATH, precision="fp32", device=DEVICE)
    _clip_transform = image_transform(_clip_model.visual.image_size, is_train=False)
    _clip_model.eval()
    _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    print(f"[GenEval] CLIP loaded in {time.time()-t0:.1f}s", flush=True)

    with open(OBJECT_NAMES_PATH) as f:
        _classnames = [line.strip() for line in f if line.strip()]
    print(f"[GenEval] Loaded {len(_classnames)} class names", flush=True)


# ============================================================
# ImageCrops dataset for CLIP color classification
# ============================================================
class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects):
        self._image = image.convert("RGB")
        if BGCOLOR == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=BGCOLOR)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        image = image.crop(box[:4])
        return (_clip_transform(image), 0)


# ============================================================
# Evaluation logic (from evaluate_images.py)
# ============================================================
def color_classification(image, bboxes, classname):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            _clip_model, _clip_tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            DEVICE
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes), batch_size=16, num_workers=0
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(_clip_model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def evaluate(image, objects, metadata):
    """Evaluate image against metadata specs. Returns (correct: bool, reason: str)."""
    correct = True
    reason = []
    matched_groups = []

    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found_objects = objects.get(classname, [])[:req['count']]
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'color' in req:
                colors = color_classification(image, found_objects, classname)
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        f"{colors.count(req['color'])} {req['color']}"
                    )
            if 'position' in req and matched:
                expected_rel, target_group = req['position']
                if target_group >= len(matched_groups) or matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    f"{' and '.join(true_rels) if true_rels else 'same position'}"
                                )
                                break
                        if not matched:
                            break
        matched_groups.append(found_objects if matched else None)

    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")

    return correct, "\n".join(reason)


def detect_objects(image_pil: Image.Image, tag: str) -> dict:
    """Run Mask2Former on a PIL image, return detected objects dict."""
    # Save to temp file (mmdet needs file path)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        image_pil.save(tmp.name)
        result = inference_detector(_object_detector, tmp.name)
    finally:
        os.unlink(tmp.name)

    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None

    confidence_threshold = THRESHOLD if tag != "counting" else COUNTING_THRESHOLD
    detected = {}

    for index, classname in enumerate(_classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
        ordering = ordering[:MAX_OBJECTS].tolist()
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append(
                (bbox[index][max_obj], None if segm is None else segm[index][max_obj])
            )
            ordering = [
                obj for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]

    return detected


def score_single(img_b64: str, metadata: dict) -> float:
    """Score a single image against its metadata. Returns 1.0 (correct) or 0.0 (incorrect)."""
    if not img_b64:
        raise ValueError("Missing image payload")
    try:
        image = ImageOps.exif_transpose(
            Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
        )
        tag = metadata.get("tag", "")
        detected = detect_objects(image, tag)
        correct, reason = evaluate(image, detected, metadata)
        return 1.0 if correct else 0.0
    except Exception as e:
        print(f"[GenEval Error] {repr(e)}", flush=True)
        raise


def score_batch(img_b64_list: List[str], metadata_list: List[dict]) -> List[float]:
    """Score a batch of images. Returns list of 0.0/1.0 scores."""
    n = len(img_b64_list)
    print(f"[GenEval] === Batch: {n} images ===", flush=True)
    t0 = time.time()

    scores = []
    for i in range(n):
        with _lock:
            s = score_single(img_b64_list[i], metadata_list[i])
        scores.append(s)

    elapsed = time.time() - t0
    avg = sum(scores) / max(n, 1)
    n_correct = int(sum(scores))
    print(f"[GenEval] Scored {n} in {elapsed:.1f}s, correct={n_correct}/{n} ({avg*100:.1f}%)", flush=True)
    return scores


# ============================================================
# HTTP Server
# ============================================================
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not authorized(self.headers):
            self._json({"error": "Unauthorized"}, 401)
            return
        self._json({"ok": True} if self.path == "/health" else {"error": "Not found"},
                   200 if self.path == "/health" else 404)

    def _json(self, obj, status=200):
        out = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        if not authorized(self.headers):
            self._json({"error": "Unauthorized"}, 401)
            return
        try:
            data = read_json_body(self)
            img_b64_list = data.get("images_b64", [])
            metadata_list = data.get("metadata_list", [])
            if len(metadata_list) != len(img_b64_list):
                self._json({"error": "images_b64 and metadata_list must have same length"}, 400)
                return
            scores = score_batch(img_b64_list, metadata_list)
            self._json({"scores": scores})
        except Exception as e:
            import traceback
            print(f"[GenEval Error] {traceback.format_exc()}", flush=True)
            self._json({"error": str(e), "scores": []}, 500)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("GENEVAL_PORT", "7000"))
    print(f"[GenEval Server] port={port} model={MODEL_PATH}", flush=True)
    print(f"[GenEval Server] thresholds: det={THRESHOLD} counting={COUNTING_THRESHOLD} "
          f"nms={NMS_THRESHOLD} position={POSITION_THRESHOLD}", flush=True)
    preload_models()
    print(f"[GenEval Server] HTTP服务启动，端口 {port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
