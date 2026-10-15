"""Use MMCV's reference attention implementation on GPUs unsupported by its wheel."""
import os

import torch
from mmcv.ops import multi_scale_deform_attn as attention


def configure_attention():
    backend = os.environ.get("GENEVAL_ATTN_BACKEND", "torch")
    if backend == "mmcv":
        return
    if backend != "torch":
        raise ValueError("GENEVAL_ATTN_BACKEND must be torch or mmcv")

    def apply(value, shapes, level_starts, locations, weights, im2col_step):
        return attention.multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)

    # Only this process is adapted; no installed library files are modified.
    attention.MultiScaleDeformableAttnFunction.apply = staticmethod(apply)


def check_backend():
    import open_clip
    from mmdet.apis import init_detector

    configure_attention()
    print("GenEval torch:", torch.__version__, flush=True)
    if not torch.cuda.is_available():
        print("GenEval imports passed; GPU execution requires a GPU task", flush=True)
        return
    torch.manual_seed(0)
    value = torch.rand(1, 5, 2, 4)
    shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long)
    locations = torch.rand(1, 3, 2, 2, 2, 2)
    weights = torch.rand(1, 3, 2, 4).softmax(-1).reshape(1, 3, 2, 2, 2)
    expected = attention.multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
    actual = attention.MultiScaleDeformableAttnFunction.apply(
        value.cuda(), shapes.cuda(), torch.tensor([0, 4], device="cuda"),
        locations.cuda(), weights.cuda(), 64)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    matrix = torch.ones(2, 2, device="cuda")
    torch.testing.assert_close((matrix @ matrix).cpu(), torch.full((2, 2), 2.0))
    torch.cuda.synchronize()
    print("GENEVAL_GPU_BACKEND_OK", flush=True)


if __name__ == "__main__":
    check_backend()
