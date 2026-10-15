#!/bin/bash
# Download the public Mask2Former Swin-S COCO checkpoint used by GenEval.
set -euo pipefail

DEST="${1:-mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth}"
if [ -f "$DEST" ]; then
    echo "Using local Mask2Former checkpoint: $DEST"
    exit 0
fi
mkdir -p "$(dirname "$DEST")"
curl -L --fail -o "$DEST" \
    "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
echo "Saved Mask2Former checkpoint: $DEST"
