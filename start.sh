#!/bin/bash
set -euo pipefail

cd /root/TRELLIS

# Normal startup is deliberately offline. Model downloads must be completed
# separately before this script is run.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRELLIS_OFFLINE=1
export GRADIO_ANALYTICS_ENABLED=False
export TRELLIS_SERVER_NAME="${TRELLIS_SERVER_NAME:-0.0.0.0}"
export TRELLIS_SERVER_PORT="${TRELLIS_SERVER_PORT:-7860}"
export TORCH_HOME=/root/.cache/torch
export TRELLIS_DINOV2_REPO=/root/.cache/torch/hub/facebookresearch_dinov2_main

# Use an isolated environment for watertight OBJ reconstruction. The first
# path is suitable for portable installations; the fallback matches the
# currently verified AutoDL deployment.
if [ -z "${TRELLIS_POSTPROCESS_PYTHON:-}" ]; then
    if [ -x /opt/trellis-postprocess/bin/python ]; then
        export TRELLIS_POSTPROCESS_PYTHON=/opt/trellis-postprocess/bin/python
    elif [ -x /root/autodl-tmp/conda-envs/gggs/bin/python ]; then
        export TRELLIS_POSTPROCESS_PYTHON=/root/autodl-tmp/conda-envs/gggs/bin/python
    fi
fi
if [ -z "${TRELLIS_PYMESHLAB_PYTHONPATH:-}" ] && [ -d /root/autodl-tmp/tools/pymeshlab_py ]; then
    export TRELLIS_PYMESHLAB_PYTHONPATH=/root/autodl-tmp/tools/pymeshlab_py
fi

test -f /root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/refs/main || {
    echo "TRELLIS model cache is missing; run the one-time online download first." >&2
    exit 1
}
test -f "$TRELLIS_DINOV2_REPO/hubconf.py" || {
    echo "Local DINOv2 repository is missing; run the one-time online download first." >&2
    exit 1
}
test -f "$TORCH_HOME/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth" || {
    echo "DINOv2 checkpoint is missing; run the one-time online download first." >&2
    exit 1
}
test -f /root/.u2net/u2net.onnx || {
    echo "U2Net background-removal model is missing; run the one-time online download first." >&2
    exit 1
}

if ! /root/miniconda3/bin/python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "No usable NVIDIA GPU was detected inside the container." >&2
    echo "Start the image with Docker GPU access enabled: --gpus all" >&2
    exit 1
fi

echo "TRELLIS is starting at http://127.0.0.1:${TRELLIS_SERVER_PORT}"
exec /root/miniconda3/bin/python app.py "$@"
