#!/bin/bash
# TRELLIS offline launcher -- works with or without root.
#
# Path resolution order (first match wins, no root required):
#   1. explicit environment variable, e.g. TRELLIS_HOME, TORCH_HOME, ...
#   2. TRELLIS_PREFIX if set
#   3. the directory containing THIS script (auto-detected)
#
# Historically every path was hardcoded under /root, which forced `sudo` and
# broke any deployment on a machine where /root is not writable. The script now
# derives a prefix and only falls back to /root when nothing else is available.
set -euo pipefail

# ---------------------------------------------------------------- path layout
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# TRELLIS_PREFIX is the install root: <prefix>/TRELLIS holds the code and
# <prefix>/start.sh is this script. Default: the script's own directory.
PREFIX="${TRELLIS_PREFIX:-$SCRIPT_DIR}"

TRELLIS_CODE_DIR="${TRELLIS_CODE_DIR:-$PREFIX/TRELLIS}"
# Model caches: honour the standard env vars first, then keep them inside the
# prefix so an unprivileged user never needs to touch /root.
export TORCH_HOME="${TORCH_HOME:-$PREFIX/.cache/torch}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PREFIX/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-$PREFIX/.cache/huggingface}"
U2NET_HOME="${U2NET_HOME:-$PREFIX/.u2net}"
export TRELLIS_DINOV2_REPO="${TRELLIS_DINOV2_REPO:-$TORCH_HOME/hub/facebookresearch_dinov2_main}"

# Python interpreter: explicit > python3 on PATH > python on PATH.
# An explicit TRELLIS_PYTHON that is not usable must fail loudly instead of being
# silently ignored -- otherwise the user sees "no interpreter found" while having
# just set one, which is impossible to diagnose.
if [[ -n "${TRELLIS_PYTHON:-}" ]]; then
    if [[ ! -e "${TRELLIS_PYTHON}" ]]; then
        echo "[FAIL] TRELLIS_PYTHON points to a missing file: ${TRELLIS_PYTHON}" >&2
        exit 1
    fi
    if [[ ! -x "${TRELLIS_PYTHON}" ]]; then
        echo "[FAIL] TRELLIS_PYTHON is not executable: ${TRELLIS_PYTHON}" >&2
        exit 1
    fi
    if ! "${TRELLIS_PYTHON}" -c 'import sys' >/dev/null 2>&1; then
        echo "[FAIL] TRELLIS_PYTHON cannot run: ${TRELLIS_PYTHON}" >&2
        exit 1
    fi
    PYTHON="${TRELLIS_PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    cat >&2 <<MSG
[FAIL] No python interpreter found on PATH.

This package assumes an existing Python 3.10 environment with PyTorch 2.1.2+cu121
and the TRELLIS CUDA extensions already installed. Point the launcher at it:

    TRELLIS_PYTHON=/path/to/miniconda3/bin/python ./start.sh

If your conda lives under /root, note that /root is typically mode 0700, so other
users cannot reach it. Install or copy the environment to a shared location such
as /opt/conda or \$HOME/miniconda3 and point TRELLIS_PYTHON there.
MSG
    exit 1
fi

# ------------------------------------------------------------- sanity checks
fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ -d "$TRELLIS_CODE_DIR" ]] || fail "code directory not found: $TRELLIS_CODE_DIR"
[[ -f "$TRELLIS_CODE_DIR/app.py" ]] || fail "app.py not found in $TRELLIS_CODE_DIR"

MODEL_CACHE="$HUGGINGFACE_HUB_CACHE/models--JeffreyXiang--TRELLIS-image-large"
[[ -f "$MODEL_CACHE/refs/main" ]] || fail \
"TRELLIS model cache missing at $MODEL_CACHE
       Run ./restore_offline_assets.sh first, or set HUGGINGFACE_HUB_CACHE."

[[ -f "$TRELLIS_DINOV2_REPO/hubconf.py" ]] || fail \
"DINOv2 repository missing at $TRELLIS_DINOV2_REPO
       Run ./restore_offline_assets.sh first, or set TRELLIS_DINOV2_REPO."

[[ -f "$TORCH_HOME/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth" ]] || fail \
"DINOv2 checkpoint missing under $TORCH_HOME
       Run ./restore_offline_assets.sh first, or set TORCH_HOME."

[[ -f "$U2NET_HOME/u2net.onnx" ]] || fail \
"U2Net model missing at $U2NET_HOME/u2net.onnx
       Run ./restore_offline_assets.sh first, or set U2NET_HOME."

if ! "$PYTHON" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    fail "No usable NVIDIA GPU detected by $PYTHON.
       Check nvidia-smi inside the container and that Docker was started with --gpus all."
fi

# -------------------------------------------------------------------- launch
cd "$TRELLIS_CODE_DIR"

# Deliberately offline: model download must have happened beforehand.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRELLIS_OFFLINE=1
export GRADIO_ANALYTICS_ENABLED=False
export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-${TMPDIR:-/tmp}/gradio}"
export TRELLIS_SERVER_NAME="${TRELLIS_SERVER_NAME:-0.0.0.0}"
export TRELLIS_SERVER_PORT="${TRELLIS_SERVER_PORT:-7860}"

# rembg reads the u2net location from this variable in rembg >= 2.0.50;
# older builds only look in ~/.u2net, so we also expose U2NET_HOME.
export U2NET_HOME

mkdir -p "$GRADIO_TEMP_DIR" 2>/dev/null || true

echo "TRELLIS code    : $TRELLIS_CODE_DIR"
echo "model cache     : $MODEL_CACHE"
echo "torch home      : $TORCH_HOME"
echo "u2net home      : $U2NET_HOME"
echo "python          : $PYTHON"
echo "listening on    : http://${TRELLIS_SERVER_NAME}:${TRELLIS_SERVER_PORT}"
echo

exec "$PYTHON" app.py "$@"
