#!/usr/bin/env bash
set -euo pipefail

ASSET_DIR="${1:-$(pwd)}"

mkdir -p /root/.cache/huggingface/hub
mkdir -p /root/.cache/torch/hub/checkpoints
mkdir -p /root/.cache/torch/hub
mkdir -p /root/.u2net
mkdir -p /root

if ls "$ASSET_DIR"/TRELLIS-image-large-hf-cache.tar.part-* >/dev/null 2>&1; then
  cat "$ASSET_DIR"/TRELLIS-image-large-hf-cache.tar.part-* > "$ASSET_DIR"/TRELLIS-image-large-hf-cache.tar
  tar -xf "$ASSET_DIR"/TRELLIS-image-large-hf-cache.tar -C /root/.cache/huggingface/hub
fi

tar -xf "$ASSET_DIR"/dinov2_vitl14_reg4_pretrain.pth.tar -C /root/.cache/torch/hub/checkpoints
tar -xzf "$ASSET_DIR"/facebookresearch_dinov2_main.tar.gz -C /root/.cache/torch/hub
tar -xf "$ASSET_DIR"/u2net.onnx.tar -C /root/.u2net

tar -xzf "$ASSET_DIR"/trellis_offline_source.tar.gz -C /root
chmod +x /root/start.sh || true

echo "Restored TRELLIS offline code and model assets."
echo "Start web demo with: cd /root && ./start.sh"
