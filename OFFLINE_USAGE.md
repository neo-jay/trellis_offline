# TRELLIS 离线版本使用命令说明

## 1. 目录约定

当前离线版本默认目录如下：

```bash
/root/TRELLIS                         # TRELLIS 项目代码
/root/start.sh                        # 离线推理启动脚本
/root/.cache/huggingface/hub          # HuggingFace 模型缓存
/root/.cache/torch                    # Torch / DINOv2 缓存
/root/.u2net                          # rembg / U2Net 背景移除模型
```

当前代码包：

```bash
/root/autodl-tmp/code_packages/code_only_with_training.tar.gz
```

该代码包包含训练脚本，但不包含模型权重、检查点、运行结果。

---

## 2. 离线推理需要的模型文件

当前离线推理依赖以下模型和缓存：

```bash
/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large
/root/.cache/torch/hub/facebookresearch_dinov2_main
/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
/root/.u2net/u2net.onnx
```

大小大致如下：

```text
TRELLIS-image-large       约 2.9G
DINOv2 checkpoint         约 1.2G
U2Net / rembg             约 168M
DINOv2 本地代码仓库       约 2.5M
合计                     约 4.3G
```

TRELLIS 主模型里主要包含：

```text
slat_flow_img_dit_L_64l8p2_fp16.safetensors
ss_flow_img_dit_L_16l8_fp16.safetensors
slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors
slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors
slat_dec_rf_swin8_B_64l8r16_fp16.safetensors
ss_dec_conv3d_16l8_fp16.safetensors
```

---

## 3. 启动网页推理

进入远程后执行：

```bash
cd /root
./start.sh
```

指定端口启动：

```bash
cd /root
TRELLIS_SERVER_NAME=0.0.0.0 TRELLIS_SERVER_PORT=7860 ./start.sh
```

后台启动：

```bash
cd /root
nohup env TRELLIS_SERVER_NAME=0.0.0.0 TRELLIS_SERVER_PORT=7860 ./start.sh > /root/trellis_start.log 2>&1 &
```

查看日志：

```bash
tail -f /root/trellis_start.log
```

检查端口：

```bash
ss -ltnp | grep 7860
```

如果平台需要网页预览或端口转发，转发端口填：

```text
7860
```

---

## 4. 命令行推理

不启动网页，仅跑官方示例：

```bash
cd /root/TRELLIS

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRELLIS_OFFLINE=1
export TORCH_HOME=/root/.cache/torch
export TRELLIS_DINOV2_REPO=/root/.cache/torch/hub/facebookresearch_dinov2_main

/root/miniconda3/bin/python example.py
```

多图示例：

```bash
cd /root/TRELLIS

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRELLIS_OFFLINE=1
export TORCH_HOME=/root/.cache/torch
export TRELLIS_DINOV2_REPO=/root/.cache/torch/hub/facebookresearch_dinov2_main

/root/miniconda3/bin/python example_multi_image.py
```

---

## 5. 启动训练

当前代码已经补齐官方训练入口：

```bash
/root/TRELLIS/train.py
/root/TRELLIS/configs/
/root/TRELLIS/trellis/datasets/
/root/TRELLIS/trellis/trainers/
```

训练前需要准备训练数据集，并确认训练依赖已安装。

单卡训练示例：

```bash
cd /root/TRELLIS

/root/miniconda3/bin/python train.py \
  --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir outputs/slat_flow_img_train \
  --data_dir /path/to/your/dataset \
  --num_gpus 1
```

VAE 训练示例：

```bash
cd /root/TRELLIS

/root/miniconda3/bin/python train.py \
  --config configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json \
  --output_dir outputs/slat_vae_mesh_train \
  --data_dir /path/to/your/dataset \
  --num_gpus 1
```

多卡训练示例：

```bash
cd /root/TRELLIS

/root/miniconda3/bin/python train.py \
  --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir outputs/slat_flow_img_train \
  --data_dir /path/to/your/dataset \
  --num_gpus 4
```

恢复训练示例：

```bash
cd /root/TRELLIS

/root/miniconda3/bin/python train.py \
  --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir outputs/slat_flow_img_train_resume \
  --data_dir /path/to/your/dataset \
  --load_dir outputs/slat_flow_img_train \
  --ckpt latest \
  --num_gpus 1
```

注意：当前离线推理权重不等同于完整训练数据和训练 checkpoint。真正训练还需要符合 TRELLIS 数据格式的数据集。

---

## 6. `start.sh` 中和模型路径相关的位置

当前 `/root/start.sh` 里和离线模型路径相关的内容主要是这些：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TRELLIS_OFFLINE=1
export GRADIO_ANALYTICS_ENABLED=False
export TRELLIS_SERVER_NAME="${TRELLIS_SERVER_NAME:-0.0.0.0}"
export TRELLIS_SERVER_PORT="${TRELLIS_SERVER_PORT:-7860}"
export TORCH_HOME=/root/.cache/torch
export TRELLIS_DINOV2_REPO=/root/.cache/torch/hub/facebookresearch_dinov2_main
```

模型完整性检查：

```bash
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
```

---

## 7. 如果模型存放位置变了，需要修改哪里

### 7.1 TRELLIS 主模型位置变化

默认检查路径：

```bash
/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/refs/main
```

如果仍使用 HuggingFace cache 结构，建议保持这个目录不变。

如果要改到其他位置，例如：

```bash
/data/models/models--JeffreyXiang--TRELLIS-image-large
```

需要修改 `start.sh` 里的检查项：

```bash
test -f /data/models/models--JeffreyXiang--TRELLIS-image-large/refs/main || {
    echo "TRELLIS model cache is missing." >&2
    exit 1
}
```

同时最好设置：

```bash
export HF_HOME=/data/models
```

不过更推荐保持 HuggingFace 默认缓存结构，避免 `from_pretrained("JeffreyXiang/TRELLIS-image-large")` 离线找不到模型。

### 7.2 DINOv2 repo 位置变化

默认路径：

```bash
/root/.cache/torch/hub/facebookresearch_dinov2_main
```

如果改到：

```bash
/data/models/facebookresearch_dinov2_main
```

修改：

```bash
export TRELLIS_DINOV2_REPO=/data/models/facebookresearch_dinov2_main
```

并修改或保留检查：

```bash
test -f "$TRELLIS_DINOV2_REPO/hubconf.py" || {
    echo "Local DINOv2 repository is missing." >&2
    exit 1
}
```

### 7.3 DINOv2 checkpoint 位置变化

默认路径：

```bash
/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

如果只是移动整个 torch cache，例如：

```bash
/data/torch_cache
```

修改：

```bash
export TORCH_HOME=/data/torch_cache
```

并确保文件位于：

```bash
/data/torch_cache/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

### 7.4 U2Net / rembg 位置变化

默认路径：

```bash
/root/.u2net/u2net.onnx
```

如果改到其他位置，最稳妥的做法是仍然创建软链接到默认路径：

```bash
mkdir -p /root/.u2net
ln -sf /data/models/u2net.onnx /root/.u2net/u2net.onnx
```

这样无需修改 Python 代码。

如果直接修改路径，则需要确认 `rembg.new_session('u2net')` 是否能读取新的模型目录；一般不建议改。

---

## 8. 推荐的路径迁移方式

建议不要改代码，优先用软链接保持默认路径：

```bash
mkdir -p /root/.cache/huggingface/hub
mkdir -p /root/.cache/torch/hub/checkpoints
mkdir -p /root/.cache/torch/hub
mkdir -p /root/.u2net

ln -s /data/models/models--JeffreyXiang--TRELLIS-image-large \
  /root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large

ln -s /data/models/facebookresearch_dinov2_main \
  /root/.cache/torch/hub/facebookresearch_dinov2_main

ln -s /data/models/dinov2_vitl14_reg4_pretrain.pth \
  /root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth

ln -s /data/models/u2net.onnx \
  /root/.u2net/u2net.onnx
```

这样 `/root/start.sh` 基本不用改。

---

## 9. 常用检查命令

检查模型文件是否存在：

```bash
test -f /root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/refs/main && echo ok
test -f /root/.cache/torch/hub/facebookresearch_dinov2_main/hubconf.py && echo ok
test -f /root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth && echo ok
test -f /root/.u2net/u2net.onnx && echo ok
```

检查 GPU：

```bash
nvidia-smi
```

检查 PyTorch CUDA：

```bash
/root/miniconda3/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

检查网页端口：

```bash
ss -ltnp | grep 7860
```

查看当前模型占用大小：

```bash
du -sh /root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large
du -sh /root/.cache/torch/hub/checkpoints
du -sh /root/.u2net
```

---

## 10. 注意事项

1. 当前 `code_only_with_training.tar.gz` 只包含代码，不包含模型权重。
2. 离线推理必须提前准备 TRELLIS 主模型、DINOv2、本地 DINOv2 repo、U2Net。
3. 网页推理使用 `/root/start.sh`。
4. 训练使用 `/root/TRELLIS/train.py`。
5. 训练需要额外准备符合 TRELLIS 格式的数据集。
6. 如果只是部署推理，不建议修改模型路径；如果必须修改，优先使用软链接。
