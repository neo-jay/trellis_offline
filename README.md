# TRELLIS Offline Release v1.0.0

This release contains the TRELLIS offline code package and required inference model assets.

## Included assets

- `dinov2_vitl14_reg4_pretrain.pth.tar`: DINOv2 checkpoint.
- `facebookresearch_dinov2_main.tar.gz`: local DINOv2 torch hub repository.
- `u2net.onnx.tar`: U2Net/rembg background removal model.
- `restore_offline_assets.sh`: restore script.
- `SHA256SUMS.txt`: checksum file.

## Restore

Download all release assets to the same directory, then run:

```bash
chmod +x restore_offline_assets.sh
sudo ./restore_offline_assets.sh .
```

## Start web inference

```bash
cd /root
./start.sh
```

Default port: `7860`.

OBJ 水密后处理的算法、额外依赖和部署变量见 [POSTPROCESSING.md](POSTPROCESSING.md)。

## Train example

```bash
cd /root/TRELLIS
/root/miniconda3/bin/python train.py \
  --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir outputs/slat_flow_img_train \
  --data_dir /path/to/your/dataset \
  --num_gpus 1
```

Training requires a dataset in the expected TRELLIS format and training dependencies.
