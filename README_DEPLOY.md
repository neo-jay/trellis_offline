# TRELLIS 离线版部署说明（无 root 权限）

本包**不需要 root 权限**即可完成部署与运行。所有路径由安装目录自动推导，不再写死 `/root`。

## 一、前置条件

| 项 | 要求 |
| --- | --- |
| 操作系统 | Linux x86_64 |
| GPU | NVIDIA，显存 ≥ 24 GB（推荐 32 GB），驱动 ≥ 550 |
| Python | **3.10.x**，且已安装 PyTorch 2.1.2+cu121（其余依赖见下） |
| 磁盘 | ≥ 20 GB 可用（模型约 4.5 GB，运行输出另计） |
| 权限 | **普通用户即可**，无需 sudo |

### Python 环境必须可被当前用户访问

这是最容易踩的坑：如果 conda 装在 `/root/miniconda3`，而 `/root` 权限是 `0700`，
**其他用户无法执行其中的 python**（即使文件本身有执行位）。请把环境放在当前用户可读的位置：

```bash
# 推荐：放在自己家目录
~/miniconda3/bin/python
# 或系统级共享位置（需管理员）
/opt/conda/bin/python
```

## 二、部署步骤

```bash
# 1. 把 Release 里的全部文件下载到同一个目录，例如 ~/trellis-setup
cd ~/trellis-setup

# 2. 环境自检（强烈建议先做，会指出所有阻塞项）
chmod +x check_env.sh
./check_env.sh --python ~/miniconda3/bin/python

# 3. 校验完整性
chmod +x restore_offline_assets.sh
./restore_offline_assets.sh --verify

# 4. 部署（默认装到当前目录；也可用 --prefix 指定）
./restore_offline_assets.sh

#    指定安装目录的例子：
./restore_offline_assets.sh --prefix ~/trellis

# 5. 启动
cd ~/trellis-setup          # 或你指定的 --prefix 目录
TRELLIS_PYTHON=~/miniconda3/bin/python ./start.sh
```

### 环境自检会检查什么

`check_env.sh` 逐项检查 7 类条件，任何 `[FAIL]` 都会阻止部署：

| # | 检查项 | 阻塞 |
| --- | --- | :---: |
| 1 | CPU 架构（需 x86_64，ARM 会明确报错） | ✅ |
| 2 | GPU 与驱动（显存 <24 GB 仅警告） | ✅ |
| 3 | Python 可执行性（含 0700 目录导致的无权限访问） | ✅ |
| 4 | PyTorch 版本与 `cuda.is_available()` | ✅ |
| 5 | 8 个编译扩展能否导入 | ✅ |
| 6 | `ninja` / `nvcc`（nvdiffrast JIT 需要） | 仅警告 |
| 7 | `/root` 等目录是否对其他用户可穿行 | 仅警告 |

浏览器打开 `http://<服务器IP>:7860`。

## 三、Release 文件清单

| 文件 | 大小 | 必须 |
| --- | ---: | :---: |
| `TRELLIS-image-large-hf-cache.tar.part-00` | 1992 MB | ✅ |
| `TRELLIS-image-large-hf-cache.tar.part-01` | 1015 MB | ✅ |
| `dinov2_vitl14_reg4_pretrain.pth.tar` | 1218 MB | ✅ |
| `u2net.onnx.tar` | 176 MB | ✅ |
| `facebookresearch_dinov2_main.tar.gz` | 1 MB | ✅ |
| `trellis_offline_source.tar.gz` | 0.2 MB | ✅ |
| `restore_offline_assets.sh` | — | ✅ |
| `start.sh` | — | ✅ |
| `check_env.sh` | — | ✅ |
| `SHA256SUMS.txt` | — | 建议 |
| `README_RELEASE.md` | — | 说明 |

**分卷必须两段都下载，并按 part-00、part-01 顺序拼接**（脚本会自动做）。

## 四、部署后的目录结构

```
<prefix>/
├── start.sh                     启动脚本
├── restore_offline_assets.sh    部署脚本
├── TRELLIS/                     代码
│   └── app.py
├── .cache/
│   ├── huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/
│   └── torch/hub/
│       ├── facebookresearch_dinov2_main/
│       └── checkpoints/dinov2_vitl14_reg4_pretrain.pth
└── .u2net/u2net.onnx
```

## 五、可选环境变量

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `TRELLIS_PREFIX` | 脚本所在目录 | 安装根目录 |
| `TRELLIS_PYTHON` | PATH 上的 python3 | 指定解释器 |
| `TRELLIS_SERVER_PORT` | 7860 | 服务端口 |
| `TRELLIS_SERVER_NAME` | 0.0.0.0 | 监听地址 |
| `TORCH_HOME` | `<prefix>/.cache/torch` | Torch 缓存 |
| `HUGGINGFACE_HUB_CACHE` | `<prefix>/.cache/huggingface/hub` | HF 缓存 |
| `U2NET_HOME` | `<prefix>/.u2net` | 抠图模型 |

## 六、常见问题

**Q: 提示 `No usable NVIDIA GPU detected`**
容器内未挂载 GPU。检查 `nvidia-smi`，Docker 启动需加 `--gpus all`。

**Q: 提示 `TRELLIS_PYTHON cannot run` 或 `Permission denied`**
Python 环境在当前用户不可访问的位置（典型是 `/root` 下的 conda）。
把环境移到 `~` 或 `/opt` 下，或请管理员放宽目录权限。

**Q: 部署前想确认环境是否合适**
运行 `./check_env.sh --python <你的python路径>`，它会逐项报告架构、GPU、PyTorch、
CUDA 扩展、ninja/nvcc 与目录权限。

**Q: 提示模型缓存缺失**
`restore_offline_assets.sh` 未执行或执行失败，先跑 `--verify` 检查校验和。

**Q: 已有旧版本，想升级**
脚本默认拒绝覆盖。确认后用 `--force`，旧代码会自动备份为 `TRELLIS.bak.<时间戳>`。

**Q: 端口被占用**
`TRELLIS_SERVER_PORT=7861 ./start.sh`
