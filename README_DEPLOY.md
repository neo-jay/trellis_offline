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
| `README_DEPLOY.md` | — | 说明 |

### ⚠️ 不要下载页面的 "Source code"

GitHub 页面最下方自动生成的 **Source code (zip)** / **Source code (tar.gz)**：

- **不含任何模型权重**，只有 git 仓库快照
- 且指向旧 tag，代码与本次交付版本不一致

**只下载上方 Assets 列表里的 11 个文件。**

### 为什么模型要切成两片

GitHub Release **单个文件上限 2 GB**，而 TRELLIS 主模型打包后约 **2.9 GB**，因此拆成两片：

| 文件 | 大小 | 内容 |
| --- | ---: | --- |
| `TRELLIS-image-large-hf-cache.tar.part-00` | 1992 MB | 同一个 tar 的前半段 |
| `TRELLIS-image-large-hf-cache.tar.part-01` | 1015 MB | 同一个 tar 的后半段 |
| 合计 | 2.83 GB | 拼接后才是完整归档 |

**两片必须全部下载**，缺任何一片都会导致解压失败（`tar: Unexpected EOF in archive`）。

### 部署脚本会自动拼接，无需手动操作

`restore_offline_assets.sh` 内部的逻辑：

```bash
PARTS=( "$ASSET_DIR"/TRELLIS-image-large-hf-cache.tar.part-* )   # 按名字排序
JOINED="$(mktemp "${TMPDIR:-/tmp}/hf-cache.XXXXXX.tar")"
cat "${PARTS[@]}" > "$JOINED"                                    # 拼成完整 tar
tar -xf "$JOINED" -C "$HF_CACHE"                                 # 解压到 HF 缓存
rm -f "$JOINED"                                                  # 删除临时文件
```

你只需把 11 个文件放在同一目录并执行：

```bash
./restore_offline_assets.sh
```

**三个注意点：**

1. **不要改文件名**。脚本靠 `part-*` 通配符匹配并按字典序拼接，改名会导致找不到或顺序错误。
2. **拼接需要约 2.9 GB 临时空间**，默认使用 `/tmp`。若 `/tmp` 空间不足，请指定其他位置：
   ```bash
   TMPDIR=/data/tmp ./restore_offline_assets.sh
   ```
3. **`SHA256SUMS.txt` 记录的是两片各自的哈希**，不是拼接后的。所以 `--verify` 校验的是分片完整性；两片都对，拼出来必然完整。

### 手动拼接（仅在脚本异常时使用）

```bash
cd ~/trellis-setup

# 1. 按顺序拼接
cat TRELLIS-image-large-hf-cache.tar.part-00 \
    TRELLIS-image-large-hf-cache.tar.part-01 > hf.tar

# 2. 解压到 HF 缓存目录
mkdir -p ~/trellis/.cache/huggingface/hub
tar -xf hf.tar -C ~/trellis/.cache/huggingface/hub

# 3. 清理临时文件
rm hf.tar

# 4. 确认模型就位（有输出即成功）
ls ~/trellis/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/refs/main
```

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
