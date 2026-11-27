# TRELLIS 单图重建 + 三维模型语义尺寸调整

在 [TRELLIS](https://github.com/microsoft/TRELLIS) 基础上扩展的两个命令行工具：

1. **单图目标重建** —— 一张目标照片生成带贴图的三维网格与 3DGS
2. **三维模型语义尺寸调整** —— 用中文短句对已有 OBJ 做局部尺寸微调，保留原 UV、材质与拓扑

两个入口共用**一个 conda 环境**，无需切换。

---

## 快速开始

### 环境

```bash
conda create -y -n trellis python=3.10
conda activate trellis

# 1) PyTorch 必须最先装
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) CUDA 扩展
pip install spconv-cu120==2.3.6
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install nvdiffrast==0.3.3 kaolin==0.17.0 xatlas==0.0.9

# 3) 其余依赖
pip install -r requirements.txt

# 4) 语义调整的自相交检测（必须加 --no-deps，否则 numpy 会被升级）
pip install --no-deps pymeshlab==2025.7.post1
```

完整说明（含无需 root 的安装方式、GPU 架构兼容、常见问题）见
[`docs/环境配置说明_无需root.txt`](docs/环境配置说明_无需root.txt)。

### 模型权重

权重不随代码分发，需单独放置。程序会自动探测，也可用环境变量指定：

```bash
export HF_HOME=$HOME/.cache/huggingface      # TRELLIS-image-large
export TORCH_HOME=$HOME/.cache/torch         # DINOv2
export U2NET_HOME=$HOME/.u2net               # 抠图模型
```

---

## 用法

### 单图重建

```bash
python reconstruct_cli.py \
    --input-image /data/img/ship_001.jpg \
    --output-dir /data/out/ship_001
```

输出 8 项：

```
sample.obj  sample.mtl  material_0.png  sample_obj_package.zip
sample.glb  gaussian.ply  mesh_quality.json  log/
```

### 语义尺寸调整

```bash
python trellis/utils/mesh_text_edit.py \
    --input-obj /data/out/ship_001/sample.obj \
    --instruction "船头加宽 8%" \
    --output-dir /data/out/ship_001_edit
```

输出 7 项：

```
edited.obj  edited.mtl  material_0.png  edited_obj_package.zip
edit_params.json  mesh_quality.json  log/
```

输入必须是由重建导出的**带贴图 OBJ**（连同 `.mtl` 与贴图同目录）。
不依赖重新运行重建，任意符合要求的 OBJ 均可。

### 先预览再执行

```bash
# 只分析候选区域，不写文件
python trellis/utils/mesh_text_edit.py \
    --input-obj model.obj --instruction "船头加宽 8%" --plan

# 确认后正式输出
python trellis/utils/mesh_text_edit.py \
    --input-obj model.obj --instruction "船头加宽 8%" \
    --output-dir /data/out/edit_001
```

---

## 参数

### reconstruct_cli.py（8 个）

| 参数 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | :---: | --- | --- |
| `--input-image` | 路径 | 是 | — | 单张 RGB/RGBA 图像 |
| `--output-dir` | 路径 | 是 | — | 输出目录，必须新建或为空 |
| `--seed` | 整数 | 否 | 42 | 随机种子 |
| `--num-samples` | 整数 | 否 | 1 | 每张图的候选数 |
| `--formats` | 字符串 | 否 | mesh,gaussian | 输出表示类型 |
| `--no-preprocess` | 开关 | 否 | 关 | 跳过自动抠图与居中裁切 |
| `--simplify` | 浮点 | 否 | 0.8 | **删除**的面比例，0.8 约删 80% |
| `--texture-size` | 整数 | 否 | 1024 | 贴图边长 |

### mesh_text_edit.py（7 个）

| 参数 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | :---: | --- | --- |
| `--input-obj` | 路径 | 是 | — | 带材质 OBJ；MTL 与贴图需可解析 |
| `--instruction` | 字符串 | 是 | — | 如「把船头变宽 8%」 |
| `--output-dir` | 路径 | 正式编辑必填 | — | 新建目录；`--plan` 时不需要 |
| `--region-hint-json` | JSON | 条件 | `{}` | 目标部件、方向或包围盒提示 |
| `--magnitude` | 浮点 | 否 | 按指令解析 | 0.08 表示 +8% |
| `--plan` | 开关 | 否 | 关 | 只分析，不生成文件 |
| `--no-quality-gate` | 开关 | 否 | 关 | 仅诊断用，交付勿用 |

---

## 支持能力与边界

### 尺寸调整支持的部件

| 类别 | 部件 |
| --- | --- |
| 船舶 | 船头、船尾、（船体/甲板/桅杆需提供区域提示） |
| 飞机 | 机翼、左翼、右翼、机身、机头、尾翼 |
| 汽车 | 车头、车尾、车身、车顶、车轮、货厢 |

### 支持的操作

加宽 / 变窄 / 加长 / 缩短 / 加高 / 降低，以及旋转倾斜。

### 明确不支持

* **不新增部件**，不做生成式重建
* **不修改贴图内容**，UV 与材质索引原样保留
* 形变幅度上限 30%，超出破坏薄壁结构
* 无法理解隐含意图（如「看起来更威武」），会明确拒绝而不是猜测

### 返回码

| 码 | 含义 |
| ---: | --- |
| 0 | 成功 |
| 1 | 参数错误 |
| 2 | 输出目录冲突 |
| 3 | 输入文件缺失或格式不符 |
| 5 | 需确认区域或朝向（非错误，需用户澄清） |
| 9 | 质量闸门未通过（形变会破坏网格，拒绝输出） |

---

## 质量保证

语义调整在导出前执行严格检查，任一不通过即拒绝输出：

* 不新增开放边、非流形边、退化面
* 不出现面片翻转
* 局部面片面积变化率在 [0.2, 3.0] 内
* 顶点最大位移不超过对角线 12%
* 非编辑区顶点位移为 0
* 面片与 UV 索引与输入完全一致
* 可选自相交检测

---

## 接口一致性自检

```bash
python tools/check_api_contract.py
```

校验参数名、默认值、必填标记、输出文件清单是否与接口文档一致，
并反向检查是否存在文档未规定的多余参数。

---

## 目录结构

```
reconstruct_cli.py                     单图重建入口
trellis/utils/
  mesh_text_edit.py                    语义调整入口
  auto_vehicle_parts.py                部件自动定位
  semantic_size_support.py             编辑计划与形变
  ground_vehicle_parts.py              地面载具部件规则
  photo_direction_bridge.py            照片辅助方向判定
  mesh_intersection_check.py           自相交检测
  obj_topology_export.py               OBJ 导出
  meshfix_worker.py                    PyMeshFix 子进程隔离
trellis/representations/mesh/
  inference_quality.py                 网格拓扑质量统计
tools/
  check_api_contract.py                接口一致性检查
  prepare_part_masks.py                部件掩码准备
  render_size_comparison.py            调参对比渲染
  render_auto_parts_report.py          自动定位可视化
tests/                                 回归测试
docs/                                  接口与部署文档
```

---

## 与上游 TRELLIS 的关系

本仓库基于 [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)
（arXiv:2412.01506）。新增的 reconstruction CLI 与 semantic size adjustment
为本次扩展，未改动上游的模型结构与推理数学。

上游许可证见 [LICENSE](LICENSE)。

---

## 已知限制

* 单图重建产物的几何质量取决于输入照片；薄壁、细长结构易有伪影
* 语义调整对重建伪影敏感，大幅形变可能被质量闸门拒绝。
  **提高 `--simplify` 值（多删面，如 0.9）或换更好的输入照片可改善**。
  实测：未减面（约 28 万面）时 5% 加宽被拒；删面 0.5/0.8/0.9 三档均可通过
* 部件自动定位对船体、甲板、桅杆覆盖不足，这三类需提供区域提示
* 单一环境已验证；如需重新用 PartField 计算分割，需额外 torch 2.4 环境

---

## 发布资产（Release Assets）

模型权重不随代码分发，通过 GitHub Release 提供：

| 文件 | 说明 |
| --- | --- |
| `dinov2_vitl14_reg4_pretrain.pth.tar` | DINOv2 权重 |
| `facebookresearch_dinov2_main.tar.gz` | DINOv2 torch hub 仓库 |
| `u2net.onnx.tar` | U2Net / rembg 抠图模型 |
| `TRELLIS-image-large-hf-cache.tar.part-00/01` | TRELLIS 主模型（分卷） |
| `restore_offline_assets.sh` | 资产恢复脚本 |
| `SHA256SUMS.txt` | 校验和 |

### 恢复步骤（不需要 root）

```bash
# 1. 下载全部资产到同一目录
# 2. 校验
sha256sum -c SHA256SUMS.txt

# 3. 恢复（按提示设置安装前缀，默认当前目录）
chmod +x restore_offline_assets.sh
./restore_offline_assets.sh .

# 4. 检查环境
./check_env.sh

# 5. 启动
./start.sh
```

`restore_offline_assets.sh` 从 `TRELLIS_PREFIX`（默认脚本所在目录）派生所有路径，
不需要 `sudo`。历史版本曾要求 `sudo ./restore_offline_assets.sh .` 并把权重写入
`/root`，该方式已废弃。

若需把权重放到其他位置，用环境变量指定：

```bash
export TRELLIS_PREFIX=/opt/trellis        # 安装根目录
export TORCH_HOME=$TRELLIS_PREFIX/.cache/torch
export HF_HOME=$TRELLIS_PREFIX/.cache/huggingface
```
