# 带贴图 OBJ 的开放边检测与修复

## 当前导出链路

网页生成 3DGS 和原始网格后，沿用 TRELLIS 的网格简化、有限补洞、UV 展开及贴图烘焙，先生成内部 GLB；再转换成 OBJ、MTL 和 PNG。浏览器中的模型预览仍使用内部 GLB，ZIP 下载内容才是修复后的 OBJ。此版本不再运行旧版 Poisson 重新建模和纹理重烘焙流程。

1. `trellis/utils/obj_topology_export.py` 将 UV 接缝处重合的几何顶点合并为同一个 OBJ `v` 索引，同时保留独立的 `vt`、`vn` 索引及原有贴图，避免接缝被误计为几何开放边。
2. `trellis/representations/mesh/inference_quality.py` 按几何边的面使用次数统计开放边、非流形边和退化面，并统计边界闭环及非闭合区域。推理时 `cube2mesh.py` 还可对候选 SDF 做幅度受限的局部平滑，只在缺陷分数下降、表面位移受控时采用候选结果。此功能默认开启，可用 `TRELLIS_MESH_INFERENCE_OPT=0` 关闭。
3. OBJ 导出优先用 PyMeshFix 仅填充边界孔洞；如仍存在非流形问题，则尝试局部修复。修复结果必须通过零开放边、零非流形边、零退化面以及原面变更比例和顶点位移门槛，才会替换原 OBJ。新补面沿用邻近 UV；未改动面的原 UV 索引保持不变。不满足门槛时保留原 OBJ 并如实显示剩余缺陷，不强行报零。
4. 网页只在导出 OBJ 后显示最终模型的四项数值：开放边、孔洞、非闭合区域、局部拓扑异常。详细 `mesh_quality.json` 保存在远程会话目录，不进入下载 ZIP。

## 相关代码和依赖

| 文件 | 用途 |
| --- | --- |
| `TRELLIS/app.py` | 网页回调、导出流程及最终质量显示 |
| `TRELLIS/trellis/utils/obj_topology_export.py` | 保贴图的共享顶点 OBJ 导出与分级修复 |
| `TRELLIS/trellis/representations/mesh/inference_quality.py` | 网格质量统计、推理期受限 SDF 候选 |
| `TRELLIS/trellis/representations/mesh/cube2mesh.py` | 推理期候选选择与提取 |
| `start.sh` | 离线启动；将 `/root/miniconda3/bin` 加入 PATH |

新增导出代码使用 TRELLIS 主环境内的 `pymeshfix`、`trimesh`、`scipy` 和 `numpy`；仓库 `TRELLIS/setup.sh` 已包含这些依赖。远程验证版本分别为 PyMeshFix 0.17.0、Trimesh 4.5.3、SciPy 1.14.1、NumPy 1.26.3。部署到其他环境时需准备匹配平台的离线安装包，不再需要旧版 Poisson 流程的独立 Python 3.12 环境、Open3D 或 PyMeshLab。

## 验证结果与边界

2026-10-14 在当前 NVIDIA 远程用单图和 6 张多视图图片完成了生成到 OBJ 下载的端到端测试。两份测试 OBJ 的开放边、边界孔洞、非闭合区域、局部拓扑异常均为 0，OBJ/MTL/PNG 可正常读取；单图使用局部修复，多视图使用仅补孔。该结果只代表这两组样本，不保证所有图片都能修复为零。

当前检测不覆盖自相交，也不证明几何外形、厚度或贴图视觉质量完全正确；网页 GLB 预览与下载的 OBJ 可能因修复而有局部差异。针对甲方使用的 3ds Max，仍应对交付 OBJ 单独执行 STL 检查与贴图外观检查。

PyMeshFix 基于 MeshFix。用于商业交付前，应确认其许可证和交付方式满足项目要求；参见 [PyMeshFix 项目说明](https://github.com/pyvista/pymeshfix)。
