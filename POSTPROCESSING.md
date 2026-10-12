# OBJ 水密后处理

网页导出 OBJ 时会在原 TRELLIS GLB 导出流程后自动执行：

1. 从原网格表面采样带法向点云。
2. 用 Screened Poisson 生成平滑、水密的表面。
3. 保留最大连通体并使用二次误差简化。
4. 使用 xatlas 重建 UV。
5. 将原始贴图重新烘焙到新网格。
6. 导出 `sample.obj`、`sample.mtl` 和 `material_0.png` 的 ZIP 包。

## 新增依赖

建议用单独 Python 3.12 环境安装，不要改动 TRELLIS 主推理环境：

```bash
python3.12 -m venv /opt/trellis-postprocess
/opt/trellis-postprocess/bin/pip install -r /root/TRELLIS/requirements-postprocessing.txt
```

完全离线部署时，先在同平台、同 Python 版本的联网机器下载依赖及其传递依赖，再将整个 `wheelhouse` 目录复制到目标机器：

```bash
python3.12 -m pip download -r TRELLIS/requirements-postprocessing.txt -d wheelhouse
/opt/trellis-postprocess/bin/pip install --no-index --find-links wheelhouse -r /root/TRELLIS/requirements-postprocessing.txt
```

`wheelhouse`、Python 3.12 本身及可能需要的 Linux 系统包均需随离线交付物一起准备；本次代码提交不包含这些安装包。

已验证版本：Python 3.12、NumPy 2.5.4、Open3D 0.20.0、Pillow 12.3.0、PyMeshLab 2025.7.post1、Trimesh 5.1.0 和 xatlas 0.0.11。

Linux 如果报缺少 OpenGL 动态库，还需安装：

```bash
apt-get install -y libgl1 libglib2.0-0 libxrender1 libxext6 libsm6
```

## 启动配置

```bash
export TRELLIS_POSTPROCESS_PYTHON=/opt/trellis-postprocess/bin/python
```

如果 PyMeshLab 安装在额外目录而不在该 Python 环境中，再设置：

```bash
export TRELLIS_PYMESHLAB_PYTHONPATH=/path/to/pymeshlab/site-packages
```

标准 `pip install` 方式不需要设置第二个变量。当前远程的已验证配置为：

```bash
export TRELLIS_POSTPROCESS_PYTHON=/root/autodl-tmp/conda-envs/gggs/bin/python
export TRELLIS_PYMESHLAB_PYTHONPATH=/root/autodl-tmp/tools/pymeshlab_py
```

“平衡”模式采样 150,000 点并输出约 100,000 面；“高质量”模式采样 300,000 点并输出约 200,000 面。
