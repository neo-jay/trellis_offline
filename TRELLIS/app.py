import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import sys
import shutil
import zipfile
import subprocess
from typing import *
import torch
import numpy as np
import imageio
import trimesh
from trimesh.exchange.obj import export_obj
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


MAX_SEED = np.iinfo(np.int32).max
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(APP_DIR, 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)


PAGE_JS = r"""
() => {
    if (window.__trellisObjDownloadHandlerInstalled) return;
    window.__trellisObjDownloadHandlerInstalled = true;

    document.addEventListener('click', (event) => {
        const previewDownload = event.target.closest('#obj_model_viewer a[download]');
        if (!previewDownload) return;

        const filename = (previewDownload.getAttribute('download') || '').toLowerCase();
        if (!filename.endsWith('.glb')) return;

        const packageDownload =
            document.querySelector('#obj_package_download a[href]') ||
            document.querySelector('a#obj_package_download[href]');
        if (!packageDownload) return;

        event.preventDefault();
        event.stopImmediatePropagation();
        packageDownload.click();
    }, true);
}
"""


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """
    Preprocess a list of input images.
    
    Args:
        images (List[Tuple[Image.Image, str]]): The input images.
        
    Returns:
        List[Image.Image]: The preprocessed images.
    """
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: gr.Request,
) -> Tuple[dict, str]:
    """
    Convert an image to a 3D model.

    Args:
        image (Image.Image): The input image.
        multiimages (List[Tuple[Image.Image, str]]): The input images in multi-image mode.
        is_multiimage (bool): Whether is in multi-image mode.
        seed (int): The random seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.
        multiimage_algo (Literal["multidiffusion", "stochastic"]): The algorithm for multi-image generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if not is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        outputs = pipeline.run_multi_image(
            [image[0] for image in multiimages],
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


def extract_obj(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    repair_quality: str,
    req: gr.Request,
) -> Tuple[str, str]:
    """
    Extract an OBJ model package from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the internal GLB file for reliable browser preview.
        str: The path to a ZIP containing OBJ, MTL, and texture files.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)

    # Keep the existing GLB generation pipeline, then convert the generated
    # artifact to OBJ as a post-processing step.
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)

    converted_mesh = trimesh.load(glb_path, force='mesh')
    obj_text, sidecar_files = export_obj(
        converted_mesh,
        include_normals=True,
        include_texture=True,
        return_texture=True,
        mtl_name='sample.mtl',
    )

    source_obj_path = os.path.join(user_dir, 'sample_source.obj')
    with open(source_obj_path, 'w', encoding='utf-8') as f:
        f.write(obj_text)

    package_files = []
    for filename, content in sidecar_files.items():
        sidecar_path = os.path.join(user_dir, os.path.basename(filename))
        mode = 'wb' if isinstance(content, bytes) else 'w'
        kwargs = {} if mode == 'wb' else {'encoding': 'utf-8'}
        with open(sidecar_path, mode, **kwargs) as f:
            f.write(content)
        package_files.append(sidecar_path)

    # Reconstruct a smooth watertight surface without a fixed voxel grid,
    # then create a fresh UV atlas and bake the original texture into it.
    repair_dir = os.path.join(user_dir, 'watertight_obj')
    shutil.rmtree(repair_dir, ignore_errors=True)
    os.makedirs(repair_dir, exist_ok=True)
    del gs, mesh, glb
    torch.cuda.empty_cache()
    repair_env = os.environ.copy()
    pymeshlab_path = repair_env.get('TRELLIS_PYMESHLAB_PYTHONPATH', '')
    if pymeshlab_path:
        repair_env['PYTHONPATH'] = pymeshlab_path + (
            os.pathsep + repair_env['PYTHONPATH'] if repair_env.get('PYTHONPATH') else ''
        )
    repair_python = repair_env.get('TRELLIS_POSTPROCESS_PYTHON', sys.executable)
    repair_script = os.path.join(APP_DIR, 'trellis', 'utils', 'poisson_textured_repair.py')
    command = [
        repair_python,
        repair_script,
        source_obj_path,
        repair_dir,
        '--texture-size',
        str(int(texture_size)),
        '--quality',
        repair_quality,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
            env=repair_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('水密重建超过 15 分钟，请选择“平衡”质量后重试。') from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        stderr = getattr(exc, 'stderr', None)
        stdout = getattr(exc, 'stdout', None)
        details = stderr or stdout or str(exc)
        raise RuntimeError('OBJ水密重建失败：' + details[-4000:]) from exc
    print('OBJ watertight repair:', completed.stdout, flush=True)
    package_files = [
        os.path.join(repair_dir, 'sample.obj'),
        os.path.join(repair_dir, 'sample.mtl'),
        os.path.join(repair_dir, 'material_0.png'),
    ]

    zip_path = os.path.join(user_dir, 'sample_obj_package.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in package_files:
            archive.write(file_path, arcname=os.path.basename(file_path))

    torch.cuda.empty_cache()
    # LitModel3D can accept OBJ, but browsers do not reliably fetch its
    # external MTL and texture files through Gradio. Preview the identical
    # internal GLB while keeping the downloadable deliverable as OBJ.
    return glb_path, zip_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.

    Returns:
        str: The path to the extracted Gaussian file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


with gr.Blocks(delete_cache=(600, 600)) as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="单张图片", id=0) as single_image_input_tab:
                    image_prompt = gr.Image(label="输入图片", format="png", image_mode="RGBA", type="pil", height=300)
                with gr.Tab(label="多视图图片", id=1) as multiimage_input_tab:
                    multiimage_prompt = gr.Gallery(label="输入多视图图片", format="png", type="pil", height=300, columns=3)
                    gr.Markdown("""
                        请分别上传同一物体不同视角的图片。
                        
                        *注意：这是未经专门模型训练的实验性算法。当各视图中物体姿态不同或细节不一致时，生成效果可能不佳。*
                    """)
            
            with gr.Accordion(label="生成设置", open=False):
                seed = gr.Slider(0, MAX_SEED, label="随机种子", value=0, step=1)
                randomize_seed = gr.Checkbox(label="每次随机生成种子", value=True)
                gr.Markdown("阶段 1：稀疏结构生成")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="引导强度", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="采样步数", value=12, step=1)
                gr.Markdown("阶段 2：结构化潜变量生成")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="引导强度", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="采样步数", value=12, step=1)
                multiimage_algo = gr.Radio(
                    [("随机采样", "stochastic"), ("多重扩散", "multidiffusion")],
                    label="多视图算法",
                    value="stochastic",
                )

            generate_btn = gr.Button("生成3D模型")
            
            with gr.Accordion(label="OBJ导出设置", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="网格简化比例", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="纹理尺寸", value=1024, step=512)
                repair_quality = gr.Radio(
                    [("平衡（推荐）", "balanced"), ("高质量", "high")],
                    label="水密表面重建质量",
                    value="balanced",
                    info="使用平滑 Poisson 重建并重新烘焙纹理；高质量模式耗时更长。",
                )
            
            with gr.Row():
                extract_obj_btn = gr.Button("导出OBJ模型", interactive=False)
                extract_gs_btn = gr.Button("导出高斯模型", interactive=False)
            gr.Markdown("""
                        *注意：高斯模型文件可能较大（约 50 MB），预览和下载需要一定时间。*
                        """)

        with gr.Column():
            video_output = gr.Video(label="生成的3D模型", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(
                label="导出的OBJ/高斯模型",
                exposure=10.0,
                height=300,
                elem_id="obj_model_viewer",
            )
            
            with gr.Row():
                download_obj = gr.DownloadButton(
                    label="下载OBJ模型包",
                    interactive=False,
                    elem_id="obj_package_download",
                )
                download_gs = gr.DownloadButton(label="下载高斯模型", interactive=False)  
    
    is_multiimage = gr.State(False)
    output_buf = gr.State()

    # Handlers
    demo.load(start_session)
    demo.load(None, js=PAGE_JS)
    demo.unload(end_session)
    
    single_image_input_tab.select(
        lambda: False,
        outputs=[is_multiimage]
    )
    multiimage_input_tab.select(
        lambda: True,
        outputs=[is_multiimage]
    )
    
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )
    multiimage_prompt.upload(
        preprocess_images,
        inputs=[multiimage_prompt],
        outputs=[multiimage_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[image_prompt, multiimage_prompt, is_multiimage, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, multiimage_algo],
        outputs=[output_buf, video_output],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_obj_btn, extract_gs_btn],
    )

    video_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[extract_obj_btn, extract_gs_btn],
    )

    extract_obj_btn.click(
        extract_obj,
        inputs=[output_buf, mesh_simplify, texture_size, repair_quality],
        outputs=[model_output, download_obj],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_obj],
    )
    
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_obj],
    )
    

# Launch the Gradio app
if __name__ == "__main__":
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    demo.launch(
        server_name=os.environ.get("TRELLIS_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("TRELLIS_SERVER_PORT", "7860")),
        share=False,
        inbrowser=False,
        show_api=False,
        enable_monitoring=False,
    )
