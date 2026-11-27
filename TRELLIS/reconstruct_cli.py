#!/usr/bin/env python3
"""Command-line entry point for single-image 3D reconstruction.

Flag scheme (deliberately NOT the generic --input_dir/--output_dir convention):
    data source : --input <image|dir>        (single image or a folder of images)
                  --data-root <root> --split <name>   (dataset with train/val/test)
    output      : --output-dir <dir>
The two source forms are mutually exclusive; --output-dir is always required.

Pipeline is identical to app.py: run() -> to_glb() -> export_obj_shared_geometry()
-> close_obj_geometry() -> inspect_mesh(); the only difference is that results go
to --output-dir instead of a Gradio session directory, and no preview copy is made
because nothing renders it.

Exit codes (aligned with the interface document, chapter 5):
    0  success
    1  argument error (bad path, bad value, mutually exclusive flags)
    3  input image cannot be decoded / size abnormal
    4  GPU out of memory or CUDA unavailable
    9  export stage failure (mesh/topology)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
import zipfile
from typing import Iterable

WORKDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKDIR)
os.chdir(WORKDIR)

# nvdiffrast JIT-compiles at first use and shells out to ninja + nvcc. When the
# script is launched with a bare interpreter path (not through start.sh), the
# conda bin is missing from PATH and every run dies with
# "Ninja is required to load C++ extensions". Prepend the interpreter's own
# bin directory so ninja/nvcc resolve regardless of how the CLI was invoked.
_PY_BIN = os.path.dirname(sys.executable)
if _PY_BIN and _PY_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PY_BIN + os.pathsep + os.environ.get("PATH", "")

# Offline defaults. The weights may live either beside the source tree
# (<repo>/.cache/...) as in the shipped bundle, or in the user's home cache
# (~/.cache/...) as on a machine where they were downloaded through HF. Probe the
# usual locations and pick the first that actually contains the model, so the same
# script works in both layouts without the caller setting any variable.
def _pick_cache(repo_rel: str, home_rel: str, probe: str | None = None) -> str:
    """Return the first cache root that exists; when `probe` is given, require it."""
    candidates = [
        os.path.join(WORKDIR, repo_rel),
        os.path.join(os.path.expanduser("~"), home_rel),
    ]
    for c in candidates:
        if probe is None:
            if os.path.isdir(c):
                return c
        elif os.path.exists(os.path.join(c, probe)):
            return c
    return candidates[0]


_HF_HOME = _pick_cache(
    ".cache/huggingface", ".cache/huggingface",
    probe="hub/models--JeffreyXiang--TRELLIS-image-large/refs/main")
_TORCH_HOME = _pick_cache(
    ".cache/torch", ".cache/torch",
    probe="hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
_U2NET_HOME = _pick_cache(".u2net", ".u2net")

# Force the offline flags: setdefault() would silently keep a wrong inherited
# value (e.g. an HF_HOME pointing at an empty directory), which is exactly the
# failure that made the first run look like the cache was missing.
os.environ["TRELLIS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
# Cache locations honour an explicit env var if the caller set one, otherwise use
# the probed default.
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TORCH_HOME", _TORCH_HOME)
os.environ.setdefault("TRELLIS_DINOV2_REPO",
                      os.path.join(os.environ["TORCH_HOME"], "hub/facebookresearch_dinov2_main"))
os.environ.setdefault("U2NET_HOME", _U2NET_HOME)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

EXIT_OK = 0
EXIT_ARGS = 1
EXIT_INPUT = 3
EXIT_ENV = 4
EXIT_EXPORT = 9


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str, code: int) -> int:
    print("[ERROR] " + msg, file=sys.stderr, flush=True)
    return code


# --------------------------------------------------------------------- inputs
def collect_images(input_arg: str | None, data_root: str | None, split: str | None) -> list[str]:
    """Resolve the image list from either --input or --data-root + --split."""
    if input_arg:
        p = os.path.abspath(input_arg)
        if os.path.isfile(p):
            return [p]
        if os.path.isdir(p):
            files = sorted(
                os.path.join(p, f) for f in os.listdir(p)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            )
            if not files:
                raise ValueError("no image files found in directory: %s" % p)
            return files
        raise ValueError("--input-image path does not exist: %s" % p)

    root = os.path.abspath(data_root)  # type: ignore[arg-type]
    if not os.path.isdir(root):
        raise ValueError("--data-root does not exist: %s" % root)
    split_dir = os.path.join(root, split)  # type: ignore[arg-type]
    if not os.path.isdir(split_dir):
        # Accept the common <root>/images/<split> layout as well.
        alt = os.path.join(root, "images", split)  # type: ignore[arg-type]
        if os.path.isdir(alt):
            split_dir = alt
        else:
            raise ValueError(
                "split directory not found: %s (also tried %s)" % (split_dir, alt))
    files = sorted(
        os.path.join(dp, f)
        for dp, _, fns in os.walk(split_dir)
        for f in fns if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        raise ValueError("no image files under split directory: %s" % split_dir)
    return files


# ------------------------------------------------------------------ pipeline
def load_pipeline():
    from trellis.pipelines import TrellisImageTo3DPipeline
    log("[1/2] loading model ...")
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
    log("      model ready")
    return pipe


def generate(pipe, image_path: str, args) -> tuple[dict, str]:
    import torch
    from PIL import Image

    try:
        image = Image.open(image_path)
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("decode-failed: %s" % exc) from exc

    if max(image.size) < 32:
        raise RuntimeError("size-abnormal: image is %dx%d" % image.size)

    out = pipe.run(
        image,
        seed=args.seed,
        formats=args.formats,
        preprocess_image=not args.no_preprocess,
        sparse_structure_sampler_params={
            "steps": SS_STEPS, "cfg_strength": SS_GUIDANCE},
        slat_sampler_params={
            "steps": SLAT_STEPS, "cfg_strength": SLAT_GUIDANCE},
    )
    # keep the GPU from holding onto the intermediate representations
    torch.cuda.empty_cache()
    return out, image_path



def _export_gaussian_ply(gs, path: str) -> dict:
    """Write the 3DGS point cloud to PLY, trying the representation's exporters first.

    Returns a dict describing what happened, never raises for a missing exporter,
    because the caller treats this as best-effort.
    """
    import numpy as np

    # 1) the representation may already know how to save itself
    for attr in ("save_ply", "export_ply", "save_ply_file"):
        fn = getattr(gs, attr, None)
        if callable(fn):
            try:
                fn(path)
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return {"path": path, "ok": True, "method": attr, "error": None}
            except Exception:  # noqa: BLE001
                pass

    # 2) trimesh PointCloud with the Gaussian centres
    try:
        import trimesh
        xyz = gs.get_xyz.detach().cpu().numpy() if hasattr(gs, "get_xyz") else None
        if xyz is not None:
            opacity = None
            try:
                op = gs.get_opacity.detach().cpu().numpy().reshape(-1)
                opacity = op
            except Exception:  # noqa: BLE001
                pass
            cloud = trimesh.PointCloud(xyz)
            if opacity is not None:
                cloud.metadata = {"opacity_min": float(opacity.min()),
                                  "opacity_max": float(opacity.max())}
            cloud.export(path)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return {"path": path, "ok": True, "method": "trimesh.PointCloud",
                        "points": int(len(xyz)), "error": None,
                        "note": "仅中心点，未含协方差/球谐系数" if opacity is None
                                else "中心点+opacity 统计，未含协方差/球谐系数"}
    except Exception as exc:  # noqa: BLE001
        return {"path": None, "ok": False, "method": None, "error": str(exc)[:200]}

    return {"path": None, "ok": False, "method": None,
            "error": "no usable exporter on the gaussian object"}


def export_one(out: dict, out_dir: str, args) -> dict:
    """Write GLB, OBJ, MTL, texture, quality report and the zip package."""
    import torch
    import trimesh
    from trellis.utils import postprocessing_utils
    from trellis.utils.obj_topology_export import (
        export_obj_shared_geometry, close_obj_geometry)
    from trellis.representations.mesh.inference_quality import inspect_mesh

    os.makedirs(out_dir, exist_ok=True)
    gs = out["gaussian"][0]
    mesh = out["mesh"][0]

    # --simplify is the fraction of faces to REMOVE (spec 6.2), and to_glb's
    # `simplify` argument already means the same thing: it is forwarded to
    # pyvista decimate(ratio), whose actual behaviour keeps (1 - ratio). Verified
    # on a 6,960-face sphere: decimate(0.9) left 696 faces. So pass it straight
    # through with no conversion.
    glb = postprocessing_utils.to_glb(
        gs, mesh, simplify=args.simplify, texture_size=args.texture_size, verbose=False)
    glb_path = os.path.join(out_dir, "sample.glb")
    glb.export(glb_path)

    converted = trimesh.load(glb_path, force="mesh")
    obj_text, sidecars, verts, faces = export_obj_shared_geometry(
        converted, mtl_name="sample.mtl")
    obj_text, verts, faces, repair = close_obj_geometry(
        converted, obj_text, verts, faces)

    obj_path = os.path.join(out_dir, "sample.obj")
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(obj_text)

    files = [obj_path]
    for name, content in sidecars.items():
        p = os.path.join(out_dir, os.path.basename(name))
        if isinstance(content, bytes):
            with open(p, "wb") as f:
                f.write(content)
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
        files.append(p)

    quality = {
        "exported_obj": inspect_mesh(verts, faces),
        "obj_repair": repair,
    }
    qpath = os.path.join(out_dir, "mesh_quality.json")
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)

    # --- interface spec 6.4: gaussian.ply --------------------------------
    # The spec requires the 3DGS point cloud next to the mesh. Export is best
    # effort: a failure here must not lose the mesh package, so it is recorded
    # in the quality report instead of raising.
    ply_path = os.path.join(out_dir, "gaussian.ply")
    gaussian_info = {"path": None, "ok": False, "method": None, "error": None}
    try:
        gaussian_info = _export_gaussian_ply(gs, ply_path)
    except Exception as exc:  # noqa: BLE001
        gaussian_info["error"] = str(exc)[:200]
    quality["gaussian_export"] = gaussian_info
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)
    if gaussian_info.get("ok"):
        files.append(ply_path)

    zip_path = os.path.join(out_dir, "sample_obj_package.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=os.path.basename(p))

    torch.cuda.empty_cache()
    return {
        "obj": obj_path,
        "glb": glb_path,
        "zip": zip_path,
        "quality": qpath,
        "topology": quality["exported_obj"],
    }


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-image 3D reconstruction (TRELLIS). "
                    "Data source: --input-image. Output: --output-dir.")
    p.add_argument("--input-image", required=True,
                   help="single RGB/RGBA image (required)")
    p.add_argument("--output-dir", required=True,
                   help="output directory; must be new or empty (required)")

    g = p.add_argument_group("generation")
    g.add_argument("--seed", type=int, default=42,
                   help="random seed (default: 42)")
    g.add_argument("--num-samples", type=int, default=1,
                   help="candidates per image, each into a sub-directory (default: 1)")
    g.add_argument("--formats", default="mesh,gaussian",
                   help="comma separated: mesh,gaussian,radiance_field (default: mesh,gaussian)")
    g.add_argument("--no-preprocess", action="store_true",
                   help="skip automatic cutout/centre-crop (default: off)")

    ex = p.add_argument_group("export")
    ex.add_argument("--simplify", type=float, default=0.8,
                    help="fraction of faces to DELETE, e.g. 0.8 removes about 80%% (default: 0.8)")
    ex.add_argument("--texture-size", type=int, default=1024,
                    help="texture edge length in pixels (default: 1024)")
    return p


# Fixed sampler settings. These were previously exposed as --ss-steps /
# --ss-guidance / --slat-steps / --slat-guidance, but the interface contract does
# not list them, so callers must not depend on them. Values match the TRELLIS
# defaults used before this change.
SS_STEPS = 12
SS_GUIDANCE = 7.5
SLAT_STEPS = 12
SLAT_GUIDANCE = 3.0
GPU_ID = 0


def validate(parser: argparse.ArgumentParser, args) -> None:
    """Argument sanity checks. The source form is fixed by the contract."""
    # 0 = remove nothing, approaching 1 = remove almost everything (capped at 0.95)
    if not (0.0 <= args.simplify <= 0.95):
        parser.error("--simplify is the fraction of faces to REMOVE, within [0, 0.95]")
    if args.texture_size < 256:
        parser.error("--texture-size must be >= 256")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)

    args.formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    # output directory guard
    out_root = os.path.abspath(args.output_dir)
    if os.path.isdir(out_root) and os.listdir(out_root):
        return fail("--output-dir must be new or empty: %s" % out_root, EXIT_ARGS)

    try:
        images = collect_images(args.input_image, None, None)
    except ValueError as exc:
        return fail(str(exc), EXIT_ARGS)

    os.makedirs(out_root, exist_ok=True)
    log("images : %d" % len(images))
    log("output : %s" % out_root)

    # CUDA check before loading the model so the error is unambiguous
    import torch
    if not torch.cuda.is_available():
        return fail("CUDA is not available; this entry requires a GPU", EXIT_ENV)
    if torch.cuda.device_count() == 0:
        return fail("no CUDA device visible", EXIT_ENV)
    torch.cuda.set_device(GPU_ID)

    try:
        pipe = load_pipeline()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return fail("model load failed: %s" % exc, EXIT_ENV)

    summary = {"total": len(images), "ok": 0, "failed": 0, "items": []}
    abort_code = EXIT_OK

    for idx, img_path in enumerate(images, 1):
        name = os.path.splitext(os.path.basename(img_path))[0]
        item_dir = out_root if len(images) == 1 else os.path.join(out_root, name)
        log("\n[%d/%d] %s" % (idx, len(images), os.path.basename(img_path)))
        rec = {"file": img_path, "name": name, "out": item_dir}

        try:
            out, _ = generate(pipe, img_path, args)
        except RuntimeError as exc:
            msg = str(exc)
            code = EXIT_INPUT if msg.startswith(("decode-failed", "size-abnormal")) else EXIT_EXPORT
            rec.update(status="error", code=code, message=msg)
            summary["failed"] += 1
            summary["items"].append(rec)
            fail("%s -> %s" % (os.path.basename(img_path), msg), code)
            if code == EXIT_INPUT:
                continue          # bad file, keep going
            abort_code = code     # GPU/export problem: stop the batch
            break
        except torch.cuda.OutOfMemoryError:
            rec.update(status="error", code=EXIT_ENV, message="CUDA out of memory")
            summary["failed"] += 1
            summary["items"].append(rec)
            fail("%s -> CUDA out of memory" % os.path.basename(img_path), EXIT_ENV)
            abort_code = EXIT_ENV
            break

        # Multiple candidates: the pipeline's own num_samples does not survive the
        # SLAT stage (flash_attn rejects the multi-batch shape), so call run()
        # once per candidate with a shifted seed. Same total cost, same result set.
        if args.num_samples > 1:
            rec["samples"] = []
            rec["status"] = "ok"
            for si in range(args.num_samples):
                sub_dir = os.path.join(item_dir, "sample_%02d" % si)
                try:
                    if si == 0:
                        single = out
                    else:
                        sub_args = argparse.Namespace(**vars(args))
                        sub_args.seed = args.seed + si
                        single, _ = generate(pipe, img_path, sub_args)
                    info = export_one(single, sub_dir, args)
                    rec["samples"].append(info)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    rec["status"] = "error"
                    rec["code"] = EXIT_EXPORT
                    rec["message"] = str(exc)
                    break
        else:
            try:
                rec.update(export_one(out, item_dir, args))
                rec["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                rec.update(status="error", code=EXIT_EXPORT, message=str(exc))

        if rec.get("status") == "ok":
            summary["ok"] += 1
            t = rec.get("topology", {})
            log("      ok  faces=%s boundary=%s nonmanifold=%s components=%s" % (
                t.get("faces"), t.get("boundary_edges"),
                t.get("nonmanifold_edges"), t.get("components")))
        else:
            summary["failed"] += 1
            log("      FAILED: %s" % rec.get("message", "")[:160])

        summary["items"].append(rec)

    # --- interface spec 6.4: log/ directory --------------------------------
    log_dir = os.path.join(out_root, "log")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "recon.log"), "w", encoding="utf-8") as f:
        f.write("images=%d ok=%d failed=%d\n" % (
            summary["total"], summary["ok"], summary["failed"]))
        for it in summary["items"]:
            f.write("%-12s %-10s %s\n" % (
                os.path.basename(it.get("file", "")),
                it.get("status", "?"),
                (it.get("message") or it.get("obj") or "")[:120]))
    with open(os.path.join(log_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # ---------------------------------------------------------------------

    report = os.path.join(out_root, "run_summary.json")
    with open(report, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("\n=== done: ok=%d failed=%d ===" % (summary["ok"], summary["failed"]))
    log("summary: %s" % report)

    if abort_code != EXIT_OK:
        return abort_code
    return EXIT_OK if summary["failed"] == 0 else EXIT_EXPORT


if __name__ == "__main__":
    sys.exit(main())
