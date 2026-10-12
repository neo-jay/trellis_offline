#!/usr/bin/env python3
"""Smooth watertight Poisson reconstruction with a newly baked texture atlas."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
import pymeshlab
import trimesh
import xatlas


def edge_stats(mesh):
    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def topology(mesh):
    boundary, nonmanifold = edge_stats(mesh)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "watertight": bool(mesh.is_watertight),
        "components": int(len(mesh.split(only_watertight=False))),
    }


def make_scene(mesh):
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def reconstruct(source, point_count, target_faces):
    source = source.copy()
    source.remove_unreferenced_vertices()
    source.update_faces(source.nondegenerate_faces())
    source.remove_unreferenced_vertices()
    source.fix_normals(multibody=True)
    np.random.seed(1234)
    points, face_ids = trimesh.sample.sample_surface(source, point_count)
    normals = np.asarray(source.face_normals)[face_ids]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, scale=1.05, linear_fit=False, n_threads=-1
    )
    raw = trimesh.Trimesh(
        vertices=np.asarray(poisson.vertices),
        faces=np.asarray(poisson.triangles),
        process=False,
    )
    main = max(raw.split(only_watertight=False), key=lambda part: len(part.faces))
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(main.vertices)),
        o3d.utility.Vector3iVector(np.asarray(main.faces, dtype=np.int32)),
    )
    if len(main.faces) > target_faces:
        legacy = legacy.simplify_quadric_decimation(
            target_number_of_triangles=target_faces,
            maximum_error=float("inf"),
            boundary_weight=1.0,
        )
    result = trimesh.Trimesh(
        vertices=np.asarray(legacy.vertices),
        faces=np.asarray(legacy.triangles),
        process=False,
    )
    result.remove_unreferenced_vertices()
    result.fix_normals(multibody=True)
    boundary, nonmanifold = edge_stats(result)
    if boundary or nonmanifold or not result.is_watertight:
        raise RuntimeError(
            f"Poisson result is not manifold: boundary={boundary}, nonmanifold={nonmanifold}"
        )
    return result


def sample_texture(texture, uv):
    image = np.asarray(texture.convert("RGB"), dtype=np.float64) / 255.0
    height, width = image.shape[:2]
    x = np.clip(uv[:, 0], 0, 1) * (width - 1)
    y = (1.0 - np.clip(uv[:, 1], 0, 1)) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y0 = np.floor(y).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def bake(source, target, texture_path, output, texture_size):
    source_uv = np.asarray(source.visual.uv, dtype=np.float64)
    source_faces = np.asarray(source.faces, dtype=np.int64)
    scene = make_scene(source)
    nearest = scene.compute_closest_points(
        o3d.core.Tensor(np.asarray(target.vertices, dtype=np.float32))
    )
    primitive_ids = nearest["primitive_ids"].numpy().astype(np.int64)
    closest = nearest["points"].numpy()
    triangles = np.asarray(source.vertices)[source_faces[primitive_ids]]
    bary = trimesh.triangles.points_to_barycentric(triangles, closest, method="cramer")
    bary = np.nan_to_num(bary, nan=1.0 / 3.0)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
    projected_uv = np.einsum(
        "ni,nij->nj", bary, source_uv[source_faces[primitive_ids]]
    )
    rgb = sample_texture(Image.open(texture_path), projected_uv)
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)

    vmapping, indices, atlas_uv = xatlas.parametrize(
        np.asarray(target.vertices, dtype=np.float32),
        np.asarray(target.faces, dtype=np.uint32),
    )
    mapped_faces = np.asarray(vmapping, dtype=np.int64)[indices]
    wedge_uv = np.asarray(atlas_uv, dtype=np.float64)[indices].reshape(-1, 2)
    mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(target.vertices, dtype=np.float64),
        face_matrix=np.asarray(mapped_faces, dtype=np.int32),
        v_color_matrix=np.asarray(rgba, dtype=np.float64),
        w_tex_coords_matrix=wedge_uv,
    )
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(mesh, "watertight_rebaked")
    mesh_id = mesh_set.current_mesh_id()
    old_cwd = Path.cwd()
    os.chdir(output)
    try:
        mesh_set.apply_filter(
            "transfer_attributes_to_texture_per_vertex",
            sourcemesh=mesh_id,
            targetmesh=mesh_id,
            attributeenum="Vertex Color",
            upperbound=pymeshlab.PercentageValue(2.0),
            textname="material_0.png",
            textw=texture_size,
            texth=texture_size,
            overwrite=False,
            pullpush=True,
        )
        mesh_set.save_current_mesh(
            str(output / "sample.obj"),
            save_vertex_color=False,
            save_wedge_texcoord=True,
            save_textures=True,
        )
    finally:
        os.chdir(old_cwd)

    generated_mtl = output / "sample.obj.mtl"
    final_mtl = output / "sample.mtl"
    if generated_mtl.exists():
        generated_mtl.replace(final_mtl)
        obj_path = output / "sample.obj"
        text = obj_path.read_text(encoding="utf-8", errors="replace")
        obj_path.write_text(
            text.replace("mtllib ./sample.obj.mtl", "mtllib sample.mtl"),
            encoding="utf-8",
        )
    distances = np.linalg.norm(np.asarray(target.vertices) - closest, axis=1)
    return {
        "wedge_uv_count": int(len(wedge_uv)),
        "projection_mean_distance": float(np.mean(distances)),
        "projection_p95_distance": float(np.percentile(distances, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_obj", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--quality", choices=["balanced", "high"], default="balanced")
    args = parser.parse_args()
    args.source_obj = args.source_obj.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.output = args.output.resolve()

    source = trimesh.load_mesh(
        args.source_obj, force="mesh", process=False, maintain_order=True
    )
    if getattr(source.visual, "uv", None) is None:
        raise RuntimeError("Source OBJ has no UV coordinates")
    source_stats = topology(source)
    point_count, target_faces = (
        (150000, 100000) if args.quality == "balanced" else (300000, 200000)
    )
    target = reconstruct(source, point_count, target_faces)
    target.export(args.output / "sample.stl")
    texture_path = args.source_obj.parent / "material_0.png"
    bake_stats = bake(source, target, texture_path, args.output, args.texture_size)
    report = {
        "method": "screened_poisson_xatlas_rebake",
        "quality": args.quality,
        "point_count": point_count,
        "source": source_stats,
        "result": topology(target),
        **bake_stats,
    }
    (args.output / "repair_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
