"""Texture-preserving OBJ writer that does not split geometric UV seams."""

import numpy as np
from collections import defaultdict, deque
from pathlib import Path
import subprocess
import sys
import tempfile
from scipy.spatial import cKDTree
from trimesh.exchange.obj import export_obj


def export_obj_shared_geometry(mesh, mtl_name='sample.mtl'):
    """Return OBJ/MTL/texture with shared v indices and independent vt/vn indices.

    GLB stores one vertex per position/UV combination.  A plain GLB -> OBJ
    conversion often duplicates the geometric position at every UV seam,
    making topology tools report thousands of artificial open edges.  OBJ can
    instead reference one position with multiple texture and normal indices.
    """
    _, sidecars = export_obj(
        mesh, include_normals=True, include_texture=True,
        return_texture=True, mtl_name=mtl_name)
    uv = getattr(mesh.visual, 'uv', None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise ValueError('GLB mesh has no per-corner-compatible UV coordinates')

    positions, inverse = np.unique(np.asarray(mesh.vertices), axis=0, return_inverse=True)
    geometry_faces = inverse[np.asarray(mesh.faces, dtype=np.int64)]
    normals = np.asarray(mesh.vertex_normals)
    if len(normals) != len(mesh.vertices):
        raise ValueError('GLB mesh normals do not match its UV vertices')

    material_name = 'material_0'
    mtl = sidecars.get(mtl_name)
    if mtl is not None:
        source = mtl.decode('utf-8') if isinstance(mtl, bytes) else mtl
        for line in source.splitlines():
            if line.startswith('newmtl '):
                material_name = line.split(maxsplit=1)[1].strip()
                break

    lines = ['# TRELLIS textured mesh; shared geometry across UV seams',
             f'mtllib {mtl_name}', 'o geometry_0', f'usemtl {material_name}']
    lines.extend('v {:.9g} {:.9g} {:.9g}'.format(*point) for point in positions)
    lines.extend('vt {:.9g} {:.9g}'.format(*point) for point in uv)
    lines.extend('vn {:.9g} {:.9g} {:.9g}'.format(*normal) for normal in normals)
    for geo_face, uv_face in zip(geometry_faces, np.asarray(mesh.faces, dtype=np.int64)):
        lines.append('f ' + ' '.join(
            f'{int(g) + 1}/{int(t) + 1}/{int(t) + 1}' for g, t in zip(geo_face, uv_face)))
    return '\n'.join(lines) + '\n', sidecars, positions, geometry_faces


def _topology_defects(faces):
    edges = np.sort(np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def _remove_degenerate_faces(obj_text, vertices, faces, source_uv_faces):
    """Drop only invalid triangles, keeping every surviving OBJ face line/UV."""
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('OBJ geometry must contain triangular faces')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('OBJ geometry contains invalid vertex coordinates')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('OBJ geometry contains invalid face indices')
    if len(source_uv_faces) != len(faces):
        raise ValueError('OBJ UV faces do not match geometric faces')

    diagonal = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    points = vertices[faces]
    double_areas = np.linalg.norm(np.cross(
        points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    distinct = ((faces[:, 0] != faces[:, 1]) &
                (faces[:, 1] != faces[:, 2]) &
                (faces[:, 2] != faces[:, 0]))
    keep = distinct & (double_areas > diagonal * diagonal * 1e-14)
    removed = int((~keep).sum())
    if not removed:
        return obj_text, faces, source_uv_faces, 0
    if not keep.any():
        raise ValueError('OBJ geometry contains no nondegenerate faces')

    # The writer emits one face line per geometric face in the same order.
    # Preserve all vertex, UV, normal, and material declarations verbatim.
    lines = obj_text.splitlines(keepends=True)
    face_lines = [i for i, line in enumerate(lines) if line.startswith('f ')]
    if len(face_lines) != len(faces):
        raise ValueError('OBJ text faces do not match geometric faces')
    for face_index, line_index in reversed(list(enumerate(face_lines))):
        if not keep[face_index]:
            del lines[line_index]
    return ''.join(lines), faces[keep], source_uv_faces[keep], removed


def _meshfix_candidate(method, vertices, faces, timeout_sec=180):
    """Keep MeshFix's native-code failures outside the long-lived web process."""
    worker = Path(__file__).with_name('meshfix_worker.py')
    with tempfile.TemporaryDirectory(prefix='trellis_meshfix_') as temp_dir:
        input_path = Path(temp_dir) / 'input.npz'
        output_path = Path(temp_dir) / 'output.npz'
        np.savez(input_path, vertices=vertices, faces=faces.astype(np.int32))
        try:
            result = subprocess.run(
                [sys.executable, str(worker), method, str(input_path), str(output_path)],
                capture_output=True, text=True, timeout=timeout_sec, check=False)
        except subprocess.TimeoutExpired:
            return None, None, f'{method}: native repair timed out after {timeout_sec}s'
        if result.returncode != 0 or not output_path.is_file():
            detail = (result.stderr or result.stdout or '').strip()[-500:]
            return None, None, f'{method}: native repair exited {result.returncode}: {detail}'
        with np.load(output_path, allow_pickle=False) as data:
            return np.asarray(data['vertices']), np.asarray(data['faces']), None


def close_obj_geometry(mesh, obj_text, vertices, faces, max_removed_ratio=0.05):
    """Conservatively close an OBJ while retaining original textured face UVs.

    A cap-only repair is preferred.  A full PyMeshFix repair is accepted only
    when it closes every edge and makes a small, bounded change to face sets.
    The texture image is never resampled, and unchanged faces keep their
    original UV and normal indices.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    source_uv_faces = np.asarray(mesh.faces, dtype=np.int64)
    original_boundary, original_nonmanifold = _topology_defects(faces)
    metadata = {
        'method': 'none', 'original_boundary_edges': original_boundary,
        'original_nonmanifold_edges': original_nonmanifold,
    }
    obj_text, faces, source_uv_faces, removed_degenerate = _remove_degenerate_faces(
        obj_text, vertices, faces, source_uv_faces)
    metadata['removed_degenerate_faces'] = removed_degenerate
    if removed_degenerate:
        metadata['method'] = 'remove_degenerate'
    boundary_after_cleanup, nonmanifold_after_cleanup = _topology_defects(faces)
    metadata['cleaned_boundary_edges'] = boundary_after_cleanup
    metadata['cleaned_nonmanifold_edges'] = nonmanifold_after_cleanup
    if boundary_after_cleanup == 0 and nonmanifold_after_cleanup == 0:
        return obj_text, vertices, faces, metadata

    diagonal = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    original_keys = defaultdict(deque)
    for index, face in enumerate(faces):
        original_keys[tuple(sorted(face))].append(index)

    methods = ['cap_only', 'local_repair'] if nonmanifold_after_cleanup == 0 else ['local_repair']
    preferred_uv = np.zeros(len(vertices), dtype=np.int64)
    assigned = np.zeros(len(vertices), dtype=bool)
    for geometric_face, uv_face in zip(faces, source_uv_faces):
        for geometric_vertex, uv_vertex in zip(geometric_face, uv_face):
            if not assigned[geometric_vertex]:
                preferred_uv[geometric_vertex] = uv_vertex
                assigned[geometric_vertex] = True

    worker_errors = []
    for method in methods:
        candidate_vertices, candidate_faces, worker_error = _meshfix_candidate(
            method, vertices, faces)
        if worker_error:
            worker_errors.append(worker_error)
            continue
        candidate_vertices = np.asarray(candidate_vertices, dtype=np.float64)
        candidate_faces = np.asarray(candidate_faces, dtype=np.int64)
        if not len(candidate_vertices) or not len(candidate_faces):
            continue
        distances, original_vertex_ids = cKDTree(vertices).query(candidate_vertices)
        if (distances.max() / diagonal > 1e-5 or
                len(np.unique(original_vertex_ids)) != len(candidate_vertices)):
            continue
        mapped_faces = original_vertex_ids[candidate_faces]
        boundary, nonmanifold = _topology_defects(mapped_faces)
        if boundary or nonmanifold:
            continue
        areas = np.linalg.norm(np.cross(
            vertices[mapped_faces[:, 1]] - vertices[mapped_faces[:, 0]],
            vertices[mapped_faces[:, 2]] - vertices[mapped_faces[:, 0]]), axis=1)
        if np.any(areas <= diagonal * diagonal * 1e-12):
            continue

        face_queues = {key: deque(indices) for key, indices in original_keys.items()}
        matched = []
        for face in mapped_faces:
            queue = face_queues.get(tuple(sorted(face)))
            matched.append(queue.popleft() if queue else None)
        preserved = sum(index is not None for index in matched)
        removed_ratio = (len(faces) - preserved) / len(faces)
        added_ratio = (len(mapped_faces) - preserved) / len(faces)
        if removed_ratio > max_removed_ratio or added_ratio > 0.02:
            continue

        header = [line for line in obj_text.splitlines() if line.startswith(
            ('#', 'mtllib ', 'o ', 'usemtl ', 'v ', 'vt ', 'vn '))]
        normal_count = sum(line.startswith('vn ') for line in header)
        new_normals = []
        face_lines = []
        for face, source_index in zip(mapped_faces, matched):
            if source_index is not None:
                source_geometry = faces[source_index]
                source_uv = source_uv_faces[source_index]
                uv_by_vertex = dict(zip(source_geometry, source_uv))
                corners = [(int(vertex), int(uv_by_vertex[vertex]), int(uv_by_vertex[vertex]))
                           for vertex in face]
            else:
                a, b, c = vertices[face]
                normal = np.cross(b - a, c - a)
                normal /= np.linalg.norm(normal)
                new_normals.append('vn {:.9g} {:.9g} {:.9g}'.format(*normal))
                normal_index = normal_count + len(new_normals) - 1
                corners = [(int(vertex), int(preferred_uv[vertex]), normal_index)
                           for vertex in face]
            face_lines.append('f ' + ' '.join(
                f'{vertex + 1}/{uv_index + 1}/{normal_index + 1}'
                for vertex, uv_index, normal_index in corners))
        repaired_obj = '\n'.join(header + new_normals + face_lines) + '\n'
        metadata.update({
            'method': method,
            'removed_original_faces': len(faces) - preserved,
            'new_faces': len(mapped_faces) - preserved,
            'max_relative_vertex_displacement': float(distances.max() / diagonal),
        })
        return repaired_obj, vertices, mapped_faces, metadata

    metadata['reason'] = '未找到满足闭合和形状改动门槛的修复结果'
    if worker_errors:
        metadata['native_repair_errors'] = worker_errors
    return obj_text, vertices, faces, metadata
