"""Mesh quality checks and conservative, training-free SDF refinement."""

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def inspect_mesh(vertices, faces, weld_tolerance=None):
    """Report topology on geometric positions; ignore duplicated UV seam indices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    report = {
        'vertices': int(len(vertices)), 'faces': int(len(faces)),
        'boundary_edges': 0, 'nonmanifold_edges': 0,
        'degenerate_faces': 0, 'components': 0,
        'boundary_loops': 0, 'nonclosed_regions': 0,
        'local_topology_anomalies': 0,
        'watertight': False, 'winding_consistent': False,
        'self_intersections': None,
        'self_intersections_note': '未检查：当前环境未安装可靠的自相交检测后端',
    }
    if len(vertices) == 0 or len(faces) == 0:
        report['error'] = '空网格'
        return report
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        report['error'] = '坐标非有限值或面索引越界'
        return report

    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    centered = vertices - vertices.mean(axis=0)
    axis_std = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(centered.T)), 0.0))
    report['flatness_ratio'] = float(axis_std[0] / max(axis_std[-1], 1e-12))
    tolerance = weld_tolerance if weld_tolerance is not None else max(diagonal * 1e-7, 1e-9)
    _, canonical_indices, welded = np.unique(
        np.round(vertices / tolerance).astype(np.int64), axis=0,
        return_index=True, return_inverse=True)
    geometric_faces = welded[faces]
    raw_edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, raw_counts = np.unique(raw_edges, axis=0, return_counts=True)
    report['raw_index_boundary_edges'] = int((raw_counts == 1).sum())
    triangles = vertices[faces]
    doubled_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    degenerate = (doubled_area <= max(diagonal * diagonal * 1e-12, 1e-16)) | (
        (geometric_faces[:, 0] == geometric_faces[:, 1]) |
        (geometric_faces[:, 1] == geometric_faces[:, 2]) |
        (geometric_faces[:, 2] == geometric_faces[:, 0])
    )
    report['degenerate_faces'] = int(degenerate.sum())
    edges = np.sort(np.concatenate((geometric_faces[:, [0, 1]], geometric_faces[:, [1, 2]], geometric_faces[:, [2, 0]])), axis=1)
    unique_edges, edge_inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True)
    report['boundary_edges'] = int((counts == 1).sum())
    report['nonmanifold_edges'] = int((counts > 2).sum())
    report['local_topology_anomalies'] = report['nonmanifold_edges'] + report['degenerate_faces']
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges):
        boundary_vertices, boundary_inverse = np.unique(
            boundary_edges.reshape(-1), return_inverse=True)
        boundary_graph_edges = boundary_inverse.reshape(-1, 2)
        boundary_graph = coo_matrix(
            (np.ones(len(boundary_edges) * 2, dtype=np.int8),
             (np.r_[boundary_graph_edges[:, 0], boundary_graph_edges[:, 1]],
              np.r_[boundary_graph_edges[:, 1], boundary_graph_edges[:, 0]])),
            shape=(len(boundary_vertices), len(boundary_vertices))).tocsr()
        boundary_count, boundary_labels = connected_components(boundary_graph, directed=False)
        vertex_degrees = np.bincount(boundary_graph_edges.reshape(-1), minlength=len(boundary_vertices))
        for component in range(boundary_count):
            members = boundary_labels == component
            edge_count = int(np.count_nonzero(members[boundary_graph_edges[:, 0]]))
            if edge_count == int(members.sum()) and np.all(vertex_degrees[members] == 2):
                report['boundary_loops'] += 1
    report['watertight'] = bool(report['boundary_edges'] == 0 and report['nonmanifold_edges'] == 0 and report['degenerate_faces'] == 0)
    report['bbox_diagonal'] = diagonal
    report['weld_tolerance'] = tolerance
    try:
        mesh = trimesh.Trimesh(vertices=vertices[canonical_indices], faces=geometric_faces, process=False)
        report['winding_consistent'] = bool(mesh.is_winding_consistent)
        face_components = list(trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(faces))))
        report['components'] = int(len(face_components))
        face_labels = np.empty(len(faces), dtype=np.int64)
        for component, face_ids in enumerate(face_components):
            face_labels[np.asarray(face_ids, dtype=np.int64)] = component
        boundary_face_ids = np.tile(np.arange(len(faces)), 3)[counts[edge_inverse] == 1]
        report['nonclosed_regions'] = int(len(np.unique(face_labels[boundary_face_ids])))
    except Exception as exc:
        report['analysis_warning'] = str(exc)[:250]
    return report


def mesh_displacement(reference_vertices, candidate_vertices, max_samples=5000):
    """Symmetric 95th-percentile nearest-surface-vertex distance / bbox diagonal."""
    a = np.asarray(reference_vertices, dtype=np.float64)
    b = np.asarray(candidate_vertices, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    diagonal = max(float(np.linalg.norm(np.ptp(a, axis=0))), 1e-9)
    a_sample = a[np.linspace(0, len(a) - 1, min(len(a), max_samples), dtype=int)]
    b_sample = b[np.linspace(0, len(b) - 1, min(len(b), max_samples), dtype=int)]
    distances = np.r_[cKDTree(b).query(a_sample)[0], cKDTree(a).query(b_sample)[0]]
    return float(np.percentile(distances, 95) / diagonal)


def defect_score(report):
    if report.get('error'):
        return float('inf')
    return (report['boundary_edges'] + 3 * report['nonmanifold_edges'] +
            2 * report['degenerate_faces']) / max(report['faces'], 1)


@torch.no_grad()
def constrained_sdf_candidate(sdf, resolution, steps=3, blend=0.2):
    """Bounded narrow-band diffusion: minimize local field roughness, keep the initial surface.

    This is per-sample inference optimization, not model training.  The maximum
    SDF change is one quarter of a grid cell so thin parts are less likely to
    disappear.  Topology is measured after extraction; no closure is assumed.
    """
    side = resolution + 1
    original = sdf.reshape(1, 1, side, side, side).float()
    current = original.clone()
    voxel = 1.0 / resolution
    active = original.abs() <= 2.0 * voxel
    for _ in range(steps):
        neighborhood = F.avg_pool3d(current, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        proposal = torch.lerp(current, neighborhood, blend)
        current = torch.where(active, torch.clamp(proposal, original - 0.25 * voxel, original + 0.25 * voxel), original)
    return current.reshape(-1).to(dtype=sdf.dtype)
