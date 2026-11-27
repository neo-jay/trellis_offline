"""Text-guided, bounded local deformation of an existing textured OBJ.

This module is intentionally independent of the TRELLIS inference pipeline and
Gradio. It keeps OBJ face/UV/material indexing and changes geometric positions
only. The deformation is a small, smooth, weighted affine field; it is not a
generative 3D editor or a general semantic segmenter.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


AXES = {"x": 0, "y": 1, "z": 2}
MAX_MAGNITUDE = 0.15
DEFAULT_MAGNITUDE = 0.08


class EditError(Exception):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EditRequest:
    target: str
    operation: str
    magnitude: float
    instruction: str


@dataclass
class ObjData:
    lines: List[str]
    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    normal_indices: np.ndarray
    vertex_line_ids: List[int]
    normal_line_ids: List[int]
    material_names: List[str]


def _legacy_parse_instruction(instruction: str, magnitude: Optional[float] = None) -> EditRequest:
    """Normalize the supported controlled vocabulary; never infer metric scale."""
    text = str(instruction or "").strip().lower()
    if not text:
        raise EditError("编辑指令不能为空")
    if any(x in text for x in ("船头", "船首", "bow")):
        target = "bow"
    elif any(x in text for x in ("桅杆", "mast")):
        target = "mast"
    elif any(x in text for x in ("船体前段", "船身前段", "前段船体", "front hull")):
        target = "front_hull"
    else:
        raise EditError("暂不支持该部件；当前支持船头、船体前段和桅杆")
    if any(x in text for x in ("变宽", "加宽", "增宽", "扩宽", "widen", "wider")):
        operation = "widen"
    elif any(x in text for x in ("拉长", "加长", "延长", "伸长", "longer", "lengthen", "taller")):
        operation = "lengthen"
    else:
        raise EditError("暂不支持该操作；当前支持局部加宽或加长")
    if (target, operation) not in {
        ("bow", "widen"), ("front_hull", "lengthen"), ("mast", "lengthen")
    }:
        raise EditError("部件与操作不匹配：船头加宽、船体前段加长或桅杆加长")
    if re.search(r"\d+(?:\.\d+)?\s*(?:米|m\b|cm\b|厘米)", text):
        raise EditError("绝对尺寸需要现实尺度标定；请改用百分比")
    matches = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[%％]", text)
    if len(matches) > 1:
        raise EditError("指令中包含多个百分比，请只保留一个")
    parsed = float(matches[0]) / 100.0 if matches else DEFAULT_MAGNITUDE
    if magnitude is not None:
        parsed = float(magnitude)
    if not np.isfinite(parsed) or not (0.0 < parsed <= MAX_MAGNITUDE):
        raise EditError(f"相对幅度须大于 0 且不超过 {MAX_MAGNITUDE:.0%}")
    return EditRequest(target, operation, parsed, instruction)


def _resolve_index(token: str, count: int) -> int:
    raw = int(token)
    index = raw - 1 if raw > 0 else count + raw
    if index < 0 or index >= count:
        raise EditError("OBJ 面索引越界", 3)
    return index


def _read_obj(path: Path) -> ObjData:
    if not path.is_file() or path.suffix.lower() != ".obj":
        raise EditError("输入 OBJ 文件不存在或扩展名不正确", 3)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    vertices, normals, faces, normal_indices = [], [], [], []
    vertex_line_ids, normal_line_ids, material_names = [], [], []
    for line_id, line in enumerate(lines):
        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                raise EditError("OBJ 顶点行不完整", 3)
            vertices.append([float(x) for x in parts[1:4]])
            vertex_line_ids.append(line_id)
        elif line.startswith("vn "):
            parts = line.split()
            if len(parts) < 4:
                raise EditError("OBJ 法线行不完整", 3)
            normals.append([float(x) for x in parts[1:4]])
            normal_line_ids.append(line_id)
        elif line.startswith("f "):
            tokens = line.split()[1:]
            if len(tokens) != 3:
                raise EditError("当前接口仅接收三角面 OBJ", 3)
            face, face_normals = [], []
            for token in tokens:
                parts = token.split("/")
                face.append(_resolve_index(parts[0], len(vertices)))
                if len(parts) >= 3 and parts[2]:
                    face_normals.append(_resolve_index(parts[2], len(normals)))
                else:
                    face_normals.append(-1)
            faces.append(face)
            normal_indices.append(face_normals)
        elif line.startswith("mtllib "):
            names = shlex.split(line[len("mtllib "):].strip())
            material_names.extend(names)
    if not vertices or not faces:
        raise EditError("OBJ 不含有效顶点或三角面", 3)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.asarray(normals, dtype=np.float64).reshape((-1, 3))
    ni = np.asarray(normal_indices, dtype=np.int64)
    if not np.isfinite(v).all() or not np.isfinite(n).all():
        raise EditError("OBJ 含非有限坐标或法线", 3)
    return ObjData(lines, v, n, f, ni, vertex_line_ids, normal_line_ids, material_names)


def _axis(value: Any, label: str) -> int:
    if isinstance(value, int) and value in (0, 1, 2):
        return value
    if isinstance(value, str) and value.lower() in AXES:
        return AXES[value.lower()]
    raise EditError(f"{label} 必须是 x、y 或 z", 5)


def _sign(value: Any, label: str) -> int:
    if value in (1, "+", "+1", "positive", "+long_axis", "+up_axis"):
        return 1
    if value in (-1, "-", "-1", "negative", "-long_axis", "-up_axis"):
        return -1
    raise EditError(f"{label} 必须是 +1 或 -1", 5)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def _selection_mask(weights: np.ndarray, threshold: float = 1e-4) -> Dict[str, Any]:
    """Exact vertex selection in a compact, frontend-decodable bit mask."""
    packed = np.packbits(weights > threshold, bitorder="little").tobytes()
    return {
        "encoding": "base64+zlib+packbits-little",
        "vertex_count": int(len(weights)),
        "threshold": threshold,
        "data": base64.b64encode(zlib.compress(packed)).decode("ascii"),
    }


def _box_weights(v: np.ndarray, bounds: np.ndarray, feather: float, skip_axis: Optional[int] = None) -> np.ndarray:
    out = np.ones(len(v), dtype=np.float64)
    for axis in range(3):
        if axis == skip_axis:
            continue
        low, high = bounds[:, axis]
        width = max(high - low, 1e-12)
        edge = max(width * feather, 1e-12)
        out *= _smoothstep((v[:, axis] - low) / edge)
        out *= _smoothstep((high - v[:, axis]) / edge)
    return out


def _box_from_hint(hint: Dict[str, Any], minimum: np.ndarray, maximum: np.ndarray) -> Optional[np.ndarray]:
    if "bbox_fraction" in hint:
        fraction = np.asarray(hint["bbox_fraction"], dtype=np.float64)
        if fraction.shape != (2, 3) or not np.isfinite(fraction).all() or np.any(fraction < 0) or np.any(fraction > 1) or np.any(fraction[0] >= fraction[1]):
            raise EditError("bbox_fraction 应是两个 0～1 的三维坐标且下界小于上界", 5)
        return minimum + fraction * (maximum - minimum)
    if "bbox" in hint:
        box = np.asarray(hint["bbox"], dtype=np.float64)
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
            raise EditError("bbox 应是两个三维坐标且下界小于上界", 5)
        return box
    return None


def _component_summary(vertices: np.ndarray, faces: np.ndarray) -> Tuple[int, float]:
    """Use SciPy's sparse graph components; avoid expensive mesh splitting."""
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
                       shape=(len(vertices), len(vertices))).tocsr()
    count, labels = connected_components(graph, directed=False)
    face_labels = labels[faces[:, 0]]
    sizes = np.bincount(face_labels, minlength=count)
    occupied = sizes[sizes > 0]
    return int(len(occupied)), float(occupied.max() / len(faces))


def _legacy_plan(data: ObjData, request: EditRequest, region_hint: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    hint = dict(region_hint or {})
    v = data.vertices
    minimum, maximum = v.min(axis=0), v.max(axis=0)
    spans = maximum - minimum
    component_count, largest_fraction = _component_summary(v, data.faces)
    if largest_fraction < 0.55 or (component_count > 100 and largest_fraction < 0.85):
        raise EditError("模型碎片化严重，无法可靠定位目标部件；请先提供质量更好的 OBJ", 5)
    long_axis = _axis(hint["long_axis"], "long_axis") if "long_axis" in hint else int(np.argmax(spans))
    lateral_axis = _axis(hint["lateral_axis"], "lateral_axis") if "lateral_axis" in hint else int(np.argsort(spans)[-2])
    up_axis = _axis(hint["up_axis"], "up_axis") if "up_axis" in hint else int(({0, 1, 2} - {long_axis, lateral_axis}).pop())
    if len({long_axis, lateral_axis, up_axis}) != 3:
        raise EditError("长轴、横轴和竖轴必须互不相同", 5)
    if spans[long_axis] <= 1e-9 or spans[lateral_axis] <= 1e-9:
        raise EditError("输入模型的主轴范围过小，无法执行局部形变", 5)
    box = _box_from_hint(hint, minimum, maximum)
    box_feather = float(hint.get("box_feather", 0.15))
    if not 0.0 < box_feather <= 0.45:
        raise EditError("box_feather 应在 (0, 0.45] 内", 5)

    meta: Dict[str, Any] = {
        "long_axis": "xyz"[long_axis], "lateral_axis": "xyz"[lateral_axis],
        "up_axis": "xyz"[up_axis], "bounds": [minimum.tolist(), maximum.tolist()],
        "bbox": box.tolist() if box is not None else None,
        "axes_confirmed": all(key in hint for key in ("long_axis", "lateral_axis", "up_axis")),
        "components": component_count, "largest_component_face_fraction": largest_fraction,
    }
    if request.target in ("bow", "front_hull"):
        if "bow_sign" not in hint and "bow_direction" not in hint:
            raise EditError("需要确认船首方向：region_hint.bow_sign 为 +1 或 -1", 5)
        bow_sign = _sign(hint.get("bow_sign", hint.get("bow_direction")), "bow_sign")
        start = float(hint.get("bow_start", 0.60 if request.target == "bow" else 0.50))
        if not 0.30 <= start <= 0.90:
            raise EditError("bow_start 应在 0.30～0.90 内", 5)
        position = (v[:, long_axis] - minimum[long_axis]) / spans[long_axis]
        if bow_sign < 0:
            position = 1.0 - position
        weights = _smoothstep((position - start) / (1.0 - start))
        if box is not None:
            weights *= _box_weights(v, box, box_feather, skip_axis=long_axis)
        if "hull_max_height_fraction" in hint:
            if "up_sign" not in hint:
                raise EditError("使用船壳高度筛选时须提供 up_sign", 5)
            limit = float(hint["hull_max_height_fraction"])
            if not 0.1 <= limit <= 1.0:
                raise EditError("hull_max_height_fraction 应在 0.1～1.0 内", 5)
            height = (v[:, up_axis] - minimum[up_axis]) / max(spans[up_axis], 1e-12)
            if _sign(hint["up_sign"], "up_sign") < 0:
                height = 1.0 - height
            weights *= 1.0 - _smoothstep((height - limit) / 0.10)
        delta = np.zeros_like(v)
        if request.target == "bow":
            centerline = float(hint.get("centerline", np.median(v[:, lateral_axis])))
            delta[:, lateral_axis] = (v[:, lateral_axis] - centerline) * request.magnitude * weights
            meta["centerline"] = centerline
        else:
            delta[:, long_axis] = bow_sign * spans[long_axis] * request.magnitude * weights
        meta.update({"bow_sign": bow_sign, "bow_start": start})
    else:
        if box is None:
            raise EditError("桅杆目标需要 bbox 或 bbox_fraction 区域提示", 5)
        if "up_sign" not in hint:
            raise EditError("桅杆加长需要确认竖直方向：region_hint.up_sign 为 +1 或 -1", 5)
        up_sign = _sign(hint["up_sign"], "up_sign")
        height = (v[:, up_axis] - box[0, up_axis]) / (box[1, up_axis] - box[0, up_axis])
        if up_sign < 0:
            height = 1.0 - height
        weights = _smoothstep(height) * _box_weights(v, box, box_feather, skip_axis=up_axis)
        delta = np.zeros_like(v)
        delta[:, up_axis] = up_sign * (box[1, up_axis] - box[0, up_axis]) * request.magnitude * weights
        meta["up_sign"] = up_sign
    selected = int(np.count_nonzero(weights > 1e-4))
    if selected < 20 or np.max(np.linalg.norm(delta, axis=1)) < 1e-9:
        raise EditError("可编辑区域过小或幅度无效，请重新确认区域", 5)
    meta.update({"selected_vertices": selected, "selected_fraction": selected / len(v),
                 "max_weight": float(weights.max())})
    return delta, weights, meta


try:
    from .semantic_size_support import parse as _parse_size, plan as _plan_size
except ImportError:
    from semantic_size_support import parse as _parse_size, plan as _plan_size


def parse_instruction(instruction, magnitude=None):
    return _parse_size(instruction, magnitude, EditError)


def _plan(data, request, region_hint):
    return _plan_size(data, request, region_hint, EditError)


def _face_vectors(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p = vertices[faces]
    return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def _quality(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    face_area2 = np.linalg.norm(_face_vectors(vertices, faces), axis=1)
    diagonal = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    return {
        "vertices": int(len(vertices)), "faces": int(len(faces)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "degenerate_faces": int(np.count_nonzero(face_area2 <= max(diagonal * diagonal * 1e-12, 1e-16))),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "self_intersections": None,
        "self_intersections_note": "未检查；闭合边界不代表没有自相交",
    }


def _new_normals(data: ObjData, vertices: np.ndarray) -> np.ndarray:
    if not len(data.normals):
        return data.normals
    face_vectors = _face_vectors(vertices, data.faces)
    accumulated = np.zeros_like(data.normals)
    for corner in range(3):
        indices = data.normal_indices[:, corner]
        valid = indices >= 0
        np.add.at(accumulated, indices[valid], face_vectors[valid])
    lengths = np.linalg.norm(accumulated, axis=1)
    present = lengths > 1e-15
    accumulated[present] /= lengths[present, None]
    accumulated[~present] = data.normals[~present]
    return accumulated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_sidecar(root: Path, relative: str) -> Path:
    if not relative or os.path.isabs(relative):
        raise EditError("材质或贴图引用必须使用相对路径", 3)
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise EditError(f"材质或贴图文件缺失/越界：{relative}", 3)
    return candidate


def _copy_materials(data: ObjData, source_dir: Path, stage: Path) -> Tuple[Dict[str, str], List[Path]]:
    if not data.material_names:
        raise EditError("输入 OBJ 未引用 MTL；此接口要求带材质模型", 3)
    mapping: Dict[str, str] = {}
    packaged: List[Path] = []
    for i, name in enumerate(dict.fromkeys(data.material_names)):
        original = _safe_sidecar(source_dir, name)
        new_name = "edited.mtl" if len(set(data.material_names)) == 1 else f"edited_{i}.mtl"
        mtl_lines = original.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        for line in mtl_lines:
            stripped = line.strip()
            if stripped.startswith(("map_", "bump ", "norm ", "disp ", "decal ")):
                parts = shlex.split(stripped)
                if len(parts) < 2:
                    continue
                texture_name = parts[-1]
                texture = _safe_sidecar(original.parent, texture_name)
                target = stage / texture_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(texture, target)
                packaged.append(target)
        target_mtl = stage / new_name
        target_mtl.write_text("".join(mtl_lines), encoding="utf-8")
        packaged.append(target_mtl)
        mapping[name] = new_name
    return mapping, list(dict.fromkeys(packaged))


def _write_obj(data: ObjData, vertices: np.ndarray, normals: np.ndarray, mapping: Dict[str, str], output: Path) -> None:
    lines = list(data.lines)
    for i, line_id in enumerate(data.vertex_line_ids):
        if np.array_equal(vertices[i], data.vertices[i]):
            continue
        original = data.lines[line_id].split()
        extras = " " + " ".join(original[4:]) if len(original) > 4 else ""
        lines[line_id] = "v " + " ".join(f"{x:.17g}" for x in vertices[i]) + extras + "\n"
    for i, line_id in enumerate(data.normal_line_ids):
        lines[line_id] = "vn " + " ".join(f"{x:.9f}" for x in normals[i]) + "\n"
    for i, line in enumerate(lines):
        if line.startswith("mtllib "):
            names = shlex.split(line[len("mtllib "):].strip())
            lines[i] = "mtllib " + " ".join(mapping[name] for name in names) + "\n"
    output.write_text("".join(lines), encoding="utf-8")


def _face_uv_signature(lines: List[str]) -> List[str]:
    """These lines must remain byte-for-byte equivalent apart from line endings."""
    return [line.rstrip("\r\n") for line in lines if line.startswith(("f ", "vt "))]


def _verify_sidecars(stage: Path, obj: ObjData, archive_path: Path, expected_files: List[Path]) -> None:
    """Verify every OBJ/MTL/texture reference and the published ZIP members."""
    if not obj.material_names:
        raise EditError("编辑版 OBJ 缺少 MTL 引用", 9)
    for material_name in obj.material_names:
        material = _safe_sidecar(stage, material_name)
        for line in material.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith(("map_", "bump ", "norm ", "disp ", "decal ")):
                parts = shlex.split(stripped)
                if len(parts) >= 2:
                    _safe_sidecar(material.parent, parts[-1])
    expected = {str(path.relative_to(stage)) for path in expected_files}
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != expected or archive.testzip() is not None:
            raise EditError("编辑版 ZIP 内容不完整或校验失败", 9)


def plan_edit(input_obj: str, instruction: str, region_hint: Optional[Dict[str, Any]] = None,
              magnitude: Optional[float] = None) -> Dict[str, Any]:
    """Read-only candidate region summary for frontend confirmation."""
    try:
        request = parse_instruction(instruction, magnitude)
        data = _read_obj(Path(input_obj))
        delta, weights, meta = _plan(data, request, {**(region_hint or {}), '_source_obj_path':str(Path(input_obj).expanduser().resolve())})
        return {"status": "ok", "code": 0, "request": asdict(request), "region": meta,
                "selection": _selection_mask(weights),
                "requires_confirmation": not meta["axes_confirmed"] or meta.get("requires_region_confirmation", False),
                "max_displacement": float(np.max(np.linalg.norm(delta, axis=1)))}
    except EditError as exc:
        return {"status": "need_region" if exc.code == 5 else "error", "code": exc.code,
                "message": str(exc)}


def edit_obj(input_obj: str, instruction: str, output_dir: str,
             region_hint: Optional[Dict[str, Any]] = None,
             magnitude: Optional[float] = None, quality_gate: bool = True) -> Dict[str, Any]:
    """Create an edited OBJ package, or return a structured refusal without writes."""
    stage: Optional[Path] = None
    try:
        source = Path(input_obj).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        if output.exists():
            raise EditError("输出目录已存在；每次编辑须使用新的独立目录", 2)
        if source.parent == output or source.is_relative_to(output):
            raise EditError("输出目录不得覆盖输入模型目录", 2)
        request = parse_instruction(instruction, magnitude)
        data = _read_obj(source)
        delta, weights, region = _plan(data, request, {**(region_hint or {}), '_source_obj_path':str(source)})
        if region.get("requires_region_confirmation"):
            raise EditError("几何候选尚未确认；先 --plan 预览，再设置 accept_geometric_selection=true，或提供部件掩码/bbox", 5)
        proposed = data.vertices + delta
        if not np.isfinite(proposed).all():
            raise EditError("形变后出现非有限坐标", 9)
        before = _quality(data.vertices, data.faces)
        after = _quality(proposed, data.faces)
        old_vectors = _face_vectors(data.vertices, data.faces)
        new_vectors = _face_vectors(proposed, data.faces)
        old_area = np.linalg.norm(old_vectors, axis=1)
        new_area = np.linalg.norm(new_vectors, axis=1)
        valid = old_area > 1e-15
        flips = int(np.count_nonzero(np.sum(old_vectors[valid] * new_vectors[valid], axis=1) <= 0))
        area_ratio = new_area[valid] / old_area[valid]
        max_ratio = float(np.max(area_ratio)) if len(area_ratio) else 0.0
        min_ratio = float(np.min(area_ratio)) if len(area_ratio) else 0.0
        diagonal = max(float(np.linalg.norm(np.ptp(data.vertices, axis=0))), 1e-12)
        max_displacement = float(np.max(np.linalg.norm(delta, axis=1)))
        checks = {
            "face_flips": flips, "min_face_area_ratio": min_ratio,
            "max_face_area_ratio": max_ratio,
            "max_displacement_relative_to_bbox_diagonal": max_displacement / diagonal,
            "changed_vertices": int(np.count_nonzero(np.linalg.norm(delta, axis=1) > 1e-9)),
            "outside_region_max_displacement": float(np.max(np.linalg.norm(delta[weights < 1e-9], axis=1))) if np.any(weights < 1e-9) else 0.0,
            "face_and_uv_indices_preserved": False,
        }
        fixed_selection = weights > 1e-4
        axis = AXES[region["edit_axis"]]
        extent_before = float(np.ptp(data.vertices[fixed_selection, axis]))
        extent_after = float(np.ptp(proposed[fixed_selection, axis]))
        checks["fixed_region_extent_before"] = extent_before
        checks["fixed_region_extent_after"] = extent_after
        checks["fixed_region_extent_relative_change"] = extent_after / max(extent_before, 1e-15) - 1
        core = weights >= .9 * weights.max()
        core_before = float(np.ptp(data.vertices[core, axis]))
        core_after = float(np.ptp(proposed[core, axis]))
        checks["fixed_core_extent_before"] = core_before
        checks["fixed_core_extent_after"] = core_after
        checks["fixed_core_extent_relative_change"] = core_after / max(core_before, 1e-15) - 1
        checks["self_intersections_checked"] = False
        geometric_edges = np.unique(np.sort(np.vstack((data.faces[:,[0,1]], data.faces[:,[1,2]], data.faces[:,[2,0]])),axis=1),axis=0)
        original_lengths = np.linalg.norm(data.vertices[geometric_edges[:,0]]-data.vertices[geometric_edges[:,1]],axis=1)
        new_lengths = np.linalg.norm(proposed[geometric_edges[:,0]]-proposed[geometric_edges[:,1]],axis=1)
        valid_edges = original_lengths > diagonal*1e-12
        stretch = new_lengths[valid_edges]/original_lengths[valid_edges]
        checks["edge_length_ratio_p05_p50_p95"] = np.percentile(stretch,[5,50,95]).tolist() if len(stretch) else []
        cosine = np.sum(old_vectors[valid]*new_vectors[valid],axis=1)/np.maximum(old_area[valid]*new_area[valid],1e-30)
        normal_turn = np.degrees(np.arccos(np.clip(cosine,-1,1)))
        checks["face_normal_turn_degrees_p50_p95_max"] = np.percentile(normal_turn,[50,95,100]).tolist() if len(normal_turn) else []
        intersection_result = None
        if (region_hint or {}).get("self_intersection_check", False):
            with tempfile.TemporaryDirectory(prefix="trellis-intersection-") as temp:
                temp = Path(temp)
                np.savez(temp / "input.npz", before=data.vertices, after=proposed, faces=data.faces)
                worker = Path(__file__).with_name("mesh_intersection_check.py")
                try:
                    completed = subprocess.run([sys.executable, str(worker), str(temp / "input.npz"), str(temp / "result.json")],
                                               capture_output=True, text=True, timeout=60)
                except subprocess.TimeoutExpired:
                    raise EditError("自相交检测超时；未跳过检查或发布模型", 9)
                if completed.returncode != 0 or not (temp / "result.json").exists():
                    raise EditError("独立自相交检测失败；未发布模型：" + completed.stderr[-500:], 9)
                intersection_result = json.loads((temp / "result.json").read_text())
            checks["self_intersections_checked"] = True
            checks["new_intersecting_faces"] = len(intersection_result["new_intersecting_face_indices"])
            region["self_intersections_checked"] = True
        failures = []
        if intersection_result and intersection_result["new_intersecting_face_indices"]:
            failures.append("新增自相交面")
        for key in ("boundary_edges", "nonmanifold_edges", "degenerate_faces"):
            if after[key] > before[key]:
                failures.append(f"新增 {key}")
        if flips:
            failures.append("出现翻面")
        if min_ratio < 0.20 or max_ratio > 3.0:
            failures.append("局部面片面积变化过大")
        if max_displacement / diagonal > 0.12:
            failures.append("顶点最大位移过大")
        if checks["changed_vertices"] < 20:
            failures.append("有效编辑顶点不足")
        if quality_gate and failures:
            raise EditError("质量闸门未通过：" + "、".join(failures), 9)

        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
        mapping, sidecars = _copy_materials(data, source.parent, stage)
        obj_file = stage / "edited.obj"
        _write_obj(data, proposed, _new_normals(data, proposed), mapping, obj_file)
        output_check = _read_obj(obj_file)
        # The gate must also hold after OBJ decimal serialization, not only in memory.
        serialized_quality = _quality(output_check.vertices, output_check.faces)
        if np.any(weights == 0) and not np.array_equal(output_check.vertices[weights == 0], data.vertices[weights == 0]):
            raise EditError("OBJ 回读固定区域发生位移，未发布结果", 9)
        serialized_flips = int(np.count_nonzero(np.sum(old_vectors[valid] * _face_vectors(output_check.vertices, output_check.faces)[valid], axis=1) <= 0))
        if serialized_flips or serialized_quality["degenerate_faces"] > before["degenerate_faces"]:
            raise EditError("OBJ 回读出现翻面/退化，未发布结果", 9)
        if not np.array_equal(output_check.faces, data.faces) or len(output_check.vertices) != len(data.vertices):
            raise EditError("OBJ 回读校验失败，未发布编辑结果", 9)
        checks["face_and_uv_indices_preserved"] = (
            _face_uv_signature(data.lines) == _face_uv_signature(output_check.lines)
        )
        if not checks["face_and_uv_indices_preserved"]:
            raise EditError("面片或 UV 索引发生变化，未发布编辑结果", 9)
        report = {
            "request": asdict(request), "region": region,
            "input_obj": str(source), "input_sha256": _sha256(source),
            "edited_obj_sha256": _sha256(obj_file),
            "quality_before": before, "quality_after": after,
            "deformation_checks": checks, "quality_gate_passed": not failures,
            "quality_gate_notes": failures,
            "method": ("affine core with screened graph harmonic transition" if region.get("deformation_solver")=="harmonic_transition" else "bounded smoothstep-weighted local affine deformation"),
            "topology_note": "不增删面；自相交检测为可选的面集合检查，不能等同于完整几何无缺陷证明。",
            "quality_after_serialization": serialized_quality,
            "selection": _selection_mask(weights),
            "self_intersection_check": intersection_result,
            "artifact_manifest": [{"file":str(p.relative_to(stage)),"sha256":_sha256(p),"bytes":p.stat().st_size}
                                  for p in [obj_file, *sidecars]],
            "runtime": {"python":sys.version.split()[0],"numpy":np.__version__,"trimesh":trimesh.__version__},
        }
        quality_file = stage / "mesh_quality.json"
        quality_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        params_file = stage / "edit_params.json"
        params_file.write_text(json.dumps({"request": asdict(request), "region_hint": region_hint or {},
                                           "source_sha256": report["input_sha256"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        # --- interface spec 7.4: log/ directory -------------------------------
        # The spec lists log/ as part of the output contract. Write both a human
        # readable event log and a JSON summary; do it before the staging rename
        # so the published directory is always complete.
        log_dir = stage / "log"
        log_dir.mkdir(exist_ok=True)
        _events = [
            ("input", {"obj": str(source), "sha256": report["input_sha256"]}),
            ("instruction", {"text": instruction, "parsed": asdict(request)}),
            ("region", {"selected_vertices": region.get("selected_vertices"),
                        "selected_fraction": region.get("selected_fraction"),
                        "deformation_solver": region.get("deformation_solver")}),
            ("quality_gate", {"passed": not failures, "failures": failures}),
            ("result", {"status": "ok", "edited_sha256": report["edited_obj_sha256"]}),
        ]
        _log_lines = []
        for _i, (_kind, _payload) in enumerate(_events, 1):
            _log_lines.append("%02d %-12s %s" % (
                _i, _kind, json.dumps(_payload, ensure_ascii=False)))
        (log_dir / "edit.log").write_text(
            "\n".join(_log_lines) + "\n", encoding="utf-8")
        (log_dir / "run.json").write_text(json.dumps(
            {"events": [{"seq": i, "kind": k, "payload": p}
                        for i, (k, p) in enumerate(_events, 1)]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        # ---------------------------------------------------------------------

        zip_file = stage / "edited_obj_package.zip"
        with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in [obj_file, *sidecars]:
                archive.write(file, arcname=file.relative_to(stage))
        _verify_sidecars(stage, output_check, zip_file, [obj_file, *sidecars])
        if output.exists():
            raise EditError("输出目录在处理过程中已被占用", 2)
        stage.rename(output)
        stage = None
        return {"status": "ok", "code": 0, "message": "编辑完成",
                "obj_path": str(output / "edited.obj"),
                "package_path": str(output / "edited_obj_package.zip"),
                "quality_path": str(output / "mesh_quality.json"),
                "params_path": str(output / "edit_params.json"), "region": region}
    except EditError as exc:
        return {"status": "need_region" if exc.code == 5 else "error", "code": exc.code,
                "message": str(exc), "obj_path": None, "package_path": None,
                "diagnostics": {"checks": locals().get("checks"),
                                "dimension_control": locals().get("region", {}).get("dimension_control")}}
    except Exception as exc:
        return {"status": "error", "code": 9, "message": f"处理异常：{type(exc).__name__}: {exc}",
                "obj_path": None, "package_path": None}
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Edit an existing textured OBJ with a bounded local geometry change")
    parser.add_argument("--input-obj", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--region-hint-json", default="{}",
                        help="JSON object or path to JSON file (default: {})")
    parser.add_argument("--magnitude", type=float,
                        help="relative change, 0.08 = +8%% (default: parsed from --instruction)")
    parser.add_argument("--plan", action="store_true",
                        help="only inspect the candidate region, write nothing (default: off)")
    parser.add_argument("--no-quality-gate", action="store_true",
                        help="diagnostic only; delivery must keep the gate (default: off)")
    args = parser.parse_args()
    try:
        hint_text = args.region_hint_json
        if not hint_text.lstrip().startswith("{") and Path(hint_text).is_file():
            hint_text = Path(hint_text).read_text(encoding="utf-8")
        hint = json.loads(hint_text)
        if not isinstance(hint, dict):
            raise ValueError("region_hint 须为 JSON 对象")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"region_hint_json 无效：{exc}")
    if args.plan:
        result = plan_edit(args.input_obj, args.instruction, hint, args.magnitude)
    else:
        if not args.output_dir:
            parser.error("非 --plan 模式须提供 --output-dir")
        result = edit_obj(args.input_obj, args.instruction, args.output_dir, hint,
                          args.magnitude, not args.no_quality_gate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["code"]


if __name__ == "__main__":
    raise SystemExit(main())
