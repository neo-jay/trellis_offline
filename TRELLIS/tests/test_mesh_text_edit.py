"""CPU-only regression tests for the standalone OBJ semantic edit CLI."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh


SCRIPT = Path(os.environ.get("SEMANTIC_EDIT_SCRIPT", "/root/TRELLIS/trellis/utils/mesh_text_edit.py"))
# The module loads standalone (no package context), so its sibling imports like
# "from semantic_size_support import ..." need that directory on sys.path.
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("standalone_mesh_text_edit", SCRIPT)
assert spec and spec.loader
edit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = edit
spec.loader.exec_module(edit)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="
)
# accept_geometric_selection acknowledges the auto-proposed region. Without it
# edit_obj refuses with code=5 by design, so every happy-path test must set it.
HINT = {"bow_sign": 1, "long_axis": "z", "lateral_axis": "y", "up_axis": "x",
        "accept_geometric_selection": True}


def make_obj(folder: Path, *, negative_indices=False, multi_material=False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.icosphere(subdivisions=2)
    count = len(mesh.vertices)
    materials = ["a.mtl", "b.mtl"] if multi_material else ["sample.mtl"]
    lines = ["mtllib " + " ".join(materials) + "\n"]
    lines += ["v %.9f %.9f %.9f\n" % tuple(v) for v in mesh.vertices]
    lines += ["vt %.6f %.6f\n" % ((v[0] + 1) / 2, (v[1] + 1) / 2) for v in mesh.vertices]
    lines.append("vn 0 0 1\n")
    for face in mesh.faces:
        tokens = []
        for vertex in face:
            index = int(vertex) + 1
            if negative_indices:
                index = int(vertex) - count
            tokens.append(f"{index}/{index}/1")
        lines.append("f " + " ".join(tokens) + "\n")
    obj = folder / "sample.obj"
    obj.write_text("".join(lines), encoding="utf-8")
    for i, material in enumerate(materials):
        texture = f"texture_{i}.png"
        (folder / material).write_text(f"newmtl material_{i}\nmap_Kd {texture}\n", encoding="utf-8")
        (folder / texture).write_bytes(PNG)
    return obj


class MeshTextEditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.obj = make_obj(self.root / "source")

    def tearDown(self):
        self.temp.cleanup()

    def test_instruction_scope_and_magnitude(self):
        self.assertEqual(edit.parse_instruction("把船头变宽一些").magnitude, 0.08)
        self.assertEqual(edit.parse_instruction("把船体前段加长 10%").target, "front_hull")
        self.assertEqual(edit.parse_instruction("把桅杆拉长 8%").target, "mast")
        # 30% is the documented upper bound, so 20% is legal and must parse.
        # 35% exceeds it and must be refused.
        self.assertEqual(edit.parse_instruction("把船头变宽 20%").magnitude, 0.20)
        for text in ("把船头加宽 2 米", "把船头变宽 35%", "把轮船变大"):
            with self.subTest(text=text), self.assertRaises(edit.EditError):
                edit.parse_instruction(text)

    def test_plan_requires_direction_and_exposes_exact_selection(self):
        refused = edit.plan_edit(str(self.obj), "把船头变宽 8%")
        self.assertEqual((refused["status"], refused["code"]), ("need_region", 5))
        plan = edit.plan_edit(str(self.obj), "把船头变宽 8%", HINT)
        self.assertEqual(plan["status"], "ok")
        selection = plan["selection"]
        self.assertEqual(selection["encoding"], "base64+zlib+packbits-little")
        bits = np.frombuffer(zlib.decompress(base64.b64decode(selection["data"])), dtype=np.uint8)
        mask = np.unpackbits(bits, bitorder="little")[:selection["vertex_count"]]
        self.assertEqual(int(mask.sum()), plan["region"]["selected_vertices"])
        self.assertEqual(len(mask), 162)
        # Two independent confirmations are involved:
        #   1. orientation (axes / sign) -- the auto-locator cannot infer a
        #      usable long axis for an isotropic blob, so it reports need_region.
        #   2. the geometric region proposal -- accept_geometric_selection marks
        #      it as reviewed and clears requires_confirmation.
        # Without axes the request is refused for a different reason, so this
        # case only checks that the refusal is a clear request for orientation.
        orientation_needed = edit.plan_edit(
            str(self.obj), "把船头变宽 8%",
            {"bow_sign": 1, "accept_geometric_selection": True})
        self.assertEqual(orientation_needed["status"], "need_region")
        self.assertIn("长轴", orientation_needed["message"])

        # With axes supplied but no acknowledgement, the proposal still needs
        # review: status ok but flagged for confirmation.
        unconfirmed = edit.plan_edit(
            str(self.obj), "把船头变宽 8%",
            {"bow_sign": 1, "long_axis": "z", "lateral_axis": "y", "up_axis": "x"})
        self.assertEqual(unconfirmed["status"], "ok")
        self.assertTrue(unconfirmed["requires_confirmation"])

        # Acknowledging the proposal clears the confirmation flag.
        inferred = edit.plan_edit(
            str(self.obj), "把船头变宽 8%",
            {"bow_sign": 1, "long_axis": "z", "lateral_axis": "y", "up_axis": "x",
             "accept_geometric_selection": True})
        self.assertEqual(inferred["status"], "ok")
        self.assertFalse(inferred["requires_confirmation"])
        refused_edit = edit.edit_obj(str(self.obj), "把船头变宽 8%",
                                     str(self.root / "not_confirmed"), {"bow_sign": 1})
        self.assertEqual((refused_edit["status"], refused_edit["code"]), ("need_region", 5))

    def test_front_hull_and_mast_paths(self):
        hull = edit.plan_edit(str(self.obj), "船体前段加长 10%", HINT)
        self.assertEqual(hull["status"], "ok", hull)
        mast_hint = {
            "accept_geometric_selection": True,
            "long_axis": "x", "lateral_axis": "y", "up_axis": "z", "up_sign": 1,
            "bbox_fraction": [[0.1, 0.1, 0.2], [0.9, 0.9, 1.0]],
        }
        mast = edit.plan_edit(str(self.obj), "把桅杆拉长 8%", mast_hint)
        self.assertEqual(mast["status"], "ok", mast)
        self.assertGreater(mast["region"]["selected_vertices"], 20)
        mast_output = self.root / "mast_edited"
        mast_result = edit.edit_obj(str(self.obj), "把桅杆拉长 8%",
                                    str(mast_output), mast_hint)
        self.assertEqual(mast_result["status"], "ok", mast_result)

    def test_fragmented_mesh_requires_region_confirmation(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        left = mesh.vertices + [-3, 0, 0]
        right = mesh.vertices + [3, 0, 0]
        vertices = np.vstack((left, right))
        faces = np.vstack((mesh.faces, mesh.faces + len(left)))
        obj = self.root / "fragmented.obj"
        obj.write_text(
            "".join("v %.9f %.9f %.9f\n" % tuple(v) for v in vertices) +
            "".join("f %d %d %d\n" % tuple(f + 1) for f in faces), encoding="utf-8"
        )
        result = edit.plan_edit(str(obj), "把船头变宽 8%", HINT)
        self.assertEqual((result["status"], result["code"]), ("need_region", 5))

    def test_success_preserves_face_uv_and_material_package(self):
        obj = make_obj(self.root / "multi", negative_indices=True, multi_material=True)
        output = self.root / "edited"
        result = edit.edit_obj(str(obj), "把船头变宽 8%", str(output), HINT)
        self.assertEqual(result["status"], "ok", result)
        report = json.loads((output / "mesh_quality.json").read_text(encoding="utf-8"))
        self.assertTrue(report["deformation_checks"]["face_and_uv_indices_preserved"])
        self.assertEqual(edit._face_uv_signature(edit._read_obj(obj).lines),
                         edit._face_uv_signature(edit._read_obj(output / "edited.obj").lines))
        self.assertEqual(len(edit._read_obj(obj).normals),
                         len(edit._read_obj(output / "edited.obj").normals))
        self.assertNotEqual(edit._read_obj(obj).vertices.tolist(),
                            edit._read_obj(output / "edited.obj").vertices.tolist())
        with zipfile.ZipFile(output / "edited_obj_package.zip") as package:
            self.assertEqual(set(package.namelist()), {
                "edited.obj", "edited_0.mtl", "edited_1.mtl", "texture_0.png", "texture_1.png"
            })

    def test_missing_texture_and_existing_output_do_not_overwrite(self):
        (self.obj.parent / "texture_0.png").unlink()
        output = self.root / "edited"
        result = edit.edit_obj(str(self.obj), "把船头变宽 8%", str(output), HINT)
        self.assertEqual(result["code"], 3)
        self.assertFalse(output.exists())
        (self.obj.parent / "texture_0.png").write_bytes(PNG)
        output.mkdir()
        (output / "sentinel").write_text("keep", encoding="utf-8")
        result = edit.edit_obj(str(self.obj), "把船头变宽 8%", str(output), HINT)
        self.assertEqual(result["code"], 2)
        self.assertEqual((output / "sentinel").read_text(encoding="utf-8"), "keep")

    def test_quality_gate_refuses_new_defect_and_leaves_no_files(self):
        original_quality = edit._quality
        call_count = 0

        def defective_quality(vertices, faces):
            nonlocal call_count
            call_count += 1
            quality = original_quality(vertices, faces)
            if call_count == 2:
                quality["boundary_edges"] += 1
            return quality

        output = self.root / "edited"
        with mock.patch.object(edit, "_quality", side_effect=defective_quality):
            result = edit.edit_obj(str(self.obj), "把船头变宽 8%", str(output), HINT)
        self.assertEqual(result["code"], 9)
        self.assertFalse(output.exists())

    def test_failure_after_staging_cleans_up(self):
        output = self.root / "edited"
        with mock.patch.object(edit, "_verify_sidecars", side_effect=edit.EditError("test failure", 9)):
            result = edit.edit_obj(str(self.obj), "把船头变宽 8%", str(output), HINT)
        self.assertEqual(result["code"], 9)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".edited.stage-*")))


if __name__ == "__main__":
    unittest.main()
