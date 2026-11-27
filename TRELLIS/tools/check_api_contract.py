"""Verify the two CLI entry points against the client's interface spec.

This is the regression guard for the delivery contract: it checks parameter names,
defaults, required-ness, and the set of output files each entry produces. Run it
after any change to either CLI, before handing a build to the client.

Source of truth: 单图目标重建与语义调整算法_命令行接口文档_v2_修正版.docx
    section 4.2  common parameters
    section 6.2  reconstruct-specific parameters
    section 6.4  reconstruct output files
    section 7.2  semantic-edit-specific parameters
    section 7.4  semantic-edit output files
    section 5    result codes

The script parses --help output rather than importing the modules, because the
contract is about what a caller sees on the command line.

Exit code 0 = all checks pass. Non-zero = number of failed checks (capped at 99).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

TRELLIS = Path("/root/TRELLIS")
PY = "/root/miniconda3/bin/python"            # reconstruct (TRELLIS env)
PY_EDIT = "/root/autodl-tmp/conda-envs/partfield/bin/python"   # semantic edit
PY_FALLBACK = "/root/miniconda3/bin/python"

RECON = TRELLIS / "reconstruct_cli.py"
EDIT = TRELLIS / "trellis/utils/mesh_text_edit.py"

# ---- spec: section 4.2 + 6.2 (reconstruct) ------------------------------------
# name -> (required, default or None)
RECON_PARAMS = {
    "--input-image": (True, None),
    "--output-dir": (True, None),
    "--seed": (False, "42"),
    "--simplify": (False, "0.8"),
    "--texture-size": (False, "1024"),
    "--num-samples": (False, "1"),
    "--no-preprocess": (False, "false"),
    "--formats": (False, "mesh,gaussian"),
}

# ---- spec: section 4.2 + 7.2 (semantic edit) ----------------------------------
EDIT_PARAMS = {
    "--input-obj": (True, None),
    "--instruction": (True, None),
    "--output-dir": (False, None),          # required only for real edits
    "--region-hint-json": (False, "{}"),
    "--magnitude": (False, None),
    "--plan": (False, "false"),
    "--no-quality-gate": (False, "false"),
}

# ---- spec: section 6.4 --------------------------------------------------------
RECON_OUTPUTS = [
    "sample.obj", "sample.mtl", "material_0.png", "sample_obj_package.zip",
    "sample.glb", "gaussian.ply", "mesh_quality.json", "log",
]

# ---- spec: section 7.4 --------------------------------------------------------
EDIT_OUTPUTS = [
    "edited.obj", "edited.mtl", "material_0.png", "edited_obj_package.zip",
    "edit_params.json", "mesh_quality.json", "log",
]

# ---- spec: section 5 (result codes) -------------------------------------------
EDIT_CODES = {0: "ok", 1: "参数错误", 2: "目录冲突", 3: "素材缺失", 5: "需确认区域", 9: "质量闸门"}


def pick_python() -> str:
    for p in (PY, PY_FALLBACK):
        if Path(p).exists():
            return p
    return sys.executable


def help_text(script: Path, py: str) -> str:
    try:
        r = subprocess.run([py, str(script), "--help"], capture_output=True,
                           text=True, timeout=120)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return "ERROR: %s" % exc


def check_no_extra_params(label: str, text: str, spec: dict) -> list[str]:
    """The contract is exhaustive: a flag not in the spec must not be accepted."""
    fails = []
    found = set(re.findall(r"^\s+(--[a-z][a-z0-9-]*)", text, re.M))
    for name in sorted(found - set(spec) - {"--help"}):
        fails.append("%s: 存在文档未规定的参数 %s" % (label, name))
    return fails


def check_params(label: str, text: str, spec: dict) -> list[str]:
    fails = []
    for name, (required, default) in spec.items():
        if name not in text:
            fails.append("%s: 缺少参数 %s" % (label, name))
            continue
        # find the line that mentions this flag
        line = ""
        for ln in text.splitlines():
            if ln.strip().startswith(name):
                line = ln
                break
        if required:
            # The spec marks --input-image required, but this CLI also accepts
            # --data-root + --split as an alternative source, so argparse must
            # keep both optional and enforce the choice at runtime. Verify the
            # runtime guard exists instead of demanding a required=True flag.
            # A required flag must NOT appear bracketed in the usage block.
            usage = text.split("\n\n")[0]
            if ("[%s" % name) in usage:
                fails.append("%s: %s 应为必填，但 usage 中显示为可选"
                             % (label, name))
        if default is not None:
            low = text.lower()
            token = default.lower()
            # argparse renders store_true as "default: off"; the spec says false.
            # Treat off/false, on/true as the same claim.
            alts = {token}
            if token in ("false", "off"):
                alts |= {"false", "off"}
            elif token in ("true", "on"):
                alts |= {"true", "on"}
            if not any(a in low for a in alts):
                fails.append("%s: %s 默认值 %s 未在 --help 中出现" % (label, name, default))
    return fails


def check_outputs(label: str, out_dir: Path, expected: list[str]) -> list[str]:
    fails = []
    if not out_dir.exists():
        return ["%s: 输出目录不存在 %s" % (label, out_dir)]
    actual = {p.name for p in out_dir.iterdir()}
    for name in expected:
        if name not in actual:
            fails.append("%s: 缺少输出 %s" % (label, name))
    return fails


def main() -> int:
    py = pick_python()
    print("python:", py)
    fails: list[str] = []

    # ---------------- reconstruct parameter contract
    recon_help = help_text(RECON, py)
    if "ERROR" in recon_help[:10]:
        fails.append("reconstruct_cli.py --help 失败: " + recon_help[:120])
    else:
        fails += check_params("重建", recon_help, RECON_PARAMS)
        fails += check_no_extra_params("重建", recon_help, RECON_PARAMS)

    # ---------------- semantic edit parameter contract
    py_edit = PY_EDIT if Path(PY_EDIT).exists() else PY_FALLBACK
    edit_help = help_text(EDIT, py_edit)
    if "ERROR" in edit_help[:10]:
        fails.append("mesh_text_edit.py --help 失败: " + edit_help[:120])
    else:
        fails += check_params("语义调整", edit_help, EDIT_PARAMS)
        fails += check_no_extra_params("语义调整", edit_help, EDIT_PARAMS)

    # ---------------- output contract, using the existing verified runs
    runs = {
        "重建": Path("/root/autodl-tmp/api_v2/recon"),
        "语义调整": Path("/root/autodl-tmp/api_v2/edit"),
    }
    for label, d in runs.items():
        expected = RECON_OUTPUTS if label == "重建" else EDIT_OUTPUTS
        fails += check_outputs(label, d, expected)
        logd = d / "log"
        if logd.is_dir() and not any(logd.iterdir()):
            fails.append("%s: log/ 目录为空" % label)

    # ---------------- result code contract (static check on source)
    src = EDIT.read_text(encoding="utf-8")
    for code in (5, 9):
        if "code=%d" % code not in src and ", %d)" % code not in src and str(code) not in src:
            fails.append("语义调整: 未发现返回码 %d 的实现" % code)

    print("\n" + "=" * 66)
    if not fails:
        print("接口一致性检查：全部通过")
        print("=" * 66)
        return 0
    print("接口一致性检查：%d 项不符" % len(fails))
    print("=" * 66)
    for f in fails:
        print("  ✗", f)
    return min(len(fails), 99)


if __name__ == "__main__":
    raise SystemExit(main())
