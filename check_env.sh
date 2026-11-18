#!/usr/bin/env bash
# Pre-deployment environment check for the TRELLIS offline package.
#
# Run this BEFORE ./restore_offline_assets.sh on the target machine. It answers
# the questions that actually block a deployment, and it needs no root:
#   * is the CPU architecture x86_64?
#   * is there a usable NVIDIA GPU and driver?
#   * is a suitable Python + PyTorch (CUDA build) reachable by THIS user?
#   * are the compiled CUDA extensions importable?
#   * does an existing /root-style conda sit behind a 0700 directory, which would
#     make it unreachable for this user even though the files have +x?
#
# Exit code 0 = ready to deploy, 1 = at least one blocking problem.
#
# Usage:
#   ./check_env.sh [--python /path/to/python]
set -uo pipefail

PYTHON="${TRELLIS_PYTHON:-}"
if [[ "${1:-}" == "--python" ]]; then PYTHON="${2:?--python needs a path}"; fi

BLOCKERS=0
WARNINGS=0
ok()   { printf '  [ ok ]   %s\n' "$*"; }
warn() { printf '  [warn]   %s\n' "$*"; WARNINGS=$((WARNINGS+1)); }
bad()  { printf '  [FAIL]   %s\n' "$*"; BLOCKERS=$((BLOCKERS+1)); }

echo "TRELLIS offline deployment - environment check"
echo "user      : $(id -un) (uid $(id -u))"
echo "host      : $(hostname)"
echo

# ------------------------------------------------------------------ 1. arch
echo "1) CPU architecture"
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64) ok "architecture $ARCH is supported" ;;
    aarch64|arm64) bad "architecture $ARCH: the shipped wheels/extensions target x86_64.
                spconv/flash-attn/kaolin would need recompilation for ARM." ;;
    *)            bad "architecture $ARCH is untested; x86_64 is the supported target" ;;
esac
echo

# ------------------------------------------------------------------- 2. GPU
echo "2) GPU and driver"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
            | while IFS= read -r line; do ok "GPU: $line"; done
        MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [[ -n "${MEM:-}" && "$MEM" -lt 24000 ]]; then
            warn "GPU memory ${MEM} MiB is below the recommended 24000 MiB"
        fi
    else
        bad "nvidia-smi runs but reports no GPU (container started without --gpus all?)"
    fi
else
    bad "nvidia-smi not found; install the NVIDIA driver"
fi
echo

# ---------------------------------------------------------------- 3. python
echo "3) Python interpreter"
if [[ -z "$PYTHON" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    fi
fi

if [[ -z "$PYTHON" ]]; then
    bad "no python found on PATH; pass --python /path/to/python"
elif [[ ! -x "$PYTHON" ]]; then
    bad "$PYTHON is not executable by $(id -un)"
elif ! "$PYTHON" -c 'import sys' >/dev/null 2>&1; then
    bad "$PYTHON cannot run (permission denied is typical when it sits under a 0700 directory)"
else
    PYVER="$("$PYTHON" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    ok "python $PYVER at $PYTHON"
    [[ "$PYVER" == "3.10" ]] || warn "python $PYVER detected; this package is validated on 3.10"
fi
echo

# -------------------------------------------------------------- 4. pytorch
echo "4) PyTorch and CUDA"
if [[ -n "$PYTHON" && -x "$PYTHON" ]] && "$PYTHON" -c 'import sys' >/dev/null 2>&1; then
    read -r TORCH_VER CUDA_OK CUDA_VER <<<"$("$PYTHON" - <<'PY' 2>/dev/null || echo "none False none"
try:
    import torch
    print(torch.__version__, torch.cuda.is_available(), torch.version.cuda or "none")
except Exception:
    print("none", "False", "none")
PY
)"
    if [[ "$TORCH_VER" == "none" ]]; then
        bad "torch is not importable by $PYTHON"
    else
        ok "torch $TORCH_VER (cuda build $CUDA_VER)"
        if [[ "$CUDA_OK" == "True" ]]; then
            ok "torch.cuda.is_available() = True"
        else
            bad "torch.cuda.is_available() = False; PyTorch cannot see the GPU"
        fi
    fi
fi
echo

# ---------------------------------------------------------- 5. extensions
echo "5) Compiled CUDA extensions"
if [[ -n "$PYTHON" && -x "$PYTHON" ]] && "$PYTHON" -c 'import sys' >/dev/null 2>&1; then
    while IFS= read -r line; do
        case "$line" in
            ok:*)   ok   "${line#ok: }" ;;
            miss:*) bad  "${line#miss: }" ;;
        esac
    done < <("$PYTHON" - <<'PY' 2>/dev/null
mods = [("spconv", "spconv (sparse convolution)"),
        ("flash_attn", "flash_attn (attention backend)"),
        ("nvdiffrast", "nvdiffrast (rasteriser, JIT-compiles at runtime)"),
        ("kaolin", "kaolin (3D ops)"),
        ("xatlas", "xatlas (UV unwrap)"),
        ("trimesh", "trimesh"),
        ("pymeshfix", "pymeshfix"),
        ("rembg", "rembg (background removal)")]
import importlib
for mod, label in mods:
    try:
        importlib.import_module(mod)
        print("ok: %s importable" % label)
    except Exception as exc:
        print("miss: %s -> %s" % (label, type(exc).__name__))
PY
)
fi
echo

# ------------------------------------------------------- 6. build toolchain
echo "6) Build tools needed by nvdiffrast JIT"
if command -v ninja >/dev/null 2>&1; then
    ok "ninja found: $(command -v ninja)"
else
    warn "ninja not on PATH. nvdiffrast compiles at first use and will fail with
           'Ninja is required to load C++ extensions'. Install it or add conda's bin to PATH."
fi
if command -v nvcc >/dev/null 2>&1; then
    ok "nvcc found: $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1)"
else
    warn "nvcc not on PATH; nvdiffrast JIT may fail unless the extension is prebuilt"
fi
echo

# --------------------------------------------------------- 7. reachability
echo "7) Directory permissions"
for p in /root /root/miniconda3; do
    if [[ -e "$p" ]]; then
        mode="$(stat -c '%a' "$p" 2>/dev/null)"
        if [[ "$(id -u)" != "0" ]] && [[ -n "${mode:-}" ]]; then
            # a path is reachable only if every component is o+x
            if [[ "${mode: -1}" =~ [1-7] ]]; then
                ok "$p mode $mode is traversable by other users"
            else
                warn "$p mode $mode is NOT traversable by other users.
           A conda under /root cannot be used by $(id -un); install it under \$HOME or /opt."
            fi
        fi
    fi
done
echo

# ---------------------------------------------------------------- summary
echo "-----------------------------------------------------------"
if [[ $BLOCKERS -eq 0 ]]; then
    echo "RESULT: ready to deploy ($WARNINGS warning(s))"
    echo "Next: ./restore_offline_assets.sh --verify && ./restore_offline_assets.sh"
    exit 0
else
    echo "RESULT: $BLOCKERS blocking problem(s), $WARNINGS warning(s)"
    echo "Fix the [FAIL] items before deploying; the app will not start otherwise."
    exit 1
fi
