#!/usr/bin/env bash
# TRELLIS offline asset restorer -- no root required.
#
# What changed from the previous version and why:
#   * Every path used to be hardcoded under /root, so the script only worked as
#     root and silently required `sudo`.
#   * The old script unconditionally ran `tar -xzf source.tar.gz -C /root`, which
#     would overwrite an existing installation with an older code snapshot. It now
#     refuses to touch existing code unless --force is given, and backs it up.
#   * The multi-part HF cache is now joined into a temp file inside the prefix
#     instead of writing a 3 GB file next to the assets.
#
# Usage:
#   ./restore_offline_assets.sh [ASSET_DIR] [--prefix DIR] [--force] [--verify]
#
#   ASSET_DIR       where the release files were downloaded (default: script dir)
#   --prefix DIR    install root; code goes to DIR/TRELLIS, caches to DIR/.cache
#                   (default: the directory containing this script)
#   --force         allow overwriting an existing code tree (a backup is kept)
#   --verify        only check SHA-256 of the archive files, extract nothing
set -euo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

ASSET_DIR=""
PREFIX=""
FORCE=0
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)   PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
        --prefix=*) PREFIX="${1#*=}"; shift ;;
        --force)    FORCE=1; shift ;;
        --verify)   VERIFY_ONLY=1; shift ;;
        -h|--help)  sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)         echo "unknown option: $1" >&2; exit 2 ;;
        *)          ASSET_DIR="$1"; shift ;;
    esac
done

[[ -n "$ASSET_DIR" ]] || ASSET_DIR="$SELF_DIR"
PREFIX="${PREFIX:-$SELF_DIR}"

# Absolute paths, created only when actually needed.
HF_CACHE="$PREFIX/.cache/huggingface/hub"
TORCH_HUB="$PREFIX/.cache/torch/hub"
TORCH_CKPT="$TORCH_HUB/checkpoints"
U2NET_DIR="$PREFIX/.u2net"
CODE_DIR="$PREFIX/TRELLIS"

say()  { printf '%s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }

[[ -d "$ASSET_DIR" ]] || fail "asset directory not found: $ASSET_DIR"

say "assets : $ASSET_DIR"
say "prefix : $PREFIX"

# ---------------------------------------------------------------- checksums
verify_checksums() {
    local sums="$ASSET_DIR/SHA256SUMS.txt"
    [[ -f "$sums" ]] || { say "[skip] no SHA256SUMS.txt, checksums not verified"; return 0; }
    say "verifying SHA-256 ..."
    local rc=0
    while read -r want name; do
        [[ -z "${name:-}" ]] && continue
        local f="$ASSET_DIR/$name"
        if [[ ! -f "$f" ]]; then
            say "  [missing] $name"; rc=1; continue
        fi
        local got
        got="$(sha256sum "$f" | awk '{print $1}')"
        if [[ "$got" == "$want" ]]; then
            say "  [ok]      $name"
        else
            say "  [MISMATCH] $name"; rc=1
        fi
    done < "$sums"
    return $rc
}

if [[ $VERIFY_ONLY -eq 1 ]]; then
    verify_checksums && say "all checksums match" || { say "checksum verification failed" >&2; exit 1; }
    exit 0
fi

verify_checksums || fail "checksum verification failed; re-download the assets before deploying"

# ------------------------------------------------------------------ code tree
SRC_TGZ="$ASSET_DIR/trellis_offline_source.tar.gz"
[[ -f "$SRC_TGZ" ]] || fail "source archive missing: $SRC_TGZ"

if [[ -e "$CODE_DIR" && ! -d "$CODE_DIR" ]]; then
    fail "$CODE_DIR exists and is not a directory"
fi

if [[ -d "$CODE_DIR" && -n "$(ls -A "$CODE_DIR" 2>/dev/null)" ]]; then
    if [[ $FORCE -eq 0 ]]; then
        fail "existing installation found at $CODE_DIR
       Re-run with --force to overwrite it (a dated backup will be kept)."
    fi
    BACKUP="${CODE_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
    say "backing up existing code -> $BACKUP"
    mv "$CODE_DIR" "$BACKUP"
fi

mkdir -p "$PREFIX"
say "extracting source -> $PREFIX"
tar -xzf "$SRC_TGZ" -C "$PREFIX"

# The archive may nest everything under TRELLIS/ or unpack flat; normalise so
# that $PREFIX/TRELLIS/app.py always exists.
if [[ ! -f "$CODE_DIR/app.py" ]]; then
    for cand in "$PREFIX"/app.py "$PREFIX"/TRELLIS*/app.py; do
        [[ -f "$cand" ]] || continue
        mkdir -p "$CODE_DIR"
        if [[ "$(dirname "$cand")" == "$PREFIX" ]]; then
            say "normalising flat layout into $CODE_DIR"
            ( shopt -s dotglob; mv "$PREFIX"/* "$CODE_DIR"/ 2>/dev/null || true )
        fi
        break
    done
fi
[[ -f "$CODE_DIR/app.py" ]] || fail "source archive did not yield $CODE_DIR/app.py"

# ------------------------------------------------------------------- caches
mkdir -p "$HF_CACHE" "$TORCH_CKPT" "$U2NET_DIR"

# 1) HuggingFace model, shipped as split parts (GitHub limits single files).
PARTS=( "$ASSET_DIR"/TRELLIS-image-large-hf-cache.tar.part-* )
if [[ -e "${PARTS[0]}" ]]; then
    say "joining ${#PARTS[@]} HF cache part(s)"
    JOINED="$(mktemp "${TMPDIR:-/tmp}/hf-cache.XXXXXX.tar")"
    trap 'rm -f "$JOINED"' EXIT
    cat "${PARTS[@]}" > "$JOINED"
    say "extracting HF cache -> $HF_CACHE"
    tar -xf "$JOINED" -C "$HF_CACHE"
    rm -f "$JOINED"; trap - EXIT
else
    say "[skip] no HF cache parts found"
fi

# 2) DINOv2 weights and code.
if [[ -f "$ASSET_DIR/dinov2_vitl14_reg4_pretrain.pth.tar" ]]; then
    say "extracting DINOv2 checkpoint -> $TORCH_CKPT"
    tar -xf "$ASSET_DIR/dinov2_vitl14_reg4_pretrain.pth.tar" -C "$TORCH_CKPT"
else
    say "[skip] dinov2 checkpoint archive missing"
fi

if [[ -f "$ASSET_DIR/facebookresearch_dinov2_main.tar.gz" ]]; then
    say "extracting DINOv2 repo -> $TORCH_HUB"
    tar -xzf "$ASSET_DIR/facebookresearch_dinov2_main.tar.gz" -C "$TORCH_HUB"
else
    say "[skip] dinov2 repo archive missing"
fi

# 3) U2Net for background removal.
if [[ -f "$ASSET_DIR/u2net.onnx.tar" ]]; then
    say "extracting U2Net -> $U2NET_DIR"
    tar -xf "$ASSET_DIR/u2net.onnx.tar" -C "$U2NET_DIR"
else
    say "[skip] u2net archive missing"
fi

# ------------------------------------------------------------- post-checks
say ""
say "post-install checks:"
missing=0
check() { if [[ -e "$2" ]]; then say "  [ok]   $1"; else say "  [MISS] $1 -> $2"; missing=1; fi; }

check "code app.py"      "$CODE_DIR/app.py"
check "model refs/main"  "$HF_CACHE/models--JeffreyXiang--TRELLIS-image-large/refs/main"
check "dinov2 repo"      "$TORCH_HUB/facebookresearch_dinov2_main/hubconf.py"
check "dinov2 weights"   "$TORCH_CKPT/dinov2_vitl14_reg4_pretrain.pth"
check "u2net onnx"       "$U2NET_DIR/u2net.onnx"

chmod +x "$PREFIX/start.sh" 2>/dev/null || true

say ""
if [[ $missing -ne 0 ]]; then
    say "Some assets are missing; the app will refuse to start until they are restored."
    exit 1
fi

cat <<EOF
Restore complete.

Start the web demo with:
    cd "$PREFIX" && ./start.sh

If your python is not on PATH or you use a conda env, point to it explicitly:
    TRELLIS_PYTHON="$PREFIX/../miniconda3/bin/python" ./start.sh

No root privileges are required at any point.
EOF
