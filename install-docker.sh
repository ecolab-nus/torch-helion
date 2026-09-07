#!/usr/bin/env bash
# Install the torch-helion workspace inside the Loom Docker development image.
#
# torch-helion reuses Loom's toolchain image (ftod/loom_dev) and Loom's build
# steps, but keeps a single virtual environment at the torch-helion root that
# contains loom, helion-mlir (PyTorch/Helion/Triton/torch-mlir), loom-dataflow
# and torch_helion itself.
#
# Usage:
#   bash install-docker.sh [OPTIONS]
#
# Options:
#   --clean             Remove generated files before installing
#   --clean-only        Remove generated files and exit
#   --skip-mlar         Skip the loom-mlar Rust evaluator
#   --skip-dataflow     Skip loom-dataflow and loom2ttkernel
#   --skip-helion       Skip helion-mlir, and with it PyTorch and Helion
#   --skip-ttkernel     Skip the optional loom2ttkernel backend
#   --rebuild-dataflow  Force reinstalling the loom-dataflow Python extension
#   --help              Show this help
#
# Environment:
#   LOOM_REPO_URL       Git URL used when third_party/loom is missing
#   LOOM_EVAL_SYSTEM    Path to a prebuilt eval_system binary

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOOM_ROOT="$REPO_ROOT/third_party/loom"
LOOM_REPO_URL="${LOOM_REPO_URL:-https://github.com/ecolab-nus/loom.git}"

CLEAN=0
CLEAN_ONLY=0
SKIP_MLAR=0
SKIP_DATAFLOW=0
SKIP_HELION=0
SKIP_TTKERNEL=0
REBUILD_DATAFLOW=0

for arg in "$@"; do
    case "$arg" in
        --clean)             CLEAN=1 ;;
        --clean-only)        CLEAN=1; CLEAN_ONLY=1 ;;
        --skip-mlar)         SKIP_MLAR=1 ;;
        --skip-dataflow)     SKIP_DATAFLOW=1 ;;
        --skip-helion)       SKIP_HELION=1 ;;
        --skip-ttkernel)     SKIP_TTKERNEL=1 ;;
        --rebuild-dataflow)  REBUILD_DATAFLOW=1 ;;
        --help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (see --help)" >&2
            exit 1
            ;;
    esac
done

if [ "$SKIP_DATAFLOW" = "1" ]; then
    SKIP_TTKERNEL=1
fi

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_file() {
    [ -f "$1" ] || fail "required file not found: $1"
}

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

remove_generated_path() {
    local generated_path="$1"

    case "$generated_path" in
        "$REPO_ROOT"/*) ;;
        *) fail "refusing to clean path outside the repository: $generated_path" ;;
    esac

    if [ ! -e "$generated_path" ] && [ ! -L "$generated_path" ]; then
        return
    fi

    echo "  remove ${generated_path#"$REPO_ROOT"/}"
    rm -rf -- "$generated_path"
}

clean_workspace() {
    local generated_path

    echo ""
    echo "=== Cleaning generated workspace files ==="

    for generated_path in \
        "$REPO_ROOT/.venv" \
        "$REPO_ROOT/.uv-cache" \
        "$REPO_ROOT/.uv-python" \
        "$REPO_ROOT/.cargo-home" \
        "$REPO_ROOT/.pytest_cache" \
        "$REPO_ROOT/.ruff_cache" \
        "$REPO_ROOT/.mypy_cache" \
        "$REPO_ROOT/build" \
        "$REPO_ROOT/dist" \
        "$REPO_ROOT/tmp_logs" \
        "$REPO_ROOT/tmp_output" \
        "$REPO_ROOT/test"; do
        remove_generated_path "$generated_path"
    done

    while IFS= read -r -d '' generated_path; do
        remove_generated_path "$generated_path"
    done < <(
        find "$REPO_ROOT/src" "$REPO_ROOT/tests" "$REPO_ROOT/models" -xdev \
            -type d \( -name '__pycache__' -o -name '*.egg-info' \) \
            -prune -print0 2>/dev/null
    )

    # Loom owns the artifacts under third_party/loom; reuse its own cleaner so
    # this script never has to track Loom's generated paths.
    if [ -f "$LOOM_ROOT/install-docker.sh" ]; then
        echo "  delegating third_party/loom cleanup to Loom"
        bash "$LOOM_ROOT/install-docker.sh" --clean-only \
            | sed 's/^/  loom: /'
    fi

    echo "Workspace cleanup complete"
}

if [ "$CLEAN" = "1" ]; then
    clean_workspace
    if [ "$CLEAN_ONLY" = "1" ]; then
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------

echo ""
echo "=== Checking Docker toolchain ==="
for command_name in git uv cmake ninja clang++ ld.lld cargo rustc; do
    require_command "$command_name"
done

: "${MLIR_DIR:?MLIR_DIR is not set; use the loom_dev Docker image}"
: "${LLVM_DIR:?LLVM_DIR is not set; use the loom_dev Docker image}"
: "${TTMLIR_SOURCE_DIR:?TTMLIR_SOURCE_DIR is not set; use the loom_dev Docker image}"
: "${TTMLIR_BUILD_DIR:?TTMLIR_BUILD_DIR is not set; use the loom_dev Docker image}"

require_file "$MLIR_DIR/MLIRConfig.cmake"
require_file "$LLVM_DIR/LLVMConfig.cmake"
require_file "$TTMLIR_BUILD_DIR/bin/ttmlir-opt"
require_file "$TTMLIR_BUILD_DIR/lib/libMLIRTTKernelDialect.a"
require_file "$TTMLIR_BUILD_DIR/lib/libMLIRTTMetalDialect.a"

printf "  %-18s %s\n" "uv" "$(uv --version)"
printf "  %-18s %s\n" "MLIR_DIR" "$MLIR_DIR"
printf "  %-18s %s\n" "TTMLIR_BUILD_DIR" "$TTMLIR_BUILD_DIR"

# The image keeps tt-mlir's Python venv active for compiler tools. torch-helion
# uses its own .venv, so hide VIRTUAL_ENV from uv and copy from caches across
# mount boundaries instead of attempting unsupported hardlinks. Keep uv's
# managed Python and cache beside the checkout so they persist with the volume.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO_ROOT/.uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"

# The image's Cargo installation is readable but its registry is owned by root.
# Use a workspace-local Cargo home when the configured one is not writable,
# while keeping /opt/cargo/bin on PATH for the toolchain itself.
if [ -z "${CARGO_HOME:-}" ] || [ ! -w "$CARGO_HOME" ]; then
    export CARGO_HOME="$REPO_ROOT/.cargo-home"
fi

run_uv() {
    env -u VIRTUAL_ENV uv "$@"
}

# ---------------------------------------------------------------------------
# Loom checkout
# ---------------------------------------------------------------------------

echo ""
echo "=== Preparing the Loom checkout ==="
if [ ! -f "$LOOM_ROOT/pyproject.toml" ]; then
    echo "  cloning $LOOM_REPO_URL into third_party/loom"
    mkdir -p "$(dirname "$LOOM_ROOT")"
    git clone --recurse-submodules "$LOOM_REPO_URL" "$LOOM_ROOT"
else
    echo "  [OK] third_party/loom is present"
fi

if [ -f "$LOOM_ROOT/third_party/loom-dataflow/CMakeLists.txt" ] \
    && [ -f "$LOOM_ROOT/third_party/helion-mlir/pyproject.toml" ] \
    && [ -f "$LOOM_ROOT/third_party/loom-mlar/Cargo.toml" ] \
    && [ -f "$LOOM_ROOT/third_party/loom2ttkernel/CMakeLists.txt" ] \
    && [ -f "$LOOM_ROOT/third_party/adl-dialect/CMakeLists.txt" ]; then
    echo "  [OK] Loom submodules are already populated"
else
    git -c safe.directory="$LOOM_ROOT" -C "$LOOM_ROOT" \
        submodule update --init --recursive
fi

# ---------------------------------------------------------------------------
# ADL dialect
#
# Built before the Python sync: loom-dataflow's scikit-build backend consults
# ADLDialect_DIR while uv resolves its metadata, and the dialect itself needs
# nothing from the virtual environment.
# ---------------------------------------------------------------------------

if [ "$SKIP_DATAFLOW" = "0" ]; then
    ADL_SOURCE_DIR="$LOOM_ROOT/third_party/adl-dialect"
    ADL_BUILD_DIR="$ADL_SOURCE_DIR/build"
    ADL_INSTALL_DIR="$ADL_BUILD_DIR/install"
    export ADLDialect_DIR="$ADL_INSTALL_DIR/lib/cmake/ADLDialect"

    echo ""
    echo "=== Building standalone ADL dialect ==="
    cmake -S "$ADL_SOURCE_DIR" -B "$ADL_BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$ADL_INSTALL_DIR" \
        -DMLIR_DIR="$MLIR_DIR" \
        -DLLVM_USE_LINKER=lld \
        -DADL_INCLUDE_TESTS=OFF
    cmake --build "$ADL_BUILD_DIR" --target install
    require_file "$ADLDialect_DIR/ADLDialectConfig.cmake"
fi

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------

UV_BASE_ARGS=(
    sync
    --project "$REPO_ROOT"
    --inexact
)

if [ "$SKIP_HELION" = "0" ]; then
    UV_BASE_ARGS+=(--extra helion)
fi

echo ""
echo "=== Syncing the torch-helion Python environment ==="
if [ "$SKIP_DATAFLOW" = "0" ]; then
    run_uv "${UV_BASE_ARGS[@]}" --extra dataflow --no-install-package loom-dataflow
else
    run_uv "${UV_BASE_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Native components
# ---------------------------------------------------------------------------

if [ "$SKIP_DATAFLOW" = "0" ]; then
    echo ""
    echo "=== Building loom-dataflow CMake artifacts ==="
    # Loom's build script looks for lit next to the Loom checkout; point it at
    # the torch-helion environment instead.
    bash "$LOOM_ROOT/third_party/loom-dataflow/build.sh" \
        --mlir-dir="$MLIR_DIR" \
        --adl-dialect-dir="$ADLDialect_DIR" \
        --llvm-lit="$REPO_ROOT/.venv/bin/lit"

    echo ""
    echo "=== Installing the loom-dataflow Python extension ==="
    DATAFLOW_SYNC_ARGS=("${UV_BASE_ARGS[@]}" --extra dataflow)
    if [ "$REBUILD_DATAFLOW" = "1" ]; then
        DATAFLOW_SYNC_ARGS+=(--reinstall-package loom-dataflow)
    fi
    run_uv "${DATAFLOW_SYNC_ARGS[@]}"
fi

if [ "$SKIP_MLAR" = "0" ]; then
    if [ -n "${LOOM_EVAL_SYSTEM:-}" ] && [ -x "${LOOM_EVAL_SYSTEM}" ]; then
        echo ""
        echo "[SKIP] loom-mlar (using $LOOM_EVAL_SYSTEM)"
    else
        echo ""
        echo "=== Building loom-mlar ==="
        bash "$LOOM_ROOT/scripts/build-mlar.sh"
    fi
fi

if [ "$SKIP_TTKERNEL" = "0" ]; then
    echo ""
    echo "=== Building loom2ttkernel ==="
    bash "$LOOM_ROOT/third_party/loom2ttkernel/build.sh" \
        -DMLIR_DIR="$MLIR_DIR" \
        -DLLVM_DIR="$LLVM_DIR" \
        -DADLDialect_DIR="$ADLDialect_DIR" \
        -DTTMLIR_SOURCE_DIR="$TTMLIR_SOURCE_DIR" \
        -DTTMLIR_BUILD_DIR="$TTMLIR_BUILD_DIR"
fi

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

echo ""
echo "============================================"
echo "  torch-helion Docker workspace install complete"
echo "============================================"

check_import() {
    if run_uv run --project "$REPO_ROOT" --no-sync python -c "import $1" 2>/dev/null; then
        printf "  %-18s %s\n" "$1" "OK"
    else
        printf "  %-18s %s\n" "$1" "NOT INSTALLED"
    fi
}

check_import torch_helion
check_import loom
if [ "$SKIP_HELION" = "0" ]; then
    check_import helion_mlir
    check_import torch
    check_import helion
fi
if [ "$SKIP_DATAFLOW" = "0" ]; then
    check_import loom_pipeline
fi

EVAL_BIN="${LOOM_EVAL_SYSTEM:-$LOOM_ROOT/third_party/loom-mlar/tests/2d_mesh/bin/eval_system}"
if [ "$SKIP_MLAR" = "0" ]; then
    [ -x "$EVAL_BIN" ] || fail "loom-mlar evaluator was not produced"
    printf "  %-18s %s\n" "eval_system" "OK"
fi

TTKERNEL_BIN="$LOOM_ROOT/third_party/loom2ttkernel/build/bin/tileloom_to_ttkernel_opt"
if [ "$SKIP_TTKERNEL" = "0" ]; then
    [ -x "$TTKERNEL_BIN" ] || fail "loom2ttkernel executable was not produced"
    printf "  %-18s %s\n" "loom2ttkernel" "OK"
fi

echo ""
