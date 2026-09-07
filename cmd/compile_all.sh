#!/usr/bin/env bash
#
# End to end: a PyTorch model in, Loom-compiled kernels out.
#
#   ./cmd/compile_all.sh mamba --njobs 16
#
# Every argument is forwarded to cmd/compile_all.py; run with --help for the full
# list. Requires the environment install-docker.sh builds.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$ROOT/.venv" ]]; then
    echo "cmd/compile_all.sh: no .venv in $ROOT — run 'bash install-docker.sh' first" >&2
    exit 1
fi

exec uv run --project "$ROOT" --no-sync python "$ROOT/cmd/compile_all.py" "$@"
