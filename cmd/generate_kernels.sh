#!/usr/bin/env bash
#
# torch-helion only: a PyTorch model in, Helion kernels out. Loom is not run.
#
#   ./cmd/generate_kernels.sh mamba --set seq=256
#
# Every argument is forwarded to cmd/generate_kernels.py; run with --help for the full
# list. Requires the environment install-docker.sh builds.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$ROOT/.venv" ]]; then
    echo "cmd/generate_kernels.sh: no .venv in $ROOT — run 'bash install-docker.sh' first" >&2
    exit 1
fi

exec uv run --project "$ROOT" --no-sync python "$ROOT/cmd/generate_kernels.py" "$@"
