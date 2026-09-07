#!/usr/bin/env bash
#
# One generated Helion kernel through Loom, on its own.
#
#   ./cmd/run_kernel.sh mamba --kernel k4_ssd --njobs 16
#
# Every argument is forwarded to cmd/run_kernel.py; run with --help for the full
# list. Requires the environment install-docker.sh builds.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$ROOT/.venv" ]]; then
    echo "cmd/run_kernel.sh: no .venv in $ROOT — run 'bash install-docker.sh' first" >&2
    exit 1
fi

exec uv run --project "$ROOT" --no-sync python "$ROOT/cmd/run_kernel.py" "$@"
