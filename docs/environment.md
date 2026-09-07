# Development environment

torch-helion is developed inside the same container image Loom uses,
`ftod/loom_dev:latest`, so that a kernel produced here can be handed to Loom
without a second toolchain. The image supplies LLVM/MLIR, a prebuilt tt-mlir,
`uv`, Rust, Node.js and the Tenstorrent runtime; nothing in this repository
builds those from source.

## Layout

| Path                | Contents                                                        |
| ------------------- | --------------------------------------------------------------- |
| `src/torch_helion/` | Source code: lowering passes, pipeline, algorithms               |
| `tests/`            | Unit tests (`pytest`)                                            |
| `docs/`             | Function and structure manual                                    |
| `models/`           | PyTorch models used as lowering inputs                           |
| `third_party/loom/` | Local Loom checkout — untracked, cloned by `install-docker.sh`   |
| `test/`             | Generated pipeline artifacts (Loom's convention, gitignored)     |

`third_party/loom/` is deliberately **not** a git submodule. torch-helion is
meant to become a third_party module of Loom itself, and a submodule in both
directions would be circular. The installer clones it on demand; point
`LOOM_REPO_URL` elsewhere to use a fork.

## Toolchain

Everything the environment needs already exists in the image:

| Component            | Source                                             |
| -------------------- | -------------------------------------------------- |
| LLVM / MLIR          | `$MLIR_DIR`, `$LLVM_DIR` (tt-mlir toolchain)        |
| tt-mlir              | `$TTMLIR_SOURCE_DIR`, `$TTMLIR_BUILD_DIR`           |
| Python               | 3.10, managed by `uv` (`.python-version`)           |
| Rust                 | `/opt/cargo` (rustup stable)                        |

`install-docker.sh` refuses to run when those variables are missing, which is
the signal that you are outside the development image.

## Python environment

Unlike Loom, torch-helion keeps a **single** virtual environment at the
repository root (`.venv`). Loom is consumed as an editable path dependency
rather than a uv workspace member, because Loom is a uv workspace root itself
and uv rejects nested workspaces. Since uv only honours `tool.uv.sources` from
the root project, `pyproject.toml` restates Loom's sources for `helion-mlir`
and `loom-dataflow`, along with helion-mlir's package indexes for PyTorch and
torch-mlir.

The resulting environment contains:

- `torch_helion` — this repository, editable
- `loom` — the Loom pipeline and solver, editable
- `helion_mlir` — Loom's Helion frontend, which pins `torch`, `helion`,
  `triton` and `torch-mlir` (extra: `helion`)
- `loom_pipeline` — the compiled loom-dataflow extension (extra: `dataflow`)
- `lit`, `pytest`, `ruff` — the `dev` dependency group

PyTorch and Helion versions are **not** chosen here. They come from
helion-mlir's pins so that a model lowered by torch-helion targets exactly the
Helion dialect Loom consumes.

## Setup

### VS Code Dev Containers

Open the repository and run **Dev Containers: Reopen in Container**, choosing
`standard`, or `tenstorrent` to attach a Tenstorrent device. VS Code mounts the
checkout at `/workspace/torch-helion` and runs `install-docker.sh --clean`.

### Without VS Code

```bash
./docker/start-container.sh
docker exec -it torch-helion-dev bash install-docker.sh
```

The launcher bind-mounts this checkout into the container, so edits on the host
are visible immediately. It also publishes SSH on port 2223 when
`~/.ssh/authorized_keys` exists:

```bash
ssh -A -p 2223 root@localhost
```

## What the installer does

1. Clones `third_party/loom` if absent and initialises Loom's submodules
   (`adl-dialect`, `helion-mlir`, `loom-dataflow`, `loom-mlar`,
   `loom2ttkernel`).
2. Builds the standalone ADL dialect. This runs **before** the Python sync:
   loom-dataflow's scikit-build backend reads `ADLDialect_DIR` while uv
   resolves its metadata.
3. Syncs `.venv` with the `helion` and `dataflow` extras, deferring the
   loom-dataflow extension until its CMake artifacts exist.
4. Builds loom-dataflow's CMake artifacts, then installs its Python extension.
5. Builds the loom-mlar `eval_system` evaluator (Rust) and the
   `tileloom_to_ttkernel_opt` backend.
6. Reports the import status of every component.

Useful flags: `--skip-ttkernel`, `--skip-mlar`, `--skip-dataflow`,
`--skip-helion` (drops PyTorch and Helion), `--rebuild-dataflow`, `--clean`.

## Daily use

From the repository root:

```bash
uv run pytest tests                  # torch-helion unit tests
uv run pytest third_party/loom/tests # Loom's unit tests, same interpreter
uv run python -c "import torch, helion, loom, loom_pipeline"
```

Loom's kernel configs use paths relative to the Loom checkout, so run them from
there and point `uv` back at this project so the shared `.venv` is used instead
of a second Loom environment:

```bash
cd third_party/loom
uv run --project ../.. --no-sync python kernels/matmul.py \
    --config kernels/config_files/matmul.json --njobs 16 --debug
```

Outputs land in `third_party/loom/test/`. To lower the result to Tenstorrent
kernels:

```bash
./third_party/loom2ttkernel/lower.sh test/matmul_2Dmesh/IRs/p03_bufferized.mlir 1
```

## Version pins

`uv.lock` is committed and holds the same versions Loom's own lock does,
including `torch-mlir==20260531.828` — helion-mlir only requires
`>=20260122.700`, so a fresh resolve would otherwise drift onto a newer dev
wheel than Loom is tested against. Bump it deliberately with:

```bash
uv lock --upgrade-package torch-mlir
```

Because `third_party/loom` is an untracked checkout that can move independently,
the installer runs `uv sync` without `--locked`: pulling Loom re-locks whatever
its dependencies now require rather than failing.
