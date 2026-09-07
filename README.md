# Torch-helion

In this repo, we hope to deploy full model with Loom1.0 as the lower level backend. This will be the third-party module in Loom, as we want to extend the looms' capabilities to support full model. The main task of this repo is to lowering the high-level pyTorch model to the helion kernel, which act as the input to loom1.0. 


## Usage

Three entry points live in [`cmd/`](cmd/README.md), from fastest to most
complete. Every intermediate result lands in `results/<run-name>/`.

```bash
# torch-helion only: model -> OpGraph -> OpIR -> Helion kernels (seconds)
./cmd/generate_kernels.sh llama_block

# one generated kernel through Loom
./cmd/run_kernel.sh llama_block --kernel k2_gemm --njobs 16

# end to end, including Loom's full pipeline on every kernel
./cmd/compile_all.sh mamba --njobs 16
```

The first argument names a model in [`models/`](models) — `llama_block`,
`mamba` or `mamba_block`; run a script with no argument to see the list.

or from Python:

```python
from models.llama_block import build_llama_block
from torch_helion import compile_model, CompileConfig

block = build_llama_block()
result = compile_model(block, block.example_inputs(), CompileConfig(run_name="llama_block"))
print(result.program.summary())          # kernels, tiles, estimated cycles
```

`models/` also carries a full Mamba-2 backbone
(`models.mamba:build_mamba_model`), which compiles end to end into seven
kernels including the state-space scan.

The pipeline is `capture → OpGraph → optimizer (canonicalise, partition,
plan) → OpIR → codegen`, see [docs/architecture.md](docs/architecture.md).
Generated kernels are Loom CLI entry points in their own right, so they can
also be run directly:

```bash
uv run python results/llama_block/kernels/k2_gemm.py \
    --config results/llama_block/kernels/config_files/k2_gemm.json --njobs 8 --debug
```

Run the tests with `uv run pytest tests` (`-m "not slow"` skips the
multi-kernel Loom validation).

## Development environment

torch-helion develops inside Loom's own container image, `ftod/loom_dev:latest`,
so kernels produced here run against exactly the toolchain Loom consumes: LLVM/
MLIR, a prebuilt tt-mlir, Python 3.10 via `uv`, and Rust.

Loom is checked out at `third_party/loom/`. It is untracked rather than a
submodule, since torch-helion is meant to become Loom's own third_party module
and submodules in both directions would be circular — `install-docker.sh`
clones it for you.

### VS Code Dev Containers

Open this repository and run **Dev Containers: Reopen in Container**. Select
`standard` on a non-Tenstorrent machine, or `tenstorrent` to attach a device.
VS Code mounts the checkout and runs `install-docker.sh` automatically.

### Without VS Code

```bash
./docker/start-container.sh
docker exec -it torch-helion-dev bash install-docker.sh
```

The installer builds Loom's native components (ADL dialect, loom-dataflow,
loom-mlar, loom2ttkernel) and syncs a single `.venv` containing `torch_helion`,
`loom`, `helion_mlir` — which pins PyTorch, Helion, Triton and torch-mlir — and
the compiled `loom_pipeline` extension.

```bash
uv run pytest tests
```

See the [environment guide](docs/environment.md) for the repository layout,
installer flags, and how the Python environment is put together.
