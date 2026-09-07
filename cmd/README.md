# Entry points

Three ways to run the stack, from fastest to most complete. Run the `.sh`
wrapper — it finds the repository and the environment, so it works from any
directory. Each wraps a Python script of the same name; the shared plumbing
(model loading, shape overrides, printing) lives in `_common.py`.

| script | what it runs | typical time |
| --- | --- | --- |
| `generate_kernels.sh` | torch-helion only: model → OpGraph → OpIR → Helion kernels | seconds |
| `run_kernel.sh` | one generated kernel through Loom | 10 s – 2 min per kernel |
| `compile_all.sh` | everything, plus Loom's full pipeline on every kernel | ~20 s per kernel |

## Naming a model

The first argument is a model from [`models/`](../models):

| target | what it is |
| --- | --- |
| `llama_block` | one Llama-style decoder block |
| `mamba` | a Mamba-2 backbone, stacked SSD blocks with a final norm |
| `mamba_block` | a single Mamba-2 block |

The list comes from `models.TARGETS`; add a model there and every entry point
picks it up. Anything unregistered can still be named explicitly as
`module:factory`. Omit the argument and the script prints what is available.

`--set KEY=VALUE` overrides any field of that model's config dataclass, and
`results/<run>/` defaults to the model's name.

## generate_kernels.py — torch-helion only

```bash
./cmd/generate_kernels.sh mamba --set seq=256
```

Captures the model, runs the optimizer passes, plans the partition, emits the
kernels, and checks the OpIR against the original model with the reference
interpreter. Loom is never invoked, so this is the loop to use while changing
passes or templates.

`--check` additionally pushes each kernel through Loom's frontend and
exploration passes, where constraints C1–C5 are enforced. That still does not
solve for block sizes.

## run_kernel.py — one kernel through Loom

```bash
./cmd/run_kernel.sh mamba --list
./cmd/run_kernel.sh mamba --kernel k4_ssd --njobs 16
./cmd/run_kernel.sh mamba --kernel k4_ssd --assigned
./cmd/run_kernel.sh mamba --all
```

Its first argument is the run to take kernels from, which is the model name
unless `--run-name` said otherwise.

Runs Loom on a kernel this repository already generated: frontend,
exploration, ETG resolution, CP-SAT solve, materialisation. Prints the
optimal time, the chosen block sizes and the winning variant.

`--assigned` materialises the tiling torch-helion's planner chose instead of
asking Loom's solver. That skips ETG resolution and the solve, so it is much
faster — and it is the only route for a kernel whose operations have no cost
model in the hardware spec.

`--kernel` also accepts a path, so a hand-edited kernel works too:

```bash
./cmd/run_kernel.sh --kernel results/mamba/kernels/k4_ssd.py
```

## compile_all.py — end to end

```bash
./cmd/compile_all.sh mamba --njobs 16
```

Everything above in one pass, ending with Loom's optimum printed beside the
analytic estimate that drove planning, and the ratio between them. Loom's
IRs and constraints land in `results/<run>/loom/<kernel>/`.

## Running the Python directly

The wrappers only locate the repository and the environment. Inside an
activated environment the scripts run on their own:

```bash
uv run python cmd/generate_kernels.py mamba --set seq=256
```

## Exit status

0 when the reference check passes and every kernel Loom was asked about
succeeded; 1 otherwise. Safe to use in CI.
