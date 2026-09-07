# The source code structure

- capture: capture the OpGraph from the pytorch model (pytorch -> fx graph -> OpGraph)
- optimizer: canonicalisation passes, partitioning into kernels, planning; produces the OpIR (`optimizer/opir.py`)
- cost_models: cost models that link the OpIR with the hardware description (analytic estimator, tile solver, Loom backend)
- codegen: transfer the OpIR into helion kernels (Loom style), a reference interpreter, and structural validation
- pipeline.py / results.py / __main__.py: `compile_model()`, the results directory layout, and the CLI

See `docs/` for the per-stage manuals (`docs/architecture.md` is the entry point).

## OpGraph

The OpGraph is a graph from fx graph. What it realize is to erase anything else that loom1.0 compiler does not care. So actually it is a "loom-style" fx graph. Plus, it is more convinient for codegen/ to generate the required helion kernels.

After the optimizer passes the graph contains only Loom-registered ops plus
`row_sumsq` and `attention`, with token-major 2D activations
(`docs/opgraph.md`). The OpIR groups those ops into kernels: one parallel
grid, one reduction loop, accumulators, an epilogue and stores per kernel
(`docs/optimizer.md`).

## Architecture constraints

| # | Constraint | Reason |
| --- | --- | --- |
| C1 | Exactly one parallel grid — one `hl.tile([...])` statement (any number of dims). | The grid is what maps onto the 8×8 mesh. A second grid would need a global barrier mid-kernel, which the dataflow model can't express inside one kernel. |
| C2 | Exactly one loop-carried `scf.for` — non-empty `iter_args`, carrying ≥1 ranked tensor, and innermost. | Memory binding statically assigns L1 buffers by analysing that one reduction loop's double-buffered schedule. Zero such loops = nothing to bind, so pure elementwise kernels don't exist. `utils.cpp:192`, `static_memory_analyser.cpp:242` |
| | Nesting depth is unconstrained — loops without `iter_args` aren't counted. | The walk filters on `!forOp.getInitArgs().empty()`. This is why `mamba_chunk_scan` legally nests three `scf.for`. |
| C3 | Two reductions share a kernel only if they share the same loop. | Direct corollary of C2. Sibling GEMMs with identical LHS and K (QKV, gate/up) work as extra accumulators; a row reduction over K folds into the GEMM's K loop (RMSNorm). Chained GEMMs need two loops → always a kernel boundary. |
| C4 | Every body op must be registered in the hardware spec. | No entry in the perf YAML means no cost model and no lowering. Registered: `matmul`, `batch_matmul`, `add`, `sub`, `mul`, `div`, `exp`, `log`, `powf`, `max`, `cmp`, `select`, row `sum`/`max`. Not: `rsqrt`, `sigmoid`, `neg`, `tanh`, `erf`, `gelu`, `square`, `truncf` — codegen rewrites the first four into registered ops, the rest are rejected. |
| C5 | No rank-1 operands in L1; dims must be 32-aligned. | The L1 footprint estimator asserts 32×32 alignment. Per-feature gains and biases must be folded into the weights offline (that's what `param_transforms` is for). |
| C6 | Host scalars must be `hl.constexpr`. | A plain Python float becomes an unbacked symfloat the frontend can't resolve. |

## Cost models

In this part, torch-helion need to analyze each kernel's potential performance and kernel-kernel link cost to estimate the full model execution time.

Implemented in `cost_models/`: the hardware spec reader parses the ADL MLIR
and the perf YAMLs Loom uses; the analytic estimator mirrors Loom's ETG
aggregation (double-buffered load/compute per loop iteration, broadcast
reuse across the mesh, waves over the 64 cores); the tile solver picks
32-aligned tiles under the L1 capacity; the Loom backend runs the real
pipeline for authoritative numbers (`docs/cost_models.md`).

## Verified kernel contract

`docs/codegen.md` records the additional rules found by probing Loom's
frontend and exploration passes (tile-shaped scalar constants, no
duplicated operands, no elementwise producer feeding a reduction, no
exponent-form float literals, symbolic loop bounds, static 4D views for
attention).
