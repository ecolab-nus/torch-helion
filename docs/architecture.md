# Architecture

torch-helion lowers a whole PyTorch model to a sequence of Helion kernels
that Loom 1.0 can compile individually. Loom compiles *one kernel with one
parallel grid and one reduction loop*; torch-helion's job is therefore to
cut the model into such kernels, choose the cut with the lowest estimated
cost, and emit each kernel in exactly the shape Loom's frontend and
exploration passes accept.

```text
PyTorch nn.Module
   │  capture            torch.export → core ATen FX graph → OpGraph (raw)
   ▼
OpGraph (raw)            1:1 with ATen, concrete shapes, parameter values attached
   │  optimizer.passes   fold params, RoPE/RMSNorm/attention macros, commute row
   ▼                     scales through GEMMs, flatten to token-major 2D, rewrite
OpGraph (canonical)      unregistered ops → only Loom-registered ops + macro ops
   │  optimizer.partition / planner   anchors = GEMM groups & attention;
   ▼                     enumerate sibling groupings, cost with cost_models
OpIR (Program)           ordered KernelSpecs: args/views, grid, K loop, accumulators,
   │  codegen.emit       epilogue SSA, stores, tiles, cost
   ▼
kernels/*.py             Loom-style Helion kernels (LoomKernel subclass + config JSON)
   │  codegen.validate   Helion frontend + Loom exploration (C1–C5 enforced there)
   │  cost_models.loom_backend (optional)  full Loom pipeline → optimal time, block sizes
```

## Source layout

| Path | Role |
| --- | --- |
| `src/torch_helion/capture/` | `torch.export` wrapper, ATen→OpGraph converter, the `OpGraph` data model and op vocabulary |
| `src/torch_helion/optimizer/` | canonicalisation passes, layout provenance, weight folding, partitioner, legality checks, planner, the OpIR (`opir.py`) |
| `src/torch_helion/cost_models/` | hardware-spec reader (ADL MLIR + perf YAML), analytic kernel estimator, tile solver, Loom backend |
| `src/torch_helion/codegen/` | OpIR→Helion emitter, reference interpreter, structural validation |
| `src/torch_helion/pipeline.py` | `compile_model()`: runs every stage and stores results |
| `src/torch_helion/results.py` | results directory layout |
| `src/torch_helion/__main__.py` | library CLI (`python -m torch_helion`) |
| `cmd/` | shell entry points: generate only, run one kernel through Loom, or compile end to end |
| `models/` | benchmark models (`llama_block.py`, `mamba.py`), registered in `models.TARGETS` |
| `tests/` | unit tests per module + integration tests |
| `results/<run>/` | every intermediate and final result of one compilation |

## Stage documents

- [OpGraph and capture](opgraph.md)
- [Optimizer: passes, partition, planner](optimizer.md)
- [Cost models](cost_models.md)
- [Codegen and the Loom kernel contract](codegen.md)
- [Results layout](results.md)
- [Development environment](environment.md)

## Kernel decomposition of a Llama block

For the reference model (`models/llama_block.py`) the planner produces five
kernels; every one passes Loom's frontend and exploration passes:

| kernel | contents |
| --- | --- |
| `k0_gemm` | RMSNorm₁ folded + Q/K/V projections + RoPE. Five accumulators sharing the LHS (`x`): Q, K, V and the rotated Q/K (rotation folded into `W @ R`); Σx² accumulated in the same K loop; epilogue computes the row scale and `q·cos + qR·sin`. |
| `k1_attention` | flash attention over `[B, S, H, d]` views of the token-major Q/K/V |
| `k2_gemm` | output projection with the residual add in the epilogue |
| `k3_gemm` | RMSNorm₂ folded + gate/up projections + SwiGLU (sigmoid rewritten to exp/add/div) |
| `k4_gemm` | down projection with the residual add in the epilogue |

Loom compiles all five to bufferized MLIR (`results/llama_block/loom/<kernel>/IRs/p03_bufferized.mlir`);
the per-kernel optimal times are listed in [cost_models.md](cost_models.md#calibration-against-loom-llama-block-b2-s256-d256-h4-f512).

## Mamba-2 backbone

`models/mamba.py` is a full Mamba-2 stack: pre-norm residual blocks whose
mixer projects once into `z, x, B, C, dt`, applies the state-space duality
scan, gates and normalises, and projects back. It carries both evaluation
forms of the scan — the linear-time `ssd_chunked` and its exactly
equivalent quadratic dual `ssd_quadratic` — and the tests check both
against a direct recurrence.

The quadratic mode compiles end to end into seven kernels, all of which pass
Loom's frontend and exploration:

| kernel | contents |
| --- | --- |
| `k0_gemm` | RMSNorm₁ folded + the `dt` projection, softplus in the epilogue |
| `k1_gemm` | `dt` scaled by `A` and replicated across 32 columns — two GEMMs against constants |
| `k2_gemm` | the running decay: one GEMM against a constant matrix, centring folded in |
| `k3_gemm` | RMSNorm₁ folded + the `z`, `x`, `B`, `C` projections (four accumulators sharing one left operand) |
| `k4_ssd` | the scan, with the `D` skip and the `silu(z)` gate in its epilogue |
| `k5_gemm` | RMSNorm₂ folded + the output projection, residual add in the epilogue |
| `k6_gemm` | the final norm — a reduction kernel with no GEMM at all |

Two parts of the model are out of reach by construction and are documented
in its docstring: the depthwise causal conv1d (no reduction loop of its own,
so C2 has nothing to bind) is off by default, and `ngroups` defaults to one
group per head so B and C need no broadcast. The chunked scan is the
reference implementation and is not lowered; it needs four more templates
for the inter-chunk recurrence.

The RMSNorm fold uses the identity `(x·r·g) @ W = r ⊙ (x @ diag(g)W)` with
`r = (mean(x²)+ε)^-1/2`, which turns the normalisation into a row reduction
over the GEMM's own K axis (README constraint C3).
