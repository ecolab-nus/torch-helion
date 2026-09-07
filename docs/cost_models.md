# Cost models

## Hardware description (`cost_models/hw_spec.py`)

`HardwareSpec.load(path)` reads the same files Loom uses:

* the ADL/MLIR system description (`2d_mesh_torus.mlir`): mesh dimensions
  (`adl.spatial_dim "dim_x"/"dim_y"`), L1 capacity (`mem_bank` × banks →
  1,398,784 bytes, identical to the `L1_footprint.capacity` Loom writes
  into its ETG), DRAM size, and the processor modules;
* one `processors/<proc>.perf.yaml` per module with Loom's
  `SimpleTimeCost`: `cycles = fixed_latency + volume / throughput`,
  expressions over `M, N, K, B, L, P, R, effective_bandwidth`, plus
  scenario constraints (evaluated with `cost_models/expr.py`).

`OP_TO_FUNCTION` maps OpGraph ops to hardware functions (`matmul` →
`matmul_SS_f16` on the matrix lane, `exp` → `vec_exp_f16` on the vector
lane, `row_sum` → `vec_vsum_f16`, DRAM copies → `dram_to_l1_S_{f16,bcst}`,
`l1_to_dram_f16`). `effective_bandwidth` is not a free symbol in the YAML;
Loom's exploration substitutes `10 + extra` depending on the broadcast
area. The values were fitted from Loom's resolved ETG of a matmul run:
`extra = 6` when a copy is shared along the mesh's y dimension, `2–5`
otherwise (3.5 used).

## Analytic kernel estimator (`cost_models/analytic.py`)

Mirrors Loom's ETG aggregation for the emitted templates:

```
per grid tile (one core):
  loop   = iters · (max(t_load, t_compute) if double-buffered else sum) + t_load
  tile   = loop + prologue + epilogue + stores
kernel   = ceil(grid_tiles / cores) · tile
```

* `t_load` uses broadcast copies: the LHS tile is shared by every core in a
  mesh column, the RHS tile by every core in a row (both orientations are
  tried, the cheaper is reported); DRAM traffic is divided accordingly.
* `t_compute` = Σ GEMM costs + (`sumsq`: two vector multiplies over the LHS
  tile and a `[tm, 32]` GEMM against a ones matrix).
* the epilogue cost is the vector-lane cost of every `BodyOp` at the tile
  shape, plus one DRAM load per extra input; stores are one copy per output.
* the L1 footprint (accumulators + per-iteration loads ×2 for double
  buffering + epilogue temporaries) must fit `l1_bytes`.
* attention: per `(b, h, m-tile)` core, `iters = S / tile_n`, the online
  softmax ops costed individually.

The result is a cycle count for *comparing* candidates. It is not a
promise of absolute latency.

### Calibration against Loom (Llama block, B=2, S=256, D=256, H=4, F=512)

| kernel | analytic cycles | Loom optimal cycles | ratio |
| --- | --- | --- | --- |
| k0_gemm (QKV + RMSNorm + RoPE) | 57,409 | 33,792 | 1.70 |
| k1_attention | 23,846 | 19,072 | 1.25 |
| k2_gemm (o_proj + residual) | 12,249 | 10,176 | 1.20 |
| k3_gemm (gate/up + RMSNorm + SwiGLU) | 34,991 | 23,552 | 1.49 |
| k4_gemm (down + residual) | 20,155 | 14,592 | 1.38 |

The analytic model is consistently pessimistic by 1.2–1.7× (it does not
model the copy/broadcast variants Loom enumerates during exploration), but
it ranks kernels in the same order and, for `k0_gemm`, `k1_attention` and
`k3_gemm`, its tile choice coincides with Loom's CP-SAT block sizes. Loom's
numbers come from `results/llama_block/10_loom_results.json`.

### Mamba-2 block (B=2, S=256, d_model=512, d_inner=1024, 32 heads)

| kernel | analytic cycles | Loom optimal cycles | ratio |
| --- | --- | --- | --- |
| k0_gemm (dt projection + softplus) | 30,849 | 12,928 | 2.39 |
| k1_gemm (dt scaled by A, padded) | 13,426 | 13,632 | 0.98 |
| k2_gemm (running decay) | 30,808 | 30,016 | 1.03 |
| k3_gemm (z/x/B/C projections) | 120,991 | 114,240 | 1.06 |
| k4_ssd (the scan) | 66,883 | 146,944 | **0.46** |
| k5_gemm (out projection) | 67,236 | 41,088 | 1.64 |
| k6_gemm (final norm) | 17,512 | 17,984 | 0.97 |
| total | 347,705 | 376,832 | 0.92 |

The GEMM kernels land within a few percent. The scan kernel is the
outlier and in the wrong direction: the estimate is **optimistic by 2.2×**,
and the planner picks `tile_m=tile_n=256` where Loom's solver picks
`64/128`. The estimator charges one matrix product per loop iteration
against a load it assumes overlaps; for the scan the two products and the
per-iteration decay reductions dominate, and the loads do not hide them.
Treat the scan's estimate as a lower bound until that term is refitted, and
prefer Loom's own number when comparing partitions that move work into or
out of the scan.

## Tile solver (`cost_models/tile_search.py`)

Enumerates 32-aligned tile sizes dividing each extent (`tile_t, tile_n,
tile_k` / `tile_m, tile_n`) up to `max_tile`, discards assignments that do
not fit L1, and keeps the cheapest under the analytic model. The chosen
tiles are written to the kernel (`TILES`) and to
`config_files/<kernel>.assigned.json` so Loom can materialise them directly
(`assigned_block_size`), or Loom's CP-SAT solver can re-solve them
(`config_files/<kernel>.json`).

## Loom backend (`cost_models/loom_backend.py`)

`run_loom(kernel.py, config.json)` runs the generated kernel through the
full Loom pipeline in a subprocess (frontend → exploration → MLAR ETG
resolution → CP-SAT → materialisation) and parses the global best
(`Optimal T_total` in solver units × 64 cycles, block sizes, variant name).
`CompileConfig(verify_with_loom=True)` does this for every kernel and adds
the numbers to `09_cost_report.md` / `10_loom_results.json`. Loom's own
`IRs/` and `constraints/` land under `results/<run>/loom/<kernel>/`.
