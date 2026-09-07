# Codegen and the Loom kernel contract

`codegen/emit.py` turns each `KernelSpec` into a file that looks like the
kernels shipped with Loom (`third_party/loom/kernels/*.py`): a Helion
function wrapped with `helion.kernel(static_shapes=False, …)`, a
`LoomKernel` subclass with fixed argument shapes and `bind_args()`, and the
standard `__main__` block. Each kernel gets two Loom configs:
`config_files/<k>.json` (let Loom's solver pick block sizes) and
`config_files/<k>.assigned.json` (materialise the planned tiles).

## GEMM template

```python
for tile_t, tile_n in hl.tile([t, n]):            # C1: the single parallel grid
    acc_i = hl.zeros([tile_t, tile_n], ...)        # one per GEMM sharing the LHS
    ss = hl.zeros([tile_t, 32], ...)               # RMSNorm fold (optional)
    for tile_k in hl.tile(k):                      # C2: the single loop-carried loop
        xt = lhs[tile_t, tile_k]
        acc_i = hl.dot(xt, w_i[tile_k, tile_n], acc=acc_i)
        sq = xt * (xt * half); ss = hl.dot(sq, ones, acc=ss)
    <epilogue SSA over acc_i, sumsq, extra inputs loaded at [tile_t, tile_n]>
    out_j[tile_t, tile_n] = value_j
```

## Attention template

Flash attention with `tile_b = tile_h = 1`, 2D tiles per `(b, h)`, inputs
and output as `[B, S, H, d]` views (free reshapes of `[T, D]`), K/V
permuted at host level and indexed with `tile.begin` scalars, online
softmax carried in `m_i`, `l_i`, `acc`.

## Scan template

```python
for tile_b, tile_h, tile_m in hl.tile([_SSD_B, _SSD_H, _SSD_S], block_size=[1, 1, None]):
    acc0 = hl.zeros([tile_m, p], dtype=torch.float16)
    ci = torch.amax(cum_v[tile_b.begin, tile_h.begin, tile_m, :], -1, keepdim=True)
    q  = c_v[tile_b.begin, tile_h.begin, tile_m, :]
    <every epilogue tile read, hoisted>
    for tile_n in hl.tile(_SSD_S):                 # C2: the single loop-carried loop
        w  = torch.matmul(q, b_v[..., :, tile_n]) * mask[tile_m, tile_n]
        cj = torch.amax(cum_v[..., tile_n, :], -1, keepdim=True)
        dj = torch.amax(dt_v[..., tile_n, :], -1, keepdim=True)
        v  = x_v[..., tile_n, :] * broadcast(torch.exp(cj * -1.0) * dj, 1, [...])
        acc0 = torch.addmm(acc0, w, v)
    acc = acc0 * broadcast(torch.exp(ci), 1, [...]) + xm * dm
    <epilogue> ; o4[tile_b.begin, tile_h.begin, tile_m, :] = value
```

The decay is applied as two column scales rather than the outer difference
`exp(c_i − c_j)`, because a `[1, N]` row cannot be read from memory (see the
rules below). That factorisation is only numerically safe because the
running decay is centred on the sequence midpoint, which halves each
exponent's range; with the reference model's initialisation it leaves about
370 tokens of headroom before `exp` overflows fp16.

## Rules the emitter follows

Every rule below was established by pushing hand-written variants through
`helion_mlir.generate_mlir` and `loom_pipeline.run_exploration` against
`2d_mesh_torus.mlir`; the probes are summarised here so nobody has to
rediscover them.

| rule | reason |
| --- | --- |
| Scalars are materialised as tile-shaped `hl.full([...], c)` constants. | A 0-d scalar operand becomes a one-input `linalg.generic`; only the reference kernels' specific shapes are registered, and constant folding turns them into unregistered `(0,0)` keys. |
| A binary op with the same operand twice is written `a * (a * ones)` (or `a * two` for `a + a`). | Helion's device IR drops the duplicated operand (`aten.mul(load, None)`), and Helion CSEs identical loads. |
| No elementwise op may be the sole consumer of a row reduction. Σx² is accumulated with `hl.dot(sq, ones)`; epilogue reductions act on accumulators. | Loom's elementwise fusion merges the producer into the reduction, producing a mixed parallel/reduction generic; only `add`/`max` bodies exist in that class (`vec_vsum`, `vec_vmax`). |
| Float literals are never printed in exponent form; `1e-5` becomes `0.01 * 0.001`. | helion-mlir prints Python `repr` into MLIR, and `1e-05` is not valid MLIR. |
| The inner loop bound is a backed `SymInt` (a tensor size); head dim is `hl.specialize`d; attention inputs are 4D tensors, not in-kernel reshapes of 2D ones. | An `int` loop bound makes Helion emit no loop-carried `scf.for`; reshapes with Python-int dims that are first touched inside the inner loop get dynamic memrefs and a `tensor.cast` Loom cannot trace. |
| All shapes are multiples of 32; `[T, 1]` columns are allowed. | Loom's L1 footprint estimator asserts 32×32 alignment and pads static-1 dims. |
| Weights arrive as `[K, N]` constants; per-feature gains and the RoPE rotation are folded offline. | Rank-1 operands are not allowed in L1 (C5). |

## Validation and the reference interpreter

`codegen/validate.py` imports the generated module in a subprocess, runs
the Helion→MLIR frontend and Loom's exploration pass (where memory binding,
hardware mapping and ETG construction enforce C1–C5) and reports
per-kernel status (`07_validation.json`).

`codegen/interpret.py` executes the OpIR with plain PyTorch in the same
operation order as the kernels; `compile_model` compares it with the
original model (`08_reference_check.json`). For the Llama block the max
relative error is ≈5e-4 (fp16 weights folded in fp32).
