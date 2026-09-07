# OpGraph and capture

## Capture

`torch_helion.capture.export_to_fx(model, args)` runs `torch.export.export`
with static shapes and `run_decompositions()`, which lowers composite ops
(`linear`, `silu`, …) to core ATen. Parameters and buffers are lifted to
graph inputs; the converter re-attaches their values.

`fx_to_opgraph` maps every ATen node to one `OpNode`. The mapping is
literal on purpose: the raw graph stored as `02_opgraph_raw.json` is an
honest picture of the model, and all rewriting is done by the optimizer.
Unsupported ATen ops raise `UnsupportedOpError` naming the node.

Supported ATen ops: `add/sub/mul/div` (tensor or scalar), `maximum`, `pow`,
`exp`, `log`, `neg`, `rsqrt`, `sqrt`, `sigmoid`, `silu`, `tanh`, `square`,
`mean.dim`/`sum.dim_IntList`/`amax` (last dim only), `_softmax` (last dim),
`mm`, `bmm`, `addmm`, `linear`, `permute`, `transpose`, `t`, `view`,
`reshape`, `unsqueeze`/`squeeze`, `expand`, `clone`, `_to_copy`, `slice`,
`cat`, `cumsum` (last dim), `log1p`, `gt.Scalar`, `where`, and
`split_with_sizes` (the `getitem` that reads a split becomes the slice).

## The OpGraph

`OpGraph` is an ordered DAG (insertion order is topological) of `OpNode`s;
every node produces one tensor with a concrete `TensorType` (shape, dtype).

| field | meaning |
| --- | --- |
| `kind` | `input`, `param` (has a value), `const` (value created by the optimizer), `compute`, `layout` |
| `op` | operator name from `capture/ops.py` |
| `inputs` | producer node names |
| `attrs` | scalars (`scalar`, `scalar_first`), dims, shapes, eps, attention geometry |

`graph.params` holds the concrete tensors of `param`/`const` nodes so the
optimizer can fold weights offline. JSON serialisation keeps everything
except tensor values (`save_params` writes them to `params.pt`).

## Op vocabulary

| family | ops | note |
| --- | --- | --- |
| registered (Loom hardware spec) | `matmul`, `batch_matmul`, `add`, `sub`, `mul`, `div`, `max`, `powf`, `exp`, `log`, `row_sum`, `row_max`, `broadcast` | the only ops allowed in a kernel body (README C4) |
| rewritten | `neg`, `rsqrt`, `sqrt`, `sigmoid`, `silu`, `tanh`, `square`, `row_mean`, `log1p`, `softplus` | expanded into registered ops by `rewrite_unregistered` |
| macro | `rotate_half`, `rope`, `rmsnorm`, `attention`, `softmax`, `ssd` | recognised patterns; lowered to registered ops or to a kernel template |
| layout | `view`, `permute`, `expand`, `clone`, `slice`, `cat`, `cast` | never survive canonicalisation |
| internal | `row_sumsq` | Σ over the last dim of x², produced by the RMSNorm lowering; realised inside the GEMM loop |
| scan | `row_cumsum` | inclusive running sum along the last axis; lowered to a GEMM against a constant triangular matrix |
| guard | `cmp_gt`, `select` | only appear inside the `softplus` decomposition, which is matched and rewritten away |

The canonical graph (after the optimizer passes) contains only registered
ops, `row_sumsq`, `attention`, inputs and constants, and every activation
is a token-major 2D tensor `[tokens, features]`.
