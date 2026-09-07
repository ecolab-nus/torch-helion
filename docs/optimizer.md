# Optimizer

The optimizer has three layers: canonicalisation passes on the OpGraph,
the partitioner that turns a canonical graph into kernels (OpIR), and the
planner that searches over partitions with the cost model.

## Canonicalisation passes (`optimizer/passes.py`)

Run in this order by `canonicalize()`; with `verify_inputs` every pass is
checked numerically against the raw graph (max relative error stored in
`03_pass_log.json`).

| pass | effect |
| --- | --- |
| `fold_constants` | any node whose inputs are all parameters/constants is evaluated offline (weight transposes, folded gains …) |
| `match_softplus` | `where(x > t, x, log1p(exp(x)))` → `softplus(x)`; the threshold guard needs a compare and a select in shapes the templates cannot produce, and every activation sits below the threshold |
| `match_rotate_half` | `cat(neg(x[..., d/2:]), x[..., :d/2])` → `rotate_half(x)` |
| `match_rope` | `x·cos + rotate_half(x)·sin` → `rope(x, cos, sin)` |
| `lower_rope` | for `rope` on a head-split view of `y = h @ W`: `y·cos_full + (h @ (W R))·sin_full` with `R = blockdiag(rotate_half matrix)`; `cos_full`/`sin_full` are the `[S, d]` tables expanded to `[T, D]` (`param_transforms`) |
| `match_rmsnorm` | `x · rsqrt(mean(x²) + ε) · g` → `rmsnorm(x, g, ε)` |
| `push_slice_into_gemm` | `slice(a @ W, cols)` → `a @ W[:, cols]`, to a fixpoint and through identity reshapes. A fused projection that is split afterwards (Mamba's `z, x, B, C, dt`, or fused QKV) becomes sibling GEMMs sharing one left operand, which is what a single kernel can accumulate |
| `lower_rmsnorm` | with GEMM consumers, rewritten to `(x @ diag(g)W) ⊙ broadcast(r)` where `r = (row_sumsq(x)/K + ε)^-1/2`. With no GEMM to absorb the scale (a terminal norm), the normalisation stays explicit, the per-feature gain is materialised as a full 2-D constant, and the partitioner gives the reduction its own kernel |
| `match_attention` | `softmax(bmm(q', k'^T)·s) @ v'` where `q'`, `k'`, `v'` are head-split views of token-major tensors → `attention(q, k, v)` with `{batch, seq, heads, head_dim, scale}` |
| `match_ssd` | the Mamba-2 scan `(C Bᵀ ⊙ exp(cumA_i − cumA_j) ⊙ causal ⊙ dt_j) X` → one `ssd` node. Rewrites the decay so every kernel input is token-major: `dt` is projected through a constant that both scales it by `A` and replicates each head across 32 columns, and the running decay becomes one GEMM against a constant matrix with the centring folded in. Absorbs the `y + D·x` skip and the permute back to token order |
| `lower_cumsum` | `row_cumsum(x)` → `x @ P` with a constant prefix-sum matrix (`P[j,i] = 1` when `j ≤ i`). A scan becomes the one thing Loom always has: a reduction loop over a GEMM |
| `flatten_to_2d` | removes identity reshapes (verified by index-tensor provenance), drops precision casts, gives every activation a 2D type; any remaining real layout op is an error |
| `rewrite_unregistered` | `neg`→`mul(-1)`, `rsqrt`→`powf(-0.5)`, `sigmoid`→`1/(1+exp(-x))`, `silu`, `tanh`, `square`, `row_mean`; rejects rank-1 operands (C5) |
| `materialize_const_operands` | expand a rank-1 constant operand to the activation's shape. A per-feature bias cannot be folded into a weight when something non-linear sits between it and the GEMM (`softplus(dt + dt_bias)`), and rank-1 operands cannot live in L1 (C5) |
| `insert_broadcasts` | explicit `broadcast` for `[T,1]` × `[T,N]` operands |
| `dedupe_constants` | merges identical constant tensors (e.g. the RoPE tables of Q and K) |
| `cleanup` | dead-code elimination, topological order |

Layout reasoning uses `optimizer/layout.py`: a chain of layout ops is
replayed on an `arange` index tensor of its base, so "is this an identity
reshape?" and "is this the `[B,S,H,d]→[B,H,S,d]` head split?" are exact
tensor comparisons rather than stride algebra.

## Partition (`optimizer/partition.py`)

* **Anchors.** Every GEMM (`matmul` with a constant RHS) and every
  `attention` node anchors a kernel. GEMMs with the same LHS and output
  width are *siblings* and may share one kernel (extra accumulators in the
  same K loop, README C3).
* **Epilogue assignment.** Every other compute node is placed in the
  kernel that owns its inputs (accumulators or earlier epilogue values).
  If inputs come from several kernels, the later kernel wins and the other
  value is materialised in DRAM as an *extra input* (must be `[T, N]` so it
  can be loaded at the output tile).
* **Column chains.** Values derived from `row_sumsq(x)` (`[T, 1]` row
  scales) are cloned into every kernel that needs them; such a kernel must
  have `x` as its LHS and gets `sumsq = True`.
* **Outputs.** Any value consumed outside its kernel (or a graph output) is
  stored to DRAM.
* Kernels are ordered topologically by tensor dependencies.

Pure elementwise ops on DRAM tensors have no reduction loop to live in and
are rejected (README C2). An elementwise consumer of an attention output
is rejected too (the attention template has no epilogue).

## Legality (`optimizer/legality.py`)

Checks per kernel: registered epilogue ops (C4), 32-aligned extents and no
rank-1 operands (C5), `[T, N]` extra inputs and outputs, and the *fusion
rule*: Loom fuses a single-use elementwise producer into a row reduction,
producing a mixed generic that only `add`/`max` bodies satisfy, so
reductions must consume accumulators, loads or multi-use values.

## Planner (`optimizer/planner.py`)

For each sibling set the planner enumerates all set partitions
(`planner_mode="exhaustive"`, Bell numbers; 5 siblings → 52) or only
"all merged"/"all split" (`greedy`). Each candidate grouping is built,
checked, and costed: every kernel gets its cheapest tile assignment from the
analytic tile search (memoised by kernel signature), and the program cost is
Σ kernel cycles + `kernel_launch_overhead` per kernel. `05_plan/plan.md`
lists all candidates; the cheapest legal one becomes the OpIR.

## OpIR (`optimizer/opir.py`)

`Program` = ordered `KernelSpec`s + tensor table. A `gemm` kernel holds:
`args` (roles `lhs`, `rhs`, `extra`), `grid` (`rows`, `cols`), `k_extent`,
`gemms` (accumulator ↔ rhs), `sumsq`, `epilogue` (SSA `BodyOp`s over
registered ops), `outputs` (tensor ← value), `constexprs`, planned `tiles`
and `cost`. An `attention` kernel holds the q/k/v args with their
`[B, S, H, d]` views and the geometry/scale attrs.

## Kernel kinds

| kind | anchor | shape |
| --- | --- | --- |
| `gemm` | a group of GEMMs sharing one operand | grid `[T, N]`, reduction over `K`. Either operand may be the constant: a running-sum matrix multiplies an activation from the left |
| `gemm` (norm) | a `row_sumsq` with no GEMM to absorb it | same, with no GEMM at all — the sum of squares is what gives the loop a ranked tensor to carry (C2) |
| `attention` | an `attention` macro op | flash attention over `[B, S, H, d]` views |
| `ssd` | an `ssd` macro op | the Mamba-2 scan over the same views, decay instead of softmax |

An epilogue op joins the kernel whose *anchor* comes latest in graph order,
never the one whose epilogue has grown furthest — a kernel that produces a
value's other operands cannot also consume its result. It must also match
that kernel's tile frame, since an epilogue reads and writes at the output
tile.
