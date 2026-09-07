"""Canonicalisation passes: raw OpGraph → loom-style OpGraph.

The raw graph produced by :mod:`torch_helion.capture` mirrors ATen. The
passes below rewrite it into the restricted form the partitioner works on:

1. ``fold_constants``      parameter-only sub-graphs become ``const`` nodes
2. ``match_rotate_half``   ``cat(neg(slice), slice)`` → ``rotate_half``
3. ``match_rope``          ``x*cos + rotate_half(x)*sin`` → ``rope``
4. ``lower_rope``          rope on ``matmul(h, W)`` → two GEMMs sharing ``h``
                           (rotation folded into ``W @ R``) + 2D tables
5. ``match_rmsnorm``       ``x * rsqrt(mean(x^2)+eps) * g`` → ``rmsnorm``
6. ``lower_rmsnorm``       commute the row scale through the consuming GEMMs
                           (gain folded into the weights, ``row_sumsq`` left
                           on the GEMM's K axis)
7. ``match_attention``     bmm/softmax/bmm cluster → ``attention`` macro op
8. ``flatten_to_2d``       strip identity reshapes; token-major 2D activations
9. ``rewrite_unregistered`` neg/rsqrt/sigmoid/silu/... → registered ops
10. ``insert_broadcasts``  explicit ``broadcast`` for ``[T,1]`` × ``[T,N]``
11. ``cleanup``            dead-code elimination + topological order

Every pass is a pure function ``pass(graph) -> int`` (number of rewrites)
operating in place; :func:`canonicalize` runs them in order and can verify
each step numerically against the previous graph.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from ..capture.opgraph import OpGraph, OpNode, TensorType
from ..capture.ops import BINARY_OPS
from . import param_transforms as pt
from .evaluate import eval_op, graph_outputs, max_rel_err
from .layout import match_head_split, head_merge_index, trace_layout


class PassError(RuntimeError):
    """The graph contains a construct the current lowering cannot express."""


# ------------------------------------------------------------------ helpers

def _convert(node: OpNode, op: str, inputs: list[str], attrs: dict | None = None, kind: str = "compute", type: TensorType | None = None) -> None:
    """Rewrite ``node`` in place (keeps its name, so users stay valid)."""
    node.op = op
    node.inputs = list(inputs)
    node.attrs = dict(attrs or {})
    node.kind = kind
    if type is not None:
        node.type = type


def _binary_operands(g: OpGraph, node: OpNode, op: str) -> tuple[OpNode, OpNode] | None:
    if node.op != op or len(node.inputs) != 2:
        return None
    return g[node.inputs[0]], g[node.inputs[1]]


def _is_const(g: OpGraph, name: str) -> bool:
    return g[name].kind in ("param", "const")


def _twod(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 0:
        return (1, 1)
    if len(shape) == 1:
        return (1, shape[0])
    return (math.prod(shape[:-1]), shape[-1])


def _ensure_2d(g: OpGraph, name: str) -> str:
    """Return a 2D ``[prod(leading), last]`` alias of ``name``."""
    node = g[name]
    if node.type.rank == 2:
        return name
    shape = _twod(node.type.shape)
    v = g.add("view", [name], TensorType(shape, node.type.dtype), {"shape": list(shape)}, kind="layout", origin=f"ensure_2d({name})")
    return v.name


def _effective_users(g: OpGraph, name: str) -> list[tuple[OpNode, OpNode | None]]:
    """Users of ``name`` looking through identity reshapes.

    Returns ``(consumer, via)`` pairs; ``via`` is the layout node between
    ``name`` and ``consumer`` (or ``None``).
    """
    out: list[tuple[OpNode, OpNode | None]] = []
    for u in g.users(name):
        if u.kind == "layout" and u.op in ("view", "clone", "expand", "cast") and trace_layout(g, u.name).is_identity_reshape():
            for uu, _via in _effective_users(g, u.name):
                out.append((uu, u))
        else:
            out.append((u, None))
    if name in g.outputs:
        out.append((OpNode("<output>", "output", [name], g[name].type), None))
    return out


# ------------------------------------------------------------------- passes

def fold_constants(g: OpGraph) -> int:
    """Turn nodes whose inputs are all constants into ``const`` nodes."""
    n = 0
    for node in list(g):
        if node.kind in ("input", "param", "const") or not node.inputs:
            continue
        if all(_is_const(g, i) for i in node.inputs):
            value = eval_op(node.op, [g.value(i) for i in node.inputs], node.attrs)
            value = value.reshape(node.type.shape).to(node.type.torch_dtype).contiguous()
            _convert(node, "const", [], {}, kind="const")
            g.params[node.name] = value
            n += 1
    return n


def match_rotate_half(g: OpGraph) -> int:
    n = 0
    for node in list(g):
        if node.op != "cat" or len(node.inputs) != 2:
            continue
        a, b = g[node.inputs[0]], g[node.inputs[1]]
        if node.attrs["dim"] != node.type.rank - 1 or a.op != "neg" or b.op != "slice":
            continue
        sa = g[a.inputs[0]]
        if sa.op != "slice" or sa.inputs[0] != b.inputs[0]:
            continue
        x = g[sa.inputs[0]]
        d = x.type.shape[-1]
        if (sa.attrs["start"], sa.attrs["end"], b.attrs["start"], b.attrs["end"]) != (d // 2, d, 0, d // 2):
            continue
        _convert(node, "rotate_half", [x.name], {})
        n += 1
    return n


def match_rope(g: OpGraph) -> int:
    n = 0
    for node in list(g):
        ops = _binary_operands(g, node, "add")
        if ops is None:
            continue
        m1, m2 = ops
        if m1.op != "mul" or m2.op != "mul" or "scalar" in m1.attrs or "scalar" in m2.attrs:
            continue
        found = None
        for plain, rot in ((m1, m2), (m2, m1)):
            for xi, ci in ((0, 1), (1, 0)):
                x = plain.inputs[xi]
                cos = plain.inputs[ci]
                for ri, si in ((0, 1), (1, 0)):
                    r = g[rot.inputs[ri]]
                    sin = rot.inputs[si]
                    if r.op == "rotate_half" and r.inputs[0] == x and _is_const(g, cos) and _is_const(g, sin):
                        found = (x, cos, sin)
        if found is None:
            continue
        x, cos, sin = found
        _convert(node, "rope", [x, cos, sin], {"head_dim": g[x].type.shape[-1]})
        n += 1
    return n


def _rebuild_chain(g: OpGraph, prov_chain: list[str], new_base: str, target: OpNode) -> None:
    """Re-create the layout chain of ``prov_chain`` on ``new_base``.

    The last op of the chain is written into ``target`` so its users remain
    valid. ``new_base`` must have the same numel as the original base.
    """
    prev = new_base
    orig_base_shape = g[g[prov_chain[0]].inputs[0]].type.shape if prov_chain else target.type.shape
    if g[prev].type.shape != orig_base_shape:
        prev = g.add("view", [prev], TensorType(orig_base_shape, g[prev].type.dtype), {"shape": list(orig_base_shape)}, kind="layout").name
    if not prov_chain:
        _convert(target, "view", [prev], {"shape": list(target.type.shape)}, kind="layout")
        return
    for lname in prov_chain[:-1]:
        l = g[lname]
        prev = g.add(l.op, [prev], l.type, dict(l.attrs), kind="layout").name
    last = g[prov_chain[-1]]
    _convert(target, last.op, [prev], dict(last.attrs), kind="layout", type=last.type)


def lower_rope(g: OpGraph) -> int:
    n = 0
    for node in list(g):
        if node.op != "rope":
            continue
        x4, cos, sin = node.inputs
        prov = trace_layout(g, x4)
        dims = match_head_split(prov)
        if dims is None:
            raise PassError(f"rope {node.name}: input is not a head-split view of a token-major tensor")
        b, s, h, d = dims["batch"], dims["seq"], dims["heads"], dims["head_dim"]
        base = g[prov.base]
        base2d = _ensure_2d(g, base.name)
        t2 = g[base2d].type
        cos_full = pt.expand_rope_table(g.value(cos).reshape(s, d), b, h).to(t2.torch_dtype)
        sin_full = pt.expand_rope_table(g.value(sin).reshape(s, d), b, h).to(t2.torch_dtype)
        cos_c = g.add_const(cos_full, name=g.fresh_name("rope_cos"))
        sin_c = g.add_const(sin_full, name=g.fresh_name("rope_sin"))
        # rotation: fold into the producing GEMM when possible
        if base.op == "matmul" and _is_const(g, base.inputs[1]):
            wr = pt.fold_rotation_into_weight(g.value(base.inputs[1]), h, d)
            wr_c = g.add_const(wr, name=g.fresh_name(base.inputs[1] + "_rot"))
            rot = g.add("matmul", [base.inputs[0], wr_c.name], base.type, origin=f"lower_rope({node.name})")
            rot2d = _ensure_2d(g, rot.name)
        else:
            r_c = g.add_const(pt.rotate_half_heads_matrix(h, d).to(t2.torch_dtype), name=g.fresh_name("rope_R"))
            rot2d = g.add("matmul", [base2d, r_c.name], t2, origin=f"lower_rope({node.name})").name
        m1 = g.add("mul", [base2d, cos_c.name], t2)
        m2 = g.add("mul", [rot2d, sin_c.name], t2)
        y = g.add("add", [m1.name, m2.name], t2, origin=f"lower_rope({node.name})")
        _rebuild_chain(g, prov.chain, y.name, node)
        n += 1
    g.toposort()
    return n


def match_rmsnorm(g: OpGraph) -> int:
    n = 0
    for node in list(g):
        ops = _binary_operands(g, node, "mul")
        if ops is None:
            continue
        for inner, gain in (ops, ops[::-1]):
            if not _is_const(g, gain.name) or inner.op != "mul" or len(inner.inputs) != 2:
                continue
            for x, r in ((g[inner.inputs[0]], g[inner.inputs[1]]), (g[inner.inputs[1]], g[inner.inputs[0]])):
                if r.op != "rsqrt":
                    continue
                add = g[r.inputs[0]]
                if add.op != "add" or "scalar" not in add.attrs:
                    continue
                mean = g[add.inputs[0]]
                if mean.op != "row_mean":
                    continue
                sq = g[mean.inputs[0]]
                is_sq = (sq.op == "powf" and sq.attrs.get("scalar") == 2 and sq.inputs[0] == x.name) or (sq.op == "square" and sq.inputs[0] == x.name) or (sq.op == "mul" and sq.inputs == [x.name, x.name])
                if not is_sq:
                    continue
                _convert(node, "rmsnorm", [x.name, gain.name], {"eps": float(add.attrs["scalar"])})
                n += 1
                break
            if node.op == "rmsnorm":
                break
    return n


def lower_rmsnorm(g: OpGraph) -> int:
    n = 0
    for node in list(g):
        if node.op != "rmsnorm":
            continue
        x, gain = node.inputs
        eps = node.attrs["eps"]
        users = _effective_users(g, node.name)
        bad = [u.name for u, via in users if not (u.op == "matmul" and _is_const(g, u.inputs[1]) and u.inputs[0] == (via.name if via else node.name))]
        if bad:
            # No GEMM to commute the row scale into (a terminal norm, say).
            # Keep the normalisation explicit and materialise the per-feature
            # gain as a full 2-D constant, since rank-1 operands cannot live
            # in L1 (C5). The partitioner gives the reduction its own kernel.
            _lower_standalone_rmsnorm(g, node)
            n += 1
            continue
        x2d = _ensure_2d(g, x)
        r = _row_scale(g, x2d, eps, f"lower_rmsnorm({node.name})")
        gval = g.value(gain).reshape(-1)
        for mm, via in users:
            w = mm.inputs[1]
            wg = g.add_const(pt.fold_gain_into_weight(gval, g.value(w)), name=g.fresh_name(w + "_norm"))
            # The GEMM may be stated on an N-D activation; do the rescale in
            # the 2-D token-major form and view back, so the broadcast always
            # pairs a [T, 1] column with a [T, N] tile.
            flat = TensorType(_twod(mm.type.shape), mm.type.dtype)
            new_mm = g.add("matmul", [x2d, wg.name], flat, origin=f"lower_rmsnorm({mm.name})")
            bcast = g.add("broadcast", [r], flat, {"dim": 1, "shape": list(flat.shape)})
            scaled = g.add("mul", [new_mm.name, bcast.name], flat)
            _convert(mm, "view", [scaled.name], {"shape": list(mm.type.shape)}, kind="layout")
        n += 1
    g.toposort()
    return n


def _skip_casts(g: OpGraph, node: OpNode) -> OpNode:
    while node.op == "cast" and len(node.inputs) == 1:
        node = g[node.inputs[0]]
    return node


def match_softplus(g: OpGraph) -> int:
    """``where(x > t, x, log1p(exp(x)))`` -> ``softplus(x)``.

    PyTorch decomposes ``softplus`` into a threshold guard that switches to
    the identity for large inputs. The guard needs a compare and a select,
    which the hardware spec registers only in shapes the kernel templates
    cannot produce, so the pattern is folded back into one op and lowered
    to ``log(1 + exp(x))``. The two agree to fp16 precision for every
    argument below the threshold, which is where all activations sit.
    """
    n = 0
    for node in list(g):
        if node.op != "select" or len(node.inputs) != 3:
            continue
        cond, big, small = (g[i] for i in node.inputs)
        if cond.op != "cmp_gt" or "scalar" not in cond.attrs:
            continue
        x = cond.inputs[0]
        if big.name != x or small.op != "log1p":
            continue
        e = g[small.inputs[0]]
        if e.op != "exp" or e.inputs[0] != x:
            continue
        _convert(node, "softplus", [x], {"threshold": float(cond.attrs["scalar"])})
        n += 1
    return n


def lower_cumsum(g: OpGraph) -> int:
    """``row_cumsum(x)`` -> ``x @ P`` with a constant prefix-sum matrix.

    ``P[j, i] = 1`` when ``j <= i``, so the matrix product is the inclusive
    running sum along the last axis. This turns a scan into the one thing
    Loom always has: a reduction loop over a GEMM.
    """
    n = 0
    for node in list(g):
        if node.op != "row_cumsum":
            continue
        x = g[node.inputs[0]]
        k = x.type.shape[-1]
        idx = torch.arange(k)
        prefix = (idx.unsqueeze(1) <= idx.unsqueeze(0)).to(x.type.torch_dtype)
        pc = g.add_const(prefix, name=g.fresh_name("cumsum_prefix"))
        x2d = _ensure_2d(g, x.name)
        t = g[x2d].type
        mm = g.add("matmul", [x2d, pc.name], t, origin=f"lower_cumsum({node.name})")
        _convert(node, "view", [mm.name], {"shape": list(node.type.shape)}, kind="layout")
        n += 1
    g.toposort()
    return n


def _row_scale(g: OpGraph, x2d: str, eps: float, origin: str) -> str:
    """Emit ``(row_sumsq(x)/K + eps) ** -0.5`` as a ``[T, 1]`` column."""
    t, k = g[x2d].type.shape
    col = TensorType((t, 1), g[x2d].type.dtype)
    sumsq = g.add("row_sumsq", [x2d], col, origin=origin)
    m = g.add("mul", [sumsq.name], col, {"scalar": 1.0 / k})
    a = g.add("add", [m.name], col, {"scalar": eps})
    return g.add("powf", [a.name], col, {"scalar": -0.5}, origin=origin).name


def _lower_standalone_rmsnorm(g: OpGraph, node: OpNode) -> None:
    x, gain = node.inputs
    x2d = _ensure_2d(g, x)
    t, k = g[x2d].type.shape
    typ = g[x2d].type
    r = _row_scale(g, x2d, float(node.attrs["eps"]), f"rmsnorm({node.name})")
    bc = g.add("broadcast", [r], typ, {"dim": 1, "shape": [t, k]})
    scaled = g.add("mul", [x2d, bc.name], typ)
    gfull = g.add_const(
        torch.broadcast_to(g.value(gain).reshape(1, -1), (t, k)).contiguous(),
        name=g.fresh_name(gain + "_full"),
    )
    out = g.add("mul", [scaled.name, gfull.name], typ, origin=f"rmsnorm({node.name})")
    _rebuild_chain(g, [], out.name, node)


def _gemm_through_reshapes(g: OpGraph, name: str) -> OpNode | None:
    """The GEMM producing ``name``, looking through identity reshapes.

    Only reshapes that leave the last axis intact are followed, since that
    is the axis a feature slice indexes.
    """
    node = g[name]
    last = node.type.shape[-1]
    while node.kind == "layout" and node.op in ("view", "clone", "cast") and len(node.inputs) == 1:
        if node.type.shape[-1] != last or not trace_layout(g, node.name).is_identity_reshape():
            return None
        node = g[node.inputs[0]]
    if node.op == "matmul" and node.type.shape[-1] == last and _is_const(g, node.inputs[1]):
        return node
    return None


def push_slice_into_gemm(g: OpGraph) -> int:
    """``slice(a @ W, cols)`` -> ``a @ W[:, cols]``.

    A model that projects once and splits the result (``z, x, B, C, dt`` of a
    Mamba block, or fused QKV) produces feature-axis slices of a GEMM. Slicing
    the constant weight instead turns each slice into its own GEMM sharing the
    same left operand, which is exactly the sibling-GEMM form a single kernel
    can accumulate (C3). Runs to a fixpoint so slices of slices collapse too.
    """
    n = 0
    while True:
        progressed = False
        for node in list(g):
            if node.op != "slice" or node.attrs.get("dim") != g[node.inputs[0]].type.rank - 1:
                continue
            src = _gemm_through_reshapes(g, node.inputs[0])
            if src is None:
                continue
            lo, hi = node.attrs["start"], node.attrs["end"]
            wc = g.add_const(g.value(src.inputs[1])[:, lo:hi].contiguous(), name=g.fresh_name(src.inputs[1] + "_cols"))
            _convert(node, "matmul", [src.inputs[0], wc.name], {}, type=node.type)
            n += 1
            progressed = True
        if not progressed:
            break
    g.toposort()
    return n


def match_attention(g: OpGraph) -> int:
    n = 0
    for sm in list(g):
        if sm.op != "softmax":
            continue
        s = _skip_casts(g, g[sm.inputs[0]])
        scale = 1.0
        if s.op == "mul" and "scalar" in s.attrs:
            scale = float(s.attrs["scalar"])
            s = _skip_casts(g, g[s.inputs[0]])
        prov_s = trace_layout(g, s.name)
        bmm1 = g[prov_s.base]
        if bmm1.op != "batch_matmul" or not prov_s.is_identity_reshape():
            continue
        prov_q = trace_layout(g, bmm1.inputs[0])
        prov_k = trace_layout(g, bmm1.inputs[1])
        dims = match_head_split(prov_q)
        if dims is None or match_head_split(prov_k, transpose_last=True) != dims:
            continue
        # consumer: softmax -> (layout) -> bmm(p, v)
        cons = [u for u, _ in _effective_users(g, sm.name)]
        if len(cons) != 1 or cons[0].op != "batch_matmul":
            continue
        bmm2 = cons[0]
        prov_p = trace_layout(g, bmm2.inputs[0])
        if prov_p.base != sm.name or not prov_p.is_identity_reshape():
            continue
        prov_v = trace_layout(g, bmm2.inputs[1])
        if match_head_split(prov_v) != dims:
            continue
        b, sq, h, d = dims["batch"], dims["seq"], dims["heads"], dims["head_dim"]
        expected = head_merge_index(b, sq, h, d)
        # follow layout users of bmm2 until the merged token-major view appears
        final = None
        cur = bmm2
        while True:
            users = g.users(cur.name)
            if len(users) != 1 or users[0].kind != "layout":
                break
            cur = users[0]
            prov = trace_layout(g, cur.name)
            if prov.base == bmm2.name and prov.index.numel() == expected.numel() and torch.equal(prov.index.reshape(-1), expected.reshape(-1)):
                final = cur
                break
        if final is None:
            continue
        q2d = _ensure_2d(g, prov_q.base)
        k2d = _ensure_2d(g, prov_k.base)
        v2d = _ensure_2d(g, prov_v.base)
        _convert(final, "attention", [q2d, k2d, v2d], {"batch": b, "seq": sq, "heads": h, "head_dim": d, "scale": scale}, type=TensorType((b * sq, h * d), final.type.dtype))
        n += 1
    g.toposort()
    return n


SSD_PAD = 32  # a per-token scalar is stored padded to 32 columns; see match_ssd


def _through_layout(g: OpGraph, name: str) -> OpNode:
    """The first non-layout producer of ``name``."""
    node = g[name]
    while node.kind == "layout" and len(node.inputs) == 1:
        node = g[node.inputs[0]]
    return node


def _mul_factors(g: OpGraph, node: OpNode, out: list[OpNode] | None = None) -> list[OpNode]:
    """Flatten a chain of two-operand ``mul`` nodes into its factors.

    Layout nodes between the multiplies are transparent: reshapes and
    broadcasts of a factor do not change which factor it is.
    """
    out = [] if out is None else out
    node = _through_layout(g, node.name) if node.kind == "layout" else node
    if node.op == "mul" and len(node.inputs) == 2:
        for i in node.inputs:
            _mul_factors(g, g[i], out)
    else:
        out.append(node)
    return out


def _ssd_constants(g: OpGraph, batch: int, seq: int, heads: int, dtype: torch.dtype):
    """The three constant matrices the SSD lowering needs.

    ``expand``  ``[H, H*32]`` replicates each head's scalar across 32 columns,
                so the scan kernel can load it as a ``[tile, 32]`` tile and
                reduce in place. A rank-2 load with a static-1 dimension
                rank-reduces and Loom's exploration pass leaves it
                un-bufferized, so the padding is what makes the load legal.
    ``decay``   ``[T, T]`` block lower-triangular minus half the block, so one
                GEMM produces the running decay already centred on the
                sequence midpoint. Centring halves the exponent range, which
                is what keeps ``exp`` inside fp16 over a full sequence.
    ``causal``  ``[S, S]`` lower-triangular mask.
    """
    idx = torch.arange(seq)
    causal = (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(dtype)
    tok = batch * seq
    b_of = torch.arange(tok) // seq
    same = (b_of.unsqueeze(1) == b_of.unsqueeze(0)).float()
    order = (torch.arange(tok).unsqueeze(1) >= torch.arange(tok).unsqueeze(0)).float()
    decay = (same * order - 0.5 * same).to(dtype)
    expand = torch.zeros(heads, heads * SSD_PAD, dtype=dtype)
    for h in range(heads):
        expand[h, h * SSD_PAD : (h + 1) * SSD_PAD] = 1.0
    return expand, decay, causal


def match_ssd(g: OpGraph) -> int:
    """The Mamba-2 scan in quadratic form -> one ``ssd`` macro op.

    Recognises ``(C Bᵀ ⊙ exp(cumA_i - cumA_j) ⊙ causal ⊙ dt_j) X`` where the
    operands are head-major views of token-major tensors, and rewrites it so
    every input the kernel reads is token-major:

    * ``dt`` is projected through a constant that both scales it by ``A`` and
      replicates each head across 32 columns, giving ``dA`` padded;
    * the running decay becomes one GEMM against a constant matrix, with the
      centring folded into that same matrix;
    * the scan itself becomes an ``ssd`` node carrying the geometry.

    The decay is applied as two column scales, ``exp(c_i)`` outside the loop
    and ``exp(-c_j)`` on the value tile, because a row vector cannot be loaded
    from memory. That factorisation is only safe because the centring bounds
    each exponent to half the sequence's total decay.
    """
    n = 0
    for y in list(g):
        if y.op != "batch_matmul":
            continue
        prov_x = trace_layout(g, y.inputs[1])
        dims = match_head_split(prov_x)
        if dims is None:
            continue
        factors = _mul_factors(g, _through_layout(g, y.inputs[0]))
        cbs = [f for f in factors if f.op == "batch_matmul"]
        exps = [f for f in factors if f.op == "exp"]
        if len(cbs) != 1 or len(exps) != 1:
            continue
        cb = cbs[0]
        prov_c = trace_layout(g, cb.inputs[0])
        prov_b = trace_layout(g, cb.inputs[1])
        dims_c = match_head_split(prov_c)
        dims_b = match_head_split(prov_b, transpose_last=True)
        # C and B are split by the state dim, X by the head dim; the batch,
        # sequence and head counts must agree, the trailing dim need not.
        key = lambda d: None if d is None else (d["batch"], d["seq"], d["heads"])
        if key(dims_c) != key(dims) or key(dims_b) != key(dims) or dims_c != dims_b:
            continue
        # the remaining factors are the causal mask and the dt broadcast
        rest = [f for f in factors if f not in (cb, exps[0])]
        dt_bases = [trace_layout(g, f.name).base for f in rest if not _is_const(g, f.name)]
        dt_tm = next((b for b in dt_bases if g[b].type.shape[-1] == dims["heads"]), None)
        if dt_tm is None:
            continue
        b, sq, h, p = dims["batch"], dims["seq"], dims["heads"], dims["head_dim"]
        state = dims_c["head_dim"]
        dtype = g[prov_x.base].type.torch_dtype
        expand, decay, causal = _ssd_constants(g, b, sq, h, dtype)

        # dt scaled by A and padded: one GEMM against a constant
        a_scale = _ssd_head_scale(g, exps[0], h)
        dt2d = _ensure_2d(g, dt_tm)
        tok = b * sq
        pad_t = TensorType((tok, h * SSD_PAD), g[dt2d].type.dtype)
        e_dt = g.add_const(expand, name=g.fresh_name("ssd_expand"))
        e_da = g.add_const(expand * a_scale.reshape(-1, 1), name=g.fresh_name("ssd_expand_A"))
        dt_pad = g.add("matmul", [dt2d, e_dt.name], pad_t, origin=f"match_ssd({y.name})")
        da_pad = g.add("matmul", [dt2d, e_da.name], pad_t, origin=f"match_ssd({y.name})")
        q_c = g.add_const(decay, name=g.fresh_name("ssd_decay"))
        cum_pad = g.add("matmul", [q_c.name, da_pad.name], pad_t, origin=f"match_ssd({y.name})")
        mask_c = g.add_const(causal, name=g.fresh_name("ssd_causal"))

        final, d_full = _ssd_skip(g, y, prov_x.base, dims, tok)
        if d_full is None:
            d_full = torch.zeros(tok, h * p)
        d_c = g.add_const(d_full.to(dtype), name=g.fresh_name("ssd_D"))

        out_t = TensorType((tok, h * p), dtype)
        node = g.add(
            "ssd",
            [_ensure_2d(g, prov_c.base), _ensure_2d(g, prov_b.base), _ensure_2d(g, prov_x.base),
             cum_pad.name, dt_pad.name, mask_c.name, d_c.name],
            out_t,
            {"batch": b, "seq": sq, "heads": h, "head_dim": p, "state_dim": state, "pad": SSD_PAD},
            origin=f"match_ssd({y.name})",
        )
        _rebuild_chain(g, [], node.name, final)
        n += 1
    g.toposort()
    return n


def _per_head_vector(g: OpGraph, node: OpNode, heads: int) -> torch.Tensor | None:
    """A constant that varies only along a head axis, reduced to ``[heads]``.

    Constant folding may already have expanded a per-head parameter to the
    full activation shape, so the head axis is found by testing which axis of
    length ``heads`` the values are constant across.
    """
    if not _is_const(g, node.name):
        return None
    v = g.value(node.name).float()
    if v.numel() == heads:
        return v.reshape(-1)
    for axis, size in enumerate(v.shape):
        if size != heads:
            continue
        moved = v.movedim(axis, 0).reshape(heads, -1)
        if torch.allclose(moved, moved[:, :1].expand_as(moved)):
            return moved[:, 0]
    return None


def _ssd_head_scale(g: OpGraph, exp_node: OpNode, heads: int) -> torch.Tensor:
    """Per-head ``A``: the constant scaling ``dt`` on the way into the scan.

    ``dA = dt * A`` is summed by the ``row_cumsum`` feeding the decay, so the
    constant is found on that node's input, not on the exponent itself.
    """
    seen: set[str] = set()
    stack = [exp_node.inputs[0]]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        node = g[name]
        if node.op == "row_cumsum":
            for f in _mul_factors(g, _through_layout(g, node.inputs[0])):
                vec = _per_head_vector(g, f, heads)
                if vec is not None:
                    return vec
        stack.extend(node.inputs)
    return torch.ones(heads)


def _ssd_skip(
    g: OpGraph, start: OpNode, x_base: str, dims: dict, tokens: int
) -> tuple[OpNode, torch.Tensor | None]:
    """Walk forward from the scan to the token-major result.

    Two things happen downstream of the scan and both have to be absorbed,
    because both read tensors through the head-major view that no kernel can
    address: the ``y + D·x`` skip connection, and the permute back to
    token-major order. Returns the node to replace and ``D`` per feature.
    """
    b, sq, h, p = dims["batch"], dims["seq"], dims["heads"], dims["head_dim"]
    head_major, d_full = start, None

    # 1. the skip connection, if present
    frontier, seen = [start], {start.name}
    while frontier:
        node = frontier.pop()
        for u in g.users(node.name):
            if u.name in seen:
                continue
            seen.add(u.name)
            if u.kind == "layout":
                frontier.append(u)
                continue
            if u.op != "add" or len(u.inputs) != 2:
                continue
            other = next((i for i in u.inputs if i not in seen), None)
            if other is None:
                continue
            # Peel exactly one `mul`: `x` is itself a product (the RMSNorm
            # rescale), so flattening the whole tree would dissolve it.
            prod = _through_layout(g, other)
            if prod.op != "mul" or len(prod.inputs) != 2:
                continue
            for a, c in ((0, 1), (1, 0)):
                vec = _per_head_vector(g, _through_layout(g, prod.inputs[c]), h)
                if vec is None or trace_layout(g, prod.inputs[a]).base != x_base:
                    continue
                d_full = vec.repeat_interleave(p).reshape(1, -1).expand(tokens, h * p).contiguous()
                head_major = u
                break
            if d_full is not None:
                frontier = []
                break

    # 2. the layout chain back to token-major order
    expected = head_merge_index(b, sq, h, p)
    cur = head_major
    while True:
        users = g.users(cur.name)
        if len(users) != 1 or users[0].kind != "layout":
            return head_major, d_full
        cur = users[0]
        prov = trace_layout(g, cur.name)
        if (
            prov.base == head_major.name
            and prov.index.numel() == expected.numel()
            and torch.equal(prov.index.reshape(-1), expected.reshape(-1))
        ):
            return cur, d_full


def flatten_to_2d(g: OpGraph) -> int:
    """Remove identity reshapes and give every activation a 2D type."""
    n = 0
    identity: list[tuple[str, str]] = []
    for node in list(g):
        if node.kind != "layout":
            continue
        if node.op in ("view", "clone", "expand", "cast", "permute", "slice") and len(node.inputs) == 1:
            # dtype casts are precision hints of the PyTorch decomposition
            # (e.g. silu upcasts to fp32); kernels compute in the model dtype.
            prov = trace_layout(g, node.name)
            if prov.is_identity_reshape():
                identity.append((node.name, prov.base))
                continue
        prov = trace_layout(g, node.name)
        raise PassError(
            f"layout op {node.name} ({node.op}) survived canonicalisation: "
            f"{node.type} is a reshape/permute view of {prov.base} {g[prov.base].type} "
            f"via {prov.chain}. Only views a kernel template already carries "
            f"(the head split of `attention`) can be absorbed; a kernel argument "
            f"cannot yet carry a general reshape+permute view."
        )
    for name, base in identity:
        g.replace_all_uses(name, base)
        n += 1
    g.dce()
    input_shapes = {}
    for node in g:
        if node.kind in ("param", "const"):
            continue
        if node.kind == "input":
            input_shapes[node.name] = list(node.type.shape)
        if node.op == "broadcast":
            node.attrs["shape"] = list(_twod(tuple(node.attrs["shape"])))
        node.type = TensorType(_twod(node.type.shape), node.type.dtype)
    act_dtype = g[g.inputs[0]].type.dtype if g.inputs else "float16"
    for node in g:
        if node.kind in ("param", "const", "input"):
            continue
        node.type = TensorType(node.type.shape, act_dtype)
    g.metadata.setdefault("input_shapes", input_shapes)
    g.metadata.setdefault("output_shapes", g.metadata.get("output_shapes_raw", {}))
    g.metadata["dtype"] = act_dtype
    return n


def rewrite_unregistered(g: OpGraph) -> int:
    n = 0
    for node in list(g):
        if node.kind != "compute":
            continue
        t = node.type
        x = node.inputs[0] if node.inputs else None
        if node.op == "neg":
            _convert(node, "mul", [x], {"scalar": -1.0})
        elif node.op == "rsqrt":
            _convert(node, "powf", [x], {"scalar": -0.5})
        elif node.op == "sqrt":
            _convert(node, "powf", [x], {"scalar": 0.5})
        elif node.op == "square":
            _convert(node, "mul", [x, x], {})
        elif node.op in ("sigmoid", "silu", "tanh"):
            src = x
            if node.op == "tanh":
                src = g.add("mul", [x], t, {"scalar": 2.0}).name
            neg = g.add("mul", [src], t, {"scalar": -1.0})
            e = g.add("exp", [neg.name], t)
            den = g.add("add", [e.name], t, {"scalar": 1.0})
            if node.op == "sigmoid":
                _convert(node, "div", [den.name], {"scalar": 1.0, "scalar_first": True})
            elif node.op == "silu":
                sig = g.add("div", [den.name], t, {"scalar": 1.0, "scalar_first": True})
                _convert(node, "mul", [x, sig.name], {})
            else:
                sig = g.add("div", [den.name], t, {"scalar": 1.0, "scalar_first": True})
                two = g.add("mul", [sig.name], t, {"scalar": 2.0})
                _convert(node, "add", [two.name], {"scalar": -1.0})
        elif node.op == "log1p":
            _convert(node, "log", [g.add("add", [x], t, {"scalar": 1.0}).name], {})
        elif node.op == "softplus":
            e = g.add("exp", [x], t)
            one = g.add("add", [e.name], t, {"scalar": 1.0})
            _convert(node, "log", [one.name], {})
        elif node.op == "row_mean":
            k = g[x].type.shape[-1]
            s = g.add("row_sum", [x], t)
            _convert(node, "mul", [s.name], {"scalar": 1.0 / k})
        elif node.op in ("softmax", "rmsnorm", "rope", "rotate_half", "row_cumsum", "cmp_gt", "select"):
            raise PassError(f"{node.name}: '{node.op}' could not be lowered to a kernel template")
        else:
            continue
        n += 1
    for node in g:
        if node.op in BINARY_OPS and "scalar" not in node.attrs:
            for i in node.inputs:
                if g[i].kind in ("param", "const") and g[i].type.rank < 2:
                    raise PassError(f"{node.name}: rank-{g[i].type.rank} operand {i} must be folded into weights (C5)")
    g.toposort()
    return n


def materialize_const_operands(g: OpGraph) -> int:
    """Expand a rank-1 constant operand into a full 2-D constant.

    A per-feature bias or scale cannot be folded into a weight when something
    non-linear sits between it and the GEMM (Mamba's ``softplus(dt + dt_bias)``
    is the case in point). Since rank-1 operands cannot live in L1 (C5), the
    constant is materialised at the activation's shape instead. These are
    small: one row of the activation, repeated.
    """
    n = 0
    for node in list(g):
        if node.op not in BINARY_OPS or "scalar" in node.attrs or len(node.inputs) != 2:
            continue
        for idx in (0, 1):
            c, other = g[node.inputs[idx]], g[node.inputs[1 - idx]]
            if c.kind not in ("param", "const") or c.type.shape == other.type.shape:
                continue
            if other.kind in ("param", "const"):
                continue
            rows, cols = other.type.shape
            v = g.value(c.name).float()
            if v.numel() == cols:
                full = v.reshape(1, cols).expand(rows, cols)
            elif v.numel() == rows:
                full = v.reshape(rows, 1).expand(rows, cols)
            elif v.numel() == 1:
                full = v.reshape(1, 1).expand(rows, cols)
            else:
                continue
            new = g.add_const(full.contiguous().to(other.type.torch_dtype), name=g.fresh_name(c.name + "_full"))
            node.inputs[idx] = new.name
            n += 1
    g.toposort()
    return n


def insert_broadcasts(g: OpGraph) -> int:
    n = 0
    for node in list(g):
        if node.op not in BINARY_OPS or "scalar" in node.attrs or len(node.inputs) != 2:
            continue
        a, b = g[node.inputs[0]], g[node.inputs[1]]
        if a.type.shape == b.type.shape:
            continue
        for idx, (src, other) in enumerate(((a, b), (b, a))):
            if src.type.rank == 2 and other.type.rank == 2 and src.type.shape[1] == 1 and src.type.shape[0] == other.type.shape[0] and other.type.shape[1] > 1:
                bc = g.add("broadcast", [src.name], other.type, {"dim": 1, "shape": list(other.type.shape)})
                node.inputs[idx] = bc.name
                n += 1
                break
        else:
            raise PassError(f"{node.name}: cannot broadcast {a.type} with {b.type}")
    g.toposort()
    return n


def dedupe_constants(g: OpGraph) -> int:
    """Merge const nodes holding identical tensors (e.g. RoPE tables)."""
    seen: dict[tuple, str] = {}
    n = 0
    for node in list(g):
        if node.kind != "const":
            continue
        v = g.value(node.name)
        key = (node.type.shape, node.type.dtype, v.contiguous().view(-1).cpu().numpy().tobytes() if v.numel() < (1 << 22) else id(v))
        if key in seen:
            g.replace_all_uses(node.name, seen[key])
            n += 1
        else:
            seen[key] = node.name
    return n


def cleanup(g: OpGraph) -> int:
    n = g.dce()
    g.toposort()
    return n


PASSES: list[tuple[str, Callable[[OpGraph], int]]] = [
    ("fold_constants", fold_constants),
    ("match_softplus", match_softplus),
    ("match_rotate_half", match_rotate_half),
    ("match_rope", match_rope),
    ("lower_rope", lower_rope),
    ("match_rmsnorm", match_rmsnorm),
    # Before the norm is lowered, so a slice still sits directly on its GEMM.
    ("push_slice_into_gemm", push_slice_into_gemm),
    ("lower_rmsnorm", lower_rmsnorm),
    ("match_attention", match_attention),
    ("match_ssd", match_ssd),
    ("lower_cumsum", lower_cumsum),
    ("fold_constants_2", fold_constants),
    ("cleanup_1", cleanup),
    ("flatten_to_2d", flatten_to_2d),
    ("materialize_const_operands", materialize_const_operands),
    ("rewrite_unregistered", rewrite_unregistered),
    ("insert_broadcasts", insert_broadcasts),
    ("fold_constants_3", fold_constants),
    ("dedupe_constants", dedupe_constants),
    ("cleanup_2", cleanup),
]


def canonicalize(graph: OpGraph, verify_inputs: list[torch.Tensor] | None = None, tolerance: float = 2e-2) -> tuple[OpGraph, list[dict]]:
    """Run all passes on a copy of ``graph``; return ``(graph, log)``.

    With ``verify_inputs`` the graph outputs are re-evaluated after every
    pass and compared with the raw graph (max relative error).
    """
    g = graph.copy()
    g.metadata["output_shapes_raw"] = {o: list(g[o].type.shape) for o in g.outputs}
    ref = graph_outputs(g, verify_inputs) if verify_inputs is not None else None
    log: list[dict] = []
    for name, fn in PASSES:
        count = fn(g)
        entry = {"pass": name, "rewrites": count, "nodes": len(g)}
        if ref is not None:
            outs = graph_outputs(g, verify_inputs)
            err = max(max_rel_err(o.reshape(-1), r.reshape(-1)) for o, r in zip(outs, ref))
            entry["max_rel_err"] = err
            if err > tolerance:
                raise PassError(f"pass {name} changed the numerics (max rel err {err:.3e})")
        log.append(entry)
    return g, log
