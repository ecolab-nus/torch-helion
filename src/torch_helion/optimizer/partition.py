"""Partition the canonical OpGraph into Loom-compatible kernels.

Every kernel is anchored by a reduction: a group of GEMMs that share their
LHS (constraint C3) or one ``attention`` op. Non-anchor ops (elementwise,
broadcast, the RMSNorm column chain) are pulled into the epilogue of the
kernel that owns their inputs. Values that cross kernels are materialised
as DRAM tensors.

The *grouping* — which sibling GEMMs share a kernel — is the planner's
decision; :func:`build_program` is deterministic given a grouping so the
planner can evaluate many alternatives with the cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..capture.opgraph import OpGraph, OpNode, TensorType
from ..capture.ops import BINARY_OPS, UNARY_OPS
from .opir import BodyOp, GemmSpec, KernelArg, KernelOutput, KernelSpec, Program


class PartitionError(RuntimeError):
    pass


Grouping = dict[str, int]  # matmul node name -> group id


def sibling_sets(g: OpGraph) -> list[list[str]]:
    """GEMMs sharing the same LHS tensor and output width, in graph order."""
    by_lhs: dict[tuple[str, int], list[str]] = {}
    for n in g:
        if n.op == "matmul":
            by_lhs.setdefault((n.inputs[0], n.type.shape[1]), []).append(n.name)
    return list(by_lhs.values())


def default_grouping(g: OpGraph, merge_siblings: bool = True) -> Grouping:
    grouping: Grouping = {}
    gid = 0
    for group in sibling_sets(g):
        if merge_siblings:
            for m in group:
                grouping[m] = gid
            gid += 1
        else:
            for m in group:
                grouping[m] = gid
                gid += 1
    return grouping


@dataclass
class _Kernel:
    spec: KernelSpec
    owned: dict[str, str] = field(default_factory=dict)  # graph value -> in-kernel value name
    order_key: int = 0
    anchor_key: int = 0  # graph position of the anchor, never widened
    lhs_node: str | None = None
    extra_args: dict[str, str] = field(default_factory=dict)  # tensor -> arg name
    column_cache: dict[str, str] = field(default_factory=dict)


def _grid_shape(k: "_Kernel") -> tuple[int, int]:
    """The [rows, cols] tile frame a kernel's epilogue works in."""
    if k.spec.kind == "ssd":
        a = k.spec.attrs
        return (a["batch"] * a["seq"], a["heads"] * a["head_dim"])
    return (k.spec.grid["rows"], k.spec.grid["cols"])


def _fits(k: "_Kernel", node: OpNode) -> bool:
    """Whether ``node`` can be an epilogue op of ``k``.

    An epilogue reads and writes at the kernel's output tile, so a value of a
    different shape has no tile to live in.
    """
    return _grid_shape(k) == node.type.shape


def _is_column_chain(g: OpGraph, name: str, memo: dict[str, str | None]) -> str | None:
    """If ``name`` is derived solely from ``row_sumsq(x)`` and scalars, return x."""
    if name in memo:
        return memo[name]
    node = g[name]
    result: str | None = None
    if node.op == "row_sumsq":
        result = node.inputs[0]
    elif node.op in BINARY_OPS or node.op in UNARY_OPS:
        srcs = {_is_column_chain(g, i, memo) for i in node.inputs}
        srcs.discard(None)
        if len(srcs) == 1 and all(_is_column_chain(g, i, memo) is not None for i in node.inputs):
            result = srcs.pop()
    memo[name] = result
    return result


def build_program(g: OpGraph, grouping: Grouping | None = None, name: str = "program") -> Program:
    grouping = grouping if grouping is not None else default_grouping(g)
    order = {n.name: i for i, n in enumerate(g)}
    kernels: dict[int, _Kernel] = {}
    owner: dict[str, _Kernel] = {}
    column_memo: dict[str, str | None] = {}
    tensors: dict[str, TensorType] = {n.name: n.type for n in g}

    # --- anchors -----------------------------------------------------------
    for n in g:
        if n.op == "matmul":
            gid = grouping[n.name]
            lhs, rhs = n.inputs
            if g[rhs].kind not in ("param", "const") and g[lhs].kind not in ("param", "const"):
                raise PartitionError(f"{n.name}: neither GEMM operand is constant (activation-activation GEMMs need a dedicated template)")
            if gid not in kernels:
                spec = KernelSpec(name=f"k{len(kernels)}_gemm", kind="gemm", lhs="lhs", k_extent=g[lhs].type.shape[1], grid={"rows": n.type.shape[0], "cols": n.type.shape[1]})
                spec.args.append(KernelArg("lhs", lhs, "lhs"))
                kernels[gid] = _Kernel(spec, lhs_node=lhs, order_key=order[n.name], anchor_key=order[n.name])
            k = kernels[gid]
            if k.lhs_node != lhs or k.spec.grid["cols"] != n.type.shape[1]:
                raise PartitionError(f"{n.name}: grouped with GEMMs of a different LHS/width")
            arg = f"w{len(k.spec.gemms)}"
            k.spec.args.append(KernelArg(arg, rhs, "rhs"))
            acc = f"acc_{n.name}"
            k.spec.gemms.append(GemmSpec(acc, arg))
            k.owned[n.name] = acc
            owner[n.name] = k
        elif n.op == "ssd":
            spec = KernelSpec(name=f"k{len(kernels)}_ssd", kind="ssd", attrs=dict(n.attrs))
            a = n.attrs
            hp = [a["batch"], a["seq"], a["heads"], a["head_dim"]]
            hn = [a["batch"], a["seq"], a["heads"], a["state_dim"]]
            hpad = [a["batch"], a["seq"], a["heads"], a["pad"]]
            for role, src, view in zip(
                ("c", "b", "x", "cum", "dt", "mask", "dskip"), n.inputs,
                (hn, hn, hp, hpad, hpad, None, hp),
            ):
                spec.args.append(KernelArg(role, src, role, view))
            # `x` is read at two different tiles — inside the loop at tile_n and
            # in the epilogue at tile_m. Helion cannot index one viewed tensor at
            # two tiles, so the skip read gets its own argument.
            spec.args.append(KernelArg("xskip", n.inputs[2], "xskip", hp))
            k = _Kernel(spec, order_key=order[n.name], anchor_key=order[n.name])
            k.owned[n.name] = "acc"
            kernels[3000 + len(kernels)] = k
            owner[n.name] = k
        elif n.op == "attention":
            spec = KernelSpec(name=f"k{len(kernels)}_attention", kind="attention", attrs=dict(n.attrs))
            a = n.attrs
            view = [a["batch"], a["seq"], a["heads"], a["head_dim"]]
            for role, src in zip(("q", "k", "v"), n.inputs):
                spec.args.append(KernelArg(role, src, role, view))
            spec.outputs.append(KernelOutput(n.name, "out"))
            k = _Kernel(spec, order_key=order[n.name], anchor_key=order[n.name])
            k.owned[n.name] = "out"
            kernels[len(kernels) + 1000] = k
            owner[n.name] = k
        elif n.op == "batch_matmul":
            raise PartitionError(f"{n.name}: standalone batch_matmul has no kernel template")

    # --- reductions with no GEMM home anchor their own kernel ------------------
    # A standalone normalisation is a legal Loom kernel: the grid tiles the
    # rows, the loop reduces over the feature axis carrying the sum of
    # squares, and the epilogue rescales the row. It has no GEMM, which C2
    # permits as long as the loop carries a ranked tensor.
    lhs_nodes = {k.lhs_node for k in kernels.values() if k.lhs_node}
    for n in g:
        if n.op != "row_sumsq" or n.inputs[0] in lhs_nodes:
            continue
        x = n.inputs[0]
        rows, cols = g[x].type.shape
        spec = KernelSpec(name=f"k{len(kernels)}_norm", kind="gemm", lhs="lhs", k_extent=cols, grid={"rows": rows, "cols": cols})
        spec.args.append(KernelArg("lhs", x, "lhs"))
        spec.sumsq = True
        k = _Kernel(spec, lhs_node=x, order_key=order[n.name], anchor_key=order[n.name])
        k.column_cache[n.name] = "sumsq"
        kernels[2000 + len(kernels)] = k
        owner[n.name] = k

    # --- epilogue assignment -------------------------------------------------
    for n in g:
        if n.kind != "compute" or n.op in ("matmul", "attention", "ssd", "row_sumsq"):
            continue
        if _is_column_chain(g, n.name, column_memo) is not None:
            continue  # cloned into consumers on demand
        homes = [owner[i] for i in n.inputs if i in owner and owner[i].spec.kind in ("gemm", "ssd")]
        homes = [k for k in homes if _fits(k, n)]
        if any(i in owner and owner[i].spec.kind == "attention" for i in n.inputs):
            raise PartitionError(f"{n.name}: elementwise consumer of an attention output must be absorbed by a GEMM kernel (no attention epilogue template)")
        column_srcs = [_is_column_chain(g, i, column_memo) for i in n.inputs]
        column_srcs = [c for c in column_srcs if c is not None]
        if not homes and column_srcs:
            # An elementwise op combining a DRAM tensor with a row scale
            # derived from it belongs in the kernel that computes the scale.
            owners = [k for k in kernels.values() if k.lhs_node in column_srcs and _fits(k, n)]
            if owners:
                homes = [max(owners, key=lambda k: k.order_key)]
            else:
                continue  # broadcast(r): defer to whichever kernel needs it
        if not homes:
            raise PartitionError(f"{n.name} ({n.op}): pure elementwise op on DRAM tensors has no reduction loop to live in (C2)")
        # Choose by anchor position, not by how far a kernel's epilogue has
        # already grown: a kernel that produces this node's other operands
        # cannot also consume its result.
        home = max(homes, key=lambda k: k.anchor_key)
        home.order_key = max(home.order_key, order[n.name])
        args = [_materialize_input(g, i, home, owner, column_memo, tensors) for i in n.inputs]
        home.spec.epilogue.append(BodyOp(n.name, n.op, args, dict(n.attrs)))
        home.owned[n.name] = n.name
        owner[n.name] = home

    # --- outputs: values consumed outside their kernel ----------------------------
    for value, k in list(owner.items()):
        if k.spec.kind == "attention":
            continue
        if k.spec.kind == "ssd" and value not in k.owned:
            continue
        needed = value in g.outputs or any(u.name not in k.owned for u in g.users(value) if u.name in owner or u.op in ("attention", "matmul"))
        needed = needed or any(a.tensor == value for kk in kernels.values() for a in kk.spec.args if kk is not k)
        if needed:
            k.spec.outputs.append(KernelOutput(value, k.owned[value]))

    # --- program ----------------------------------------------------------------
    ordered = _order_kernels(list(kernels.values()))
    for i, k in enumerate(ordered):
        k.spec.name = f"k{i}_{k.spec.kind}"
    params = sorted({a.tensor for k in ordered for a in k.spec.args if g[a.tensor].kind in ("param", "const")})
    prog = Program(name=name, inputs=list(g.inputs), outputs=list(g.outputs), tensors=tensors, kernels=[k.spec for k in ordered], params=params, metadata=dict(g.metadata))
    prog.metadata["grouping"] = dict(grouping)
    return prog


def _materialize_input(g: OpGraph, src: str, home: _Kernel, owner: dict[str, _Kernel], column_memo: dict, tensors: dict[str, TensorType]) -> str:
    """Return the in-kernel name for graph value ``src`` used by ``home``."""
    if src in home.owned:
        return home.owned[src]
    col_src = _is_column_chain(g, src, column_memo)
    if col_src is not None:
        return _clone_column_chain(g, src, home, column_memo)
    node = g[src]
    if node.op == "broadcast":
        inner = node.inputs[0]
        if _is_column_chain(g, inner, column_memo) is not None:
            inner_name = _clone_column_chain(g, inner, home, column_memo)
            key = f"bcast:{inner}:{node.attrs.get('shape')}"
            if key not in home.column_cache:
                bname = f"{node.name}__{home.spec.name}"
                home.spec.epilogue.append(BodyOp(bname, "broadcast", [inner_name], dict(node.attrs)))
                home.column_cache[key] = bname
            home.owned[node.name] = home.column_cache[key]
            return home.column_cache[key]
    # DRAM tensor (input, const, or another kernel's value)
    if src in home.extra_args:
        return home.extra_args[src]
    arg = f"in{len(home.extra_args)}"
    # The scan kernel reads every operand through a 4-D head-major view, so an
    # epilogue input has to arrive shaped that way too: a 2-D argument reshaped
    # inside the kernel gets dynamic dimensions and a `tensor.cast` Loom's
    # shape tracer cannot follow.
    view = None
    if home.spec.kind == "ssd":
        a = home.spec.attrs
        view = [a["batch"], a["seq"], a["heads"], a["head_dim"]]
    home.spec.args.append(KernelArg(arg, src, "extra", view))
    home.extra_args[src] = arg
    return arg


def _clone_column_chain(g: OpGraph, name: str, home: _Kernel, column_memo: dict) -> str:
    if name in home.column_cache:
        return home.column_cache[name]
    node = g[name]
    x = _is_column_chain(g, name, column_memo)
    if node.op == "row_sumsq":
        if home.lhs_node != x:
            raise PartitionError(f"{home.spec.name}: needs row sum-of-squares of {x} but its LHS is {home.lhs_node}")
        home.spec.sumsq = True
        home.column_cache[name] = "sumsq"
        return "sumsq"
    inputs = [_clone_column_chain(g, i, home, column_memo) for i in node.inputs]
    local = f"{name}__col"
    home.spec.epilogue.append(BodyOp(local, node.op, inputs, dict(node.attrs)))
    home.column_cache[name] = local
    return local


def _order_kernels(kernels: list[_Kernel]) -> list[_Kernel]:
    produced = {o.tensor: k for k in kernels for o in k.spec.outputs}
    done: list[_Kernel] = []
    remaining = sorted(kernels, key=lambda k: k.order_key)
    while remaining:
        progressed = False
        for k in list(remaining):
            deps = [produced[a.tensor] for a in k.spec.args if a.tensor in produced and produced[a.tensor] is not k]
            if all(d in done for d in deps):
                done.append(k)
                remaining.remove(k)
                progressed = True
        if not progressed:
            raise PartitionError("cyclic kernel dependencies")
    return done
