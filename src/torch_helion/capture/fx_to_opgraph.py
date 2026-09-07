"""ATen FX graph → raw :class:`OpGraph`.

The conversion is deliberately literal: every ATen node becomes one OpGraph
node with the same shape, and layout ops are kept as ``layout`` nodes. All
semantic rewriting is left to :mod:`torch_helion.optimizer.passes`, so the
raw graph stored in ``results/`` is an honest picture of what the model
does.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.fx import Node

from .fx_capture import CapturedFx
from .opgraph import OpGraph, OpNode, TensorType


class UnsupportedOpError(NotImplementedError):
    """An ATen op has no OpGraph mapping."""


_HANDLERS: dict[str, Callable[["_Converter", Node], OpNode]] = {}


def _handler(*targets: str):
    def deco(fn):
        for t in targets:
            _HANDLERS[t] = fn
        return fn

    return deco


def _tt(node: Node) -> TensorType:
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise UnsupportedOpError(f"{node.name}: non-tensor value {type(val)}")
    return TensorType(tuple(int(s) for s in val.shape), str(val.dtype).replace("torch.", ""))


def _norm_dim(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


class _Converter:
    def __init__(self, captured: CapturedFx) -> None:
        self.captured = captured
        self.ep = captured.exported
        self.gm = captured.graph_module
        self.graph = OpGraph("model")
        self.names: dict[str, str] = {}  # fx node name -> opgraph node name
        # `split_with_sizes` yields a list; the `getitem` that reads it becomes
        # the slice. Records fx node name -> (source, sizes, dim).
        self.splits: dict[str, tuple[str, list[int], int]] = {}

    # ------------------------------------------------------------- helpers
    def src(self, arg: Any) -> str:
        if isinstance(arg, Node):
            return self.names[arg.name]
        raise UnsupportedOpError(f"expected tensor node, got {arg!r}")

    def emit(self, node: Node, op: str, inputs: list[str], attrs: dict | None = None, kind: str = "compute") -> OpNode:
        out = self.graph.add(op, inputs, _tt(node), attrs=attrs, kind=kind, name=node.name, origin=str(node.target))
        self.names[node.name] = out.name
        return out

    def binary(self, node: Node, op: str) -> OpNode:
        a, b = node.args[0], node.args[1]
        if isinstance(a, Node) and isinstance(b, Node):
            return self.emit(node, op, [self.src(a), self.src(b)])
        if isinstance(a, Node):
            return self.emit(node, op, [self.src(a)], {"scalar": float(b)})
        # scalar op tensor: keep operand order via 'scalar_first'
        return self.emit(node, op, [self.src(b)], {"scalar": float(a), "scalar_first": True})

    # ---------------------------------------------------------------- run
    def run(self) -> OpGraph:
        sig = self.ep.graph_signature
        params = dict(self.ep.state_dict)
        constants = getattr(self.ep, "constants", {}) or {}
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                t = _tt(node)
                if node.name in sig.inputs_to_parameters:
                    pname = sig.inputs_to_parameters[node.name]
                    self.emit_value(node, "param", pname, params[pname], t)
                elif node.name in sig.inputs_to_buffers:
                    bname = sig.inputs_to_buffers[node.name]
                    value = params.get(bname, constants.get(bname))
                    self.emit_value(node, "param", bname, value, t)
                elif node.name in getattr(sig, "inputs_to_lifted_tensor_constants", {}):
                    cname = sig.inputs_to_lifted_tensor_constants[node.name]
                    self.emit_value(node, "const", cname, constants[cname], t)
                else:
                    self.emit_value(node, "input", node.name, None, t)
            elif node.op == "call_function":
                key = str(node.target)
                if key == "aten.split_with_sizes.default":
                    src = node.args[0]
                    rank = len(_tt(src).shape)
                    dim = _norm_dim(int(node.args[2]) if len(node.args) > 2 else 0, rank)
                    self.splits[node.name] = (self.src(src), [int(v) for v in node.args[1]], dim)
                    continue
                fn = _HANDLERS.get(key)
                if fn is None:
                    raise UnsupportedOpError(f"no OpGraph mapping for {key} (node {node.name})")
                fn(self, node)
            elif node.op == "output":
                outs = node.args[0]
                if not isinstance(outs, (tuple, list)):
                    outs = (outs,)
                self.graph.outputs = [self.src(o) for o in outs]
            elif node.op == "get_attr":
                raise UnsupportedOpError(f"get_attr {node.target} not supported (export should lift it)")
            else:
                raise UnsupportedOpError(f"unsupported fx op {node.op}")
        self.graph.metadata["fx_op_histogram"] = self.captured.op_histogram()
        return self.graph

    def emit_value(self, node: Node, kind: str, pname: str, value: torch.Tensor | None, t: TensorType) -> None:
        name = pname.replace(".", "_")
        out = self.graph.add(kind, [], t, kind=kind, name=name, origin=node.name, value=value.detach().clone() if value is not None else None)
        self.names[node.name] = out.name


# ---------------------------------------------------------------- handlers
@_handler("aten.add.Tensor")
def _add(c, n):
    return c.binary(n, "add")


@_handler("aten.sub.Tensor")
def _sub(c, n):
    return c.binary(n, "sub")


@_handler("aten.mul.Tensor")
def _mul(c, n):
    return c.binary(n, "mul")


@_handler("aten.div.Tensor")
def _div(c, n):
    return c.binary(n, "div")


@_handler("aten.maximum.default")
def _max(c, n):
    return c.binary(n, "max")


@_handler("aten.pow.Tensor_Scalar")
def _pow_scalar(c, n):
    return c.emit(n, "powf", [c.src(n.args[0])], {"scalar": float(n.args[1])})


@_handler("aten.pow.Tensor_Tensor")
def _pow_tensor(c, n):
    return c.emit(n, "powf", [c.src(n.args[0]), c.src(n.args[1])])


for _name, _op in {
    "aten.exp.default": "exp",
    "aten.log.default": "log",
    "aten.neg.default": "neg",
    "aten.rsqrt.default": "rsqrt",
    "aten.sqrt.default": "sqrt",
    "aten.sigmoid.default": "sigmoid",
    "aten.silu.default": "silu",
    "aten.tanh.default": "tanh",
    "aten.square.default": "square",
    "aten.log1p.default": "log1p",
}.items():

    def _unary(c, n, _op=_op):
        return c.emit(n, _op, [c.src(n.args[0])])

    _HANDLERS[_name] = _unary


@_handler("aten.mean.dim", "aten.sum.dim_IntList", "aten.amax.default")
def _reduce(c, n):
    x = n.args[0]
    dims = n.args[1] if len(n.args) > 1 else None
    keepdim = bool(n.args[2]) if len(n.args) > 2 else bool(n.kwargs.get("keepdim", False))
    rank = len(_tt(x).shape)
    if isinstance(dims, int):
        dims = [dims]
    dims = [_norm_dim(d, rank) for d in (dims or [])]
    if dims != [rank - 1]:
        raise UnsupportedOpError(f"{n.name}: only last-dim reductions are supported (dims={dims})")
    op = {"aten.mean.dim": "row_mean", "aten.sum.dim_IntList": "row_sum", "aten.amax.default": "row_max"}[str(n.target)]
    return c.emit(n, op, [c.src(x)], {"keepdim": keepdim})


@_handler("aten._softmax.default", "aten.softmax.int")
def _softmax(c, n):
    rank = len(_tt(n.args[0]).shape)
    dim = _norm_dim(int(n.args[1]), rank)
    if dim != rank - 1:
        raise UnsupportedOpError(f"{n.name}: softmax only over the last dim")
    return c.emit(n, "softmax", [c.src(n.args[0])])


@_handler("aten.mm.default")
def _mm(c, n):
    return c.emit(n, "matmul", [c.src(n.args[0]), c.src(n.args[1])])


@_handler("aten.bmm.default")
def _bmm(c, n):
    return c.emit(n, "batch_matmul", [c.src(n.args[0]), c.src(n.args[1])])


@_handler("aten.addmm.default")
def _addmm(c, n):
    bias, a, b = n.args[:3]
    mm = c.graph.add("matmul", [c.src(a), c.src(b)], _tt(n), name=n.name + "_mm", origin="aten.addmm")
    return c.emit(n, "add", [mm.name, c.src(bias)])


@_handler("aten.permute.default")
def _permute(c, n):
    return c.emit(n, "permute", [c.src(n.args[0])], {"dims": [int(d) for d in n.args[1]]}, kind="layout")


@_handler("aten.transpose.int")
def _transpose(c, n):
    rank = len(_tt(n.args[0]).shape)
    d0, d1 = _norm_dim(int(n.args[1]), rank), _norm_dim(int(n.args[2]), rank)
    dims = list(range(rank))
    dims[d0], dims[d1] = dims[d1], dims[d0]
    return c.emit(n, "permute", [c.src(n.args[0])], {"dims": dims}, kind="layout")


@_handler("aten.t.default")
def _t(c, n):
    return c.emit(n, "permute", [c.src(n.args[0])], {"dims": [1, 0]}, kind="layout")


@_handler("aten.view.default", "aten.reshape.default", "aten._unsafe_view.default")
def _view(c, n):
    return c.emit(n, "view", [c.src(n.args[0])], {"shape": list(_tt(n).shape)}, kind="layout")


@_handler("aten.unsqueeze.default", "aten.squeeze.dim", "aten.squeeze.default", "aten.squeeze.dims")
def _squeeze(c, n):
    return c.emit(n, "view", [c.src(n.args[0])], {"shape": list(_tt(n).shape)}, kind="layout")


@_handler("aten.expand.default")
def _expand(c, n):
    return c.emit(n, "expand", [c.src(n.args[0])], {"shape": list(_tt(n).shape)}, kind="layout")


@_handler("aten.clone.default", "aten.contiguous.default", "aten.alias.default", "aten.detach.default")
def _clone(c, n):
    return c.emit(n, "clone", [c.src(n.args[0])], kind="layout")


@_handler("aten._to_copy.default", "aten.to.dtype")
def _to_copy(c, n):
    t = _tt(n)
    return c.emit(n, "cast", [c.src(n.args[0])], {"dtype": t.dtype}, kind="layout")


@_handler("aten.slice.Tensor")
def _slice(c, n):
    x = n.args[0]
    rank = len(_tt(x).shape)
    dim = _norm_dim(int(n.args[1]) if len(n.args) > 1 else 0, rank)
    size = _tt(x).shape[dim]
    start = int(n.args[2]) if len(n.args) > 2 and n.args[2] is not None else 0
    end = int(n.args[3]) if len(n.args) > 3 and n.args[3] is not None else size
    step = int(n.args[4]) if len(n.args) > 4 else 1
    if step != 1:
        raise UnsupportedOpError(f"{n.name}: strided slice not supported")
    start = max(0, start + size if start < 0 else start)
    end = min(size, end + size if end < 0 else end)
    return c.emit(n, "slice", [c.src(x)], {"dim": dim, "start": start, "end": end}, kind="layout")


@_handler("<built-in function getitem>")
def _getitem(c, n):
    src, index = n.args[0], int(n.args[1])
    if not isinstance(src, Node) or src.name not in c.splits:
        raise UnsupportedOpError(f"{n.name}: getitem on {src} is not a tensor split")
    base, sizes, dim = c.splits[src.name]
    start = sum(sizes[:index])
    return c.emit(n, "slice", [base], {"dim": dim, "start": start, "end": start + sizes[index]}, kind="layout")


@_handler("aten.cumsum.default")
def _cumsum(c, n):
    x = n.args[0]
    rank = len(_tt(x).shape)
    dim = _norm_dim(int(n.args[1]) if len(n.args) > 1 else -1, rank)
    if dim != rank - 1:
        raise UnsupportedOpError(f"{n.name}: only last-dim cumsum is supported (dim={dim})")
    return c.emit(n, "row_cumsum", [c.src(x)], {})


@_handler("aten.gt.Scalar")
def _gt(c, n):
    return c.emit(n, "cmp_gt", [c.src(n.args[0])], {"scalar": float(n.args[1])})


@_handler("aten.where.self")
def _where(c, n):
    return c.emit(n, "select", [c.src(a) for a in n.args[:3]])


@_handler("aten.cat.default")
def _cat(c, n):
    parts = [c.src(p) for p in n.args[0]]
    rank = len(_tt(n).shape)
    dim = _norm_dim(int(n.args[1]) if len(n.args) > 1 else 0, rank)
    return c.emit(n, "cat", parts, {"dim": dim}, kind="layout")


@_handler("aten.linear.default")
def _linear(c, n):
    x, w = n.args[0], n.args[1]
    wt = c.graph.add("permute", [c.src(w)], TensorType(tuple(reversed(_tt(w).shape)), _tt(w).dtype), {"dims": [1, 0]}, kind="layout", name=n.name + "_wT", origin="aten.linear")
    xt = _tt(x)
    if len(xt.shape) != 2:
        raise UnsupportedOpError(f"{n.name}: linear on rank-{len(xt.shape)} input; export should have flattened it")
    if len(n.args) > 2 and n.args[2] is not None:
        mm = c.graph.add("matmul", [c.src(x), wt.name], _tt(n), name=n.name + "_mm", origin="aten.linear")
        return c.emit(n, "add", [mm.name, c.src(n.args[2])])
    return c.emit(n, "matmul", [c.src(x), wt.name])


def fx_to_opgraph(captured: CapturedFx) -> OpGraph:
    """Convert an exported program to a raw OpGraph (params carry values)."""
    return _Converter(captured).run()


def supported_aten_ops() -> list[str]:
    return sorted(_HANDLERS)
