"""Reference evaluation of OpGraph nodes with plain PyTorch.

Used for (1) constant folding of parameter-only sub-graphs, (2) numerical
regression tests of every optimizer pass, and (3) executing the OpIR
reference interpreter. Computation happens in float32 and is cast back to
the node's dtype, mirroring how the hardware accumulates.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..capture.opgraph import OpGraph


def _scalar_binary(op: str, a: torch.Tensor, s: float, scalar_first: bool) -> torch.Tensor:
    x, y = (s, a) if scalar_first else (a, s)
    if op == "add":
        return x + y
    if op == "sub":
        return x - y
    if op == "mul":
        return x * y
    if op == "div":
        return x / y
    if op == "max":
        return torch.maximum(a, torch.full_like(a, s))
    if op == "powf":
        return torch.pow(x, y) if not scalar_first else torch.pow(torch.full_like(a, s), a)
    raise ValueError(op)


def eval_op(op: str, args: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
    """Evaluate one op on concrete tensors (float32 math)."""
    xs = [a.float() if a.is_floating_point() else a for a in args]
    if op in ("add", "sub", "mul", "div", "max", "powf"):
        if "scalar" in attrs:
            return _scalar_binary(op, xs[0], float(attrs["scalar"]), bool(attrs.get("scalar_first", False)))
        a, b = xs
        return {
            "add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div,
            "max": torch.maximum, "powf": torch.pow,
        }[op](a, b)
    if op == "exp":
        return torch.exp(xs[0])
    if op == "log":
        return torch.log(xs[0])
    if op == "neg":
        return -xs[0]
    if op == "rsqrt":
        return torch.rsqrt(xs[0])
    if op == "sqrt":
        return torch.sqrt(xs[0])
    if op == "sigmoid":
        return torch.sigmoid(xs[0])
    if op == "silu":
        return torch.nn.functional.silu(xs[0])
    if op == "tanh":
        return torch.tanh(xs[0])
    if op == "square":
        return xs[0] * xs[0]
    if op == "log1p":
        return torch.log1p(xs[0])
    if op == "softplus":
        return torch.nn.functional.softplus(xs[0])
    if op == "row_cumsum":
        return torch.cumsum(xs[0], -1)
    if op == "cmp_gt":
        return (xs[0] > float(attrs["scalar"])) if "scalar" in attrs else (xs[0] > xs[1])
    if op == "select":
        return torch.where(args[0].bool(), xs[1], xs[2])
    if op == "row_sum":
        return xs[0].sum(-1, keepdim=attrs.get("keepdim", True))
    if op == "row_max":
        return xs[0].amax(-1, keepdim=attrs.get("keepdim", True))
    if op == "row_mean":
        return xs[0].mean(-1, keepdim=attrs.get("keepdim", True))
    if op == "row_sumsq":
        return (xs[0] * xs[0]).sum(-1, keepdim=True)
    if op == "softmax":
        return torch.softmax(xs[0], -1)
    if op == "matmul":
        return xs[0] @ xs[1]
    if op == "batch_matmul":
        return torch.bmm(xs[0], xs[1])
    if op == "broadcast":
        return xs[0].expand(attrs["shape"])
    if op == "view":
        return xs[0].reshape(attrs["shape"])
    if op == "permute":
        return xs[0].permute(attrs["dims"])
    if op == "expand":
        return xs[0].expand(attrs["shape"])
    if op in ("clone", "cast"):
        return xs[0].clone()
    if op == "slice":
        return xs[0].narrow(attrs["dim"], attrs["start"], attrs["end"] - attrs["start"])
    if op == "cat":
        return torch.cat(xs, attrs["dim"])
    if op == "rotate_half":
        x = xs[0]
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), -1)
    if op == "rotate_half_heads":
        x = xs[0]
        d = attrs["head_dim"]
        x4 = x.reshape(*x.shape[:-1], x.shape[-1] // d, d)
        half = d // 2
        r = torch.cat((-x4[..., half:], x4[..., :half]), -1)
        return r.reshape(x.shape)
    if op == "rope":
        x, cos, sin = xs
        half = x.shape[-1] // 2
        rot = torch.cat((-x[..., half:], x[..., :half]), -1)
        return x * cos + rot * sin
    if op == "rmsnorm":
        x, g = xs
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + attrs["eps"]) * g
    if op == "ssd":
        # Mirrors the kernel exactly, factorised decay included, so the
        # interpreter validates the numerics the kernel actually computes.
        cm, bm, xm, cum_pad, dt_pad, mask, d_full = xs
        bs, h, s = attrs["batch"], attrs["heads"], attrs["seq"]
        p_dim, n, pad = attrs["head_dim"], attrs["state_dim"], attrs["pad"]
        c4 = cm.reshape(bs, s, h, n).transpose(1, 2)
        b4 = bm.reshape(bs, s, h, n).transpose(1, 2)
        x4 = xm.reshape(bs, s, h, p_dim).transpose(1, 2)
        cum = cum_pad.reshape(bs, s, h, pad)[..., 0].transpose(1, 2)
        dt = dt_pad.reshape(bs, s, h, pad)[..., 0].transpose(1, 2)
        w = torch.matmul(c4, b4.transpose(-1, -2)) * mask
        v = x4 * (torch.exp(-cum) * dt).unsqueeze(-1)
        y = torch.matmul(w, v) * torch.exp(cum).unsqueeze(-1)
        out = y.transpose(1, 2).reshape(bs * s, h * p_dim)
        return out + xm * d_full
    if op == "attention":
        q, k, v = xs
        b, s, h, d = attrs["batch"], attrs["seq"], attrs["heads"], attrs["head_dim"]
        q4 = q.reshape(b, s, h, d).transpose(1, 2)
        k4 = k.reshape(b, s, h, d).transpose(1, 2)
        v4 = v.reshape(b, s, h, d).transpose(1, 2)
        p = torch.softmax(torch.matmul(q4, k4.transpose(-1, -2)) * attrs["scale"], -1)
        return torch.matmul(p, v4).transpose(1, 2).reshape(b * s, h * d)
    raise NotImplementedError(f"eval_op: {op}")


def evaluate_graph(graph: OpGraph, inputs: dict[str, torch.Tensor] | list[torch.Tensor], keep_float32: bool = True) -> dict[str, torch.Tensor]:
    """Run the whole graph; returns a map name -> value for every node."""
    if isinstance(inputs, (list, tuple)):
        inputs = dict(zip(graph.inputs, inputs))
    env: dict[str, torch.Tensor] = {}
    for node in graph:
        if node.kind == "input":
            env[node.name] = inputs[node.name].reshape(node.type.shape)
        elif node.kind in ("param", "const"):
            env[node.name] = graph.value(node.name)
        else:
            args = [env[i] for i in node.inputs]
            out = eval_op(node.op, args, node.attrs)
            if tuple(out.shape) != node.type.shape:
                out = out.reshape(node.type.shape)
            env[node.name] = out if keep_float32 else out.to(node.type.torch_dtype)
    return env


def graph_outputs(graph: OpGraph, inputs: dict[str, torch.Tensor] | list[torch.Tensor]) -> list[torch.Tensor]:
    env = evaluate_graph(graph, inputs)
    return [env[o] for o in graph.outputs]


def max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / (b.abs().max() + 1e-6))


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def safe_log2(n: int) -> int:
    return int(math.log2(n))
