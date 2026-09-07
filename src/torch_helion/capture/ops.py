"""Operator vocabulary of the OpGraph.

Three families:

``LAYOUT_OPS``
    Pure re-indexing ops produced by ATen (``view``, ``permute`` ...). They
    never survive canonicalisation: the optimizer either folds them (params),
    proves them free (contiguous reshapes) or absorbs them into a macro op.

``MACRO_OPS``
    Model-level patterns recognised by the optimizer (``rmsnorm``, ``rope``,
    ``attention``). They are lowered to registered ops or to a dedicated
    kernel template.

``REGISTERED_OPS``
    Exactly the body ops the Loom hardware spec knows (constraint C4 in the
    source README). Only these may appear in a kernel body.
"""

from __future__ import annotations

LAYOUT_OPS = frozenset({"view", "permute", "expand", "clone", "slice", "cat", "cast"})

# Elementwise ops (tensor-tensor or tensor-scalar). ``attrs['scalar']`` holds
# a python number when the second operand is a scalar.
BINARY_OPS = frozenset({"add", "sub", "mul", "div", "max", "powf"})
UNARY_OPS = frozenset({"exp", "log", "neg", "rsqrt", "sigmoid", "silu", "tanh", "square", "sqrt", "log1p", "softplus"})
REDUCE_OPS = frozenset({"row_sum", "row_max", "row_mean", "row_sumsq"})
SCAN_OPS = frozenset({"row_cumsum"})
MACRO_OPS = frozenset({"rmsnorm", "rope", "rotate_half", "attention", "softmax", "ssd"})
ANCHOR_OPS = frozenset({"matmul", "batch_matmul", "attention"})

# Ops with a cost entry in the hardware spec (perf YAML) — see README C4.
REGISTERED_BINARY = frozenset({"add", "sub", "mul", "div", "max", "powf"})
REGISTERED_UNARY = frozenset({"exp", "log"})
REGISTERED_REDUCE = frozenset({"row_sum", "row_max"})
REGISTERED_OPS = REGISTERED_BINARY | REGISTERED_UNARY | REGISTERED_REDUCE | {"matmul", "batch_matmul", "broadcast"}

ELEMENTWISE_OPS = BINARY_OPS | UNARY_OPS

# Mapping of unregistered unary ops to registered equivalents; see
# :func:`torch_helion.optimizer.passes.rewrite_unregistered`.
UNARY_REWRITES = {
    "log1p": "log(add(x, 1))",
    "softplus": "log(add(1, exp(x)))",
    "neg": "mul(x, -1)",
    "rsqrt": "powf(x, -0.5)",
    "sqrt": "powf(x, 0.5)",
    "sigmoid": "div(1, add(1, exp(mul(x, -1))))",
    "silu": "mul(x, sigmoid(x))",
    "square": "mul(x, x)",
    "tanh": "sub(mul(2, sigmoid(mul(x, 2))), 1)",
}


def is_elementwise(op: str) -> bool:
    return op in ELEMENTWISE_OPS


def is_registered(op: str) -> bool:
    return op in REGISTERED_OPS
