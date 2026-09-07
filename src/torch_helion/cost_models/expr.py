"""Safe evaluation of the arithmetic expressions used in Loom perf YAMLs.

Examples of the accepted grammar::

    "M * N / 2"            "2 * B * M * N * K"      "(M * N / 8192) * 716"
    "M >= 32 && N >= 32"   "M * N < 8192"           "True"

Only numbers, names, ``+ - * /``, comparisons and ``&& || !`` are allowed.
"""

from __future__ import annotations

import ast
import operator
from functools import lru_cache
from typing import Any, Mapping

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMP = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}


class ExprError(ValueError):
    pass


def _normalize(text: str) -> str:
    return text.replace("&&", " and ").replace("||", " or ").replace("!", " not ").replace(" not =", " !=")


def symbols(text: str) -> set[str]:
    """Free symbol names of an expression."""
    tree = ast.parse(_normalize(text), mode="eval")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in ("True", "False", "true", "false")}


@lru_cache(maxsize=4096)
def _parse(text: str) -> ast.AST:
    try:
        return ast.parse(_normalize(text), mode="eval").body
    except SyntaxError as e:
        raise ExprError(f"cannot parse {text!r}: {e}") from e


def evaluate(text: str, env: Mapping[str, Any]) -> float | bool:
    """Evaluate ``text`` with ``env`` giving symbol values."""
    return _eval(_parse(str(text)), env)


def _eval(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ("True", "true"):
            return True
        if node.id in ("False", "false"):
            return False
        if node.id not in env:
            raise ExprError(f"unbound symbol {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Not):
            return not v
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True
    raise ExprError(f"unsupported syntax: {ast.dump(node)}")
