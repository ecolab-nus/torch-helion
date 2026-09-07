"""Layout provenance: which base element does each element of a view read?

ATen expresses head splitting, transposes and broadcasting as chains of
``view`` / ``permute`` / ``expand`` / ``clone`` / ``slice`` nodes. Rather
than reasoning about strides symbolically, we replay the chain on an
*index tensor* (``arange`` over the base tensor). The result maps every
element of the viewed tensor to the flat index of the base element it
reads, which makes pattern checks exact and shape-agnostic.

All shapes are static, and the tensors involved are small (a few MiB of
``int64`` at most), so this is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..capture.opgraph import OpGraph, OpNode

TRACEABLE = ("view", "permute", "expand", "clone", "slice", "cast")


@dataclass
class Provenance:
    base: str
    base_shape: tuple[int, ...]
    index: torch.Tensor  # int64, shape = viewed tensor's shape
    chain: list[str]  # layout node names, base-first

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.index.shape)

    def is_identity_reshape(self) -> bool:
        """True when the view is a contiguous re-labelling of the base."""
        n = self.index.numel()
        if n != int(torch.tensor(self.base_shape).prod()) if self.base_shape else 1:
            return False
        return bool(torch.equal(self.index.reshape(-1), torch.arange(n)))

    def equals(self, expected: torch.Tensor) -> bool:
        return tuple(self.index.shape) == tuple(expected.shape) and bool(torch.equal(self.index, expected))


def apply_layout(index: torch.Tensor, node: OpNode) -> torch.Tensor:
    if node.op == "view":
        return index.reshape(node.attrs["shape"])
    if node.op == "permute":
        return index.permute(node.attrs["dims"])
    if node.op == "expand":
        return index.expand(node.attrs["shape"])
    if node.op in ("clone", "cast"):
        return index
    if node.op == "slice":
        return index.narrow(node.attrs["dim"], node.attrs["start"], node.attrs["end"] - node.attrs["start"])
    raise ValueError(f"{node.op} is not a traceable layout op")


def trace_layout(graph: OpGraph, name: str) -> Provenance:
    """Walk back through layout ops from ``name`` and replay them."""
    chain: list[OpNode] = []
    cur = graph[name]
    while cur.kind == "layout" and cur.op in TRACEABLE and len(cur.inputs) == 1:
        chain.append(cur)
        cur = graph[cur.inputs[0]]
    base = cur
    index = torch.arange(base.type.numel).reshape(base.type.shape)
    for node in reversed(chain):
        index = apply_layout(index, node)
    return Provenance(base.name, base.type.shape, index, [n.name for n in reversed(chain)])


def head_split_index(batch: int, seq: int, heads: int, head_dim: int, transpose_last: bool = False, merge_bh: bool = False) -> torch.Tensor:
    """Index tensor of ``[T, D] -> [B, S, H, d] -> permute(0, 2, 1, 3)``.

    With ``transpose_last`` the last two dims are swapped (``[B, H, d, S]``),
    with ``merge_bh`` the leading two dims are merged (``[B*H, ...]``).
    """
    idx = torch.arange(batch * seq * heads * head_dim).reshape(batch, seq, heads, head_dim).permute(0, 2, 1, 3)
    if transpose_last:
        idx = idx.transpose(2, 3)
    if merge_bh:
        idx = idx.reshape(batch * heads, *idx.shape[2:])
    return idx


def head_merge_index(batch: int, seq: int, heads: int, head_dim: int) -> torch.Tensor:
    """Index tensor of ``[B*H, S, d] -> [B, H, S, d] -> permute(0,2,1,3) -> [T, D]``."""
    return torch.arange(batch * heads * seq * head_dim).reshape(batch, heads, seq, head_dim).permute(0, 2, 1, 3).reshape(batch * seq, heads * head_dim)


def match_head_split(prov: Provenance, transpose_last: bool = False) -> dict | None:
    """Recognise a head-split view of a token-major ``[T, D]`` base.

    Returns ``{batch, seq, heads, head_dim}`` when the provenance equals
    ``head_split_index`` for some factorisation, else ``None``. The viewed
    tensor may be ``[B, H, S, d]``, ``[B*H, S, d]`` (or their transposed
    variants).
    """
    shape = prov.shape
    if len(shape) == 4:
        b, h, s, d = shape
        if transpose_last:
            b, h, d, s = shape
        merge = False
    elif len(shape) == 3:
        bh, s, d = shape
        if transpose_last:
            bh, d, s = shape
        merge = True
        # try every factorisation of bh into (batch, heads)
        for b in range(1, bh + 1):
            if bh % b:
                continue
            h = bh // b
            if _base_matches(prov, b, s, h, d):
                expected = head_split_index(b, s, h, d, transpose_last, merge)
                if prov.equals(expected):
                    return {"batch": b, "seq": s, "heads": h, "head_dim": d}
        return None
    else:
        return None
    if not _base_matches(prov, b, s, h, d):
        return None
    expected = head_split_index(b, s, h, d, transpose_last, merge)
    return {"batch": b, "seq": s, "heads": h, "head_dim": d} if prov.equals(expected) else None


def _base_matches(prov: Provenance, b: int, s: int, h: int, d: int) -> bool:
    n = b * s * h * d
    return int(torch.tensor(prov.base_shape).prod()) == n
