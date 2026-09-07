"""Offline weight transforms (constraint C5: fold everything rank-1).

Every function returns a new tensor; nothing is done in place. They are
called by the canonicalisation passes and the resulting tensors become
``const`` nodes of the OpGraph.
"""

from __future__ import annotations

import torch


def transpose_weight(w: torch.Tensor) -> torch.Tensor:
    """``nn.Linear`` stores ``[N, K]``; kernels consume ``[K, N]``."""
    return w.t().contiguous()


def fold_gain_into_weight(gain: torch.Tensor, w_kn: torch.Tensor) -> torch.Tensor:
    """``(x * g) @ W == x @ (diag(g) @ W)`` for a per-feature gain ``g[K]``."""
    return (gain.float().unsqueeze(1) * w_kn.float()).to(w_kn.dtype)


def rotate_half_matrix(head_dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``R`` such that ``rotate_half(x) == x @ R`` for a row vector ``x[d]``.

    ``rotate_half(x) = cat(-x[d/2:], x[:d/2])``.
    """
    half = head_dim // 2
    r = torch.zeros(head_dim, head_dim, dtype=dtype)
    for i in range(half):
        r[i + half, i] = -1.0  # out[i] = -x[i + half]
        r[i, i + half] = 1.0  # out[i + half] = x[i]
    return r


def rotate_half_heads_matrix(heads: int, head_dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Block-diagonal rotation applied to every head of a ``[T, H*d]`` row."""
    return torch.block_diag(*[rotate_half_matrix(head_dim, dtype)] * heads)


def fold_rotation_into_weight(w_kn: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """``rotate_half_heads(x @ W) == x @ (W @ R)``."""
    r = rotate_half_heads_matrix(heads, head_dim)
    return (w_kn.float() @ r).to(w_kn.dtype)


def expand_rope_table(table: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    """``[S, d]`` position table → token-major ``[B*S, H*d]``."""
    seq, head_dim = table.shape
    t4 = table.reshape(1, seq, 1, head_dim).expand(batch, seq, heads, head_dim)
    return t4.reshape(batch * seq, heads * head_dim).contiguous()
