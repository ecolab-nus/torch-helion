"""A single Llama-style transformer block written in plain PyTorch.

The block is the reference workload for torch-helion. It is written with
ordinary ``torch.nn`` modules and explicit attention math (no fused SDPA
op) so that ``torch.export`` produces a graph made of core ATen ops that
the capture stage can map onto Loom's operator vocabulary.

Structure (pre-norm, no biases, as in Llama 2/3)::

    h   = rmsnorm(x) * g1
    q,k,v = h @ Wq, h @ Wk, h @ Wv
    q,k = rope(q), rope(k)
    a   = softmax(q k^T / sqrt(d)) v
    x2  = x + a @ Wo
    h2  = rmsnorm(x2) * g2
    out = x2 + (silu(h2 @ Wg) * (h2 @ Wu)) @ Wd
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LlamaBlockConfig:
    """Shape configuration of one block.

    ``hidden`` must equal ``heads * head_dim``. All dimensions should be
    multiples of 32 so that the generated kernels satisfy Loom's L1
    alignment requirement.
    """

    batch: int = 2
    seq: int = 256
    hidden: int = 256
    heads: int = 4
    intermediate: int = 512
    eps: float = 1e-5
    rope_theta: float = 10000.0
    dtype: torch.dtype = torch.float16

    @property
    def head_dim(self) -> int:
        assert self.hidden % self.heads == 0
        return self.hidden // self.heads

    @property
    def tokens(self) -> int:
        return self.batch * self.seq


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def rope_tables(seq: int, head_dim: int, theta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``cos`` and ``sin`` tables of shape ``[seq, head_dim]``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaBlockConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=False, dtype=cfg.dtype)
        self.k_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=False, dtype=cfg.dtype)
        self.v_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=False, dtype=cfg.dtype)
        self.o_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=False, dtype=cfg.dtype)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """``cos``/``sin``: RoPE tables ``[S, d]`` (broadcast over batch and heads)."""
        b, s, _ = h.shape
        q = self.q_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        # cos/sin: [s, d] -> broadcast over batch and heads
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        probs = torch.softmax(scores, dim=-1)
        a = torch.matmul(probs, v)  # [b, heads, s, d]
        a = a.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        return self.o_proj(a)


class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaBlockConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False, dtype=cfg.dtype)
        self.up_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False, dtype=cfg.dtype)
        self.down_proj = nn.Linear(cfg.intermediate, cfg.hidden, bias=False, dtype=cfg.dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))


class LlamaBlock(nn.Module):
    """One decoder block. ``forward(x)`` with ``x: [B, S, D]``.

    The RoPE tables are registered buffers (they only depend on the
    configured sequence length), so ``torch.export`` lifts them as
    constants and torch-helion can fold them into the kernels.
    """

    def __init__(self, cfg: LlamaBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_norm = RMSNorm(cfg.hidden, cfg.eps, cfg.dtype)
        self.attn = LlamaAttention(cfg)
        self.post_norm = RMSNorm(cfg.hidden, cfg.eps, cfg.dtype)
        self.mlp = LlamaMLP(cfg)
        cos, sin = rope_tables(cfg.seq, cfg.head_dim, cfg.rope_theta, cfg.dtype)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), self.rope_cos, self.rope_sin)
        x = x + self.mlp(self.post_norm(x))
        return x

    def example_inputs(self) -> tuple[torch.Tensor]:
        cfg = self.cfg
        return (torch.randn(cfg.batch, cfg.seq, cfg.hidden, dtype=cfg.dtype),)


def build_llama_block(cfg: LlamaBlockConfig | None = None, seed: int = 0) -> LlamaBlock:
    """Construct a block with deterministic random weights."""
    cfg = cfg or LlamaBlockConfig()
    torch.manual_seed(seed)
    block = LlamaBlock(cfg)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if p.ndim == 2:
                p.normal_(0.0, 0.02)
            else:
                p.uniform_(0.8, 1.2)
    return block.eval()
