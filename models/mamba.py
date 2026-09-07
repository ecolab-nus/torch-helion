"""A full Mamba-2 (SSD) language-model backbone in plain PyTorch.

The model follows the Mamba-2 architecture: each block is a pre-norm
residual block whose mixer is a state-space duality (SSD) layer computed
in the *chunked* form, which is the form Loom's own ``mamba_chunk_scan``
kernel implements.

Block structure::

    h            = rmsnorm(u)
    z,x,B,C,dt   = h @ W_in                       (one GEMM, five slices)
    dt           = softplus(dt + dt_bias)
    y            = ssd(x, dt·A, B, C)             (chunked, see ``ssd_chunked``)
    y            = y + D·x                        (skip connection)
    g            = rmsnorm(y * silu(z))           (gated norm)
    out          = u + g @ W_out

Configuration notes
-------------------
Two defaults differ from a stock Mamba-2 checkpoint, both to keep every
tensor inside Loom's kernel model, and both documented here rather than
hidden:

``ngroups = nheads``
    Stock Mamba-2 shares one (B, C) pair across all heads (multi-query
    style), which needs a broadcast of ``[T, 1, N]`` to ``[T, H, N]``.
    Giving every head its own B and C removes that broadcast. Set
    ``ngroups=1`` for the stock behaviour.

``use_conv = False``
    Mamba-2 applies a depthwise causal conv1d of width 4 to ``x, B, C``.
    A depthwise convolution mixes tokens with a *per-channel* kernel, so
    it is neither a GEMM nor an elementwise op, and it has no reduction
    loop of its own — Loom cannot express it (constraints C2/C3). The
    convolution is implemented below and enabled with ``use_conv=True``;
    the compilable default leaves it out.

``mode = "quadratic"``
    The state-space duality gives two exactly equivalent ways to evaluate
    the scan. ``ssd_chunked`` is the linear-time form a production kernel
    uses; ``ssd_quadratic`` is its dual, a decay-masked attention whose
    cost is O(S²) but whose shape is a single reduction over the sequence.
    ``test_mamba.py`` checks both against the direct recurrence.

Compilation status
------------------
torch-helion compiles this model end to end in ``quadratic`` mode: seven
kernels, all of which pass Loom's frontend and exploration, and the scan
kernel solves through Loom's full pipeline. The chunked mode is the
reference implementation and is not lowered — it needs four more kernel
templates for the inter-chunk recurrence.

The sequence length is bounded by fp16, not by the compiler. The kernel
applies the decay as two column scales, ``exp(c_i)`` and ``exp(-c_j)``,
because a row vector cannot be read from memory; the running decay is
centred on the sequence midpoint so each exponent stays within half the
total. With the default initialisation that leaves roughly 370 tokens of
headroom before ``exp`` overflows — see ``docs/codegen.md``.

All feature dimensions are multiples of 32 so the generated kernels
satisfy Loom's L1 alignment requirement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MambaConfig:
    """Shapes of a Mamba-2 backbone.

    ``d_inner = expand * d_model`` and ``nheads = d_inner // headdim``;
    every one of ``d_model``, ``d_inner``, ``headdim``, ``d_state``,
    ``nheads`` and ``chunk`` should be a multiple of 32.
    """

    batch: int = 2
    seq: int = 256
    d_model: int = 512
    expand: int = 2
    headdim: int = 32
    d_state: int = 32
    chunk: int = 64
    n_layers: int = 2
    mode: str = "quadratic"  # "quadratic" (compiled) or "chunked" (reference)
    ngroups: int | None = None  # None -> one group per head
    d_conv: int = 4
    use_conv: bool = False
    eps: float = 1e-5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dtype: torch.dtype = torch.float16

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def nheads(self) -> int:
        assert self.d_inner % self.headdim == 0
        return self.d_inner // self.headdim

    @property
    def groups(self) -> int:
        return self.nheads if self.ngroups is None else self.ngroups

    @property
    def d_bc(self) -> int:
        return self.groups * self.d_state

    @property
    def d_in_proj(self) -> int:
        return 2 * self.d_inner + 2 * self.d_bc + self.nheads

    @property
    def tokens(self) -> int:
        return self.batch * self.seq

    @property
    def nchunks(self) -> int:
        assert self.seq % self.chunk == 0
        return self.seq // self.chunk


class RMSNorm(nn.Module):
    """Root-mean-square norm with a learned per-feature gain."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def causal_chunk_masks(chunk: int, nchunks: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constant masks used by the chunked scan.

    Returns ``(within, across, prefix)``:

    ``within[i, j]``  1 when ``j <= i`` — the causal mask inside one chunk.
    ``across[z, c]``  1 when ``c < z``  — chunk ``z`` sees earlier chunks only.
    ``prefix[j, i]``  1 when ``j <= i`` — running-sum matrix, so that
                      ``v @ prefix`` is the inclusive cumulative sum of ``v``
                      along its last axis.
    """
    idx = torch.arange(chunk)
    within = (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(dtype)
    # right-multiply prefix sum: (v @ prefix)[i] = sum_{j <= i} v[j]
    prefix = (idx.unsqueeze(1) <= idx.unsqueeze(0)).to(dtype)
    cidx = torch.arange(nchunks)
    across = (cidx.unsqueeze(1) > cidx.unsqueeze(0)).to(dtype)
    return within, across, prefix


def ssd_chunked(
    x: torch.Tensor,
    dA: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    within: torch.Tensor,
    across: torch.Tensor,
    prefix: torch.Tensor,
) -> torch.Tensor:
    """Chunked state-space duality scan.

    Shapes (chunk-major throughout; ``b`` batch, ``h`` heads, ``c`` chunks,
    ``l`` chunk length, ``p`` head dim, ``n`` state dim)::

        x, out : [b, h, c, l, p]
        dA, dt : [b, h, c, l]
        B, C   : [b, h, c, l, n]

    The scan is the standard four-step decomposition. Every step is a
    matrix multiplication over one axis, with the decay factors applied as
    elementwise scales:

    1. ``cumA`` — inclusive cumulative sum of ``dA`` inside each chunk,
       written as a multiply by the constant prefix matrix.
    2. diagonal block — the within-chunk part, a decay-weighted causal
       attention: ``((C Bᵀ) ⊙ L) X`` with ``L[i,j] = exp(cumA_i - cumA_j)``.
    3. chunk states — each chunk summarised into ``[p, n]``.
    4. inter-chunk recurrence and the off-diagonal output.
    """
    n = B.shape[-1]

    # 1. running decay inside each chunk: cumA = dA @ prefix
    cumA = torch.matmul(dA.unsqueeze(-2), prefix).squeeze(-2)   # [b,h,c,l]
    last = cumA[..., -1:]                                        # [b,h,c,1]

    # 2. diagonal block: decay-weighted causal attention within the chunk
    cb = torch.matmul(C, B.transpose(-1, -2))                    # [b,h,c,l,l]
    # Masking inside the exponent keeps every argument <= 0: above the
    # diagonal the difference would be positive and could overflow fp16,
    # and inf * 0 is NaN. Multiplying by the mask twice is exact.
    decay_in = torch.exp((cumA.unsqueeze(-1) - cumA.unsqueeze(-2)) * within) * within
    m = cb * decay_in * dt.unsqueeze(-2)                         # mask + input scale
    y_diag = torch.matmul(m, x)                                  # [b,h,c,l,p]

    # 3. per-chunk state: [p, n] summary of every chunk
    w = (dt * torch.exp(last - cumA)).unsqueeze(-1)              # [b,h,c,l,1]
    states = torch.matmul((x * w).transpose(-1, -2), B)          # [b,h,c,p,n]

    # 4. inter-chunk recurrence, then the off-diagonal output
    tot = last.squeeze(-1)                                       # [b,h,c]
    cumTot = torch.matmul(tot.unsqueeze(-2), prefix[: tot.shape[-1], : tot.shape[-1]]).squeeze(-2)
    # The state entering chunk z accumulates the decay of chunks c+1..z-1,
    # so the row index uses the *exclusive* running total.
    cumTotEx = cumTot - tot
    decay_ch = torch.exp((cumTotEx.unsqueeze(-1) - cumTot.unsqueeze(-2)) * across) * across
    flat = states.flatten(-2)                                    # [b,h,c,p*n]
    carried = torch.matmul(decay_ch, flat)                       # [b,h,c,p*n]
    S = carried.unflatten(-1, (x.shape[-1], n))                  # [b,h,c,p,n]
    y_off = torch.matmul(C, S.transpose(-1, -2)) * torch.exp(cumA).unsqueeze(-1)
    return y_diag + y_off


def ssd_quadratic(
    x: torch.Tensor,
    dA: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    causal: torch.Tensor,
) -> torch.Tensor:
    """Quadratic (attention-dual) form of the same scan.

    Shapes are head-major: ``x: [b, h, s, p]``, ``dA, dt: [b, h, s]``,
    ``B, C: [b, h, s, n]``, ``causal: [s, s]`` a constant 0/1 mask.

    The scan matrix is ``M[i,j] = (C_i·B_j) · exp(cumA_i - cumA_j) · dt_j``
    below the diagonal and zero above it, and the output is ``M X``. Since
    ``cumA`` decreases, every exponent below the diagonal is negative, so
    no online rescaling pass is needed — the whole layer is two matrix
    multiplies with an elementwise decay between them.
    """
    cumA = torch.cumsum(dA, dim=-1)                                   # [b,h,s]
    cb = torch.matmul(C, B.transpose(-1, -2))                         # [b,h,s,s]
    # The mask goes inside the exponent so its argument is never positive:
    # above the diagonal exp would overflow fp16 and inf * 0 is NaN.
    delta = (cumA.unsqueeze(-1) - cumA.unsqueeze(-2)) * causal
    m = cb * torch.exp(delta) * causal * dt.unsqueeze(-2)
    return torch.matmul(m, x)                                         # [b,h,s,p]


class Mamba2Mixer(nn.Module):
    """The SSD mixer of one Mamba-2 block."""

    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dtype
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_in_proj, bias=False, dtype=d)
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False, dtype=d)
        self.norm = RMSNorm(cfg.d_inner, cfg.eps, d)
        self.A_log = nn.Parameter(torch.zeros(cfg.nheads, dtype=d))
        self.D = nn.Parameter(torch.ones(cfg.nheads, dtype=d))
        self.dt_bias = nn.Parameter(torch.zeros(cfg.nheads, dtype=d))
        if cfg.use_conv:
            width = 2 * cfg.d_bc + cfg.d_inner
            self.conv_weight = nn.Parameter(torch.zeros(width, cfg.d_conv, dtype=d))
        within, across, prefix = causal_chunk_masks(cfg.chunk, cfg.nchunks, d)
        self.register_buffer("mask_within", within)
        self.register_buffer("mask_across", across)
        self.register_buffer("mask_prefix", prefix)
        idx = torch.arange(cfg.seq)
        self.register_buffer("mask_causal", (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(d))
        self.register_buffer("mask_seq_prefix", (idx.unsqueeze(1) <= idx.unsqueeze(0)).to(d))

    def _conv(self, xbc: torch.Tensor) -> torch.Tensor:
        """Depthwise causal conv1d over the sequence (not Loom-expressible)."""
        b, s, w = xbc.shape
        pad = torch.nn.functional.pad(xbc.transpose(1, 2), (self.cfg.d_conv - 1, 0))
        out = torch.nn.functional.conv1d(pad, self.conv_weight.unsqueeze(1), groups=w)
        return torch.nn.functional.silu(out.transpose(1, 2))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, s, _ = u.shape
        c, l, h, p, n = cfg.nchunks, cfg.chunk, cfg.nheads, cfg.headdim, cfg.d_state
        g = cfg.groups

        proj = self.in_proj(u)
        z, xbc, dt = torch.split(proj, [cfg.d_inner, cfg.d_inner + 2 * cfg.d_bc, h], dim=-1)
        if cfg.use_conv:
            xbc = self._conv(xbc)
        x, B, C = torch.split(xbc, [cfg.d_inner, cfg.d_bc, cfg.d_bc], dim=-1)

        dt = torch.nn.functional.softplus(dt + self.dt_bias)          # [b,s,h]
        A = -torch.exp(self.A_log)                                    # [h]

        # chunk-major views: [b, h, c, l, ...]
        xc = x.reshape(b, c, l, h, p).permute(0, 3, 1, 2, 4)
        Bc = B.reshape(b, c, l, g, n).permute(0, 3, 1, 2, 4)
        Cc = C.reshape(b, c, l, g, n).permute(0, 3, 1, 2, 4)
        dtc = dt.reshape(b, c, l, h).permute(0, 3, 1, 2)
        if g != h:  # stock Mamba-2 shares one (B, C) across heads
            Bc = Bc.expand(b, h, c, l, n)
            Cc = Cc.expand(b, h, c, l, n)
        dA = dtc * A.reshape(1, h, 1, 1)

        if cfg.mode == "chunked":
            y = ssd_chunked(xc, dA, dtc, Bc, Cc, self.mask_within, self.mask_across, self.mask_prefix)
            y = y + xc * self.D.reshape(1, h, 1, 1, 1)
            y = y.permute(0, 2, 3, 1, 4).reshape(b, s, cfg.d_inner)
        else:
            # head-major [b, h, s, ...] views; the chunk axis is not needed
            xq = x.reshape(b, s, h, p).permute(0, 2, 1, 3)
            Bq = B.reshape(b, s, g, n).permute(0, 2, 1, 3)
            Cq = C.reshape(b, s, g, n).permute(0, 2, 1, 3)
            dtq = dt.permute(0, 2, 1)
            dAq = dtq * A.reshape(1, h, 1)
            if g != h:
                Bq = Bq.expand(b, h, s, n)
                Cq = Cq.expand(b, h, s, n)
            y = ssd_quadratic(xq, dAq, dtq, Bq, Cq, self.mask_causal)
            y = y + xq * self.D.reshape(1, h, 1, 1)
            y = y.permute(0, 2, 1, 3).reshape(b, s, cfg.d_inner)
        gated = self.norm(y * torch.nn.functional.silu(z))
        return self.out_proj(gated)


class Mamba2Block(nn.Module):
    """Pre-norm residual block: ``u + mixer(rmsnorm(u))``."""

    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.eps, cfg.dtype)
        self.mixer = Mamba2Mixer(cfg)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u + self.mixer(self.norm(u))


class MambaModel(nn.Module):
    """A stack of Mamba-2 blocks with a final norm. ``forward(u)``, ``u: [B, S, D]``."""

    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([Mamba2Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.eps, cfg.dtype)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            u = layer(u)
        return self.norm_f(u)

    def example_inputs(self) -> tuple[torch.Tensor]:
        cfg = self.cfg
        return (torch.randn(cfg.batch, cfg.seq, cfg.d_model, dtype=cfg.dtype),)


def build_mamba_model(cfg: MambaConfig | None = None, seed: int = 0) -> MambaModel:
    """Construct a Mamba-2 backbone with deterministic, well-scaled weights."""
    cfg = cfg or MambaConfig()
    torch.manual_seed(seed)
    model = MambaModel(cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2 and "conv" not in name:
                param.normal_(0.0, 0.02)
            elif "A_log" in name:
                # A in [-1, -1/16] after -exp(A_log), the usual Mamba init range
                param.copy_(torch.log(torch.rand(param.shape, dtype=torch.float32) * 0.9 + 0.1).to(param.dtype))
            elif "dt_bias" in name:
                dt = torch.rand(param.shape, dtype=torch.float32) * (math.log(cfg.dt_max) - math.log(cfg.dt_min)) + math.log(cfg.dt_min)
                dt = dt.exp().clamp(min=1e-4)
                param.copy_((dt + torch.log(-torch.expm1(-dt))).to(param.dtype))
            elif "D" in name and param.ndim == 1:
                param.fill_(1.0)
            elif param.ndim == 1:
                param.uniform_(0.9, 1.1)
            else:
                param.normal_(0.0, 0.02)
    return model.eval()


def build_mamba_block(cfg: MambaConfig | None = None, seed: int = 0) -> MambaModel:
    """One-layer Mamba-2 model, the smallest useful compilation target."""
    cfg = cfg or MambaConfig(n_layers=1)
    cfg.n_layers = 1
    return build_mamba_model(cfg, seed)
