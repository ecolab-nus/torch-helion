"""Analytic kernel cost model.

The estimator mirrors the structure of Loom's ETG cost aggregation for the
kernel templates torch-helion emits, using the very same per-function
time costs from the hardware perf YAMLs:

* per core, one grid tile at a time (``waves = ceil(tiles / cores)``);
* the single reduction loop runs ``iters`` times; loads of the next tile
  overlap with compute when double buffering fits in L1
  (``t_iter = max(t_load, t_compute)``, else the sum);
* the prologue/epilogue and stores are added once per grid tile;
* spatial reuse: the LHS tile is shared by every core in one mesh column
  and the RHS tile by every core in one mesh row, so their DRAM copies
  are broadcasts (cheaper per core, and the DRAM traffic is divided by
  the mesh dimension). Both grid→mesh orientations are evaluated and the
  cheaper one is reported.

The result is a cycle count comparable between candidate partitions and
tile assignments; it is *not* a promise of absolute latency. Use
:mod:`torch_helion.cost_models.loom_backend` to obtain Loom's own number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

from ..capture.opgraph import TensorType
from ..capture.ops import REGISTERED_BINARY, REGISTERED_UNARY
from ..optimizer.opir import KernelSpec
from .hw_spec import HardwareSpec

SUMSQ_COLS = 32  # width of the ones matrix used for the RMSNorm fold


@dataclass
class CostBreakdown:
    cycles: float
    tiles: dict[str, int]
    grid_tiles: int
    waves: int
    per_tile: float
    loop_iters: int
    t_load_iter: float
    t_compute_iter: float
    t_prologue: float
    t_epilogue: float
    t_store: float
    double_buffer: bool
    l1_bytes: int
    dram_read_bytes: float
    dram_write_bytes: float
    orientation: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CostModelError(ValueError):
    pass


def _div(a: int, b: int) -> int:
    if a % b:
        raise CostModelError(f"{b} does not divide {a}")
    return a // b


def _tile_shape(name: str, spec: KernelSpec, tensors: dict[str, TensorType], tm: int, tn: int, env: dict[str, tuple[int, int]]) -> tuple[int, int]:
    if name in env:
        return env[name]
    for a in spec.args:
        if a.name == name:
            return (tm, tn)
    raise CostModelError(f"{spec.name}: epilogue reads unknown value {name}")


def epilogue_cycles(spec: KernelSpec, hw: HardwareSpec, tm: int, tn: int, env0: dict[str, tuple[int, int]] | None = None) -> tuple[float, dict[str, tuple[int, int]]]:
    """Vector-lane cycles of the epilogue at tile ``[tm, tn]``.

    ``env0`` seeds values the kernel produces outside its GEMM accumulators —
    the scan kernel's single ``acc``, for instance.
    """
    env: dict[str, tuple[int, int]] = dict(env0 or {})
    env.update({g.acc: (tm, tn) for g in spec.gemms})
    total = 0.0
    if spec.sumsq:
        env["sumsq"] = (tm, 1)
        total += hw.reduce_cycles("row_max", tm, SUMSQ_COLS)
    for op in spec.epilogue:
        shapes = [_tile_shape(i, spec, {}, tm, tn, env) for i in op.inputs]
        if op.op == "broadcast":
            env[op.name] = (tm, tn)
            continue
        if op.op in REGISTERED_BINARY or op.op in REGISTERED_UNARY:
            out = max(shapes, key=lambda s: s[0] * s[1])
            env[op.name] = out
            total += hw.elementwise_cycles(op.op, out[0] * out[1])
        elif op.op in ("row_sum", "row_max"):
            env[op.name] = (shapes[0][0], 1)
            total += hw.reduce_cycles(op.op, shapes[0][0], shapes[0][1])
        else:
            raise CostModelError(f"{spec.name}: epilogue op {op.op} is not registered (C4)")
    return total, env


def gemm_cost(spec: KernelSpec, tensors: dict[str, TensorType], hw: HardwareSpec, tiles: dict[str, int], force_db: bool | None = None) -> CostBreakdown:
    tm, tn, tk = tiles["tile_t"], tiles["tile_n"], tiles["tile_k"]
    T, N = spec.grid["rows"], spec.grid["cols"]
    K = spec.k_extent
    iters = _div(K, tk)
    grid_tiles = _div(T, tm) * _div(N, tn)
    mx, my = hw.mesh
    eb = hw.dtype_bytes
    n_gemm = len(spec.gemms)
    extras = [a for a in spec.args if a.role == "extra"]

    # compute per loop iteration
    t_comp = n_gemm * hw.matmul_cycles(tm, tn, tk)
    if spec.sumsq:
        t_comp += 2 * hw.elementwise_cycles("mul", tm * tk) + hw.matmul_cycles(tm, SUMSQ_COLS, tk)

    best: CostBreakdown | None = None
    for orientation in ("t_on_x", "t_on_y"):
        lhs_area = (1, my) if orientation == "t_on_x" else (mx, 1)
        rhs_area = (mx, 1) if orientation == "t_on_x" else (1, my)
        t_load = hw.dram_load_cycles(tm, tk, lhs_area) + n_gemm * hw.dram_load_cycles(tk, tn, rhs_area)
        # L1 footprint
        acc_bytes = n_gemm * tm * tn * eb + (tm * SUMSQ_COLS * eb if spec.sumsq else 0)
        iter_bytes = (tm * tk + n_gemm * tk * tn) * eb + (tm * tk * eb + tk * SUMSQ_COLS * eb if spec.sumsq else 0)
        epi_bytes = (len(extras) + 2) * tm * tn * eb
        for db in ((True, False) if force_db is None else (force_db,)):
            l1 = acc_bytes + iter_bytes * (2 if db else 1) + epi_bytes
            if l1 > hw.l1_bytes:
                continue
            t_iter = max(t_load, t_comp) if db else t_load + t_comp
            t_loop = iters * t_iter + (t_load if db else 0.0)
            t_epi_ops, _ = epilogue_cycles(spec, hw, tm, tn)
            t_epi = t_epi_ops + sum(hw.dram_load_cycles(tm, tn, (1, 1)) for _ in extras)
            t_store = sum(hw.dram_store_cycles(tm, tn) for _ in spec.outputs)
            per_tile = t_loop + t_epi + t_store
            waves = math.ceil(grid_tiles / hw.cores)
            cycles = waves * per_tile
            lhs_share = my if orientation == "t_on_x" else mx
            rhs_share = mx if orientation == "t_on_x" else my
            dram_read = grid_tiles * iters * (tm * tk * eb / lhs_share + n_gemm * tk * tn * eb / rhs_share) + grid_tiles * len(extras) * tm * tn * eb
            dram_write = grid_tiles * len(spec.outputs) * tm * tn * eb
            cb = CostBreakdown(cycles, dict(tiles), grid_tiles, waves, per_tile, iters, t_load, t_comp, 0.0, t_epi, t_store, db, l1, dram_read, dram_write, orientation,
                               {"n_gemm": n_gemm, "sumsq": spec.sumsq, "extras": len(extras), "outputs": len(spec.outputs)})
            if best is None or cb.cycles < best.cycles:
                best = cb
    if best is None:
        raise CostModelError(f"{spec.name}: tiles {tiles} exceed L1 ({hw.l1_bytes} bytes)")
    return best


def attention_cost(spec: KernelSpec, tensors: dict[str, TensorType], hw: HardwareSpec, tiles: dict[str, int], force_db: bool | None = None) -> CostBreakdown:
    tm, tn = tiles["tile_m"], tiles["tile_n"]
    b, s, h, d = spec.attrs["batch"], spec.attrs["seq"], spec.attrs["heads"], spec.attrs["head_dim"]
    eb = hw.dtype_bytes
    iters = _div(s, tn)
    grid_tiles = b * h * _div(s, tm)
    mx, my = hw.mesh
    # every core works on its own (b, h, m-tile): no operand sharing
    t_load = hw.dram_load_cycles(d, tn) + hw.dram_load_cycles(tn, d)
    t_comp = (
        hw.matmul_cycles(tm, tn, d)
        + hw.elementwise_cycles("mul", tm * tn)  # scale
        + hw.reduce_cycles("row_max", tm, tn)
        + hw.elementwise_cycles("max", tm)
        + hw.elementwise_cycles("sub", tm * tn)
        + hw.elementwise_cycles("exp", tm * tn)
        + hw.elementwise_cycles("sub", tm) + hw.elementwise_cycles("exp", tm)
        + hw.elementwise_cycles("mul", tm * d)
        + hw.matmul_cycles(tm, d, tn)
        + hw.elementwise_cycles("add", tm * d)
        + hw.reduce_cycles("row_sum", tm, tn)
        + hw.elementwise_cycles("mul", tm) + hw.elementwise_cycles("add", tm)
    )
    best = None
    for db in ((True, False) if force_db is None else (force_db,)):
        l1 = (2 * tm * d + 3 * tm * tn + 4 * tm) * eb + (d * tn + tn * d) * eb * (2 if db else 1)
        if l1 > hw.l1_bytes:
            continue
        t_iter = max(t_load, t_comp) if db else t_load + t_comp
        t_loop = iters * t_iter + (t_load if db else 0.0)
        t_pro = hw.dram_load_cycles(tm, d)
        t_epi = hw.elementwise_cycles("div", tm * d)
        t_store = hw.dram_store_cycles(tm, d)
        per_tile = t_loop + t_pro + t_epi + t_store
        waves = math.ceil(grid_tiles / hw.cores)
        cycles = waves * per_tile
        dram_read = grid_tiles * (tm * d + iters * 2 * tn * d) * eb
        dram_write = grid_tiles * tm * d * eb
        cb = CostBreakdown(cycles, dict(tiles), grid_tiles, waves, per_tile, iters, t_load, t_comp, t_pro, t_epi, t_store, db, l1, dram_read, dram_write, "bh_on_mesh",
                           {"batch": b, "heads": h, "seq": s, "head_dim": d})
        if best is None or cb.cycles < best.cycles:
            best = cb
    if best is None:
        raise CostModelError(f"{spec.name}: tiles {tiles} exceed L1")
    return best


def ssd_cost(spec: KernelSpec, tensors: dict[str, TensorType], hw: HardwareSpec, tiles: dict[str, int], force_db: bool | None = None) -> CostBreakdown:
    """Cost of the Mamba scan kernel.

    Same skeleton as attention: one grid tile per (batch, head, row block),
    a loop over the sequence, two matrix products per iteration with the
    decay applied between them. The decay costs two padded reductions and a
    handful of vector ops on the value tile.
    """
    tm, tn = tiles["tile_m"], tiles["tile_n"]
    a = spec.attrs
    b, s, h, p, n, pad = a["batch"], a["seq"], a["heads"], a["head_dim"], a["state_dim"], a["pad"]
    eb = hw.dtype_bytes
    iters = _div(s, tn)
    grid_tiles = b * h * _div(s, tm)
    extras = [x for x in spec.args if x.role == "extra"]

    t_load = hw.dram_load_cycles(n, tn) + hw.dram_load_cycles(tn, p) + 2 * hw.dram_load_cycles(tn, pad)
    t_comp = (
        hw.matmul_cycles(tm, tn, n)
        + hw.elementwise_cycles("mul", tm * tn)                     # causal mask
        + 2 * hw.reduce_cycles("row_max", tn, pad)                  # cum_j and dt_j
        + hw.elementwise_cycles("mul", tn) + hw.elementwise_cycles("exp", tn) + hw.elementwise_cycles("mul", tn)
        + hw.elementwise_cycles("mul", tn * p)                      # scale the value tile
        + hw.matmul_cycles(tm, p, tn)
    )
    epi_ops, _ = epilogue_cycles(spec, hw, tm, p, {"acc": (tm, p)})
    best = None
    for db in ((True, False) if force_db is None else (force_db,)):
        l1 = (tm * n + tm * p + tm * pad + tm * tn) * eb + (n * tn + tn * p + 2 * tn * pad) * eb * (2 if db else 1)
        if l1 > hw.l1_bytes:
            continue
        t_iter = max(t_load, t_comp) if db else t_load + t_comp
        t_loop = iters * t_iter + (t_load if db else 0.0)
        t_pro = hw.dram_load_cycles(tm, n) + hw.dram_load_cycles(tm, pad) + hw.reduce_cycles("row_max", tm, pad)
        t_epi = (
            hw.elementwise_cycles("exp", tm) + hw.elementwise_cycles("mul", tm * p)
            + 2 * hw.dram_load_cycles(tm, p) + 2 * hw.elementwise_cycles("mul", tm * p)
            + epi_ops + sum(hw.dram_load_cycles(tm, p) for _ in extras)
        )
        t_store = hw.dram_store_cycles(tm, p)
        per_tile = t_loop + t_pro + t_epi + t_store
        waves = math.ceil(grid_tiles / hw.cores)
        dram_read = grid_tiles * ((tm * (n + p + pad)) * eb + iters * (n * tn + tn * p + 2 * tn * pad) * eb)
        dram_write = grid_tiles * tm * p * eb
        cb = CostBreakdown(waves * per_tile, dict(tiles), grid_tiles, waves, per_tile, iters, t_load, t_comp, t_pro, t_epi, t_store, db, l1, dram_read, dram_write, "bh_on_mesh",
                           {"batch": b, "heads": h, "seq": s, "head_dim": p, "state_dim": n, "extras": len(extras)})
        if best is None or cb.cycles < best.cycles:
            best = cb
    if best is None:
        raise CostModelError(f"{spec.name}: tiles {tiles} exceed L1")
    return best


def kernel_cost(spec: KernelSpec, tensors: dict[str, TensorType], hw: HardwareSpec, tiles: dict[str, int]) -> CostBreakdown:
    if spec.kind == "gemm":
        return gemm_cost(spec, tensors, hw, tiles)
    if spec.kind == "attention":
        return attention_cost(spec, tensors, hw, tiles)
    if spec.kind == "ssd":
        return ssd_cost(spec, tensors, hw, tiles)
    raise CostModelError(f"unknown kernel kind {spec.kind}")
