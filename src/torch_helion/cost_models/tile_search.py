"""Tile-size solver for the analytic cost model.

Loom's CP-SAT solver picks block sizes for one kernel from the resolved
ETG. During partition planning we need the same decision for many
candidate kernels quickly, so this module enumerates the (small) domain
of 32-aligned tile sizes that divide the kernel extents and fit in L1 and
returns the cheapest assignment under :mod:`analytic`. The chosen tiles
can later be handed to Loom as ``assigned_block_size`` or re-solved by
Loom's CP-SAT for the final kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..capture.opgraph import TensorType
from ..optimizer.opir import KernelSpec
from .analytic import CostBreakdown, CostModelError, kernel_cost
from .hw_spec import HardwareSpec


@dataclass
class TileSearchResult:
    best: CostBreakdown
    evaluated: int
    feasible: int
    top: list[CostBreakdown]


def aligned_divisors(extent: int, alignment: int, max_tile: int, allow_full: bool = True) -> list[int]:
    out = []
    for t in range(alignment, min(extent, max_tile) + 1, alignment):
        if extent % t == 0:
            out.append(t)
    if allow_full and extent <= max_tile and extent % alignment == 0 and extent not in out:
        out.append(extent)
    if not out:  # extent smaller than alignment or not aligned: use it whole
        out = [extent]
    return out


def tile_domains(spec: KernelSpec, alignment: int, max_tile: int) -> dict[str, list[int]]:
    if spec.kind == "gemm":
        return {
            "tile_t": aligned_divisors(spec.grid["rows"], alignment, max_tile),
            "tile_n": aligned_divisors(spec.grid["cols"], alignment, max_tile),
            "tile_k": aligned_divisors(spec.k_extent, alignment, max_tile),
        }
    if spec.kind in ("attention", "ssd"):
        s = spec.attrs["seq"]
        return {"tile_m": aligned_divisors(s, alignment, max_tile), "tile_n": aligned_divisors(s, alignment, max_tile)}
    raise CostModelError(spec.kind)


def search_tiles(spec: KernelSpec, tensors: dict[str, TensorType], hw: HardwareSpec, alignment: int = 32, max_tile: int = 512, keep_top: int = 5) -> TileSearchResult:
    domains = tile_domains(spec, alignment, max_tile)
    names = list(domains)
    results: list[CostBreakdown] = []
    evaluated = 0
    for values in product(*(domains[n] for n in names)):
        tiles = dict(zip(names, values))
        evaluated += 1
        try:
            results.append(kernel_cost(spec, tensors, hw, tiles))
        except CostModelError:
            continue
    if not results:
        raise CostModelError(f"{spec.name}: no feasible tile assignment (domains {domains})")
    results.sort(key=lambda r: (r.cycles, -r.tiles.get("tile_k", 0)))
    return TileSearchResult(results[0], evaluated, len(results), results[:keep_top])
