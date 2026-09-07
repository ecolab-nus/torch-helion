"""Planner: choose the kernel grouping and tile sizes with the cost model.

Search space
------------
The only structural freedom left after canonicalisation is *which sibling
GEMMs share a kernel* (constraint C3 says they may, not that they must).
Merging saves one pass over the shared LHS and lets the epilogue combine
their accumulators; splitting reduces the L1 footprint and allows larger
tiles. For every sibling set the planner enumerates all set partitions
(``exhaustive``), or just "all merged" vs "all split" (``greedy``).

Every candidate grouping is turned into a :class:`Program` by the
partitioner, checked for legality, and costed: each kernel gets its best
tile assignment from :mod:`torch_helion.cost_models.tile_search`; the
program cost is the sum of kernel cycles plus a fixed launch/link overhead
per kernel boundary. The cheapest legal candidate wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

from ..capture.opgraph import OpGraph
from ..config import CompileConfig
from ..cost_models.hw_spec import HardwareSpec
from ..cost_models.tile_search import search_tiles
from ..cost_models.analytic import CostModelError
from .legality import check_program
from .opir import Program
from .partition import Grouping, PartitionError, build_program, sibling_sets


@dataclass
class Candidate:
    index: int
    grouping: Grouping
    program: Program | None
    legal: bool
    total_cycles: float
    kernel_cycles: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "grouping": self.grouping,
            "legal": self.legal,
            "total_cycles": self.total_cycles,
            "kernel_cycles": self.kernel_cycles,
            "kernels": [k.name for k in self.program.kernels] if self.program else [],
            "violations": self.violations,
            "error": self.error,
        }


@dataclass
class PlanResult:
    best: Candidate
    candidates: list[Candidate]

    @property
    def program(self) -> Program:
        assert self.best.program is not None
        return self.best.program


def set_partitions(items: list[str]) -> list[list[list[str]]]:
    """All partitions of ``items`` into non-empty blocks (Bell number many)."""
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    out = []
    for part in set_partitions(rest):
        out.append([[first]] + part)
        for i in range(len(part)):
            out.append(part[:i] + [[first] + part[i]] + part[i + 1 :])
    return out


def enumerate_groupings(g: OpGraph, mode: str = "exhaustive", max_candidates: int = 512) -> list[Grouping]:
    sets = sibling_sets(g)
    per_set: list[list[list[list[str]]]] = []
    for s in sets:
        if mode == "exhaustive" and len(s) <= 6:
            per_set.append(set_partitions(s))
        else:
            per_set.append([[list(s)], [[m] for m in s]] if len(s) > 1 else [[list(s)]])
    groupings: list[Grouping] = []
    for combo in product(*per_set):
        grouping: Grouping = {}
        gid = 0
        for part in combo:
            for block in part:
                for m in block:
                    grouping[m] = gid
                gid += 1
        groupings.append(grouping)
        if len(groupings) >= max_candidates:
            break
    return groupings


def kernel_signature(spec, tensors) -> str:
    """Structural key of a kernel (independent of its name) for cost memoisation."""
    import json

    d = spec.to_dict()
    d.pop("name")
    d.pop("tiles", None)
    d.pop("cost", None)
    d["shapes"] = {a.name: list(tensors[a.tensor].shape) for a in spec.args}
    d["args"] = [[a.name, a.role, a.view] for a in spec.args]
    d["outputs"] = [o.value for o in spec.outputs]
    return json.dumps(d, sort_keys=True, default=str)


def cost_program(program: Program, hw: HardwareSpec, config: CompileConfig, memo: dict[str, Any] | None = None) -> tuple[float, dict[str, float]]:
    total = 0.0
    per_kernel: dict[str, float] = {}
    memo = memo if memo is not None else {}
    for spec in program.kernels:
        key = kernel_signature(spec, program.tensors)
        res = memo.get(key)
        if res is None:
            res = search_tiles(spec, program.tensors, hw, alignment=config.tile_alignment, max_tile=config.max_tile)
            memo[key] = res
        spec.tiles = dict(res.best.tiles)
        spec.cost = {"cycles": res.best.cycles, "breakdown": res.best.to_dict(), "evaluated": res.evaluated, "feasible": res.feasible,
                     "alternatives": [{"tiles": r.tiles, "cycles": r.cycles} for r in res.top]}
        per_kernel[spec.name] = res.best.cycles
        total += res.best.cycles + config.kernel_launch_overhead
    return total, per_kernel


def plan(g: OpGraph, hw: HardwareSpec, config: CompileConfig, name: str = "program") -> PlanResult:
    groupings = enumerate_groupings(g, config.planner_mode)
    candidates: list[Candidate] = []
    memo: dict[str, Any] = {}
    for i, grouping in enumerate(groupings):
        try:
            program = build_program(g, grouping, name=name)
        except PartitionError as e:
            candidates.append(Candidate(i, grouping, None, False, float("inf"), error=str(e)))
            continue
        report = check_program(program)
        if not report.ok:
            candidates.append(Candidate(i, grouping, program, False, float("inf"), violations=[str(v) for v in report.violations]))
            continue
        try:
            total, per_kernel = cost_program(program, hw, config, memo)
        except CostModelError as e:
            candidates.append(Candidate(i, grouping, program, False, float("inf"), error=str(e)))
            continue
        candidates.append(Candidate(i, grouping, program, True, total, per_kernel))
    legal = [c for c in candidates if c.legal]
    if not legal:
        details = "\n".join(f"  candidate {c.index}: {c.error or c.violations}" for c in candidates[:10])
        raise PartitionError(f"no legal partition found among {len(candidates)} candidates:\n{details}")
    best = min(legal, key=lambda c: (c.total_cycles, len(c.program.kernels)))
    best.program.metadata["plan"] = {"candidates": len(candidates), "legal": len(legal), "total_cycles": best.total_cycles, "kernel_cycles": best.kernel_cycles, "launch_overhead": config.kernel_launch_overhead}
    return PlanResult(best, candidates)
