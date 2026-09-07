"""Hardware description consumed by the cost model.

Loom describes hardware as an ADL/MLIR module (topology, memories, data
movers, processor functionality) plus one ``<processor>.perf.yaml`` per
processor with symbolic time costs. :class:`HardwareSpec` reads exactly
those files so the analytic estimator and Loom's own evaluator agree on
the numbers.

Time model of one hardware function (Loom's ``SimpleTimeCost``)::

    cycles = fixed_latency + volume / throughput

where the three fields are expressions over the function's symbols
(``M``, ``N``, ``K``, ``B``, ``L``, ``P``, ``R``, ``effective_bandwidth``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import expr as E
from .yaml_lite import load_yaml


@dataclass
class TimeCost:
    fixed_latency: str
    volume: str
    throughput: str

    def cycles(self, env: dict[str, float]) -> float:
        fixed = float(E.evaluate(self.fixed_latency, env))
        vol = float(E.evaluate(self.volume, env))
        thr = float(E.evaluate(self.throughput, env))
        return fixed + (vol / thr if thr else float("inf"))


@dataclass
class Scenario:
    constraints: str
    time_cost: TimeCost


@dataclass
class HwFunction:
    name: str
    processor: str
    constraints: str
    scenarios: list[Scenario]
    symbols: list[str] = field(default_factory=list)

    def cycles(self, **env: float) -> float:
        env = {k: float(v) for k, v in env.items()}
        if self.constraints and not E.evaluate(self.constraints, env):
            raise ValueError(f"{self.name}: constraints {self.constraints!r} violated by {env}")
        for sc in self.scenarios:
            if not sc.constraints or E.evaluate(sc.constraints, env):
                return sc.time_cost.cycles(env)
        raise ValueError(f"{self.name}: no scenario matches {env}")


# loom-style op -> (processor, hw function)
OP_TO_FUNCTION: dict[str, tuple[str, str]] = {
    "matmul": ("matrix_lane", "matmul_SS_f16"),
    "batch_matmul": ("matrix_lane", "batch_matmul_SS_f16"),
    "row_sum": ("matrix_lane", "vec_vsum_f16"),
    "row_max": ("matrix_lane", "vec_vmax_f16"),
    "add": ("vector_lane", "vec_add_f16"),
    "sub": ("vector_lane", "vec_sub_f16"),
    "mul": ("vector_lane", "vec_mul_f16"),
    "div": ("vector_lane", "vec_div_f16"),
    "max": ("vector_lane", "vec_max_f16"),
    "powf": ("vector_lane", "vec_powf_f16"),
    "exp": ("vector_lane", "vec_exp_f16"),
    "log": ("vector_lane", "vec_log_f16"),
    "cmp": ("vector_lane", "vec_cmpf_ogt_f16"),
    "select": ("vector_lane", "vec_select_f16"),
    "dram_to_l1": ("dram_l1_noc0", "dram_to_l1_S_f16"),
    "dram_to_l1_bcast": ("dram_l1_noc0", "dram_to_l1_S_bcst"),
    "l1_to_dram": ("l1_dram_noc1", "l1_to_dram_f16"),
}


@dataclass
class HardwareSpec:
    path: Path
    mesh: tuple[int, int]
    l1_bytes: int
    dram_bytes: int
    functions: dict[str, HwFunction]
    dtype_bytes: int = 2

    # ----------------------------------------------------------- loading
    @staticmethod
    def load(path: str | Path) -> "HardwareSpec":
        path = Path(path).resolve()
        text = path.read_text()
        mesh = _parse_mesh(text)
        l1 = _parse_memory(text, "mem_bank", "dim_nbank")
        dram = _parse_memory(text, "mem_DRAM_bank", "dim_dram_channel")
        functions: dict[str, HwFunction] = {}
        for proc in re.findall(r"module @proc_(\w+)", text):
            yaml_path = path.parent / "processors" / f"{proc}.perf.yaml"
            if not yaml_path.exists():
                continue
            for fn in _load_perf_yaml(yaml_path, proc):
                functions[fn.name] = fn
        return HardwareSpec(path, mesh, l1, dram, functions)

    # ------------------------------------------------------------ queries
    @property
    def cores(self) -> int:
        return self.mesh[0] * self.mesh[1]

    def function_for(self, op: str) -> HwFunction:
        if op not in OP_TO_FUNCTION:
            raise KeyError(f"op {op!r} has no hardware function (C4)")
        _, fname = OP_TO_FUNCTION[op]
        if fname not in self.functions:
            raise KeyError(f"hardware spec {self.path.name} lacks {fname}")
        return self.functions[fname]

    def has_op(self, op: str) -> bool:
        return op in OP_TO_FUNCTION and OP_TO_FUNCTION[op][1] in self.functions

    def op_cycles(self, op: str, **dims: float) -> float:
        return self.function_for(op).cycles(**dims)

    def elementwise_cycles(self, op: str, numel: int) -> float:
        return self.op_cycles(op, L=numel)

    def reduce_cycles(self, op: str, rows: int, cols: int) -> float:
        return self.op_cycles(op, P=rows, R=cols)

    def matmul_cycles(self, m: int, n: int, k: int, batch: int = 1) -> float:
        if batch > 1:
            return self.op_cycles("batch_matmul", B=batch, M=m, N=n, K=k)
        return self.op_cycles("matmul", M=m, N=n, K=k)

    def dram_load_cycles(self, rows: int, cols: int, area: tuple[int, int] = (1, 1)) -> float:
        """DRAM→L1 copy of a ``[rows, cols]`` fp16 tile.

        ``effective_bandwidth`` is not a free symbol in Loom's YAML; the
        exploration pass substitutes ``10 + extra`` where ``extra`` depends
        on the broadcast area (fitted from Loom's resolved ETG: 6 when the
        transfer is shared along the mesh's y dimension, 2–5 otherwise).
        """
        extra = 6.0 if area[1] >= 2 else 3.5
        op = "dram_to_l1_bcast" if area != (1, 1) else "dram_to_l1"
        return self.op_cycles(op, M=rows, N=cols, effective_bandwidth=10.0 + extra, bcst_x=area[0], bcst_y=area[1])

    def dram_store_cycles(self, rows: int, cols: int) -> float:
        return self.op_cycles("l1_to_dram", M=rows, N=cols)

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "mesh": list(self.mesh),
            "cores": self.cores,
            "l1_bytes": self.l1_bytes,
            "dram_bytes": self.dram_bytes,
            "functions": sorted(self.functions),
        }


# ------------------------------------------------------------------ parsing

def _parse_mesh(text: str) -> tuple[int, int]:
    dims = dict(re.findall(r'adl\.spatial_dim "(dim_[xy])", (\d+)', text))
    return int(dims.get("dim_x", 1)), int(dims.get("dim_y", 1))


def _parse_memory(text: str, bank: str, dim: str) -> int:
    m = re.search(rf'adl\.memory\.bank "{bank}", \{{bsize = (\d+), nblk = (\d+)\}}', text)
    if not m:
        return 0
    per_bank = int(m.group(1)) * int(m.group(2))
    d = re.search(rf'adl\.spatial_dim "{dim}", (\d+)', text)
    return per_bank * (int(d.group(1)) if d else 1)


def _load_perf_yaml(path: Path, proc: str) -> list[HwFunction]:
    data = load_yaml(path.read_text())
    out = []
    for name, spec in (data.get("functions") or {}).items():
        scenarios = []
        for sc in spec.get("scenarios", []):
            tc = sc["time_cost"]["simple"]
            scenarios.append(Scenario(str(sc.get("constraints", "")), TimeCost(str(tc["fixed_latency"]), str(tc["volume"]), str(tc["throughput"]))))
        out.append(HwFunction(name, proc, str(spec.get("constraints", "")), scenarios, list(spec.get("symbols", []))))
    return out
