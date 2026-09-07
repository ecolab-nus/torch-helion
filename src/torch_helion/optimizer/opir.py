"""OpIR: kernel-level program produced by the optimizer.

A :class:`Program` is an ordered list of :class:`KernelSpec`. Every kernel
is a complete description of one Helion kernel — argument tensors and how
they are viewed, the parallel grid, the single reduction loop, the
accumulators, an epilogue expressed as a tiny SSA program over registered
ops, and the stores. The codegen turns it into Python source without any
further analysis; the interpreter executes the same description with plain
PyTorch for validation.

Kernel kinds
------------
``gemm``
    ``for tile_t, tile_n in hl.tile([T, N])`` + ``for tile_k in hl.tile(K)``.
    One LHS ``[T, K]`` shared by all GEMMs; each GEMM adds an accumulator
    ``acc_i = lhs @ rhs_i`` (``rhs_i: [K, N]``). ``sumsq`` adds the RMSNorm
    fold ``ss = sum_k lhs^2`` (accumulated with ``hl.dot`` against a ones
    matrix, see codegen). The epilogue reads accumulators, ``sumsq`` (a
    ``[T, 1]`` column), and extra ``[T, N]`` tensors loaded at the output
    tile, and yields the stored outputs.
``attention``
    Flash-attention template over ``[B, S, H, d]`` views of ``[T, D]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..capture.opgraph import TensorType


@dataclass
class KernelArg:
    name: str  # argument name inside the kernel
    tensor: str  # program tensor it binds to
    role: str  # lhs | rhs | extra | q | k | v
    view: list[int] | None = None  # shape the kernel sees (free reshape), None = as is


@dataclass
class BodyOp:
    name: str
    op: str
    inputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GemmSpec:
    acc: str
    rhs: str  # kernel argument name of the [K, N] operand


@dataclass
class KernelOutput:
    tensor: str  # program tensor
    value: str  # epilogue value / accumulator stored


@dataclass
class KernelSpec:
    name: str
    kind: str
    args: list[KernelArg] = field(default_factory=list)
    outputs: list[KernelOutput] = field(default_factory=list)
    # gemm kind
    lhs: str | None = None
    k_extent: int = 0
    grid: dict[str, int] = field(default_factory=dict)  # {"rows": T, "cols": N}
    gemms: list[GemmSpec] = field(default_factory=list)
    sumsq: bool = False
    epilogue: list[BodyOp] = field(default_factory=list)
    # attention kind
    attrs: dict[str, Any] = field(default_factory=dict)
    constexprs: dict[str, float] = field(default_factory=dict)
    # filled by the cost model / planner
    tiles: dict[str, int] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def arg(self, name: str) -> KernelArg:
        for a in self.args:
            if a.name == name:
                return a
        raise KeyError(name)

    def input_tensors(self) -> list[str]:
        return [a.tensor for a in self.args]

    def output_tensors(self) -> list[str]:
        return [o.tensor for o in self.outputs]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "KernelSpec":
        k = KernelSpec(name=d["name"], kind=d["kind"])
        k.args = [KernelArg(**a) for a in d.get("args", [])]
        k.outputs = [KernelOutput(**o) for o in d.get("outputs", [])]
        k.lhs = d.get("lhs")
        k.k_extent = d.get("k_extent", 0)
        k.grid = dict(d.get("grid", {}))
        k.gemms = [GemmSpec(**g) for g in d.get("gemms", [])]
        k.sumsq = d.get("sumsq", False)
        k.epilogue = [BodyOp(**b) for b in d.get("epilogue", [])]
        k.attrs = dict(d.get("attrs", {}))
        k.constexprs = dict(d.get("constexprs", {}))
        k.tiles = dict(d.get("tiles", {}))
        k.cost = dict(d.get("cost", {}))
        k.notes = list(d.get("notes", []))
        return k


@dataclass
class Program:
    name: str
    inputs: list[str]
    outputs: list[str]
    tensors: dict[str, TensorType]
    kernels: list[KernelSpec]
    params: list[str] = field(default_factory=list)  # tensors with compile-time values
    metadata: dict[str, Any] = field(default_factory=dict)

    def intermediates(self) -> list[str]:
        produced = {o for k in self.kernels for o in k.output_tensors()}
        return [t for t in produced if t not in self.outputs]

    def producer(self, tensor: str) -> KernelSpec | None:
        for k in self.kernels:
            if tensor in k.output_tensors():
                return k
        return None

    def consumers(self, tensor: str) -> list[KernelSpec]:
        return [k for k in self.kernels if tensor in k.input_tensors()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tensors": {n: {"shape": list(t.shape), "dtype": t.dtype} for n, t in self.tensors.items()},
            "params": self.params,
            "kernels": [k.to_dict() for k in self.kernels],
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Program":
        return Program(
            name=d["name"],
            inputs=list(d["inputs"]),
            outputs=list(d["outputs"]),
            tensors={n: TensorType(tuple(t["shape"]), t["dtype"]) for n, t in d["tensors"].items()},
            kernels=[KernelSpec.from_dict(k) for k in d["kernels"]],
            params=list(d.get("params", [])),
            metadata=dict(d.get("metadata", {})),
        )

    @staticmethod
    def from_json(path: str | Path) -> "Program":
        return Program.from_dict(json.loads(Path(path).read_text()))

    def summary(self) -> str:
        lines = [f"Program {self.name}: {len(self.kernels)} kernels, inputs={self.inputs}, outputs={self.outputs}"]
        for k in self.kernels:
            ins = ", ".join(f"{a.name}<-{a.tensor}" for a in k.args)
            outs = ", ".join(o.tensor for o in k.outputs)
            extra = f" gemms={len(k.gemms)} sumsq={k.sumsq} epilogue={len(k.epilogue)}" if k.kind == "gemm" else f" attrs={k.attrs}"
            cost = f" est={k.cost.get('cycles'):,.0f}cyc tiles={k.tiles}" if k.cost.get("cycles") else ""
            lines.append(f"  [{k.kind}] {k.name}({ins}) -> {outs}{extra}{cost}")
        return "\n".join(lines)
