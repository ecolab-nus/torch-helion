"""Kernel legality: the Loom constraints C1–C6 plus the fusion rule.

The partitioner constructs kernels that satisfy C1/C2/C3 by construction
(one grid, one reduction loop, GEMMs sharing LHS and K). This module
checks everything that depends on shapes and op choice:

* C4  every epilogue op has a hardware function
* C5  every tile-able extent is a multiple of 32; no rank-1 operands
* fusion rule: Loom fuses a single-use elementwise producer into a row
  reduction, producing a mixed parallel/reduction generic that only
  ``add``/``max`` bodies satisfy. Reductions must therefore consume
  accumulators, loads, or multi-use values.
* every extra epilogue input has the kernel's ``[T, N]`` shape
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..capture.opgraph import TensorType
from ..capture.ops import REGISTERED_BINARY, REGISTERED_UNARY
from .opir import KernelSpec, Program

ALIGN = 32


@dataclass
class Violation:
    kernel: str
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.kernel}] {self.rule}: {self.message}"


@dataclass
class LegalityReport:
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, kernel: str, rule: str, message: str) -> None:
        self.violations.append(Violation(kernel, rule, message))

    def __str__(self) -> str:
        return "legal" if self.ok else "\n".join(str(v) for v in self.violations)


def _aligned(n: int) -> bool:
    return n % ALIGN == 0


def check_kernel(spec: KernelSpec, tensors: dict[str, TensorType], report: LegalityReport | None = None) -> LegalityReport:
    rep = report or LegalityReport()
    k = spec.name
    if spec.kind == "gemm":
        T, N, K = spec.grid["rows"], spec.grid["cols"], spec.k_extent
        for label, v in (("T", T), ("N", N), ("K", K)):
            if not _aligned(v):
                rep.add(k, "C5", f"{label}={v} is not a multiple of {ALIGN}")
        if not spec.gemms and not spec.sumsq:
            # A GEMM-less kernel is legal only when the sum-of-squares gives
            # the loop a ranked tensor to carry (a standalone normalisation).
            rep.add(k, "C2", "kernel has neither a GEMM nor a reduction to carry the loop")
        for g in spec.gemms:
            rhs = tensors[spec.arg(g.rhs).tensor]
            if rhs.shape != (K, N):
                rep.add(k, "C3", f"rhs {g.rhs} has shape {rhs.shape}, expected {(K, N)}")
        lhs = tensors[spec.arg(spec.lhs).tensor] if spec.lhs else None
        if lhs is not None and lhs.shape != (T, K):
            rep.add(k, "C3", f"lhs has shape {lhs.shape}, expected {(T, K)}")
        for a in spec.args:
            if a.role == "extra":
                t = tensors[a.tensor]
                if t.shape != (T, N):
                    rep.add(k, "C5", f"epilogue input {a.tensor} has shape {t.shape}; must be [{T}, {N}] to load at the output tile")
        # epilogue ops
        produced: dict[str, str] = {g.acc: "acc" for g in spec.gemms}
        if spec.sumsq:
            produced["sumsq"] = "sumsq"
        uses: dict[str, int] = {}
        for op in spec.epilogue:
            for i in op.inputs:
                uses[i] = uses.get(i, 0) + 1
        for o in spec.outputs:
            uses[o.value] = uses.get(o.value, 0) + 1
        for op in spec.epilogue:
            if op.op in ("row_sum", "row_max"):
                src = op.inputs[0]
                if produced.get(src) == "elementwise" and uses.get(src, 0) == 1:
                    rep.add(k, "fusion", f"{op.name}: reduction of single-use elementwise value {src} would be fused by Loom into an unregistered mixed generic")
                produced[op.name] = "reduce"
            elif op.op in REGISTERED_BINARY or op.op in REGISTERED_UNARY:
                produced[op.name] = "elementwise"
            elif op.op == "broadcast":
                produced[op.name] = "broadcast"
            else:
                rep.add(k, "C4", f"epilogue op {op.name} = {op.op} has no hardware function")
        for o in spec.outputs:
            t = tensors[o.tensor]
            if t.shape != (T, N):
                rep.add(k, "C5", f"output {o.tensor} has shape {t.shape}, expected [{T}, {N}]")
    elif spec.kind == "ssd":
        a = spec.attrs
        for label in ("seq", "head_dim", "state_dim", "pad"):
            if not _aligned(a[label]):
                rep.add(k, "C5", f"{label}={a[label]} is not a multiple of {ALIGN}")
        tok = a["batch"] * a["seq"]
        widths = {"c": a["heads"] * a["state_dim"], "b": a["heads"] * a["state_dim"], "x": a["heads"] * a["head_dim"],
                  "cum": a["heads"] * a["pad"], "dt": a["heads"] * a["pad"], "dskip": a["heads"] * a["head_dim"], "xskip": a["heads"] * a["head_dim"]}
        for arg in spec.args:
            t = tensors[arg.tensor]
            if arg.role == "mask":
                if t.shape != (a["seq"], a["seq"]):
                    rep.add(k, "C5", f"causal mask {arg.tensor} has shape {t.shape}, expected [{a['seq']}, {a['seq']}]")
            elif arg.role in widths:
                if t.shape != (tok, widths[arg.role]):
                    rep.add(k, "C5", f"{arg.role} operand {arg.tensor} has shape {t.shape}, expected [{tok}, {widths[arg.role]}]")
            elif arg.role == "extra" and t.shape != (tok, a["heads"] * a["head_dim"]):
                rep.add(k, "C5", f"epilogue input {arg.tensor} has shape {t.shape}; must match the scan output")
        for op in spec.epilogue:
            if op.op not in REGISTERED_BINARY and op.op not in REGISTERED_UNARY and op.op != "broadcast":
                rep.add(k, "C4", f"epilogue op {op.name} = {op.op} has no hardware function")
    elif spec.kind == "attention":
        a = spec.attrs
        for label in ("seq", "head_dim"):
            if not _aligned(a[label]):
                rep.add(k, "C5", f"{label}={a[label]} is not a multiple of {ALIGN}")
        for arg in spec.args:
            t = tensors[arg.tensor]
            if t.shape != (a["batch"] * a["seq"], a["heads"] * a["head_dim"]):
                rep.add(k, "C5", f"{arg.role} operand {arg.tensor} has shape {t.shape}")
    else:
        rep.add(k, "kind", f"unknown kernel kind {spec.kind}")
    return rep


def check_program(program: Program) -> LegalityReport:
    rep = LegalityReport()
    for spec in program.kernels:
        check_kernel(spec, program.tensors, rep)
    return rep
