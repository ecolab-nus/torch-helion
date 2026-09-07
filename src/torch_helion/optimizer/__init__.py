"""Optimizer: canonicalisation, partitioning, planning → OpIR."""

from .legality import LegalityReport, check_kernel, check_program
from .opir import BodyOp, GemmSpec, KernelArg, KernelOutput, KernelSpec, Program
from .partition import Grouping, PartitionError, build_program, default_grouping, sibling_sets
from .passes import PassError, canonicalize
from .planner import Candidate, PlanResult, enumerate_groupings, plan

__all__ = [
    "BodyOp", "Candidate", "GemmSpec", "Grouping", "KernelArg", "KernelOutput", "KernelSpec", "LegalityReport",
    "PartitionError", "PassError", "PlanResult", "Program", "build_program", "canonicalize", "check_kernel",
    "check_program", "default_grouping", "enumerate_groupings", "plan", "sibling_sets",
]
