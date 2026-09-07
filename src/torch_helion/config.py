"""Compile-time configuration for the torch-helion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOM_ROOT = REPO_ROOT / "third_party" / "loom"
DEFAULT_HW_SPEC = LOOM_ROOT / "third_party" / "loom-mlar" / "tests" / "2d_mesh" / "2d_mesh_torus.mlir"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


@dataclass
class CompileConfig:
    """Knobs of :func:`torch_helion.compile_model`.

    Attributes
    ----------
    run_name:
        Sub-directory of ``results_dir`` receiving every intermediate result.
    hw_spec:
        Hardware description consumed by Loom (ADL/MLIR). The perf YAMLs next
        to it feed the analytic cost model.
    results_dir:
        Root of all results; ``results/<run_name>/`` is created.
    planner_mode:
        ``"exhaustive"`` enumerates every legal fusion decision, ``"greedy"``
        takes the locally cheapest merge (fallback for very large graphs).
    tile_alignment / max_tile:
        Domain of tile sizes searched by the analytic tile solver.
    verify_with_loom:
        Additionally run the real Loom pipeline (frontend → exploration →
        ETG → CP-SAT) on every generated kernel and record its optimal time.
    loom_njobs:
        Parallel workers handed to Loom when ``verify_with_loom`` is set.
    check_frontend:
        Run the Helion→MLIR frontend and Loom's exploration pass on every
        generated kernel (seconds per kernel; catches codegen mistakes without
        the full pipeline).
    validate_workers:
        Kernels validated concurrently (one subprocess each).
    """

    run_name: str = "run"
    hw_spec: Path = field(default_factory=lambda: Path(os.environ.get("TORCH_HELION_HW_SPEC", DEFAULT_HW_SPEC)))
    results_dir: Path = field(default_factory=lambda: Path(os.environ.get("TORCH_HELION_RESULTS", DEFAULT_RESULTS_DIR)))
    planner_mode: str = "exhaustive"
    tile_alignment: int = 32
    max_tile: int = 512
    kernel_launch_overhead: float = 2000.0
    verify_with_loom: bool = False
    loom_njobs: int = 8
    loom_topk_candidates: int = 1
    check_frontend: bool = True
    validate_workers: int = 4
    save_params: bool = False
    dtype: str = "float16"

    @property
    def run_dir(self) -> Path:
        return Path(self.results_dir) / self.run_name

    def hw_spec_dir(self) -> Path:
        return Path(self.hw_spec).resolve().parent
