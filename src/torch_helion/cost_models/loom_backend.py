"""Exact cost via Loom itself.

Runs the generated kernel module through the full Loom pipeline
(Helion frontend → exploration → ETG resolution → CP-SAT block-size
solve → materialisation) and returns Loom's optimal time and block sizes.
Slow (seconds to minutes per kernel) but authoritative; the planner uses
the analytic model and this backend verifies the final choice.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO_ROOT

SOLVER_UNIT_CYCLES = 64  # loom.loom_utils.modeling.TIME_COST_SCALE


@dataclass
class LoomResult:
    kernel: str
    ok: bool
    output_dir: Path
    best_variant: str | None = None
    solver_units: float | None = None
    block_sizes: dict[str, int] = field(default_factory=dict)
    n_variants: int | None = None
    error: str | None = None
    log_path: Path | None = None
    wall_seconds: float = 0.0

    @property
    def cycles(self) -> float | None:
        return None if self.solver_units is None else self.solver_units * SOLVER_UNIT_CYCLES

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["output_dir"] = str(self.output_dir)
        d["log_path"] = str(self.log_path) if self.log_path else None
        d["cycles"] = self.cycles
        return d


def run_loom(kernel_py: Path, config_json: Path, njobs: int = 8, topk_candidates: int = 1, timeout: int = 1800, log_dir: Path | None = None) -> LoomResult:
    import json
    import time

    cfg = json.loads(Path(config_json).read_text())
    out_dir = Path(cfg["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(kernel_py), "--config", str(config_json), "--njobs", str(njobs), "--debug", "--topk-candidates", str(topk_candidates)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return LoomResult(kernel_py.stem, False, out_dir, error=f"timeout after {timeout}s", wall_seconds=time.time() - t0)
    log_path = (log_dir or out_dir) / f"{kernel_py.stem}.loom.log"
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    res = LoomResult(kernel_py.stem, proc.returncode == 0, out_dir, log_path=log_path, wall_seconds=time.time() - t0)
    if proc.returncode != 0:
        res.error = "\n".join((proc.stderr or proc.stdout).splitlines()[-20:])
        return res
    res.n_variants = _int(re.search(r"Solving (\d+) variants", proc.stdout))
    m = re.search(r"GLOBAL BEST\s*\nVariant \[\d+/\d+\]: (\S+)\s*\nOptimal T_total: ([\d,]+) solver units((?:\n  \w+ = \d+)*)", proc.stdout)
    if m:
        res.best_variant = m.group(1)
        res.solver_units = float(m.group(2).replace(",", ""))
        res.block_sizes = {k: int(v) for k, v in re.findall(r"  (\w+) = (\d+)", m.group(3))}
    else:
        m = re.search(r"Optimal T_total: ([\d,]+) solver units", proc.stdout)
        if m:
            res.solver_units = float(m.group(1).replace(",", ""))
    return res


def _int(m):
    return int(m.group(1)) if m else None
