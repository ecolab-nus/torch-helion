"""Results directory layout for one compilation run.

::

    results/<run_name>/
      00_model.txt                 model repr and configuration
      01_fx_graph.txt              exported ATen graph (torch.export)
      02_opgraph_raw.json          OpGraph straight from the FX graph
      03_opgraph_canonical.json    after the optimizer passes
      03_pass_log.json             per-pass rewrite counts / numerics check
      04_hw_spec.json              hardware summary used by the cost model
      05_plan/candidates.json      every partition candidate with its cost
      05_plan/plan.md              human-readable planning report
      06_opir.json                 the chosen Program (OpIR)
      07_validation.json           frontend / exploration checks per kernel
      08_reference_check.json      OpIR interpreter vs PyTorch model
      09_cost_report.{md,json}     analytic model estimate
      10_loom_results.json         (optional) Loom pipeline results
      kernels/<kernel>.py          generated Helion kernels (Loom style)
      kernels/config_files/*.json  Loom configs (with/without assigned tiles)
      params.pt                    (optional) transformed weights
      loom/<kernel>/               (optional) Loom outputs (IRs/, constraints/)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultsWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.files: list[Path] = []

    def path(self, name: str) -> Path:
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.write_text(text)
        self.files.append(p)
        return p

    def write_json(self, name: str, obj: Any) -> Path:
        return self.write_text(name, json.dumps(obj, indent=2, default=_default))

    def listing(self) -> list[str]:
        return sorted(str(p.relative_to(self.run_dir)) for p in self.run_dir.rglob("*") if p.is_file())


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
