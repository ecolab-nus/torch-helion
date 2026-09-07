"""Structural validation of generated kernels with Loom's own tools.

``frontend_check`` runs the Helion→MLIR frontend on the generated module
(sub-second). ``exploration_check`` additionally runs Loom's exploration
pipeline (memory binding, hardware mapping, ETG construction), which is
where constraints C1–C5 are enforced. Both run in a subprocess so a
``report_fatal_error`` inside the C++ passes cannot take the compiler
process down.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO_ROOT

_CHECK_SCRIPT = r'''
import importlib.util, json, sys, traceback
sys.dont_write_bytecode = True
path, hw_spec, do_explore = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
res = {"frontend": False, "exploration": None, "error": None, "variants": None, "symbols": None}
try:
    import pathlib
    modname = "torch_helion_generated_" + pathlib.Path(path).stem
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod  # Helion resolves kernel globals through sys.modules
    spec.loader.exec_module(mod)
    from loom import LoomKernel
    cls = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, LoomKernel) and v is not LoomKernel][0]
    mlir = cls.generate_mlir()
    res["frontend"] = True
    res["mlir_chars"] = len(mlir)
    if do_explore:
        from loom_pipeline import run_exploration
        explored, etg = run_exploration(input_mlir=mlir, hw_spec_file=hw_spec, produce_etg=True, skip_etg=False)
        variants = json.loads(etg)
        res["exploration"] = True
        res["variants"] = len(variants)
        res["symbols"] = sorted(variants[0]["constraint_scope"]["metadata"]["symbols"]) if variants else []
except Exception as e:
    res["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-2000:]
    if res["frontend"] and do_explore:
        res["exploration"] = False
print("__RESULT__" + json.dumps(res))
'''


@dataclass
class ValidationResult:
    kernel: str
    frontend_ok: bool
    exploration_ok: bool | None
    error: str | None = None
    variants: int | None = None
    symbols: list[str] = field(default_factory=list)
    stderr_tail: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def validate_kernels(paths: dict[str, Path], hw_spec: Path, explore: bool = True, timeout: int = 600, workers: int = 4) -> dict[str, ValidationResult]:
    """Validate several kernels concurrently (each in its own subprocess)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {name: pool.submit(validate_kernel, path, hw_spec, explore, timeout) for name, path in paths.items()}
        return {name: f.result() for name, f in futures.items()}


def validate_kernel(py_path: Path, hw_spec: Path, explore: bool = True, timeout: int = 600) -> ValidationResult:
    cmd = [sys.executable, "-c", _CHECK_SCRIPT, str(py_path), str(hw_spec), "1" if explore else "0"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    name = Path(py_path).stem
    line = next((l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")), None)
    tail = "\n".join(proc.stderr.splitlines()[-15:])
    if line is None:
        return ValidationResult(name, False, False if explore else None, error=f"validator crashed (exit {proc.returncode})", stderr_tail=tail)
    res = json.loads(line[len("__RESULT__"):])
    return ValidationResult(name, res["frontend"], res["exploration"], res["error"], res["variants"], res["symbols"] or [], tail if res["error"] else "")
