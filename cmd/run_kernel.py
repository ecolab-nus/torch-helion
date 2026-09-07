#!/usr/bin/env python
"""One generated Helion kernel through Loom, on its own.

    ./cmd/run_kernel.sh mamba --list
    ./cmd/run_kernel.sh mamba --kernel k4_ssd --njobs 16
    ./cmd/run_kernel.sh --kernel results/mamba/kernels/k4_ssd.py

Takes a kernel this repository already generated and runs Loom's pipeline on
just that one: Helion frontend, exploration, ETG resolution, CP-SAT solve,
materialisation. Use it to iterate on a single kernel without recompiling the
whole model.

``--assigned`` uses the tiling torch-helion's planner chose instead of asking
Loom's solver, which skips ETG resolution and the solve and is much faster.
``--all`` runs every kernel of the run in sequence.

Loom writes its IRs and constraints under ``results/<run>/loom/<kernel>/``.
Exit status is 0 when every kernel run succeeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import DEFAULT_RESULTS_DIR, rel, setup_logging

from torch_helion.cost_models.loom_backend import LoomResult, run_loom


def _kernels_dir(args) -> Path:
    if args.kernel and args.kernel.endswith(".py"):
        return Path(args.kernel).resolve().parent
    return (Path(args.results_dir) / args.run_name / "kernels").resolve()


def _available(kernels_dir: Path) -> list[Path]:
    return sorted(p for p in kernels_dir.glob("*.py") if not p.name.startswith("_"))


def _resolve(args, kernels_dir: Path) -> list[Path]:
    available = _available(kernels_dir)
    if not available:
        raise SystemExit(f"no generated kernels in {rel(kernels_dir)}; run cmd/generate_kernels.py first")
    if args.all:
        return available
    if not args.kernel:
        names = ", ".join(p.stem for p in available)
        raise SystemExit(f"pass --kernel NAME, --all, or --list. Available: {names}")
    if args.kernel.endswith(".py"):
        path = Path(args.kernel).resolve()
        if not path.exists():
            raise SystemExit(f"no such kernel file: {args.kernel}")
        return [path]
    match = [p for p in available if p.stem == args.kernel]
    if not match:
        names = ", ".join(p.stem for p in available)
        raise SystemExit(f"no kernel {args.kernel!r} in {rel(kernels_dir)}. Available: {names}")
    return match


def _report(result: LoomResult) -> None:
    if not result.ok:
        print(f"  FAILED after {result.wall_seconds:.0f}s")
        for line in (result.error or "").splitlines()[-6:]:
            print(f"    {line[:160]}")
        if result.log_path:
            print(f"    full log: {rel(result.log_path)}")
        return
    print(f"  ok in {result.wall_seconds:.0f}s")
    if result.n_variants is not None:
        print(f"  variants explored : {result.n_variants}")
    if result.solver_units is not None:
        print(f"  optimal T_total   : {result.solver_units:,.0f} solver units = {result.cycles:,.0f} cycles")
    if result.block_sizes:
        print(f"  block sizes       : {result.block_sizes}")
    if result.best_variant:
        print(f"  best variant      : {result.best_variant[:110]}")
    print(f"  bufferized MLIR   : {rel(result.output_dir / 'IRs' / 'p03_bufferized.mlir')}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_kernel", description=__doc__.splitlines()[0])
    p.add_argument("run_name", nargs="?", metavar="RUN",
                   help="which results/<RUN>/ to take kernels from, usually the model name")
    p.add_argument("--kernel", help="kernel name (k4_ssd) or path to a generated .py")
    p.add_argument("--run-name", dest="run_name_opt", help="same as the positional RUN")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--all", action="store_true", help="run every kernel of the run, in order")
    p.add_argument("--list", action="store_true", help="list the kernels available and exit")
    p.add_argument("--assigned", action="store_true",
                   help="materialise the planner's tiling instead of running Loom's solver (much faster)")
    p.add_argument("--njobs", type=int, default=8)
    p.add_argument("--topk-candidates", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800, help="seconds per kernel")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    args.run_name = args.run_name or args.run_name_opt
    if not args.run_name and not (args.kernel or "").endswith(".py"):
        runs = sorted(d.name for d in Path(args.results_dir).glob("*/kernels") for d in [d.parent])
        listing = "\n".join(f"  {r}" for r in runs) or "  (none yet — run ./cmd/generate_kernels.sh first)"
        p.exit(2, f"which run? Available under {rel(Path(args.results_dir))}:\n{listing}\n")

    kernels_dir = _kernels_dir(args)
    if args.list:
        available = _available(kernels_dir)
        if not available:
            raise SystemExit(f"no generated kernels in {rel(kernels_dir)}")
        print(f"kernels in {rel(kernels_dir)}:")
        for path in available:
            print(f"  {path.stem}")
        return 0

    suffix = ".assigned.json" if args.assigned else ".json"
    failures = 0
    for path in _resolve(args, kernels_dir):
        config = kernels_dir / "config_files" / f"{path.stem}{suffix}"
        if not config.exists():
            print(f"{path.stem}: missing config {rel(config)}")
            failures += 1
            continue
        mode = "planner tiling" if args.assigned else "Loom solver"
        print(f"\n{path.stem} ({mode}, njobs={args.njobs})")
        result = run_loom(path, config, njobs=args.njobs, topk_candidates=args.topk_candidates,
                          timeout=args.timeout, log_dir=kernels_dir.parent / "loom")
        _report(result)
        failures += 0 if result.ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
