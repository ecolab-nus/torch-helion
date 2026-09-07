"""Command line entry point.

::

    uv run python -m torch_helion --model models.llama_block:build_llama_block \
        --run-name llama_block [--verify-with-loom] [--no-check] [--planner greedy]

``--model`` names a ``module:factory`` whose factory returns an
``nn.Module`` with an ``example_inputs()`` method (see ``models/``).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

from .config import CompileConfig, DEFAULT_HW_SPEC, DEFAULT_RESULTS_DIR, REPO_ROOT
from .pipeline import compile_model


def _load_model(spec: str, kwargs: dict[str, int]):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    mod_name, _, factory = spec.partition(":")
    module = importlib.import_module(mod_name)
    fn = getattr(module, factory or "build")
    if kwargs and hasattr(module, "LlamaBlockConfig"):
        return fn(module.LlamaBlockConfig(**kwargs))
    return fn()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="torch_helion", description="Lower a PyTorch model to Loom-style Helion kernels.")
    p.add_argument("--model", default="models.llama_block:build_llama_block", help="module:factory building the model")
    p.add_argument("--run-name", default="run", help="results/<run-name>/ receives all intermediate results")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--hw-spec", type=Path, default=DEFAULT_HW_SPEC)
    p.add_argument("--planner", choices=["exhaustive", "greedy"], default="exhaustive")
    p.add_argument("--no-check", action="store_true", help="skip Helion frontend / Loom exploration validation of the generated kernels")
    p.add_argument("--verify-with-loom", action="store_true", help="run the full Loom pipeline on every kernel")
    p.add_argument("--njobs", type=int, default=8)
    p.add_argument("--save-params", action="store_true")
    p.add_argument("--shape", action="append", default=[], metavar="KEY=INT", help="model config override, e.g. --shape seq=512 (llama block)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(message)s")

    kwargs = {}
    for item in args.shape:
        k, _, v = item.partition("=")
        kwargs[k] = int(v)
    model = _load_model(args.model, kwargs)
    cfg = CompileConfig(
        run_name=args.run_name, results_dir=args.results_dir, hw_spec=args.hw_spec, planner_mode=args.planner,
        check_frontend=not args.no_check, verify_with_loom=args.verify_with_loom, loom_njobs=args.njobs, save_params=args.save_params,
    )
    res = compile_model(model, model.example_inputs(), cfg)
    print(res.program.summary())
    print(f"reference check: max rel err {res.reference_check['max_rel_err']:.3e} ({'ok' if res.reference_check['ok'] else 'FAILED'})")
    for name, v in res.validation.items():
        status = "frontend ok" if v.frontend_ok else "frontend FAILED"
        if v.exploration_ok is not None:
            status += ", exploration ok" if v.exploration_ok else ", exploration FAILED"
        print(f"  {name}: {status}" + (f" — {v.error[:200]}" if v.error else ""))
    for name, r in res.loom_results.items():
        print(f"  {name}: Loom {'ok' if r.ok else 'FAILED'} T={r.cycles} cycles blocks={r.block_sizes}" + (f" — {r.error[:200]}" if r.error else ""))
    print(f"estimated total: {res.estimated_cycles:,.0f} cycles; results in {res.run_dir}")
    return 0 if res.reference_check["ok"] and all(v.frontend_ok for v in res.validation.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
