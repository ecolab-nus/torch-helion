#!/usr/bin/env python
"""End to end: a PyTorch model in, Loom-compiled kernels out.

    ./cmd/compile_all.sh mamba --njobs 16

Everything ``cmd/generate_kernels.py`` does, and then every generated kernel
goes through Loom's full pipeline — Helion frontend, dataflow exploration,
MLAR ETG resolution, CP-SAT block-size solve, materialisation — so each one
ends as bufferized MLIR with a solved tiling.

Loom's outputs land in ``results/<run>/loom/<kernel>/`` and its optimum is
printed beside the analytic estimate that drove planning. Budget a couple of
minutes per kernel.

Exit status is 0 when the reference check passes, every kernel is accepted,
and Loom succeeds on all of them.
"""

from __future__ import annotations

import argparse
import time

from _common import (
    add_model_args,
    add_output_args,
    default_run_name,
    load_model,
    model_epilog,
    parse_overrides,
    print_cost_table,
    print_program,
    print_validation,
    rel,
    require_model,
    setup_logging,
)

from torch_helion import CompileConfig, compile_model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compile_all", description=__doc__.splitlines()[0],
                                epilog=model_epilog(), formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_output_args(p)
    p.add_argument("--njobs", type=int, default=8, help="parallel workers for Loom's ETG resolution and solver")
    p.add_argument("--topk-candidates", type=int, default=1, help="Loom candidates to materialise per kernel")
    p.add_argument("--no-check", action="store_true", help="skip the standalone frontend/exploration check")
    args = p.parse_args(argv)
    setup_logging(args.verbose)

    target = require_model(p, args)
    model = load_model(target, parse_overrides(args.set))
    config = CompileConfig(
        run_name=args.run_name or default_run_name(target),
        results_dir=args.results_dir,
        hw_spec=args.hw_spec,
        planner_mode=args.planner,
        check_frontend=not args.no_check,
        verify_with_loom=True,
        loom_njobs=args.njobs,
        loom_topk_candidates=args.topk_candidates,
        save_params=args.save_params,
    )
    started = time.time()
    result = compile_model(model, model.example_inputs(), config)

    print_program(result)
    validated = print_validation(result)
    loom_ok = print_cost_table(result)
    print(f"\nresults in {rel(result.run_dir)}, Loom IRs under {rel(result.run_dir / 'loom')}")
    print(f"wall time {time.time() - started:.0f}s")
    return 0 if result.reference_check["ok"] and validated and loom_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
