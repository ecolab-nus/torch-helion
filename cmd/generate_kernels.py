#!/usr/bin/env python
"""torch-helion only: a PyTorch model in, Helion kernels out. Loom is not run.

    ./cmd/generate_kernels.sh mamba_block --set seq=256

Runs capture, the optimizer passes, partitioning, planning and codegen, then
checks the OpIR against the original model with the reference interpreter.
Nothing here shells out to Loom, so it finishes in seconds and is the loop to
use while changing passes or templates.

Pass ``--check`` to additionally push each generated kernel through Loom's
frontend and exploration passes, which is where constraints C1–C5 are
enforced. That still does not solve for block sizes; use
``cmd/compile_all.py`` for that.

Exit status is 0 when the reference check passes (and, with ``--check``,
every kernel is accepted).
"""

from __future__ import annotations

import argparse

from _common import (
    add_model_args,
    add_output_args,
    default_run_name,
    load_model,
    model_epilog,
    parse_overrides,
    print_program,
    print_validation,
    rel,
    require_model,
    setup_logging,
)

from torch_helion import CompileConfig, compile_model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="generate_kernels", description=__doc__.splitlines()[0],
                                epilog=model_epilog(), formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_output_args(p)
    p.add_argument("--check", action="store_true",
                   help="also run Loom's frontend and exploration on each kernel (no block-size solve)")
    p.add_argument("--validate-workers", type=int, default=4, help="kernels validated concurrently with --check")
    args = p.parse_args(argv)
    setup_logging(args.verbose)

    target = require_model(p, args)
    run_name = args.run_name or default_run_name(target)
    model = load_model(target, parse_overrides(args.set))
    config = CompileConfig(
        run_name=run_name,
        results_dir=args.results_dir,
        hw_spec=args.hw_spec,
        planner_mode=args.planner,
        check_frontend=args.check,
        validate_workers=args.validate_workers,
        verify_with_loom=False,
        save_params=args.save_params,
    )
    result = compile_model(model, model.example_inputs(), config)

    print_program(result)
    validated = print_validation(result)
    print(f"\nestimated total: {result.estimated_cycles:,.0f} cycles (analytic model)")
    print(f"kernels in {rel(result.run_dir / 'kernels')}, full run in {rel(result.run_dir)}")
    print("\nrun one through Loom with:")
    first = result.program.kernels[0].name
    print(f"  ./cmd/run_kernel.sh {run_name} --kernel {first}")
    return 0 if result.reference_check["ok"] and validated else 1


if __name__ == "__main__":
    raise SystemExit(main())
