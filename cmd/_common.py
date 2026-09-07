"""Shared plumbing for the entry points in ``cmd/``.

Each script here is a thin front end over :func:`torch_helion.compile_model`
and :func:`torch_helion.cost_models.loom_backend.run_loom`; everything they
have in common — locating a model, parsing shape overrides, and printing a
run the same way — lives here.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch_helion.config import DEFAULT_HW_SPEC, DEFAULT_RESULTS_DIR  # noqa: E402

__all__ = [
    "DEFAULT_HW_SPEC", "DEFAULT_RESULTS_DIR", "REPO_ROOT", "add_model_args", "add_output_args",
    "default_run_name", "load_model", "model_epilog", "parse_overrides", "print_cost_table",
    "print_program", "print_validation", "rel", "require_model", "setup_logging",
]


def _config_class(module) -> type | None:
    """The module's own config dataclass, if it has one."""
    for value in vars(module).values():
        if isinstance(value, type) and dataclasses.is_dataclass(value) and value.__name__.endswith("Config"):
            return value
    return None


def _coerce(text: str):
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_overrides(items: list[str]) -> dict[str, object]:
    """``["seq=512", "mode=quadratic"]`` -> ``{"seq": 512, "mode": "quadratic"}``."""
    out: dict[str, object] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        out[key.strip()] = _coerce(value.strip())
    return out


def load_model(target: str, overrides: dict[str, object] | None = None):
    """Build a model from ``models/``.

    ``target`` is a name registered in ``models.TARGETS`` (``mamba``), or an
    explicit ``module:factory`` for anything not registered there.
    """
    import models

    if ":" not in target:
        if target not in models.TARGETS:
            raise SystemExit(
                f"unknown model {target!r}. Available models:\n{models.describe()}\n"
                "or pass an explicit module:factory."
            )
        module_name = models.TARGETS[target][0]
        factory = models.TARGETS[target][1]
    else:
        module_name, _, factory = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"cannot import {module_name!r}: {exc}") from exc
    try:
        build = getattr(module, factory or "build")
    except AttributeError as exc:
        raise SystemExit(f"{module_name!r} has no factory {factory or 'build'!r}") from exc
    if not overrides:
        return build()
    config_cls = _config_class(module)
    if config_cls is None:
        raise SystemExit(f"{module_name!r} exposes no *Config dataclass, so --set cannot apply")
    fields = {f.name for f in dataclasses.fields(config_cls)}
    unknown = sorted(set(overrides) - fields)
    if unknown:
        raise SystemExit(f"{config_cls.__name__} has no field(s): {', '.join(unknown)}")
    return build(config_cls(**overrides))


def model_epilog() -> str:
    import models

    return "models:\n" + models.describe()


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", nargs="?", metavar="MODEL",
                        help="a model in models/ (see below), or an explicit module:factory")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a field of the model's config dataclass, e.g. --set seq=512")


def require_model(parser: argparse.ArgumentParser, args) -> str:
    """The model to compile, or a listing of what is available."""
    import models

    if not args.model:
        parser.exit(2, f"which model? Available:\n{models.describe()}\n")
    return args.model


def default_run_name(target: str) -> str:
    """``results/<run>/`` defaults to the model's own name."""
    return target.rpartition(":")[2].removeprefix("build_") if ":" in target else target


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", default=None,
                        help="results/<run-name>/ receives every intermediate result (default: the model name)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--hw-spec", type=Path, default=DEFAULT_HW_SPEC)
    parser.add_argument("--planner", choices=["exhaustive", "greedy"], default="exhaustive",
                        help="partition search: exhaustive enumerates every sibling grouping")
    parser.add_argument("--save-params", action="store_true", help="write the transformed weights to params.pt")
    parser.add_argument("-v", "--verbose", action="store_true")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def print_program(result) -> None:
    print(result.program.summary())
    check = result.reference_check
    status = "ok" if check["ok"] else "FAILED"
    print(f"\nreference check vs PyTorch: max rel err {check['max_rel_err']:.2e} ({status})")


def print_validation(result) -> bool:
    """Print per-kernel frontend/exploration status. Returns True if all passed."""
    if not result.validation:
        return True
    print("\nLoom frontend and exploration:")
    ok = True
    for name, v in result.validation.items():
        marks = [f"frontend {'ok' if v.frontend_ok else 'FAILED'}"]
        if v.exploration_ok is not None:
            marks.append(f"exploration {'ok' if v.exploration_ok else 'FAILED'}")
        if v.variants:
            marks.append(f"{v.variants} variants")
        print(f"  {name:14s} {', '.join(marks)}")
        if v.error:
            print(f"    {v.error.splitlines()[0][:160]}")
        ok = ok and v.frontend_ok and v.exploration_ok is not False
    return ok


def print_cost_table(result) -> bool:
    """Print the analytic estimate beside Loom's optimum. Returns True if Loom succeeded everywhere."""
    loom = result.loom_results
    header = f"\n{'kernel':<14}{'kind':<6}{'analytic':>12}{'Loom':>12}{'ratio':>8}  block sizes"
    print(header)
    print("-" * len(header))
    ok, total_a, total_l = True, 0.0, 0.0
    for spec in result.program.kernels:
        est = spec.cost.get("cycles", 0.0)
        res = loom.get(spec.name)
        total_a += est
        if res is None or not res.ok or res.cycles is None:
            ok = ok and (res is None)
            print(f"{spec.name:<14}{spec.kind:<6}{est:>12,.0f}{'FAILED':>12}{'':>8}  {(res.error or '').splitlines()[0][:60] if res else ''}")
            continue
        total_l += res.cycles
        print(f"{spec.name:<14}{spec.kind:<6}{est:>12,.0f}{res.cycles:>12,.0f}{est / res.cycles:>8.2f}  {res.block_sizes}")
    print("-" * len(header))
    print(f"{'total':<14}{'':<6}{total_a:>12,.0f}{total_l:>12,.0f}{(total_a / total_l if total_l else 0):>8.2f}")
    return ok
