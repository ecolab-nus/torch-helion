"""End-to-end pipeline: PyTorch model → OpGraph → OpIR → Helion kernels.

::

    from torch_helion import compile_model, CompileConfig
    result = compile_model(model, example_args, CompileConfig(run_name="llama_block"))
    print(result.program.summary())

Every stage stores its result under ``results/<run_name>/`` (see
:mod:`torch_helion.results`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from .capture import CapturedFx, OpGraph, capture
from .codegen.emit import emit_program
from .codegen.interpret import program_outputs
from .codegen.validate import ValidationResult, validate_kernels
from .config import CompileConfig
from .cost_models.hw_spec import HardwareSpec
from .cost_models.loom_backend import LoomResult, run_loom
from .optimizer.evaluate import max_rel_err
from .optimizer.opir import Program
from .optimizer.passes import canonicalize
from .optimizer.planner import PlanResult, plan
from .results import ResultsWriter

log = logging.getLogger("torch_helion")


@dataclass
class CompileResult:
    config: CompileConfig
    run_dir: Path
    fx: CapturedFx
    raw_graph: OpGraph
    graph: OpGraph
    hw: HardwareSpec
    plan: PlanResult
    program: Program
    kernel_files: dict[str, Path]
    validation: dict[str, ValidationResult] = field(default_factory=dict)
    reference_check: dict[str, Any] = field(default_factory=dict)
    loom_results: dict[str, LoomResult] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def estimated_cycles(self) -> float:
        return self.program.metadata.get("plan", {}).get("total_cycles", float("nan"))


def compile_model(model: torch.nn.Module, example_args: Sequence[Any], config: CompileConfig | None = None) -> CompileResult:
    config = config or CompileConfig()
    rw = ResultsWriter(config.run_dir)
    timings: dict[str, float] = {}

    def stage(name: str):
        class _T:
            def __enter__(self_):
                self_.t = time.time()
                log.info("stage %s", name)

            def __exit__(self_, *a):
                timings[name] = time.time() - self_.t

        return _T()

    # 1. capture ----------------------------------------------------------------
    with stage("capture"):
        fx, raw = capture(model, example_args)
        rw.write_text("00_model.txt", f"{model!r}\n\nconfig: {config!r}\n")
        rw.write_text("01_fx_graph.txt", fx.text())
        rw.write_json("01_fx_op_histogram.json", fx.op_histogram())
        raw.to_json(rw.path("02_opgraph_raw.json"))
        rw.write_text("02_opgraph_raw.txt", raw.summary())

    # 2. optimize: canonical graph -----------------------------------------------
    with stage("canonicalize"):
        graph, pass_log = canonicalize(raw, verify_inputs=list(example_args))
        graph.to_json(rw.path("03_opgraph_canonical.json"))
        rw.write_text("03_opgraph_canonical.txt", graph.summary())
        rw.write_json("03_pass_log.json", pass_log)

    # 3. hardware + planning --------------------------------------------------------
    with stage("plan"):
        hw = HardwareSpec.load(config.hw_spec)
        rw.write_json("04_hw_spec.json", hw.summary())
        result = plan(graph, hw, config, name=config.run_name)
        program = result.program
        rw.write_json("05_plan/candidates.json", [c.to_dict() for c in result.candidates])
        rw.write_text("05_plan/plan.md", plan_report(result, hw, config))
        program.to_json(rw.path("06_opir.json"))
        rw.write_text("06_opir.txt", program.summary())

    # 4. codegen ---------------------------------------------------------------------
    with stage("codegen"):
        kernel_files = emit_program(program, config, rw.path("kernels"))
        if config.save_params:
            torch.save({n: graph.params[n] for n in program.params}, rw.path("params.pt"))
        rw.write_text("kernels/README.md", kernels_readme(program, kernel_files, config))

    # 5. reference check: OpIR interpreter vs PyTorch --------------------------------
    with stage("reference_check"):
        with torch.no_grad():
            ref = model(*example_args)
        outs = program_outputs(program, graph.params, list(example_args))
        refs = ref if isinstance(ref, (list, tuple)) else [ref]
        errs = [max_rel_err(o.reshape(-1), r.reshape(-1)) for o, r in zip(outs, refs)]
        reference_check = {"max_rel_err": max(errs), "per_output": errs, "tolerance": 2e-2, "ok": max(errs) <= 2e-2}
        rw.write_json("08_reference_check.json", reference_check)

    # 6. structural validation with the Helion frontend / Loom exploration ------------
    validation: dict[str, ValidationResult] = {}
    if config.check_frontend:
        with stage("validate"):
            validation = validate_kernels(kernel_files, Path(config.hw_spec), explore=True, workers=config.validate_workers)
            rw.write_json("07_validation.json", {k: v.to_dict() for k, v in validation.items()})

    # 7. cost report -------------------------------------------------------------------
    rw.write_json("09_cost_report.json", cost_report_json(program, hw, config))
    rw.write_text("09_cost_report.md", cost_report_md(program, hw, config, validation))

    # 8. optional: Loom in the loop --------------------------------------------------------
    loom_results: dict[str, LoomResult] = {}
    if config.verify_with_loom:
        with stage("loom"):
            for spec in program.kernels:
                cfg = rw.path(f"kernels/config_files/{spec.name}.json")
                res = run_loom(kernel_files[spec.name], cfg, njobs=config.loom_njobs, topk_candidates=config.loom_topk_candidates, log_dir=rw.path("loom"))
                loom_results[spec.name] = res
                spec.cost["loom"] = res.to_dict()
            rw.write_json("10_loom_results.json", {k: v.to_dict() for k, v in loom_results.items()})
            program.to_json(rw.path("06_opir.json"))
            rw.write_text("09_cost_report.md", cost_report_md(program, hw, config, validation, loom_results))

    rw.write_json("timings.json", timings)
    return CompileResult(config, rw.run_dir, fx, raw, graph, hw, result, program, kernel_files, validation, reference_check, loom_results, timings)


# ------------------------------------------------------------------ reports

def plan_report(result: PlanResult, hw: HardwareSpec, config: CompileConfig) -> str:
    lines = [f"# Partition plan ({config.run_name})", "", f"Hardware: {hw.path.name}, mesh {hw.mesh}, L1 {hw.l1_bytes:,} bytes", f"Candidates: {len(result.candidates)} (legal: {sum(c.legal for c in result.candidates)}), planner mode: {config.planner_mode}", ""]
    lines.append("| # | kernels | est. cycles | legal | note |")
    lines.append("|---|---------|-------------|-------|------|")
    for c in sorted(result.candidates, key=lambda c: (not c.legal, c.total_cycles)):
        note = c.error or "; ".join(c.violations[:2]) or ("best" if c is result.best else "")
        cyc = f"{c.total_cycles:,.0f}" if c.legal else "-"
        lines.append(f"| {c.index} | {len(c.program.kernels) if c.program else '-'} | {cyc} | {'yes' if c.legal else 'no'} | {note} |")
    lines += ["", "## Chosen program", "", "```", result.program.summary(), "```"]
    return "\n".join(lines)


def cost_report_json(program: Program, hw: HardwareSpec, config: CompileConfig) -> dict[str, Any]:
    kernels = []
    for k in program.kernels:
        bd = k.cost.get("breakdown", {})
        kernels.append({"name": k.name, "kind": k.kind, "tiles": k.tiles, "cycles": k.cost.get("cycles"), "breakdown": bd, "loom": k.cost.get("loom")})
    inter = program.intermediates()
    link_bytes = {t: program.tensors[t].nbytes for t in inter}
    return {"hw": hw.summary(), "kernel_launch_overhead": config.kernel_launch_overhead, "kernels": kernels, "total_cycles": program.metadata.get("plan", {}).get("total_cycles"), "intermediate_tensors": link_bytes, "intermediate_bytes": sum(link_bytes.values())}


def cost_report_md(program: Program, hw: HardwareSpec, config: CompileConfig, validation: dict[str, ValidationResult] | None = None, loom: dict[str, LoomResult] | None = None) -> str:
    total = program.metadata.get("plan", {}).get("total_cycles", 0.0)
    lines = [f"# Cost report ({program.name})", "", f"Hardware `{hw.path.name}`: {hw.mesh[0]}x{hw.mesh[1]} mesh, L1 {hw.l1_bytes:,} bytes/core.", f"Estimated end-to-end time: **{total:,.0f} cycles** (analytic model, incl. {config.kernel_launch_overhead:,.0f} cycles per kernel boundary).", ""]
    lines.append("| kernel | kind | tiles | est. cycles | waves | iters | load/iter | compute/iter | epilogue | store | L1 bytes | frontend | exploration | Loom cycles |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in program.kernels:
        bd = k.cost.get("breakdown", {})
        v = (validation or {}).get(k.name)
        fe = "-" if v is None else ("ok" if v.frontend_ok else "FAIL")
        ex = "-" if v is None or v.exploration_ok is None else ("ok" if v.exploration_ok else "FAIL")
        lr = (loom or {}).get(k.name)
        lc = "-" if lr is None or lr.cycles is None else f"{lr.cycles:,.0f}"
        lines.append(f"| {k.name} | {k.kind} | {k.tiles} | {k.cost.get('cycles', 0):,.0f} | {bd.get('waves', '-')} | {bd.get('loop_iters', '-')} | {bd.get('t_load_iter', 0):,.0f} | {bd.get('t_compute_iter', 0):,.0f} | {bd.get('t_epilogue', 0):,.0f} | {bd.get('t_store', 0):,.0f} | {bd.get('l1_bytes', 0):,} | {fe} | {ex} | {lc} |")
    inter = program.intermediates()
    lines += ["", f"Kernel-to-kernel links: {len(inter)} intermediate tensors, {sum(program.tensors[t].nbytes for t in inter):,} bytes through DRAM.", ""]
    for t in inter:
        p = program.producer(t)
        cs = [c.name for c in program.consumers(t)]
        lines.append(f"- `{t}` {program.tensors[t]}: {p.name if p else '?'} -> {', '.join(cs) or 'output'}")
    return "\n".join(lines) + "\n"


def kernels_readme(program: Program, files: dict[str, Path], config: CompileConfig) -> str:
    lines = ["# Generated kernels", "", f"Program `{program.name}`: {len(program.kernels)} Helion kernels in execution order.", ""]
    for k in program.kernels:
        lines.append(f"## {k.name} ({k.kind})")
        lines.append("")
        lines.append("Inputs: " + ", ".join(f"`{a.name}` ← `{a.tensor}` {program.tensors[a.tensor]}" + (f" viewed as {a.view}" if a.view else "") for a in k.args))
        lines.append("Outputs: " + ", ".join(f"`{o.tensor}` {program.tensors[o.tensor]}" for o in k.outputs))
        lines.append("")
        lines.append(f"    uv run python {files[k.name]} --config {files[k.name].parent / 'config_files' / (k.name + '.json')} --njobs {config.loom_njobs} --debug --topk-candidates 1")
        lines.append("")
    return "\n".join(lines)
