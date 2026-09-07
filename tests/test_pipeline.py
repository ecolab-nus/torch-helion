import importlib.util
import json
import os

import pytest

from torch_helion import CompileConfig, compile_model

HAVE_LOOM = importlib.util.find_spec("helion_mlir") is not None and importlib.util.find_spec("loom_pipeline") is not None


def test_compile_model_end_to_end(block, example_args, tmp_path):
    cfg = CompileConfig(run_name="unit", results_dir=tmp_path, check_frontend=False, planner_mode="greedy")
    res = compile_model(block, example_args, cfg)
    assert len(res.program.kernels) == 5
    assert res.reference_check["ok"]
    files = {p.name for p in res.run_dir.rglob("*") if p.is_file()}
    for expected in ("01_fx_graph.txt", "02_opgraph_raw.json", "03_opgraph_canonical.json", "03_pass_log.json", "04_hw_spec.json", "candidates.json", "plan.md", "06_opir.json", "08_reference_check.json", "09_cost_report.md", "09_cost_report.json", "k0_gemm.py", "k1_attention.py", "k0_gemm.json", "README.md", "timings.json"):
        assert expected in files, expected
    opir = json.loads((res.run_dir / "06_opir.json").read_text())
    assert [k["kind"] for k in opir["kernels"]] == ["gemm", "attention", "gemm", "gemm", "gemm"]
    assert res.estimated_cycles > 0


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_LOOM, reason="Loom toolchain not available")
def test_all_generated_kernels_pass_loom_exploration(block, example_args, tmp_path):
    cfg = CompileConfig(run_name="unit_validate", results_dir=tmp_path, check_frontend=True, planner_mode="greedy")
    res = compile_model(block, example_args, cfg)
    for name, v in res.validation.items():
        assert v.frontend_ok and v.exploration_ok, f"{name}: {v.error}"


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_LOOM or os.environ.get("TORCH_HELION_RUN_LOOM") != "1", reason="set TORCH_HELION_RUN_LOOM=1 to run the full Loom pipeline")
def test_loom_pipeline_on_generated_kernel(block, example_args, tmp_path):
    from torch_helion.cost_models.loom_backend import run_loom

    cfg = CompileConfig(run_name="unit_loom", results_dir=tmp_path, check_frontend=False, planner_mode="greedy")
    res = compile_model(block, example_args, cfg)
    spec = res.program.kernels[2]
    out = run_loom(res.kernel_files[spec.name], res.run_dir / "kernels" / "config_files" / f"{spec.name}.json", njobs=8)
    assert out.ok, out.error
    assert out.solver_units and out.solver_units > 0 and out.block_sizes
