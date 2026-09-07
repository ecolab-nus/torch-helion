import importlib.util
import re

import pytest
import torch

from torch_helion.codegen.emit import emit_kernel_module, emit_program
from torch_helion.codegen.interpret import program_outputs
from torch_helion.codegen.validate import validate_kernel
from torch_helion.config import CompileConfig
from torch_helion.optimizer.evaluate import max_rel_err
from torch_helion.optimizer.partition import build_program
from torch_helion.optimizer.planner import cost_program

HAVE_LOOM = importlib.util.find_spec("helion_mlir") is not None and importlib.util.find_spec("loom_pipeline") is not None


@pytest.fixture(scope="module")
def program(canonical, hw):
    g, _ = canonical
    prog = build_program(g, name="test")
    cost_program(prog, hw, CompileConfig())
    return prog


def test_emitted_sources_are_valid_python(program):
    for spec in program.kernels:
        src = emit_kernel_module(spec, program)
        compile(src, spec.name, "exec")
        assert "class " in src and "LoomKernel" in src and "hl.tile(" in src
        code = src.split('"""', 2)[2]  # skip the module docstring (it prints raw attrs)
        assert not re.search(r"\d+e-\d+", code), "exponent-form float leaked into kernel source"
        if spec.kind == "gemm":
            assert src.count("hl.dot(xt,") == len(spec.gemms)
            assert ("ss = hl.dot(sq, ones" in src) == spec.sumsq
        else:
            assert "tile_b.begin, tile_h.begin" in src


def test_epsilon_constant_is_product_of_clean_floats(program):
    k0 = program.kernels[0]
    src = emit_kernel_module(k0, program)
    assert "0.001" in src and "0.01" in src  # 1e-5 = 0.001 * 0.01


def test_emit_program_writes_files(program, tmp_path):
    cfg = CompileConfig(results_dir=tmp_path, run_name="r")
    files = emit_program(program, cfg, tmp_path / "r" / "kernels")
    assert set(files) == {k.name for k in program.kernels}
    for name, path in files.items():
        assert path.exists()
        assert (path.parent / "config_files" / f"{name}.json").exists()
        assert (path.parent / "config_files" / f"{name}.assigned.json").exists()


def test_interpreter_matches_model(program, canonical, block, example_args):
    g, _ = canonical
    with torch.no_grad():
        ref = block(*example_args)
    out = program_outputs(program, g.params, list(example_args))[0]
    assert out.shape == ref.shape
    assert max_rel_err(out.reshape(-1), ref.reshape(-1)) < 2e-2


@pytest.mark.skipif(not HAVE_LOOM, reason="Loom toolchain not available")
def test_generated_gemm_kernel_passes_frontend_and_exploration(program, tmp_path):
    cfg = CompileConfig(results_dir=tmp_path, run_name="r")
    files = emit_program(program, cfg, tmp_path / "r" / "kernels")
    spec = program.kernels[2]  # o_proj + residual: smallest gemm
    res = validate_kernel(files[spec.name], cfg.hw_spec, explore=True)
    assert res.frontend_ok, res.error
    assert res.exploration_ok, res.error
    assert set(res.symbols) == {"tile_t", "tile_n", "tile_k"}
