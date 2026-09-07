"""The entry points in cmd/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CMD = REPO / "cmd"
if str(CMD) not in sys.path:
    sys.path.insert(0, str(CMD))

import _common  # noqa: E402

SCRIPTS = ["generate_kernels.py", "compile_all.py", "run_kernel.py"]
SHELL = ["generate_kernels.sh", "compile_all.sh", "run_kernel.sh"]


def test_parse_overrides_coerces_types():
    out = _common.parse_overrides(["seq=512", "eps=1e-5", "mode=quadratic"])
    assert out == {"seq": 512, "eps": 1e-05, "mode": "quadratic"}
    assert isinstance(out["seq"], int)
    with pytest.raises(SystemExit):
        _common.parse_overrides(["seq"])


def test_registered_targets_resolve():
    import models

    assert set(models.available()) == {"llama_block", "mamba", "mamba_block"}
    for name in models.available():
        factory, spec = models.resolve(name)
        assert callable(factory) and spec.startswith("models.")
    assert "mamba" in models.describe()


def test_load_model_by_registered_name():
    model = _common.load_model("mamba_block", {"seq": 64, "n_layers": 1})
    assert model.cfg.seq == 64
    assert type(model).__name__ == "MambaModel"


def test_load_model_rejects_unknown_target():
    with pytest.raises(SystemExit, match="unknown model"):
        _common.load_model("not_a_model", {})


def test_default_run_name():
    assert _common.default_run_name("mamba") == "mamba"
    assert _common.default_run_name("models.mamba:build_mamba_model") == "mamba_model"


def test_load_model_applies_overrides():
    model = _common.load_model("models.mamba:build_mamba_block", {"seq": 64, "n_layers": 1})
    assert model.cfg.seq == 64
    assert model.cfg.n_layers == 1
    assert model.example_inputs()[0].shape[1] == 64


def test_load_model_rejects_unknown_field():
    with pytest.raises(SystemExit, match="no field"):
        _common.load_model("models.mamba:build_mamba_block", {"nope": 1})


def test_load_model_rejects_bad_spec():
    with pytest.raises(SystemExit, match="cannot import"):
        _common.load_model("models.not_a_model:build", {})
    with pytest.raises(SystemExit, match="no factory"):
        _common.load_model("models.mamba:not_a_factory", {})


def test_config_class_discovery():
    import models.mamba as mamba

    assert _common._config_class(mamba) is mamba.MambaConfig


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help(script):
    proc = subprocess.run([sys.executable, str(CMD / script), "--help"], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    assert "MODEL" in proc.stdout or "--kernel" in proc.stdout


@pytest.mark.parametrize("script", SHELL)
def test_shell_wrapper_is_executable(script):
    path = CMD / script
    assert path.exists(), f"{script} missing"
    assert path.stat().st_mode & 0o111, f"{script} is not executable"
    text = path.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert f"cmd/{path.stem}.py" in text


@pytest.mark.parametrize("script", ["generate_kernels.py", "compile_all.py"])
def test_missing_model_lists_targets(script):
    proc = subprocess.run([sys.executable, str(CMD / script)], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "which model?" in out
    for name in ("llama_block", "mamba", "mamba_block"):
        assert name in out


def test_missing_run_lists_runs(tmp_path):
    (tmp_path / "demo" / "kernels").mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, str(CMD / "run_kernel.py"), "--results-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode != 0
    assert "which run?" in proc.stdout + proc.stderr
    assert "demo" in proc.stdout + proc.stderr


def test_run_kernel_lists_generated_kernels(tmp_path):
    kernels = tmp_path / "demo" / "kernels"
    (kernels / "config_files").mkdir(parents=True)
    (kernels / "k0_gemm.py").write_text("# generated\n")
    (kernels / "k1_ssd.py").write_text("# generated\n")
    proc = subprocess.run(
        [sys.executable, str(CMD / "run_kernel.py"), "demo", "--results-dir", str(tmp_path), "--list"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert "k0_gemm" in proc.stdout and "k1_ssd" in proc.stdout


def test_run_kernel_reports_unknown_kernel(tmp_path):
    kernels = tmp_path / "demo" / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "k0_gemm.py").write_text("# generated\n")
    proc = subprocess.run(
        [sys.executable, str(CMD / "run_kernel.py"), "demo", "--results-dir", str(tmp_path), "--kernel", "nope"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode != 0
    assert "k0_gemm" in proc.stderr


def test_generate_kernels_end_to_end(tmp_path):
    """The fast path: no Loom, kernels on disk, exit 0."""
    proc = subprocess.run(
        [sys.executable, str(CMD / "generate_kernels.py"), "mamba_block",
         "--results-dir", str(tmp_path), "--planner", "greedy",
         "--set", "seq=64", "--set", "n_layers=1"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "reference check vs PyTorch" in proc.stdout
    # the run directory defaults to the model name
    kernels = sorted(p.stem for p in (tmp_path / "mamba_block" / "kernels").glob("*.py"))
    assert any(k.endswith("_ssd") for k in kernels)
    assert (tmp_path / "mamba_block" / "06_opir.json").exists()


def test_shell_wrapper_runs_the_model(tmp_path):
    """The .sh entry point works from an unrelated working directory."""
    proc = subprocess.run(
        [str(CMD / "generate_kernels.sh"), "llama_block", "--results-dir", str(tmp_path),
         "--planner", "greedy", "--set", "seq=64", "--set", "hidden=128", "--set", "heads=2",
         "--set", "intermediate=128"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert (tmp_path / "llama_block" / "kernels").is_dir()
