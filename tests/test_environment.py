"""Smoke tests that the Loom toolchain is importable from torch-helion."""

import importlib

import pytest


def test_torch_helion_imports():
    import torch_helion

    assert torch_helion.__version__


@pytest.mark.parametrize("module_name", ["loom", "torch", "helion", "helion_mlir"])
def test_loom_frontend_stack_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_dataflow_pipeline_imports():
    pytest.importorskip("loom_pipeline")
