"""Shared fixtures: a small llama block and its capture/canonical graphs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.llama_block import LlamaBlockConfig, build_llama_block  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg() -> LlamaBlockConfig:
    return LlamaBlockConfig(batch=2, seq=128, hidden=128, heads=2, intermediate=256)


@pytest.fixture(scope="session")
def block(small_cfg):
    return build_llama_block(small_cfg)


@pytest.fixture(scope="session")
def example_args(block):
    torch.manual_seed(1)
    return block.example_inputs()


@pytest.fixture(scope="session")
def captured(block, example_args):
    from torch_helion.capture import capture

    return capture(block, example_args)


@pytest.fixture(scope="session")
def canonical(captured, example_args):
    from torch_helion.optimizer.passes import canonicalize

    _, raw = captured
    graph, log = canonicalize(raw, verify_inputs=list(example_args))
    return graph, log


@pytest.fixture(scope="session")
def hw():
    from torch_helion.config import DEFAULT_HW_SPEC
    from torch_helion.cost_models.hw_spec import HardwareSpec

    if not DEFAULT_HW_SPEC.exists():
        pytest.skip("Loom hardware spec not available")
    return HardwareSpec.load(DEFAULT_HW_SPEC)


@pytest.fixture
def scratch(tmp_path):
    return tmp_path
