"""Capture: PyTorch model → FX graph → OpGraph."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .fx_capture import CapturedFx, export_to_fx
from .fx_to_opgraph import UnsupportedOpError, fx_to_opgraph, supported_aten_ops
from .opgraph import OpGraph, OpNode, TensorType


def capture(model: torch.nn.Module, example_args: Sequence[Any]) -> tuple[CapturedFx, OpGraph]:
    """Export ``model`` and build the raw OpGraph in one call."""
    fx = export_to_fx(model, example_args)
    return fx, fx_to_opgraph(fx)


__all__ = [
    "CapturedFx",
    "OpGraph",
    "OpNode",
    "TensorType",
    "UnsupportedOpError",
    "capture",
    "export_to_fx",
    "fx_to_opgraph",
    "supported_aten_ops",
]
