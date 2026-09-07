"""PyTorch model → ATen FX graph via ``torch.export``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.export import ExportedProgram


@dataclass
class CapturedFx:
    """Result of :func:`export_to_fx`."""

    exported: ExportedProgram
    example_args: tuple[Any, ...]

    @property
    def graph_module(self) -> torch.fx.GraphModule:
        return self.exported.graph_module

    def text(self) -> str:
        """Readable ATen graph (stored under ``results/``)."""
        gm = self.graph_module
        return gm.graph.python_code("self").src

    def op_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for n in self.graph_module.graph.nodes:
            if n.op == "call_function":
                key = str(n.target)
                hist[key] = hist.get(key, 0) + 1
        return dict(sorted(hist.items()))


def export_to_fx(model: torch.nn.Module, example_args: Sequence[Any]) -> CapturedFx:
    """Export ``model`` with static shapes and decompose to core ATen.

    ``torch.export`` traces the model with fake tensors, lifting parameters
    and buffers to graph inputs. ``run_decompositions()`` lowers composite
    ops (``linear``, ``silu`` ...) to the core ATen set that
    :mod:`torch_helion.capture.fx_to_opgraph` understands.
    """
    model = model.eval()
    args = tuple(example_args)
    with torch.no_grad():
        ep = torch.export.export(model, args)
        ep = ep.run_decompositions()
    return CapturedFx(exported=ep, example_args=args)
