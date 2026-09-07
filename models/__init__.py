"""Benchmark models used as lowering inputs for torch-helion.

``TARGETS`` names the models the ``cmd/`` entry points accept, so
``./cmd/compile_all.sh mamba`` is enough to say what to compile. Add a model
here and it becomes available to every entry point.
"""

from __future__ import annotations

import importlib
from typing import Any

# target name -> (module, factory, one-line description)
TARGETS: dict[str, tuple[str, str, str]] = {
    "llama_block": (
        "models.llama_block",
        "build_llama_block",
        "one Llama-style decoder block: RMSNorm, QKV + RoPE, attention, SwiGLU MLP",
    ),
    "mamba": (
        "models.mamba",
        "build_mamba_model",
        "a Mamba-2 backbone: stacked SSD blocks with a final norm",
    ),
    "mamba_block": (
        "models.mamba",
        "build_mamba_block",
        "a single Mamba-2 block, the smallest useful SSD target",
    ),
}


def available() -> list[str]:
    """Registered target names, in listing order."""
    return list(TARGETS)


def describe() -> str:
    """One line per target, for CLI help and error messages."""
    width = max(len(name) for name in TARGETS)
    return "\n".join(f"  {name:<{width}}  {desc}" for name, (_, _, desc) in TARGETS.items())


def resolve(name: str) -> tuple[Any, str]:
    """Return ``(factory, spec)`` for a registered target name."""
    if name not in TARGETS:
        raise KeyError(name)
    module_name, factory, _ = TARGETS[name]
    return getattr(importlib.import_module(module_name), factory), f"{module_name}:{factory}"
