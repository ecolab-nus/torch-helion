"""Codegen: OpIR → Helion kernels (Loom style), reference interpreter, validation."""

from .emit import CodegenError, emit_kernel_module, emit_program, emit_ssd_kernel, safe_float_factors
from .interpret import program_outputs, run_kernel, run_program
from .validate import ValidationResult, validate_kernel, validate_kernels

__all__ = ["CodegenError", "ValidationResult", "emit_kernel_module", "emit_program",
    "emit_ssd_kernel", "program_outputs", "run_kernel", "run_program", "safe_float_factors", "validate_kernel", "validate_kernels"]
