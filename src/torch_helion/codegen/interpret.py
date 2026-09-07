"""Reference interpreter for the OpIR.

Executes a :class:`Program` with plain PyTorch, kernel by kernel, using the
same operation order the generated Helion kernels use (accumulators, the
RMSNorm sum of squares, the epilogue SSA program, the attention template).
It is the oracle for validating the optimizer + codegen against the
original PyTorch model without a device.
"""

from __future__ import annotations

import torch

from ..optimizer.evaluate import eval_op
from ..optimizer.opir import KernelSpec, Program


def run_kernel(spec: KernelSpec, env: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Execute one kernel; returns the tensors it produces (float32)."""
    args = {a.name: env[a.tensor].float() for a in spec.args}
    outs: dict[str, torch.Tensor] = {}
    if spec.kind == "gemm":
        lhs = args["lhs"]
        vals: dict[str, torch.Tensor] = {}
        for g in spec.gemms:
            vals[g.acc] = lhs @ args[g.rhs]
        if spec.sumsq:
            vals["sumsq"] = (lhs * lhs).sum(-1, keepdim=True)
        for a in spec.args:
            if a.role == "extra":
                vals[a.name] = args[a.name]
        for op in spec.epilogue:
            vals[op.name] = eval_op(op.op, [vals[i] for i in op.inputs], op.attrs)
        for o in spec.outputs:
            outs[o.tensor] = vals[o.value]
    elif spec.kind == "ssd":
        order = ["c", "b", "x", "cum", "dt", "mask", "dskip"]
        vals = {"acc": eval_op("ssd", [args[r] for r in order], spec.attrs)}
        for a in spec.args:
            if a.role == "extra":
                vals[a.name] = args[a.name]
        for op in spec.epilogue:
            vals[op.name] = eval_op(op.op, [vals[i] for i in op.inputs], op.attrs)
        for o in spec.outputs:
            outs[o.tensor] = vals[o.value]
    elif spec.kind == "attention":
        q, k, v = args["q"], args["k"], args["v"]
        outs[spec.outputs[0].tensor] = eval_op("attention", [q, k, v], spec.attrs)
    else:
        raise ValueError(spec.kind)
    return outs


def run_program(program: Program, params: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor] | list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run all kernels in order; returns every produced tensor (float32)."""
    if isinstance(inputs, (list, tuple)):
        inputs = dict(zip(program.inputs, inputs))
    env: dict[str, torch.Tensor] = {}
    for name in program.inputs:
        env[name] = inputs[name].reshape(program.tensors[name].shape).float()
    for name in program.params:
        env[name] = params[name].float()
    for spec in program.kernels:
        env.update(run_kernel(spec, env))
    return env


def program_outputs(program: Program, params: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor] | list[torch.Tensor]) -> list[torch.Tensor]:
    env = run_program(program, params, inputs)
    shapes = program.metadata.get("output_shapes", {})
    outs = []
    for o in program.outputs:
        t = env[o]
        if o in shapes:
            t = t.reshape(shapes[o])
        outs.append(t)
    return outs
