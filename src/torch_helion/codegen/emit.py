"""OpIR → Helion kernel source in Loom style.

Each :class:`KernelSpec` becomes one ``*.py`` file that mirrors the
reference kernels shipped with Loom (``kernels/matmul.py`` etc.):

* a plain Helion function wrapped with ``helion.kernel(static_shapes=False)``
* a ``LoomKernel`` subclass with the concrete shapes and ``bind_args()``
* the standard ``__main__`` block so the file is a Loom CLI entry point

Codegen rules encoded here (all verified against Loom's exploration pass):

* scalars are materialised as tile-shaped ``hl.full`` constants (a 0-d
  scalar operand yields an unregistered one-input generic);
* a binary op with the same operand twice is written ``a * (a * ones)``
  (Helion drops the duplicated operand otherwise);
* the RMSNorm sum of squares is accumulated with ``hl.dot`` against a ones
  matrix so no elementwise op feeds a row reduction (Loom would fuse them);
* float literals are never printed in exponent form (the frontend prints
  them verbatim into MLIR): tiny constants are emitted as products;
* attention kernels take ``[B, S, H, d]`` views and index them with
  ``tile.begin`` scalars so all memrefs stay static.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import CompileConfig
from ..optimizer.opir import KernelSpec, Program

F16 = "torch.float16"


class CodegenError(RuntimeError):
    pass


def safe_float_factors(value: float) -> list[float]:
    """Split ``value`` into factors whose ``repr`` has no exponent."""
    value = float(value)
    if value == 0.0 or "e" not in repr(value):
        return [value]
    factors: list[float] = []
    remaining = value
    for _ in range(6):
        if "e" not in repr(remaining):
            factors.append(remaining)
            return factors
        # peel off a power of ten that prints cleanly
        for step in (0.01, 0.001, 0.0001, 100.0, 1000.0, 10000.0):
            cand = remaining / step
            if "e" not in repr(cand) or abs(cand) >= 1e-4:
                factors.append(step)
                remaining = cand
                break
    factors.append(remaining)
    if any("e" in repr(f) for f in factors):
        raise CodegenError(f"cannot represent constant {value} without exponent notation")
    return factors


@dataclass
class _Val:
    expr: str
    shape: str  # "tn" (tile_t x tile_n), "col" (tile_t x 1), "k" (tile_t x tile_k)


class _GemmEmitter:
    SHAPES = {"tn": "[tile_t, tile_n]", "col": "[tile_t, 1]", "k": "[tile_t, tile_k]"}

    def __init__(self, spec: KernelSpec, program: Program) -> None:
        self.spec = spec
        self.program = program
        self.lines: list[str] = []
        self.vals: dict[str, _Val] = {}
        self.consts: dict[tuple[str, float], str] = {}
        self.counter = 0
        self.constexprs: list[tuple[str, float]] = []

    def var(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def const(self, value: float, shape: str) -> str:
        key = (shape, float(value))
        if key in self.consts:
            return self.consts[key]
        factors = safe_float_factors(value)
        names = []
        for f in factors:
            v = self.var("c")
            self.lines.append(f"        {v} = hl.full({self.SHAPES[shape]}, {f!r}, dtype={F16})")
            names.append(v)
        if len(names) == 1:
            out = names[0]
        else:
            out = self.var("c")
            self.lines.append(f"        {out} = {' * '.join(names)}")
        self.consts[key] = out
        return out

    def operand(self, name: str) -> _Val:
        if name in self.vals:
            return self.vals[name]
        for a in self.spec.args:
            if a.name == name:
                v = self.var("ld")
                self.lines.append(f"        {v} = {a.name}[tile_t, tile_n]")
                self.vals[name] = _Val(v, "tn")
                return self.vals[name]
        raise CodegenError(f"{self.spec.name}: unknown epilogue operand {name}")

    def emit_body(self) -> None:
        spec = self.spec
        for g in spec.gemms:
            self.vals[g.acc] = _Val(g.acc, "tn")
        if spec.sumsq:
            # ss accumulates 0.5 * sum(x^2) (see loop); restore the factor 2 here
            self.lines.append("        ss_col = torch.amax(ss, -1, keepdim=True)")
            two = self.const(2.0, "col")
            self.lines.append(f"        sumsq = ss_col * {two}")
            self.vals["sumsq"] = _Val("sumsq", "col")
        for op in spec.epilogue:
            self.emit_op(op)
        for o in spec.outputs:
            v = self.operand(o.value)
            self.lines.append(f"        out_{_ident(o.tensor)}[tile_t, tile_n] = {v.expr}")

    def emit_op(self, op) -> None:
        name = op.name
        out = self.var(_ident(name)[:24])
        if op.op == "broadcast":
            src = self.operand(op.inputs[0])
            self.lines.append(f"        {out} = broadcast({src.expr}, 1, [{src.expr}.size(0), tile_n])")
            self.vals[name] = _Val(out, "tn")
            return
        if op.op in ("row_sum", "row_max"):
            src = self.operand(op.inputs[0])
            fn = "torch.sum" if op.op == "row_sum" else "torch.amax"
            self.lines.append(f"        {out} = {fn}({src.expr}, -1, keepdim=True)")
            self.vals[name] = _Val(out, "col")
            return
        if op.op in ("exp", "log"):
            src = self.operand(op.inputs[0])
            self.lines.append(f"        {out} = torch.{op.op}({src.expr})")
            self.vals[name] = _Val(out, src.shape)
            return
        if op.op in ("add", "sub", "mul", "div", "max", "powf"):
            if "scalar" in op.attrs:
                a = self.operand(op.inputs[0])
                c = self.const(float(op.attrs["scalar"]), a.shape)
                lhs, rhs = (c, a.expr) if op.attrs.get("scalar_first") else (a.expr, c)
                shape = a.shape
            else:
                a, b = self.operand(op.inputs[0]), self.operand(op.inputs[1])
                lhs, rhs = a.expr, b.expr
                shape = a.shape if a.shape == b.shape else ("tn" if "tn" in (a.shape, b.shape) else a.shape)
                if lhs == rhs:
                    # identical operands: Helion drops the duplicate, so rewrite
                    ones = self.const(1.0, a.shape)
                    if op.op == "mul":
                        rhs = f"({lhs} * {ones})"
                    elif op.op == "add":
                        two = self.const(2.0, a.shape)
                        self.lines.append(f"        {out} = {lhs} * {two}")
                        self.vals[name] = _Val(out, shape)
                        return
                    else:
                        rhs = f"({lhs} * {ones})"
            expr = {
                "add": f"{lhs} + {rhs}", "sub": f"{lhs} - {rhs}", "mul": f"{lhs} * {rhs}", "div": f"{lhs} / {rhs}",
                "max": f"torch.maximum({lhs}, {rhs})", "powf": f"torch.pow({lhs}, {rhs})",
            }[op.op]
            self.lines.append(f"        {out} = {expr}")
            self.vals[name] = _Val(out, shape)
            return
        raise CodegenError(f"{self.spec.name}: cannot emit op {op.op}")


def _ident(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if not out[0].isdigit() else "_" + out


def emit_gemm_kernel(spec: KernelSpec, program: Program) -> str:
    em = _GemmEmitter(spec, program)
    args = [a.name for a in spec.args]
    sig = ", ".join(f"{a}: torch.Tensor" for a in args)
    outs = [f"out_{_ident(o.tensor)}" for o in spec.outputs]
    fn = f"_{spec.name}"
    head = [
        f"def {fn}({sig}) -> tuple[torch.Tensor, ...]:",
        "    t, k = lhs.size()",
        # A normalisation kernel has no weight to take the output width from;
        # its grid columns are the feature axis it reduces over.
        f"    _, n = {spec.gemms[0].rhs}.size()" if spec.gemms else "    n = k",
    ]
    for o in outs:
        head.append(f"    {o} = torch.empty([t, n], dtype={F16}, device=lhs.device)")
    head.append("    for tile_t, tile_n in hl.tile([t, n]):")
    for g in spec.gemms:
        head.append(f"        {g.acc} = hl.zeros([tile_t, tile_n], dtype={F16})")
    if spec.sumsq:
        head.append(f"        ss = hl.zeros([tile_t, 32], dtype={F16})")
    head.append("        for tile_k in hl.tile(k):")
    head.append("            xt = lhs[tile_t, tile_k]")
    for g in spec.gemms:
        head.append(f"            {g.acc} = hl.dot(xt, {g.rhs}[tile_k, tile_n], acc={g.acc})")
    if spec.sumsq:
        head += [
            f"            half = hl.full([tile_t, tile_k], 0.5, dtype={F16})",
            "            sq = xt * (xt * half)",
            f"            ones = hl.full([tile_k, 32], 1.0, dtype={F16})",
            "            ss = hl.dot(sq, ones, acc=ss)",
        ]
    em.emit_body()
    tail = ["    return " + ", ".join(outs) + ("," if len(outs) == 1 else "")]
    return "\n".join(head + em.lines + tail)


class _SsdEmitter(_GemmEmitter):
    """Epilogue emitter for the scan kernel: tiles are ``[tile_m, head_dim]``."""

    def __init__(self, spec: KernelSpec, program: Program) -> None:
        super().__init__(spec, program)
        self.SHAPES = {"tn": "[tile_m, p]", "col": "[tile_m, 1]"}
        self.vals["acc"] = _Val("acc", "tn")
        self.pre: list[str] = []

    def operand(self, name: str) -> _Val:
        if name in self.vals:
            return self.vals[name]
        for a in self.spec.args:
            if a.name == name:
                v = self.var("ld")
                # Hoisted above the reduction loop: Helion rejects a tile read
                # issued after the loop has closed.
                self.pre.append(f"        {v} = {a.name}_v[tile_b.begin, tile_h.begin, tile_m, :]")
                self.vals[name] = _Val(v, "tn")
                return self.vals[name]
        raise CodegenError(f"{self.spec.name}: unknown epilogue operand {name}")

    def emit_body(self) -> None:
        for op in self.spec.epilogue:
            self.emit_op(op)
        for o in self.spec.outputs:
            v = self.operand(o.value)
            self.lines.append(f"        o4_{_ident(o.tensor)}[tile_b.begin, tile_h.begin, tile_m, :] = {v.expr}")


SSD_TEMPLATE = '''def _{name}({sig}, p: hl.constexpr) -> tuple[torch.Tensor, ...]:
{views}
{allocs}
    for tile_b, tile_h, tile_m in hl.tile([_SSD_B, _SSD_H, _SSD_S], block_size=[1, 1, None]):
        acc0 = hl.zeros([tile_m, p], dtype={f16})
        ci = torch.amax(cum_v[tile_b.begin, tile_h.begin, tile_m, :], -1, keepdim=True)
        q = c_v[tile_b.begin, tile_h.begin, tile_m, :]
        xm = xskip_v[tile_b.begin, tile_h.begin, tile_m, :]
        dm = dskip_v[tile_b.begin, tile_h.begin, tile_m, :]
{pre}
        for tile_n in hl.tile(_SSD_S):
            kt = b_v[tile_b.begin, tile_h.begin, :, tile_n]
            w = torch.matmul(q, kt) * mask[tile_m, tile_n]
            cj = torch.amax(cum_v[tile_b.begin, tile_h.begin, tile_n, :], -1, keepdim=True)
            dj = torch.amax(dt_v[tile_b.begin, tile_h.begin, tile_n, :], -1, keepdim=True)
            scale = torch.exp(cj * -1.0) * dj
            v = x_v[tile_b.begin, tile_h.begin, tile_n, :] * broadcast(scale, 1, [scale.size(0), p])
            acc0 = torch.addmm(acc0, w, v)
        acc0 = acc0 * broadcast(torch.exp(ci), 1, [ci.size(0), p])
        acc = acc0 + xm * dm
{body}
    return {rets}'''


def emit_ssd_kernel(spec: KernelSpec, program: Program) -> str:
    """Render the Mamba-2 scan kernel.

    Every operand is read through a host-level head-major view of a
    token-major tensor, and the two per-token scalars (the running decay and
    ``dt``) are stored padded to 32 columns and reduced in the kernel: a
    rank-2 load with a static-1 dimension rank-reduces and Loom's exploration
    pass leaves it un-bufferized.
    """
    a = spec.attrs
    b, s, h, p, n, pad = a["batch"], a["seq"], a["heads"], a["head_dim"], a["state_dim"], a["pad"]
    # The head dim is passed as a constexpr and used as the trailing extent of
    # every view built from it. Without that specialisation Helion leaves the
    # reshaped dimensions symbolic, which yields dynamic memrefs, a
    # `tensor.cast` Loom's shape tracer rejects, and an unrolled reduction.
    # Shapes are emitted as module-level names, not literals. Helion's host
    # tracer specialises a name lookup consistently across every view; with
    # bare literals some reshaped extents stay symbolic, which yields dynamic
    # memrefs, a `tensor.cast` Loom's shape tracer rejects, and a reduction
    # loop unrolled away entirely.
    head_roles = {"x", "dskip", "xskip", "extra"}
    consts = f"_SSD_B, _SSD_S, _SSD_H, _SSD_N, _SSD_PAD = {b}, {s}, {h}, {n}, {pad}\n_SSD_T, _SSD_HP = {b * s}, {h * p}\n\n"
    names = ["_SSD_B", "_SSD_S", "_SSD_H"]
    views = []
    for arg in spec.args:
        if arg.role == "mask":
            continue
        last = "p" if arg.role in head_roles else ("_SSD_PAD" if arg.role in ("cum", "dt") else "_SSD_N")
        perm = "(0, 2, 3, 1)" if arg.role == "b" else "(0, 2, 1, 3)"
        views.append(f"    {arg.name}_v = {arg.name}.reshape({', '.join(names)}, {last}).permute{perm}")
    allocs, rets = [], []
    for o in spec.outputs:
        name = f"out_{_ident(o.tensor)}"
        allocs.append(f"    {name} = torch.empty([_SSD_T, _SSD_HP], dtype={F16}, device=c.device)")
        allocs.append(f"    o4_{_ident(o.tensor)} = {name}.reshape(_SSD_B, _SSD_S, _SSD_H, p).permute(0, 2, 1, 3)")
        rets.append(name)
    spec.constexprs = {"p": p}
    em = _SsdEmitter(spec, program)
    em.emit_body()
    return consts + SSD_TEMPLATE.format(
        name=spec.name,
        sig=", ".join(f"{x.name}: torch.Tensor" for x in spec.args),
        views="\n".join(views), allocs="\n".join(allocs),
        B=b, H=h, S=s, P=p, f16=F16,
        pre="\n".join(em.pre), body="\n".join(em.lines),
        rets=", ".join(rets) + ("," if len(rets) == 1 else ""),
    )


ATTENTION_TEMPLATE = '''def _{name}(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: hl.constexpr) -> torch.Tensor:
    batch = q.size(0)
    seq = q.size(1)
    heads = q.size(2)
    hd = hl.specialize(q.size(3))
    out_{out} = torch.empty_like(q)
    q4 = q.permute(0, 2, 1, 3)
    k4 = k.permute(0, 2, 3, 1)
    v4 = v.permute(0, 2, 1, 3)
    o4 = out_{out}.permute(0, 2, 1, 3)
    for tile_b, tile_h, tile_m in hl.tile([batch, heads, seq], block_size=[1, 1, None]):
        m_i = hl.full([tile_m, 1], float("-inf"), dtype={f16})
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_m, hd], dtype={f16})
        qt = q4[tile_b.begin, tile_h.begin, tile_m, :]
        for tile_n in hl.tile(seq):
            kt = k4[tile_b.begin, tile_h.begin, :, tile_n]
            sc = hl.full([tile_m, tile_n], scale, dtype={f16})
            qk = torch.matmul(qt, kt) * sc
            m_ij = torch.maximum(m_i, torch.amax(qk, -1, keepdim=True))
            qk = qk - broadcast(m_ij, 1, [m_ij.size(0), tile_n])
            p = torch.exp(qk)
            alpha = torch.exp(m_i - m_ij)
            vt = v4[tile_b.begin, tile_h.begin, tile_n, :]
            acc = acc * broadcast(alpha, 1, [alpha.size(0), hd])
            acc = acc + torch.matmul(p, vt)
            l_i = l_i * alpha + torch.sum(p, -1, keepdim=True)
            m_i = m_ij
        acc = acc / broadcast(l_i, 1, [l_i.size(0), hd])
        o4[tile_b.begin, tile_h.begin, tile_m, :] = acc
    return out_{out}'''


def emit_attention_kernel(spec: KernelSpec, program: Program) -> str:
    scale = float(spec.attrs["scale"])
    if "e" in repr(scale):
        raise CodegenError(f"{spec.name}: attention scale {scale} cannot be printed without exponent")
    spec.constexprs = {"scale": scale}
    return ATTENTION_TEMPLATE.format(name=spec.name, out=_ident(spec.outputs[0].tensor), f16=F16)


MODULE_TEMPLATE = '''"""{name} — generated by torch-helion (do not edit).

{doc}

Run through Loom from the repository root:

    uv run python {rel_path} --config {rel_config} --njobs 8 --debug --topk-candidates 1
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast  # noqa: F401  (registers the op)


{body}


class {cls}(LoomKernel):
    """Loom entry point for kernel ``{name}``."""

    kernel_name = "{name}"
    assume_divisible: bool = True

    ARG_SHAPES = {arg_shapes}
    CONSTEXPRS = {constexprs}
    TILES = {tiles}

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={{
            "range_unroll_factors": {tune},
            "range_num_stages": {tune},
        }},
    )(_{name})

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            raise SystemExit("generated kernels have fixed shapes; regenerate with torch-helion instead")

    @classmethod
    def bind_args(cls) -> tuple:
        args = [torch.empty(shape, dtype=torch.float16) for _, shape in cls.ARG_SHAPES]
        return tuple(args + [v for _, v in cls.CONSTEXPRS])


if __name__ == "__main__":
    try:
        shape, normalized_argv = resolve_kernel_shape_args(sys.argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sys.argv = normalized_argv
    {cls}(shape).run()
'''


def kernel_doc(spec: KernelSpec, program: Program) -> str:
    lines = []
    if spec.kind == "gemm":
        lines.append(f"GEMM kernel: grid [T={spec.grid['rows']}, N={spec.grid['cols']}], reduction K={spec.k_extent}")
        for g in spec.gemms:
            lines.append(f"  {g.acc} = lhs @ {g.rhs}  ({spec.arg(g.rhs).tensor})")
        if spec.sumsq:
            lines.append("  sumsq = row sum of lhs^2 (RMSNorm fold)")
        for op in spec.epilogue:
            lines.append(f"  {op.name} = {op.op}({', '.join(op.inputs)}) {op.attrs if op.attrs else ''}")
        for o in spec.outputs:
            lines.append(f"  store {o.tensor} <- {o.value}")
    elif spec.kind == "ssd":
        a = spec.attrs
        lines.append(f"Mamba-2 scan (quadratic form): batch={a['batch']} seq={a['seq']} heads={a['heads']} head_dim={a['head_dim']} state_dim={a['state_dim']}")
        lines.append("  acc = exp(c_i) * sum_j (C_i.B_j) causal_ij exp(-c_j) dt_j x_j  +  D * x_i")
        for op in spec.epilogue:
            lines.append(f"  {op.name} = {op.op}({', '.join(op.inputs)}) {op.attrs if op.attrs else ''}")
        for o in spec.outputs:
            lines.append(f"  store {o.tensor} <- {o.value}")
    else:
        a = spec.attrs
        lines.append(f"Flash attention: batch={a['batch']} seq={a['seq']} heads={a['heads']} head_dim={a['head_dim']} scale={a['scale']}")
    if spec.tiles:
        lines.append(f"Planned tiles: {spec.tiles}  (analytic estimate {spec.cost.get('cycles', 0):,.0f} cycles)")
    return "\n".join(lines)


def emit_kernel_module(spec: KernelSpec, program: Program, rel_path: str = "", rel_config: str = "") -> str:
    if spec.kind == "gemm":
        body = emit_gemm_kernel(spec, program)
    elif spec.kind == "ssd":
        body = emit_ssd_kernel(spec, program)
    elif spec.kind == "attention":
        body = emit_attention_kernel(spec, program)
    else:
        raise CodegenError(spec.kind)
    if spec.kind == "ssd":
        # The scan kernel takes flat token-major tensors and builds its
        # head-major views inside. Binding 4-D arguments instead makes Helion
        # specialise the grid to a single block and unroll the reduction loop
        # away, which leaves no loop-carried `scf.for` for memory binding (C2).
        arg_shapes = [(a.name, list(program.tensors[a.tensor].shape)) for a in spec.args]
    else:
        arg_shapes = [(a.name, list(a.view) if a.view else list(program.tensors[a.tensor].shape)) for a in spec.args]
    cls = "".join(p.capitalize() for p in spec.name.split("_"))
    # One entry per loop level. The scan kernel nests a 3-D grid over a
    # reduction loop; a short list leaves Helion free to specialise the
    # reshaped extents, which turns every view argument into a dynamic memref.
    levels = 4 if spec.kind == "ssd" else 2
    return MODULE_TEMPLATE.format(
        tune=repr([0] * levels),
        name=spec.name, doc=kernel_doc(spec, program), body=body, cls=cls, arg_shapes=repr(arg_shapes),
        constexprs=repr(list(spec.constexprs.items())), tiles=repr(spec.tiles), rel_path=rel_path or f"{spec.name}.py", rel_config=rel_config or f"config_files/{spec.name}.json",
    )


def loom_config(spec: KernelSpec, config: CompileConfig, output_path: Path, assigned: bool = False) -> dict[str, Any]:
    cfg: dict[str, Any] = {"output_path": str(output_path), "hw_spec": str(Path(config.hw_spec).resolve())}
    if assigned and spec.tiles:
        block = dict(spec.tiles)
        if spec.kind in ("attention", "ssd"):
            block.update({"tile_b": 1, "tile_h": 1})
        block["is_double_buffer"] = 1 if spec.cost.get("breakdown", {}).get("double_buffer", True) else 0
        cfg["assigned_block_size"] = {"ALL": block}
    return cfg


def emit_program(program: Program, config: CompileConfig, out_dir: Path) -> dict[str, Path]:
    """Write ``kernels/<name>.py`` + ``kernels/config_files/<name>.json``."""
    out_dir = Path(out_dir)
    cfg_dir = out_dir / "config_files"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    repo = config.results_dir.resolve().parent if (config.results_dir / "..").exists() else Path.cwd()
    for spec in program.kernels:
        py = out_dir / f"{spec.name}.py"
        js = cfg_dir / f"{spec.name}.json"
        js_assigned = cfg_dir / f"{spec.name}.assigned.json"
        try:
            rel_py, rel_js = str(py.relative_to(repo)), str(js.relative_to(repo))
        except ValueError:
            rel_py, rel_js = str(py), str(js)
        py.write_text(emit_kernel_module(spec, program, rel_py, rel_js))
        loom_out = out_dir.parent / "loom" / spec.name
        js.write_text(json.dumps(loom_config(spec, config, loom_out), indent=2))
        js_assigned.write_text(json.dumps(loom_config(spec, config, loom_out.parent / (spec.name + "_assigned"), assigned=True), indent=2))
        written[spec.name] = py
    return written
