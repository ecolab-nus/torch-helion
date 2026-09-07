"""The Mamba-2 model and the compiler support it needs."""

from __future__ import annotations

import pytest
import torch

from models.mamba import (
    MambaConfig,
    build_mamba_block,
    build_mamba_model,
    causal_chunk_masks,
    ssd_chunked,
    ssd_quadratic,
)
from torch_helion.capture import capture
from torch_helion.optimizer import passes as P
from torch_helion.optimizer.evaluate import max_rel_err


@pytest.fixture(scope="module")
def small_mamba_cfg():
    # d_inner = nheads * headdim with both 32-aligned forces d_inner >= 1024
    return MambaConfig(batch=2, seq=64, d_model=512, expand=2, headdim=32, d_state=32, chunk=32, n_layers=1)


def _recurrence(x, dA, dt, B, C):
    """Direct O(S) SSD recurrence: the definition the two forms must match."""
    b, h, s, p = x.shape
    n = B.shape[-1]
    state = torch.zeros(b, h, p, n, dtype=x.dtype)
    ys = []
    for t in range(s):
        decay = torch.exp(dA[:, :, t]).reshape(b, h, 1, 1)
        upd = dt[:, :, t].reshape(b, h, 1, 1) * x[:, :, t].unsqueeze(-1) * B[:, :, t].unsqueeze(-2)
        state = decay * state + upd
        ys.append((state * C[:, :, t].unsqueeze(-2)).sum(-1))
    return torch.stack(ys, dim=2)


def _ssd_inputs(b=2, h=3, s=24, p=4, n=5, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(b, h, s, p)
    B = torch.randn(b, h, s, n)
    C = torch.randn(b, h, s, n)
    dt = torch.rand(b, h, s) * 0.1 + 0.01
    A = -(torch.rand(b, h, 1) * 0.9 + 0.1)
    return x, dt * A, dt, B, C


def test_both_ssd_forms_match_the_recurrence():
    x, dA, dt, B, C = _ssd_inputs()
    ref = _recurrence(x, dA, dt, B, C)
    s = x.shape[2]
    idx = torch.arange(s)
    causal = (idx.unsqueeze(1) >= idx.unsqueeze(0)).float()
    yq = ssd_quadratic(x, dA, dt, B, C, causal)
    assert max_rel_err(yq.reshape(-1), ref.reshape(-1)) < 1e-5

    l = 6
    c = s // l
    within, across, prefix = causal_chunk_masks(l, c, torch.float32)
    r4 = lambda t: t.reshape(*t.shape[:2], c, l, t.shape[-1])
    r3 = lambda t: t.reshape(*t.shape[:2], c, l)
    yc = ssd_chunked(r4(x), r3(dA), r3(dt), r4(B), r4(C), within, across, prefix)
    assert max_rel_err(yc.reshape(-1), ref.reshape(-1)) < 1e-5


def test_prefix_mask_is_a_running_sum():
    within, across, prefix = causal_chunk_masks(5, 4, torch.float32)
    v = torch.randn(3, 5)
    assert torch.allclose(v @ prefix, v.cumsum(-1), atol=1e-6)
    assert torch.equal(within, torch.tril(torch.ones(5, 5)))
    assert torch.equal(across, torch.tril(torch.ones(4, 4), diagonal=-1))


def test_model_modes_agree(small_mamba_cfg):
    outs = {}
    for mode in ("quadratic", "chunked"):
        cfg = MambaConfig(**{**small_mamba_cfg.__dict__, "mode": mode})
        model = build_mamba_model(cfg)
        torch.manual_seed(3)
        x = torch.randn(cfg.batch, cfg.seq, cfg.d_model, dtype=cfg.dtype)
        with torch.no_grad():
            outs[mode] = model(x).float()
    assert torch.isfinite(outs["quadratic"]).all()
    assert max_rel_err(outs["quadratic"].reshape(-1), outs["chunked"].reshape(-1)) < 2e-2


@pytest.mark.parametrize("cfg", [MambaConfig(), MambaConfig(batch=2, seq=64, d_model=512, expand=2, headdim=32, d_state=32, chunk=32)])
def test_config_shapes_are_aligned(cfg):
    for name in ("d_model", "d_inner", "headdim", "d_state", "nheads", "d_bc", "d_in_proj"):
        assert getattr(cfg, name) % 32 == 0, name
    assert cfg.d_inner == cfg.expand * cfg.d_model
    assert cfg.nheads * cfg.headdim == cfg.d_inner
    assert cfg.groups == cfg.nheads


def test_capture_produces_the_expected_ops(small_mamba_cfg):
    model = build_mamba_block(MambaConfig(**{**small_mamba_cfg.__dict__}))
    fx, g = capture(model, model.example_inputs())
    ops = {n.op for n in g}
    # the ops that only the Mamba path exercises
    assert {"row_cumsum", "log1p", "cmp_gt", "select", "batch_matmul", "slice"} <= ops
    assert g.inputs == ["u"]


def test_passes_canonicalise_the_whole_model(small_mamba_cfg):
    """Every pass runs, and the graph ends up in registered ops only."""
    from torch_helion.capture.ops import REGISTERED_OPS

    model = build_mamba_block(MambaConfig(**{**small_mamba_cfg.__dict__}))
    args = model.example_inputs()
    _, raw = capture(model, args)
    graph, log = P.canonicalize(raw, verify_inputs=list(args))
    ran = {e["pass"] for e in log}
    assert {"match_softplus", "push_slice_into_gemm", "lower_cumsum", "match_ssd", "materialize_const_operands"} <= ran
    assert all(e.get("max_rel_err", 0.0) < 2e-2 for e in log)
    ops = {n.op for n in graph if n.kind == "compute"}
    assert ops <= REGISTERED_OPS | {"row_sumsq", "ssd"}, ops - (REGISTERED_OPS | {"row_sumsq", "ssd"})
    assert sum(n.op == "ssd" for n in graph) == 1
    assert all(n.kind != "layout" for n in graph)


def test_ssd_node_carries_token_major_operands(small_mamba_cfg):
    model = build_mamba_block(MambaConfig(**{**small_mamba_cfg.__dict__}))
    args = model.example_inputs()
    _, raw = capture(model, args)
    graph, _ = P.canonicalize(raw)
    ssd = next(n for n in graph if n.op == "ssd")
    cfg = small_mamba_cfg
    tok = cfg.batch * cfg.seq
    a = ssd.attrs
    assert (a["batch"], a["seq"], a["heads"]) == (cfg.batch, cfg.seq, cfg.nheads)
    assert (a["head_dim"], a["state_dim"]) == (cfg.headdim, cfg.d_state)
    widths = [cfg.nheads * cfg.d_state, cfg.nheads * cfg.d_state, cfg.d_inner,
              cfg.nheads * a["pad"], cfg.nheads * a["pad"], cfg.seq, cfg.d_inner]
    for src, width in zip(ssd.inputs, widths):
        shape = graph[src].type.shape
        assert shape == ((cfg.seq, cfg.seq) if width == cfg.seq else (tok, width)), (src, shape)
    assert ssd.type.shape == (tok, cfg.d_inner)


def test_compiles_to_kernels(small_mamba_cfg, tmp_path):
    from torch_helion import CompileConfig, compile_model

    model = build_mamba_block(MambaConfig(**{**small_mamba_cfg.__dict__}))
    cfg = CompileConfig(run_name="mamba", results_dir=tmp_path, check_frontend=False, planner_mode="greedy")
    res = compile_model(model, model.example_inputs(), cfg)
    kinds = [k.kind for k in res.program.kernels]
    assert kinds.count("ssd") == 1
    assert set(kinds) == {"gemm", "ssd"}
    assert res.reference_check["ok"], res.reference_check
    scan = next(k for k in res.program.kernels if k.kind == "ssd")
    assert {a.role for a in scan.args} >= {"c", "b", "x", "cum", "dt", "mask", "dskip", "xskip"}
    assert scan.tiles and scan.cost["cycles"] > 0
    src = (res.run_dir / "kernels" / f"{scan.name}.py").read_text()
    # the shape constants must be module-level names, not literals
    assert "_SSD_B, _SSD_S, _SSD_H" in src
    assert "hl.tile([_SSD_B, _SSD_H, _SSD_S]" in src
    assert "p: hl.constexpr" in src
    # every epilogue tile read is hoisted above the reduction loop
    body = src[src.index("for tile_b"):]
    inner = body[body.index("for tile_n"):].split("\n", 1)[1].split("        acc0 = acc0")[0]
    assert "tile_m, :]" not in inner


def test_conv_is_available_but_off_by_default(small_mamba_cfg):
    assert small_mamba_cfg.use_conv is False
    cfg = MambaConfig(**{**small_mamba_cfg.__dict__, "use_conv": True})
    model = build_mamba_model(cfg)
    with torch.no_grad():
        y = model(*model.example_inputs())
    assert y.shape == (cfg.batch, cfg.seq, cfg.d_model)
