import pytest
import torch

from torch_helion.capture.opgraph import OpGraph, TensorType
from torch_helion.capture.ops import REGISTERED_OPS
from torch_helion.optimizer import passes as P
from torch_helion.optimizer.evaluate import graph_outputs, max_rel_err


def test_canonical_graph_is_loom_style(canonical):
    g, log = canonical
    assert all(n.kind != "layout" for n in g)
    compute_ops = {n.op for n in g if n.kind == "compute"}
    allowed = REGISTERED_OPS | {"row_sumsq", "attention"}
    assert compute_ops <= allowed, compute_ops - allowed
    assert sum(n.op == "attention" for n in g) == 1
    assert sum(n.op == "row_sumsq" for n in g) == 2  # two RMSNorms
    assert all(n.type.rank == 2 for n in g if n.kind == "compute")
    assert all(e.get("max_rel_err", 0.0) < 2e-2 for e in log)
    names = {e["pass"] for e in log}
    assert {"lower_rope", "lower_rmsnorm", "match_attention", "flatten_to_2d"} <= names


def test_rope_rotation_folded_into_weights(canonical):
    g, _ = canonical
    rot = [n for n in g if n.kind == "const" and "_rot" in n.name]
    assert len(rot) == 2  # q and k projections
    # rope tables deduplicated: one cos and one sin table
    tables = [n.name for n in g if n.kind == "const" and n.name.startswith("rope_")]
    assert len(tables) == 2


def _elementwise_graph(op, attrs=None, shape=(4, 8)):
    g = OpGraph()
    x = g.add("input", [], TensorType(shape, "float32"), kind="input", name="x")
    y = g.add(op, [x.name], TensorType(shape, "float32"), attrs or {})
    g.outputs = [y.name]
    return g


@pytest.mark.parametrize("op", ["neg", "rsqrt", "sigmoid", "silu", "tanh", "square", "sqrt"])
def test_rewrite_unregistered_preserves_numerics(op):
    g = _elementwise_graph(op)
    x = torch.rand(4, 8) + 0.5
    ref = graph_outputs(g, [x])[0]
    P.rewrite_unregistered(g)
    assert all(n.op in REGISTERED_OPS for n in g if n.kind == "compute")
    out = graph_outputs(g, [x])[0]
    assert max_rel_err(out, ref) < 1e-5


def test_row_mean_rewrite():
    g = OpGraph()
    x = g.add("input", [], TensorType((4, 8), "float32"), kind="input", name="x")
    m = g.add("row_mean", [x.name], TensorType((4, 1), "float32"), {"keepdim": True})
    g.outputs = [m.name]
    xv = torch.randn(4, 8)
    P.rewrite_unregistered(g)
    assert torch.allclose(graph_outputs(g, [xv])[0], xv.mean(-1, keepdim=True), atol=1e-6)


def test_insert_broadcasts_and_errors():
    g = OpGraph()
    x = g.add("input", [], TensorType((4, 8), "float32"), kind="input", name="x")
    c = g.add("row_sum", [x.name], TensorType((4, 1), "float32"))
    m = g.add("mul", [x.name, c.name], TensorType((4, 8), "float32"))
    g.outputs = [m.name]
    assert P.insert_broadcasts(g) == 1
    assert g[m.inputs[1]].op == "broadcast" and g[m.inputs[1]].attrs["shape"] == [4, 8]
    bad = OpGraph()
    a = bad.add("input", [], TensorType((4, 8), "float32"), kind="input", name="a")
    b = bad.add("input", [], TensorType((8, 4), "float32"), kind="input", name="b")
    bad.add("add", [a.name, b.name], TensorType((4, 8), "float32"))
    with pytest.raises(P.PassError):
        P.insert_broadcasts(bad)


def test_rotate_half_and_rope_matching():
    g = OpGraph()
    x = g.add("input", [], TensorType((2, 8), "float32"), kind="input", name="x")
    s1 = g.add("slice", [x.name], TensorType((2, 4), "float32"), {"dim": 1, "start": 0, "end": 4}, kind="layout")
    s2 = g.add("slice", [x.name], TensorType((2, 4), "float32"), {"dim": 1, "start": 4, "end": 8}, kind="layout")
    n = g.add("neg", [s2.name], TensorType((2, 4), "float32"))
    cat = g.add("cat", [n.name, s1.name], TensorType((2, 8), "float32"), {"dim": 1}, kind="layout")
    cos = g.add_const(torch.rand(2, 8), name="cos")
    sin = g.add_const(torch.rand(2, 8), name="sin")
    m1 = g.add("mul", [x.name, cos.name], TensorType((2, 8), "float32"))
    m2 = g.add("mul", [cat.name, sin.name], TensorType((2, 8), "float32"))
    add = g.add("add", [m1.name, m2.name], TensorType((2, 8), "float32"))
    g.outputs = [add.name]
    xv = torch.randn(2, 8)
    ref = graph_outputs(g, [xv])[0]
    assert P.match_rotate_half(g) == 1 and g[cat.name].op == "rotate_half"
    assert P.match_rope(g) == 1 and g[add.name].op == "rope"
    assert torch.allclose(graph_outputs(g, [xv])[0], ref, atol=1e-6)


def test_rmsnorm_without_a_gemm_consumer_becomes_a_standalone_norm():
    g = OpGraph()
    x = g.add("input", [], TensorType((4, 8), "float32"), kind="input", name="x")
    gain = g.add_const(torch.rand(8) + 0.5, name="g")
    n = g.add("rmsnorm", [x.name, gain.name], TensorType((4, 8), "float32"), {"eps": 1e-5})
    g.outputs = [n.name]
    xv = torch.randn(4, 8)
    ref = graph_outputs(g, [xv])[0]
    assert P.lower_rmsnorm(g) == 1
    assert g[g[n.name].inputs[0]].op == "mul"
    # the per-feature gain is materialised as a full 2-D constant (C5)
    full = [c for c in g if c.kind == "const" and c.type.shape == (4, 8)]
    assert len(full) == 1
    assert torch.allclose(graph_outputs(g, [xv])[0], ref, atol=1e-5)
    assert sum(nd.op == "row_sumsq" for nd in g) == 1


def test_softplus_pattern_and_rewrite():
    g = OpGraph()
    t = TensorType((4, 8), "float32")
    x = g.add("input", [], t, kind="input", name="x")
    e = g.add("exp", [x.name], t)
    l = g.add("log1p", [e.name], t)
    c = g.add("cmp_gt", [x.name], t, {"scalar": 20.0})
    sel = g.add("select", [c.name, x.name, l.name], t)
    g.outputs = [sel.name]
    xv = torch.randn(4, 8)
    assert P.match_softplus(g) == 1
    assert g[sel.name].op == "softplus"
    g.dce()
    P.rewrite_unregistered(g)
    assert all(nd.op in REGISTERED_OPS for nd in g if nd.kind == "compute")
    out = graph_outputs(g, [xv])[0]
    assert torch.allclose(out, torch.nn.functional.softplus(xv), atol=1e-4)


def test_lower_cumsum_is_a_triangular_gemm():
    g = OpGraph()
    t = TensorType((4, 8), "float32")
    x = g.add("input", [], t, kind="input", name="x")
    cs = g.add("row_cumsum", [x.name], t)
    g.outputs = [cs.name]
    xv = torch.randn(4, 8)
    assert P.lower_cumsum(g) == 1
    mm = [nd for nd in g if nd.op == "matmul"]
    assert len(mm) == 1
    prefix = g.value(mm[0].inputs[1])
    assert torch.equal(prefix, torch.triu(torch.ones(8, 8)))
    assert torch.allclose(graph_outputs(g, [xv])[0], xv.cumsum(-1), atol=1e-5)


def test_push_slice_into_gemm():
    g = OpGraph()
    x = g.add("input", [], TensorType((4, 6), "float32"), kind="input", name="x")
    w = g.add_const(torch.randn(6, 10), name="w")
    mm = g.add("matmul", [x.name, w.name], TensorType((4, 10), "float32"))
    a = g.add("slice", [mm.name], TensorType((4, 4), "float32"), {"dim": 1, "start": 0, "end": 4}, kind="layout")
    b = g.add("slice", [mm.name], TensorType((4, 6), "float32"), {"dim": 1, "start": 4, "end": 10}, kind="layout")
    g.outputs = [a.name, b.name]
    xv = torch.randn(4, 6)
    ref = [o.clone() for o in graph_outputs(g, [xv])]
    assert P.push_slice_into_gemm(g) == 2
    assert g[a.name].op == "matmul" and g[b.name].op == "matmul"
    assert g.value(g[a.name].inputs[1]).shape == (6, 4)
    outs = graph_outputs(g, [xv])
    assert all(torch.allclose(o, r, atol=1e-5) for o, r in zip(outs, ref))


def test_fold_constants_and_dedupe():
    g = OpGraph()
    w = g.add("param", [], TensorType((2, 3), "float32"), kind="param", name="w", value=torch.arange(6.0).reshape(2, 3))
    t = g.add("permute", [w.name], TensorType((3, 2), "float32"), {"dims": [1, 0]}, kind="layout")
    t2 = g.add("permute", [w.name], TensorType((3, 2), "float32"), {"dims": [1, 0]}, kind="layout")
    x = g.add("input", [], TensorType((1, 3), "float32"), kind="input", name="x")
    m = g.add("matmul", [x.name, t.name], TensorType((1, 2), "float32"))
    m2 = g.add("matmul", [x.name, t2.name], TensorType((1, 2), "float32"))
    g.outputs = [m.name, m2.name]
    assert P.fold_constants(g) == 2
    assert g[t.name].kind == "const" and torch.equal(g.value(t.name), torch.arange(6.0).reshape(2, 3).t())
    assert P.dedupe_constants(g) == 1
    assert g[m2.name].inputs[1] == t.name
