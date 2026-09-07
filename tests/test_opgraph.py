import json

import pytest
import torch

from torch_helion.capture.opgraph import OpGraph, TensorType


def _tiny():
    g = OpGraph("t")
    x = g.add("input", [], TensorType((4, 8)), kind="input", name="x")
    w = g.add("param", [], TensorType((8, 8)), kind="param", name="w", value=torch.eye(8, dtype=torch.float16))
    mm = g.add("matmul", [x.name, w.name], TensorType((4, 8)))
    e = g.add("exp", [mm.name], TensorType((4, 8)))
    dead = g.add("log", [mm.name], TensorType((4, 8)))
    g.outputs = [e.name]
    return g, x, w, mm, e, dead


def test_tensor_type_basics():
    t = TensorType((2, 3), "float16")
    assert t.rank == 2 and t.numel == 6 and t.nbytes == 12
    assert str(t) == "float16[2, 3]"
    assert TensorType.of(torch.zeros(2, 3, dtype=torch.float32)).dtype == "float32"


def test_add_users_and_dce():
    g, x, w, mm, e, dead = _tiny()
    assert [u.name for u in g.users(mm.name)] == [e.name, dead.name]
    assert g.num_uses(mm.name) == 2
    assert g.dce() == 1
    assert dead.name not in g
    assert x.name in g  # inputs survive


def test_replace_uses_and_remove():
    g, x, w, mm, e, dead = _tiny()
    g.replace_all_uses(mm.name, x.name)
    assert e.inputs == [x.name]
    with pytest.raises(ValueError):
        g.remove(x.name)
    g.dce()
    assert mm.name not in g


def test_toposort_and_insert_before():
    g, x, w, mm, e, dead = _tiny()
    late = g.add("exp", [x.name], TensorType((4, 8)))
    g.insert_before(mm.name, late)
    order = list(g.nodes)
    assert order.index(late.name) < order.index(mm.name)
    g.toposort()
    order = list(g.nodes)
    for n in g:
        for i in n.inputs:
            assert order.index(i) < order.index(n.name)


def test_json_roundtrip(tmp_path):
    g, *_ = _tiny()
    p = tmp_path / "g.json"
    g.to_json(p)
    g2 = OpGraph.from_json(p)
    assert [n.to_dict() for n in g2] == [n.to_dict() for n in g]
    assert g2.outputs == g.outputs
    assert json.loads(p.read_text())["name"] == "t"


def test_copy_keeps_params():
    g, x, w, *_ = _tiny()
    c = g.copy()
    assert torch.equal(c.value(w.name), g.value(w.name))
    assert c.summary().startswith("OpGraph t")


def test_unknown_input_rejected():
    g = OpGraph()
    with pytest.raises(KeyError):
        g.add("exp", ["nope"], TensorType((1, 1)))
