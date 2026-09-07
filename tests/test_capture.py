import pytest
import torch

from torch_helion.capture import UnsupportedOpError, capture, supported_aten_ops


def test_capture_llama_block(captured):
    fx, g = captured
    hist = fx.op_histogram()
    assert "aten.mm.default" in hist and "aten.bmm.default" in hist and "aten._softmax.default" in hist
    assert g.inputs == ["x"]
    assert len(g.outputs) == 1
    params = [n.name for n in g if n.kind == "param"]
    assert "attn_q_proj_weight" in params and "rope_cos" in params
    assert all(n.name in g.params for n in g if n.kind in ("param", "const"))
    # every node has a concrete 2D+ type
    assert all(n.type.numel > 0 for n in g)
    assert "mm" in [n.name for n in g if n.op == "matmul"]


def test_raw_graph_is_topological(captured):
    _, g = captured
    order = list(g.nodes)
    for n in g:
        for i in n.inputs:
            assert order.index(i) < order.index(n.name)


def test_unsupported_op_reported():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.erf(x)

    with pytest.raises(UnsupportedOpError):
        capture(M(), (torch.randn(4, 4),))


def test_supported_ops_listing():
    ops = supported_aten_ops()
    assert "aten.mm.default" in ops and "aten.view.default" in ops
