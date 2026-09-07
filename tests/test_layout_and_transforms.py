import torch

from torch_helion.capture.opgraph import OpGraph, TensorType
from torch_helion.optimizer import param_transforms as pt
from torch_helion.optimizer.layout import head_merge_index, head_split_index, match_head_split, trace_layout


def _chain(b, s, h, d, transpose_last=False, merge=True):
    g = OpGraph()
    base = g.add("input", [], TensorType((b * s, h * d)), kind="input", name="x")
    v1 = g.add("view", [base.name], TensorType((b, s, h, d)), {"shape": [b, s, h, d]}, kind="layout")
    p = g.add("permute", [v1.name], TensorType((b, h, s, d)), {"dims": [0, 2, 1, 3]}, kind="layout")
    last = p
    if transpose_last:
        last = g.add("permute", [p.name], TensorType((b, h, d, s)), {"dims": [0, 1, 3, 2]}, kind="layout")
    if merge:
        shp = list(last.type.shape)
        last = g.add("view", [last.name], TensorType((b * h, shp[2], shp[3])), {"shape": [b * h, shp[2], shp[3]]}, kind="layout")
    return g, last


def test_trace_layout_identity_reshape():
    g = OpGraph()
    x = g.add("input", [], TensorType((4, 8)), kind="input", name="x")
    v = g.add("view", [x.name], TensorType((2, 2, 8)), {"shape": [2, 2, 8]}, kind="layout")
    c = g.add("clone", [v.name], TensorType((2, 2, 8)), kind="layout")
    prov = trace_layout(g, c.name)
    assert prov.base == "x" and prov.is_identity_reshape() and prov.chain == [v.name, c.name]


def test_head_split_match_and_transpose():
    b, s, h, d = 2, 16, 4, 8
    g, last = _chain(b, s, h, d)
    prov = trace_layout(g, last.name)
    assert match_head_split(prov) == {"batch": b, "seq": s, "heads": h, "head_dim": d}
    assert not prov.is_identity_reshape()
    g2, last2 = _chain(b, s, h, d, transpose_last=True)
    assert match_head_split(trace_layout(g2, last2.name), transpose_last=True) == {"batch": b, "seq": s, "heads": h, "head_dim": d}
    assert match_head_split(trace_layout(g2, last2.name)) is None


def test_head_merge_inverts_split():
    b, s, h, d = 2, 8, 2, 4
    split = head_split_index(b, s, h, d)  # [B,H,S,d] of arange(T*D)
    merged = head_merge_index(b, s, h, d)  # [T,D] of arange(B*H*S*d)
    # merging the split view recovers the identity
    x = torch.arange(b * s * h * d)
    assert torch.equal(x.reshape(b, s, h, d).permute(0, 2, 1, 3).reshape(-1)[merged.reshape(-1)], x)
    assert split.reshape(-1).tolist() != list(range(b * s * h * d))


def test_rotate_half_matrix_matches_definition():
    d = 8
    x = torch.randn(3, d)
    half = d // 2
    ref = torch.cat((-x[:, half:], x[:, :half]), -1)
    assert torch.allclose(x @ pt.rotate_half_matrix(d), ref)
    heads = 2
    xh = torch.randn(3, heads * d)
    ref_h = torch.cat([torch.cat((-xh[:, i * d + half : (i + 1) * d], xh[:, i * d : i * d + half]), -1) for i in range(heads)], -1)
    assert torch.allclose(xh @ pt.rotate_half_heads_matrix(heads, d), ref_h)


def test_weight_folds():
    k, n = 8, 6
    w = torch.randn(k, n, dtype=torch.float16)
    g = torch.rand(k, dtype=torch.float16) + 0.5
    x = torch.randn(4, k, dtype=torch.float16)
    ref = ((x.float() * g.float()) @ w.float())
    assert torch.allclose(x.float() @ pt.fold_gain_into_weight(g, w).float(), ref, atol=2e-2, rtol=2e-2)
    heads, d = 2, 4
    w2 = torch.randn(k, heads * d, dtype=torch.float16)
    rot = pt.fold_rotation_into_weight(w2, heads, d)
    y = x.float() @ w2.float()
    yr = torch.cat([torch.cat((-y[:, i * d + 2 : (i + 1) * d], y[:, i * d : i * d + 2]), -1) for i in range(heads)], -1)
    assert torch.allclose(x.float() @ rot.float(), yr, atol=5e-2, rtol=5e-2)
    assert pt.transpose_weight(torch.arange(6).reshape(2, 3)).shape == (3, 2)


def test_expand_rope_table():
    s, d, b, h = 4, 2, 2, 3
    tab = torch.arange(s * d, dtype=torch.float32).reshape(s, d)
    full = pt.expand_rope_table(tab, b, h)
    assert full.shape == (b * s, h * d)
    assert torch.equal(full[s + 1, h * d - d :], tab[1])
