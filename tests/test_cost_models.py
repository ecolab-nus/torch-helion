import pytest

from torch_helion.capture.opgraph import TensorType
from torch_helion.cost_models.analytic import CostModelError, attention_cost, gemm_cost
from torch_helion.cost_models.tile_search import aligned_divisors, search_tiles
from torch_helion.optimizer.opir import BodyOp, GemmSpec, KernelArg, KernelOutput, KernelSpec


def _gemm(T=256, K=256, N=256, gemms=1, sumsq=False, extra=False):
    spec = KernelSpec(name="k", kind="gemm", lhs="lhs", k_extent=K, grid={"rows": T, "cols": N}, sumsq=sumsq)
    tensors = {"x": TensorType((T, K))}
    spec.args.append(KernelArg("lhs", "x", "lhs"))
    for i in range(gemms):
        spec.args.append(KernelArg(f"w{i}", f"w{i}", "rhs"))
        tensors[f"w{i}"] = TensorType((K, N))
        spec.gemms.append(GemmSpec(f"acc{i}", f"w{i}"))
    if extra:
        spec.args.append(KernelArg("in0", "r", "extra"))
        tensors["r"] = TensorType((T, N))
        spec.epilogue.append(BodyOp("y", "add", ["acc0", "in0"]))
        spec.outputs.append(KernelOutput("y", "y"))
    else:
        spec.outputs.append(KernelOutput("out", "acc0"))
    tensors["y"] = tensors["out"] = TensorType((T, N))
    return spec, tensors


def test_gemm_cost_scales_with_k(hw):
    s1, t1 = _gemm(K=256)
    s2, t2 = _gemm(K=512)
    tiles = {"tile_t": 64, "tile_n": 64, "tile_k": 64}
    c1, c2 = gemm_cost(s1, t1, hw, tiles), gemm_cost(s2, t2, hw, tiles)
    assert c1.cycles > 0 and c2.loop_iters == 2 * c1.loop_iters
    assert c2.cycles > c1.cycles
    assert c1.l1_bytes < hw.l1_bytes and c1.orientation in ("t_on_x", "t_on_y")


def test_sumsq_and_extras_add_cost(hw):
    tiles = {"tile_t": 64, "tile_n": 64, "tile_k": 64}
    base = gemm_cost(*_gemm(), hw, tiles)
    with_sumsq = gemm_cost(*_gemm(sumsq=True), hw, tiles)
    with_extra = gemm_cost(*_gemm(extra=True), hw, tiles)
    assert with_sumsq.t_compute_iter > base.t_compute_iter
    assert with_extra.t_epilogue > base.t_epilogue and with_extra.dram_read_bytes > base.dram_read_bytes


def test_l1_overflow_raises(hw):
    spec, tensors = _gemm(T=4096, K=4096, N=4096, gemms=4)
    with pytest.raises(CostModelError):
        gemm_cost(spec, tensors, hw, {"tile_t": 512, "tile_n": 512, "tile_k": 512})


def test_aligned_divisors_and_search(hw):
    assert aligned_divisors(256, 32, 512) == [32, 64, 128, 256]
    assert aligned_divisors(96, 32, 512) == [32, 96]
    assert aligned_divisors(16, 32, 512) == [16]
    spec, tensors = _gemm()
    res = search_tiles(spec, tensors, hw)
    assert res.feasible > 0 and res.best.cycles == min(r.cycles for r in res.top)
    assert all(256 % v == 0 for v in res.best.tiles.values())


def test_attention_cost(hw):
    spec = KernelSpec(name="a", kind="attention", attrs={"batch": 2, "seq": 256, "heads": 4, "head_dim": 64, "scale": 0.125})
    tensors = {}
    c = attention_cost(spec, tensors, hw, {"tile_m": 64, "tile_n": 64})
    assert c.grid_tiles == 2 * 4 * 4 and c.loop_iters == 4 and c.cycles > 0
    res = search_tiles(spec, tensors, hw)
    assert res.best.tiles["tile_m"] % 32 == 0
