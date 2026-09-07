import pytest

from torch_helion.cost_models import expr as E
from torch_helion.cost_models.yaml_lite import _MiniYaml, load_yaml


def test_evaluate_arithmetic():
    assert E.evaluate("M * N / 2", {"M": 4, "N": 6}) == 12
    assert E.evaluate("(M * N / 8192) * 716", {"M": 64, "N": 128}) == pytest.approx(716)
    assert E.evaluate("2 * B * M * N * K", {"B": 2, "M": 1, "N": 1, "K": 3}) == 12


def test_evaluate_constraints():
    assert E.evaluate("M >= 32 && N >= 32 && K >= 32", {"M": 32, "N": 64, "K": 32}) is True
    assert E.evaluate("M * N < 8192", {"M": 64, "N": 64}) is True
    assert E.evaluate("M * N >= 8192", {"M": 64, "N": 64}) is False
    assert E.evaluate("True", {}) is True


def test_symbols_and_errors():
    assert E.symbols("M * N * 2 * effective_bandwidth") == {"M", "N", "effective_bandwidth"}
    with pytest.raises(E.ExprError):
        E.evaluate("M + ", {"M": 1})
    with pytest.raises(E.ExprError):
        E.evaluate("M + Q", {"M": 1})
    with pytest.raises(E.ExprError):
        E.evaluate("__import__('os')", {})


PERF = '''
time_costs:
  a: &a
    simple:
      fixed_latency: "1"
      volume: "L"
      throughput: "1024"
functions:
  vec_add_f16:
    scenarios:
      - time_cost: *a
  matmul_SS_f16:
    constraints: "M >= 32 && N >= 32"
    scenarios:
      - constraints: "M * N >= 8192"
        time_cost: *a
      - constraints: "M * N < 8192"
        time_cost: *a
  bcst:
    symbols: ["M", "N", "bcst_x"]
    scenarios:
      - time_cost: *a
'''


def test_mini_yaml_parser_matches_pyyaml():
    mini = _MiniYaml(PERF).parse()
    assert mini["functions"]["vec_add_f16"]["scenarios"][0]["time_cost"]["simple"]["throughput"] == "1024"
    assert mini["functions"]["matmul_SS_f16"]["scenarios"][1]["constraints"] == "M * N < 8192"
    assert mini["functions"]["bcst"]["symbols"] == ["M", "N", "bcst_x"]
    assert load_yaml(PERF) == mini


def test_hw_spec_loads_loom_mesh(hw):
    assert hw.mesh == (8, 8) and hw.cores == 64
    assert hw.l1_bytes == 16 * 5464 * 16  # matches Loom's ETG L1 capacity
    assert hw.has_op("matmul") and hw.has_op("exp") and not hw.has_op("rsqrt")


def test_hw_spec_costs_follow_perf_yaml(hw):
    # matmul_large: M*N/2 + 2*M*N*K/716
    m, n, k = 128, 128, 256
    assert hw.matmul_cycles(m, n, k) == pytest.approx(m * n / 2 + 2 * m * n * k / 716)
    # small scenario when M*N < 8192
    assert hw.matmul_cycles(32, 32, 32) == pytest.approx(32 * 32 / 2 + 2 * 32 * 32 * 32 / ((32 * 32 / 8192) * 716))
    assert hw.elementwise_cycles("exp", 7000) == pytest.approx(1 + 7000 / 7)
    assert hw.reduce_cycles("row_sum", 64, 64) == pytest.approx(1 + 64 * 64 / 128)
    assert hw.dram_store_cycles(64, 64) == pytest.approx(454 + 64 * 64 * 2 / 150)
    assert hw.dram_load_cycles(64, 64, (1, 8)) > hw.dram_load_cycles(64, 64, (8, 1))
    with pytest.raises(ValueError):
        hw.matmul_cycles(16, 16, 16)  # violates M >= 32
    with pytest.raises(KeyError):
        hw.op_cycles("rsqrt", L=1)
