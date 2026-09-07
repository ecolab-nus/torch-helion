import torch

from torch_helion.codegen.interpret import program_outputs
from torch_helion.config import CompileConfig
from torch_helion.optimizer.evaluate import max_rel_err
from torch_helion.optimizer.legality import check_program
from torch_helion.optimizer.partition import build_program, default_grouping, sibling_sets
from torch_helion.optimizer.planner import enumerate_groupings, plan, set_partitions


def test_sibling_sets(canonical):
    g, _ = canonical
    sizes = sorted(len(s) for s in sibling_sets(g))
    assert sizes == [1, 1, 2, 5]  # o_proj, down_proj, gate/up, q/qR/k/kR/v


def test_default_partition_is_five_kernels_and_legal(canonical):
    g, _ = canonical
    prog = build_program(g)
    kinds = [k.kind for k in prog.kernels]
    assert kinds == ["gemm", "attention", "gemm", "gemm", "gemm"]
    assert check_program(prog).ok
    k0 = prog.kernels[0]
    assert len(k0.gemms) == 5 and k0.sumsq and len(k0.outputs) == 3
    assert prog.kernels[3].sumsq and len(prog.kernels[3].gemms) == 2
    assert [len(k.args) for k in prog.kernels[1:]] == [3, 3, 3, 3]
    assert prog.intermediates() and prog.outputs == g.outputs


def test_split_partition_interpreter_matches_model(canonical, block, example_args):
    g, _ = canonical
    with torch.no_grad():
        ref = block(*example_args)
    for merge in (True, False):
        prog = build_program(g, default_grouping(g, merge_siblings=merge))
        assert check_program(prog).ok
        out = program_outputs(prog, g.params, list(example_args))[0]
        assert max_rel_err(out.reshape(-1), ref.reshape(-1)) < 2e-2
    assert len(build_program(g, default_grouping(g, merge_siblings=False)).kernels) == 10  # 5 + attention + o_proj + gate + up + down


def test_set_partitions_counts():
    assert [len(set_partitions(list("abcde")[:n])) for n in range(6)] == [1, 1, 2, 5, 15, 52]


def test_enumerate_groupings(canonical):
    g, _ = canonical
    assert len(enumerate_groupings(g, "exhaustive")) == 52 * 2
    assert len(enumerate_groupings(g, "greedy")) == 4


def test_plan_picks_cheapest_legal(canonical, hw):
    g, _ = canonical
    cfg = CompileConfig(planner_mode="greedy")
    res = plan(g, hw, cfg)
    legal = [c for c in res.candidates if c.legal]
    assert legal and res.best.total_cycles == min(c.total_cycles for c in legal)
    assert all(k.tiles for k in res.program.kernels)
    assert all(k.cost["cycles"] > 0 for k in res.program.kernels)
    assert res.program.metadata["plan"]["legal"] == len(legal)


def test_program_json_roundtrip(canonical, tmp_path):
    from torch_helion.optimizer.opir import Program

    g, _ = canonical
    prog = build_program(g)
    p = tmp_path / "opir.json"
    prog.to_json(p)
    back = Program.from_json(p)
    assert back.to_dict() == prog.to_dict()
    assert "k0_gemm" in back.summary()
