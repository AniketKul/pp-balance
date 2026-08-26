import pytest
from pp_balance.report import compare

def test_compare_reports_both_splits_and_the_gap():
    costs = [1.0] * 4 + [20.0] + [1.0] * 3
    mems = [0] * 8
    row = compare(costs, mems, pp_size=4, mem_cap=float("inf"))
    assert row.even_counts == [2, 2, 2, 2]
    assert row.even_max_cost == pytest.approx(21.0)  # stage holding the 20
    assert row.balanced_max_cost <= row.even_max_cost
    assert row.reduction_pct >= 0.0

def test_compare_reports_zero_gap_on_uniform_costs():
    costs = [1.0] * 8
    row = compare(costs, [0] * 8, pp_size=4, mem_cap=float("inf"))
    assert row.reduction_pct == pytest.approx(0.0, abs=1e-6)

def test_compare_surfaces_memory_infeasible():
    row = compare([1.0] * 4, [100] * 4, pp_size=2, mem_cap=150)
    assert row.memory_infeasible is True
