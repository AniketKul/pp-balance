import pytest
from pp_balance.partitioner import balance, cut_points, even_split, feasible

def test_even_split_matches_vllm_for_nemotron_h_pp8():
    # 52 layers, pp_size=8: upstream nudges the remainder toward middle ranks
    assert even_split(52, 8) == [6, 6, 6, 7, 7, 7, 7, 6]

def test_even_split_divisible_is_uniform():
    assert even_split(36, 4) == [9, 9, 9, 9]

def test_even_split_always_sums_to_num_layers():
    for n in range(1, 60):
        for pp in range(1, min(n, 16) + 1):
            assert sum(even_split(n, pp)) == n

COSTS = [1.0, 2.0, 3.0, 4.0, 5.0]
NOMEM = [0] * 5
INF = float("inf")

def test_feasible_when_budget_is_total():
    assert feasible(15.0, COSTS, NOMEM, pp_size=1, mem_cap=INF) is True

def test_infeasible_when_budget_below_largest_element():
    assert feasible(4.0, COSTS, NOMEM, pp_size=5, mem_cap=INF) is False

def test_feasible_splits_into_enough_stages():
    # [1,2,3] [4,5] -> max 9; budget 9 with 2 stages fits
    assert feasible(9.0, COSTS, NOMEM, pp_size=2, mem_cap=INF) is True
    assert feasible(8.0, COSTS, NOMEM, pp_size=2, mem_cap=INF) is False

def test_memory_cap_forces_extra_stages():
    mems = [10, 10, 10, 10, 10]
    # 20 bytes/stage -> 2 layers per stage -> needs 3 stages regardless of time
    assert feasible(INF, COSTS, mems, pp_size=3, mem_cap=20) is True
    assert feasible(INF, COSTS, mems, pp_size=2, mem_cap=20) is False

def test_single_element_exceeding_mem_cap_is_never_feasible():
    mems = [10, 10, 999, 10, 10]
    assert feasible(INF, COSTS, mems, pp_size=5, mem_cap=20) is False

def test_feasible_is_monotone_in_x():
    mems = [10, 10, 10, 10, 10]
    seen_true = False
    for x in [i * 0.5 for i in range(1, 40)]:
        ok = feasible(x, COSTS, mems, pp_size=3, mem_cap=30)
        if ok:
            seen_true = True
        else:
            assert not seen_true, f"feasibility flipped back to False at x={x}"

def dp_optimal_max_stage(costs, mems, pp_size, mem_cap):
    """Exact minimum achievable max-stage cost, by dynamic programming."""
    n = len(costs)
    best = [[INF] * (pp_size + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(1, n + 1):
        for k in range(1, pp_size + 1):
            for j in range(i):
                if sum(mems[j:i]) > mem_cap:
                    continue
                cand = max(best[j][k - 1], sum(costs[j:i]))
                if cand < best[i][k]:
                    best[i][k] = cand
    return min(best[n][k] for k in range(1, pp_size + 1))

def test_balance_matches_dp_optimum_on_hybrid_like_vector():
    # 3 cost classes, mimicking mamba / mlp / attention
    costs = [1.0, 1.0, 8.0, 1.0, 1.0, 8.0, 3.0, 3.0, 1.0, 1.0, 8.0, 2.0]
    mems = [0] * len(costs)
    for pp in (2, 3, 4):
        got = balance(costs, mems, pp_size=pp, mem_cap=INF)
        want = dp_optimal_max_stage(costs, mems, pp, INF)
        assert got.max_stage_cost == pytest.approx(want, rel=1e-6)

def test_balance_matches_dp_under_memory_pressure():
    costs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    mems = [10, 10, 10, 10, 10, 10, 10, 10]
    got = balance(costs, mems, pp_size=4, mem_cap=30)
    want = dp_optimal_max_stage(costs, mems, 4, 30)
    assert got.max_stage_cost == pytest.approx(want, rel=1e-6)

def test_balance_partition_is_contiguous_and_complete():
    costs = [1.0, 5.0, 2.0, 8.0, 3.0]
    res = balance(costs, [0] * 5, pp_size=3, mem_cap=INF)
    assert sum(res.partition) == 5
    assert all(p >= 0 for p in res.partition)
    assert len(res.partition) <= 3

def test_balance_beats_or_ties_even_split_on_skewed_vector():
    costs = [1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0]
    res = balance(costs, [0] * 8, pp_size=4, mem_cap=INF)
    counts = even_split(8, 4)
    idx, even_max = 0, 0.0
    for c in counts:
        even_max = max(even_max, sum(costs[idx:idx + c]))
        idx += c
    assert res.max_stage_cost <= even_max

def test_balance_reports_memory_infeasible_instead_of_returning_hi():
    costs = [1.0, 1.0, 1.0]
    mems = [10, 10, 10]
    res = balance(costs, mems, pp_size=1, mem_cap=15)
    assert res.memory_infeasible is True
    assert res.partition == []

def test_cut_points_reconstructs_partition_for_a_budget():
    costs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cut_points(9.0, costs, [0] * 5, mem_cap=INF) == [3, 2]
