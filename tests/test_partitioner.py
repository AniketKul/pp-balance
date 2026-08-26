from pp_balance.partitioner import even_split, feasible

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
