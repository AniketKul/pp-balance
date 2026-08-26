from pp_balance.partitioner import even_split

def test_even_split_matches_vllm_for_nemotron_h_pp8():
    # 52 layers, pp_size=8: upstream nudges the remainder toward middle ranks
    assert even_split(52, 8) == [6, 6, 6, 7, 7, 7, 7, 6]

def test_even_split_divisible_is_uniform():
    assert even_split(36, 4) == [9, 9, 9, 9]

def test_even_split_always_sums_to_num_layers():
    for n in range(1, 60):
        for pp in range(1, min(n, 16) + 1):
            assert sum(even_split(n, pp)) == n
