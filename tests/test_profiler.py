import pytest
from pp_balance.profiler import profile_model

@pytest.mark.gpu
def test_profile_model_measures_every_layer_in_both_regimes():
    result = profile_model(
        "Qwen/Qwen3-8B", prompt_tokens=512, decode_steps=4, repeats=3,
        gpu_memory_utilization=0.30,
    )
    n = result.n_layers
    assert n == 36  # Qwen3-8B has 36 hidden layers
    assert len(result.prefill_ms) == n
    assert len(result.decode_ms) == n
    assert len(result.prefill_iqr) == n
    assert all(t > 0 for t in result.prefill_ms)
    assert all(t > 0 for t in result.decode_ms)
    # A prefill forward processes the whole prompt; a decode step processes one
    # token. Per layer, prefill must cost strictly more.
    assert sum(result.prefill_ms) > sum(result.decode_ms)
