import pytest
from pp_balance.report import compare
from pp_balance.profiler import profile_model
from pp_balance.validate import check_reconstruction


@pytest.mark.gpu
def test_qwen3_8b_control_shows_no_material_gap():
    res = profile_model("Qwen/Qwen3-8B", prompt_tokens=2048,
                        decode_steps=8, repeats=5)
    check_reconstruction(sum(res.prefill_ms), res.model_fwd_ms)
    costs = res.prefill_ms
    row = compare(costs, [0] * len(costs), pp_size=4, mem_cap=float("inf"))
    assert row.reduction_pct < 2.0, (
        f"control model shows {row.reduction_pct:.1f}% reduction; "
        "a uniform dense stack should show none — suspect a partitioner bug"
    )
