import pytest
from pp_balance.validate import reconstruction_error, check_reconstruction

def test_reconstruction_error_is_relative():
    assert reconstruction_error(per_layer_sum=95.0, model_fwd=100.0) == \
        pytest.approx(0.05)

def test_check_passes_within_ten_percent():
    check_reconstruction(per_layer_sum=95.0, model_fwd=100.0)  # no raise

def test_check_raises_beyond_ten_percent():
    with pytest.raises(ValueError, match="reconstruction"):
        check_reconstruction(per_layer_sum=50.0, model_fwd=100.0)
