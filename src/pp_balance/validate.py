"""Guards that stop us reporting numbers we cannot stand behind."""

RECONSTRUCTION_TOLERANCE = 0.10  # spec: within 10%


def reconstruction_error(per_layer_sum: float, model_fwd: float) -> float:
    """Relative gap between summed per-layer times and the model forward."""
    return abs(per_layer_sum - model_fwd) / model_fwd


def check_reconstruction(per_layer_sum: float, model_fwd: float) -> None:
    """Raise unless per-layer timings account for the whole model forward.

    The denominator is the TOP-LEVEL MODEL FORWARD, not the wall time of
    `generate()`. Measured on Nemotron-H-8B at 4K context, summed per-layer
    time is 99.8% of the model forward but only 65.1% of the `generate()`
    call, because that call also carries scheduling, sampling and
    detokenisation. Comparing against `generate()` would fail this check on
    every healthy run.
    """
    err = reconstruction_error(per_layer_sum, model_fwd)
    if err > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"reconstruction check failed: per-layer sum {per_layer_sum:.2f}ms "
            f"vs model forward {model_fwd:.2f}ms ({err:.1%} off, tolerance "
            f"{RECONSTRUCTION_TOLERANCE:.0%}). Hook overhead or missed work — "
            "all downstream numbers are void."
        )
