"""Group the measured cost vector by layer type to sanity-check it."""
import statistics
import sys

from pp_balance.model_access import layer_labels
from pp_balance.profiler import profile_model

PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"

res = profile_model("nvidia/Nemotron-H-8B-Base-8K", prompt_tokens=8000,
                    decode_steps=8, repeats=5)
labels = layer_labels(PATTERN, res.n_layers)
for regime, costs in (("prefill", res.prefill_ms), ("decode", res.decode_ms)):
    by_type: dict[str, list[float]] = {}
    for label, c in zip(labels, costs):
        by_type.setdefault(label, []).append(c)
    print(regime)
    for label, vals in sorted(by_type.items()):
        print(f"  {label:10s} n={len(vals):2d} "
              f"median={statistics.median(vals):7.3f}ms")

attn = statistics.median([c for l, c in zip(labels, res.prefill_ms)
                          if l == "attention"])
mamba = statistics.median([c for l, c in zip(labels, res.prefill_ms)
                           if l == "mamba"])
if abs(attn - mamba) / max(attn, mamba) < 0.05:
    sys.exit("attention and mamba layers measure the same at 8K context — "
             "the hooks are almost certainly on the wrong modules")
