"""Measure per-layer latency inside vLLM's own runtime."""
import os
import statistics
from dataclasses import dataclass, field

import torch

from pp_balance.model_access import find_layers

# Keep the engine in-process so hook state is easy to reason about.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_ATTR = "_ppbalance_records"
TOP = "TOP"  # key for the whole-model forward


@dataclass
class ProfileResult:
    model_id: str
    n_layers: int
    prefill_ms: list[float] = field(default_factory=list)
    decode_ms: list[float] = field(default_factory=list)
    prefill_iqr: list[float] = field(default_factory=list)
    decode_iqr: list[float] = field(default_factory=list)
    model_fwd_ms: float = 0.0   # top-level model forward, prefill (denominator)
    end_to_end_ms: float = 0.0  # whole generate() call, informational only


def install_hooks(model):
    """Attach CUDA-event pairs to every decoder layer, plus the whole model.

    The `"TOP"` entry times the top-level model forward and is the denominator
    for the reconstruction check. VERIFIED: a hook on the model *body*
    (`model.model`) never fires under vLLM's runtime, so it cannot be used;
    a hook on the top-level module does.
    """
    parts = find_layers(model)
    keys = list(range(len(parts.layers))) + [TOP]
    setattr(model, _ATTR, {k: [] for k in keys})
    records = getattr(model, _ATTR)

    def attach(module, key):
        def pre(mod, args):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            mod._ppb_start = start

        def post(mod, args, out, key=key):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            # Index the dict at call time so drain_timings can safely rebind.
            records[key].append((mod._ppb_start, end))
            return out

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    for idx, layer in enumerate(parts.layers):
        attach(layer, idx)
    attach(model, TOP)
    return len(parts.layers)


def drain_timings(model) -> dict:
    """Synchronise, convert events to milliseconds, and clear. In-worker."""
    torch.cuda.synchronize()
    records = getattr(model, _ATTR)
    out = {k: [s.elapsed_time(e) for s, e in evs] for k, evs in records.items()}
    for k in records:
        records[k] = []
    return out


def _median_iqr(samples: list[list[float]]) -> tuple[list[float], list[float]]:
    med, iqr = [], []
    for s in samples:
        s = sorted(s)
        med.append(statistics.median(s))
        if len(s) >= 4:
            q1 = s[len(s) // 4]
            q3 = s[(3 * len(s)) // 4]
            iqr.append(q3 - q1)
        else:
            iqr.append(0.0)
    return med, iqr


def profile_model(
    model_id: str,
    prompt_tokens: int = 4096,
    decode_steps: int = 8,
    repeats: int = 5,
    max_model_len: int | None = None,
) -> ProfileResult:
    """Load through vLLM, drive real requests, return per-layer medians."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        max_model_len=max_model_len or (prompt_tokens + decode_steps + 64),
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # CUDA graphs would swallow per-layer boundaries
        # Force the prompt through in ONE forward. Without this vLLM chunks
        # prefill, and "the first call per layer is the prefill" — which the
        # loop below relies on — silently becomes false.
        max_num_batched_tokens=prompt_tokens + decode_steps + 64,
        # VERIFIED REQUIRED: with prefix caching on (the default), the warmup
        # call populates the cache and every measured repeat of the same prompt
        # skips prefill entirely, recording ZERO prefill samples per layer.
        enable_prefix_caching=False,
    )
    n_layers = llm.apply_model(install_hooks)[0]

    # A real prompt of the requested token length.
    tok = llm.get_tokenizer()
    prompt_ids = tok.encode("The history of computing " * 4000)[:prompt_tokens]
    prompt = tok.decode(prompt_ids)

    prefill_s: list[list[float]] = [[] for _ in range(n_layers)]
    decode_s: list[list[float]] = [[] for _ in range(n_layers)]

    # Warmup, then discard.
    llm.generate([prompt], SamplingParams(max_tokens=decode_steps))
    llm.apply_model(drain_timings)

    top_s: list[float] = []
    e2e: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        llm.generate([prompt], SamplingParams(max_tokens=decode_steps,
                                              ignore_eos=True))
        t1.record()
        torch.cuda.synchronize()
        e2e.append(t0.elapsed_time(t1))

        timings = llm.apply_model(drain_timings)[0]
        for key, series in timings.items():
            if not series:
                continue
            if key == TOP:
                top_s.append(series[0])  # prefill forward of the whole model
                continue
            # First call per layer is the prefill forward; the rest are decodes.
            prefill_s[key].append(series[0])
            decode_s[key].extend(series[1:])

    if not top_s or any(not v for v in prefill_s):
        raise RuntimeError(
            "no prefill samples recorded — check that prefix caching is "
            "disabled and that hooks are attached to the executing model"
        )

    p_med, p_iqr = _median_iqr(prefill_s)
    d_med, d_iqr = _median_iqr(decode_s)
    return ProfileResult(model_id, n_layers, p_med, d_med, p_iqr, d_iqr,
                         model_fwd_ms=statistics.median(top_s),
                         end_to_end_ms=statistics.median(e2e))
