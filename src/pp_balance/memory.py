"""Measured memory footprint per layer."""
import torch.nn as nn


def module_bytes(module: nn.Module) -> int:
    """Bytes held by this module's real parameters and buffers."""
    total = sum(p.numel() * p.element_size() for p in module.parameters())
    total += sum(b.numel() * b.element_size() for b in module.buffers())
    return total


def layer_param_bytes(layers: list[nn.Module]) -> list[int]:
    """Parameter bytes for each decoder layer, in order."""
    return [module_bytes(layer) for layer in layers]


def kv_state_bytes_per_token(runner) -> dict[str, int]:
    """Per-layer KV/SSM bytes per token, read from vLLM's own cache spec.

    Returns an empty dict if the runtime does not expose a spec, which the
    caller must surface as 'weights-only' rather than papering over with a
    config-derived estimate.
    """
    get_spec = getattr(runner, "get_kv_cache_spec", None)
    if get_spec is None:
        return {}
    out = {}
    for name, spec in get_spec().items():
        page = getattr(spec, "page_size_bytes", None)
        block = getattr(spec, "block_size", None)
        if page is None or not block:
            continue
        out[name] = page // block
    return out
