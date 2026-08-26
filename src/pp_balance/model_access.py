"""Locate the parts of a loaded vLLM model that we need to measure."""
from dataclasses import dataclass

import torch.nn as nn

# From vLLM's ALL_DECODER_LAYER_TYPES in model_executor/models/nemotron_h.py
LAYER_TYPE_CODES = {"M": "mamba", "-": "mlp", "*": "attention", "E": "moe"}


@dataclass
class ModelParts:
    layers: list[nn.Module]
    embed: nn.Module
    lm_head: nn.Module


def find_layers(model: nn.Module) -> ModelParts:
    """Both NemotronHForCausalLM and Qwen3ForCausalLM use this layout."""
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise AttributeError(
            f"{type(model).__name__} has no .model.layers; "
            "this architecture is not supported"
        )
    return ModelParts(
        layers=list(inner.layers),
        embed=inner.embed_tokens,
        lm_head=model.lm_head,
    )


def layer_labels(hybrid_pattern: str | None, n_layers: int) -> list[str]:
    """Human-readable type per layer, from config.hybrid_override_pattern."""
    if hybrid_pattern is None:
        return ["dense"] * n_layers
    if len(hybrid_pattern) != n_layers:
        raise ValueError(
            f"pattern length {len(hybrid_pattern)} != n_layers {n_layers}"
        )
    return [LAYER_TYPE_CODES[c] for c in hybrid_pattern]
