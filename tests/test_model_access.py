import torch.nn as nn
import pytest
from pp_balance.model_access import find_layers, layer_labels

class FakeInner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])

class FakeModel(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.model = FakeInner(n)
        self.lm_head = nn.Linear(4, 10)

def test_find_layers_returns_layers_embed_and_head():
    m = FakeModel(3)
    parts = find_layers(m)
    assert len(parts.layers) == 3
    assert parts.embed is m.model.embed_tokens
    assert parts.lm_head is m.lm_head

def test_find_layers_raises_on_unknown_shape():
    with pytest.raises(AttributeError):
        find_layers(nn.Linear(4, 4))

def test_layer_labels_from_hybrid_pattern():
    labels = layer_labels("M-*", n_layers=3)
    assert labels == ["mamba", "mlp", "attention"]

def test_layer_labels_without_pattern_are_uniform():
    assert layer_labels(None, n_layers=3) == ["dense", "dense", "dense"]

def test_layer_labels_rejects_length_mismatch():
    with pytest.raises(ValueError):
        layer_labels("M-*", n_layers=4)
