import torch
import torch.nn as nn
import pytest
from pp_balance.memory import module_bytes, layer_param_bytes

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False, dtype=torch.float32)

def test_module_bytes_counts_real_tensors():
    assert module_bytes(Tiny()) == 4 * 4 * 4  # 16 params x 4 bytes

def test_module_bytes_respects_dtype():
    m = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    assert module_bytes(m) == 4 * 4 * 2

def test_layer_param_bytes_one_entry_per_layer():
    layers = [Tiny(), Tiny(), Tiny()]
    assert layer_param_bytes(layers) == [64, 64, 64]
