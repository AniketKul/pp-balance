"""Report whether vLLM exposes per-layer KV/SSM byte attribution."""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import json

from vllm import LLM

from pp_balance.memory import layer_param_bytes
from pp_balance.model_access import find_layers

llm = LLM(model="nvidia/Nemotron-H-8B-Base-8K", max_model_len=2048,
          gpu_memory_utilization=0.85, enforce_eager=True,
          max_num_batched_tokens=2048, enable_prefix_caching=False)

def probe(model):
    parts = find_layers(model)
    return {
        "param_bytes": layer_param_bytes(parts.layers),
        "embed_bytes": sum(p.numel() * p.element_size()
                           for p in parts.embed.parameters()),
        "lm_head_bytes": sum(p.numel() * p.element_size()
                             for p in parts.lm_head.parameters()),
    }

info = llm.apply_model(probe)[0]
pb = info["param_bytes"]
print(f"MEM: layers={len(pb)} min={min(pb)/1e6:.1f}MB max={max(pb)/1e6:.1f}MB "
      f"ratio={max(pb)/min(pb):.2f}")
print(f"MEM: embed={info['embed_bytes']/1e9:.3f}GB "
      f"lm_head={info['lm_head_bytes']/1e9:.3f}GB")

spec_available = False
try:
    from pp_balance.memory import kv_state_bytes_per_token
    def runner_probe(model):
        return None  # placeholder; runner is not reachable via apply_model
    print("MEM: kv spec via runner not reachable through apply_model")
except Exception as exc:  # pragma: no cover - diagnostic only
    print(f"MEM: kv spec probe failed: {exc}")

with open("results/memory-probe.json", "w") as f:
    json.dump({"param_bytes": pb,
               "embed_bytes": info["embed_bytes"],
               "lm_head_bytes": info["lm_head_bytes"],
               "kv_spec_available": spec_available}, f, indent=2)
print("MEM: wrote results/memory-probe.json")
