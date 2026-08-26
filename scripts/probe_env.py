"""Assert the installed vLLM can serve both subject models."""
import torch
import vllm
from vllm.model_executor.models.registry import ModelRegistry


def main():
    print(f"vllm.__version__ = {vllm.__version__}")
    print(f"vllm.__file__    = {vllm.__file__}")

    archs = ModelRegistry.get_supported_archs()
    for arch in ("NemotronHForCausalLM", "Qwen3ForCausalLM"):
        assert arch in archs, f"{arch} not supported by this vLLM build"
        print(f"OK  {arch}")
    assert torch.cuda.is_available(), "no CUDA device"
    print(f"OK  {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
