"""Layer-to-stage partitioning: vLLM's baseline and a cost-aware alternative."""


def even_split(num_layers: int, pp_size: int) -> list[int]:
    """Reproduce upstream vLLM `get_pp_indices` layer counts.

    Mirrors vllm/distributed/utils.py on main: an even split, with any
    remainder pushed onto the ranks just before the last one.
    """
    per = num_layers // pp_size
    partitions = [per] * pp_size
    remaining = num_layers % pp_size
    if remaining:
        for i in range(2, remaining + 2):
            partitions[-i] += 1
    return partitions
