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


def feasible(
    x: float,
    costs: list[float],
    mems: list[int],
    pp_size: int,
    mem_cap: float,
) -> bool:
    """Can these elements be cut into <= pp_size contiguous stages, where every
    stage costs <= x and holds <= mem_cap bytes?

    Greedy: extend the current stage until either accumulator would overflow,
    then cut. Optimal for a fixed x because costs are non-negative, so a suffix
    of a feasible segment is always feasible.
    """
    stages, acc_t, acc_m = 1, 0.0, 0
    for c, b in zip(costs, mems):
        if c > x or b > mem_cap:
            return False  # a single element fits nowhere
        if acc_t + c > x or acc_m + b > mem_cap:
            stages += 1
            if stages > pp_size:
                return False
            acc_t, acc_m = c, b
        else:
            acc_t, acc_m = acc_t + c, acc_m + b
    return True
