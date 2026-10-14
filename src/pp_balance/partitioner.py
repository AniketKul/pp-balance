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


from dataclasses import dataclass, field


@dataclass
class Partition:
    """Result of a partitioning attempt."""
    partition: list[int] = field(default_factory=list)  # layers per stage
    max_stage_cost: float = float("inf")
    max_stage_bytes: int = 0
    memory_infeasible: bool = False


def cut_points(
    x: float, costs: list[float], mems: list[int], mem_cap: float
) -> list[int]:
    """Greedily cut for budget x, returning the element count per stage."""
    counts, acc_t, acc_m, n = [], 0.0, 0, 0
    for c, b in zip(costs, mems):
        if n and (acc_t + c > x or acc_m + b > mem_cap):
            counts.append(n)
            acc_t, acc_m, n = c, b, 1
        else:
            acc_t, acc_m, n = acc_t + c, acc_m + b, n + 1
    if n:
        counts.append(n)
    return counts


def balance(
    costs: list[float], mems: list[int], pp_size: int, mem_cap: float
) -> Partition:
    """Minimise the cost of the most expensive contiguous stage.

    Binary searches the bottleneck cost. Monotone in x because raising the
    budget can only push cuts later, never earlier.
    """
    if not feasible(float("inf"), costs, mems, pp_size, mem_cap):
        # Memory alone rules out every budget; say so rather than return hi.
        return Partition(memory_infeasible=True)

    lo, hi = max(costs), sum(costs)
    for _ in range(100):  # ~1e-30 relative precision, ample for ms-scale costs
        mid = (lo + hi) / 2
        if feasible(mid, costs, mems, pp_size, mem_cap):
            hi = mid
        else:
            lo = mid

    counts = cut_points(hi, costs, mems, mem_cap)
    idx, worst_t, worst_m = 0, 0.0, 0
    for n in counts:
        worst_t = max(worst_t, sum(costs[idx:idx + n]))
        worst_m = max(worst_m, sum(mems[idx:idx + n]))
        idx += n
    return Partition(counts, worst_t, worst_m, False)


def evaluate_split(
    counts: list[int], costs: list[float], mems: list[int]
) -> Partition:
    """Score a given per-stage element count against a cost/memory vector."""
    if sum(counts) != len(costs):
        raise ValueError(
            f"counts sum to {sum(counts)} but there are {len(costs)} elements"
        )
    idx, worst_t, worst_m = 0, 0.0, 0
    for n in counts:
        worst_t = max(worst_t, sum(costs[idx:idx + n]))
        worst_m = max(worst_m, sum(mems[idx:idx + n]))
        idx += n
    return Partition(list(counts), worst_t, worst_m, False)
