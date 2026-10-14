"""Compare vLLM's split against the balanced split on a measured vector."""
from dataclasses import dataclass

from pp_balance.partitioner import balance, evaluate_split, even_split


@dataclass
class ComparisonRow:
    pp_size: int
    even_counts: list[int]
    even_max_cost: float
    even_max_bytes: int
    balanced_counts: list[int]
    balanced_max_cost: float
    balanced_max_bytes: int
    reduction_pct: float
    memory_infeasible: bool = False


def compare(
    costs: list[float], mems: list[int], pp_size: int, mem_cap: float
) -> ComparisonRow:
    """Score both partitions with identical code so the gap is real."""
    even_counts = even_split(len(costs), pp_size)
    even = evaluate_split(even_counts, costs, mems)
    bal = balance(costs, mems, pp_size, mem_cap)

    if bal.memory_infeasible:
        return ComparisonRow(
            pp_size, even_counts, even.max_stage_cost, even.max_stage_bytes,
            [], float("inf"), 0, 0.0, memory_infeasible=True,
        )

    reduction = 0.0
    if even.max_stage_cost > 0:
        reduction = 100.0 * (
            even.max_stage_cost - bal.max_stage_cost
        ) / even.max_stage_cost
    return ComparisonRow(
        pp_size, even_counts, even.max_stage_cost, even.max_stage_bytes,
        bal.partition, bal.max_stage_cost, bal.max_stage_bytes, reduction,
    )
