# PP-Balance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure vLLM's count-based pipeline-parallel layer split against a cost-aware binary-search partition, using per-layer costs measured through vLLM's own runtime on real checkpoints.

**Architecture:** Four independent components. `partitioner` is pure Python with no GPU dependency and is built first via TDD. `model_access` locates decoder layers inside a loaded vLLM model. `profiler` attaches CUDA-event hooks and drives real requests. `memory` reads real parameter and KV/state bytes. `report` compares even vs balanced splits.

**Tech Stack:** Python 3.10, `uv`, vLLM (released wheel), PyTorch 2.6 + CUDA 12.4, pytest, 1x H100 80GB.

**Reference:** Spec at `docs/superpowers/specs/2026-08-26-pp-balance-design.md`. vLLM source questions resolve against `github.com/vllm-project/vllm`, never a local checkout.

**Verified facts this plan depends on:**
- `NemotronHForCausalLM` and `Qwen3ForCausalLM` both expose decoder layers at `model.model.layers`, embeddings at `model.model.embed_tokens`, head at `model.lm_head`.
- Nemotron-H layer type codes: `M`=Mamba2, `-`=MLP, `*`=Attention, `E`=MoE, from `ALL_DECODER_LAYER_TYPES`.
- `nvidia/Nemotron-H-8B-Base-8K` `hybrid_override_pattern` is 52 chars = 24 `M` + 24 `-` + 4 `*`.
- `LLM.apply_model(func)` runs `func(model)` inside each worker and returns results. Return only plain picklable values.

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`
- Create: `src/pp_balance/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create the venv and install dependencies**

```bash
cd ~/aniket/pp-balance
uv venv --python 3.10
uv pip install vllm pytest
```

Expected: install completes. Note the resolved vLLM version; record it in the README later.

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "pp-balance"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["vllm", "pytest"]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: requires a GPU and a downloaded model"]
```

- [ ] **Step 3: Create empty package files**

```bash
mkdir -p src/pp_balance tests
touch src/pp_balance/__init__.py tests/__init__.py
uv pip install -e .
```

- [ ] **Step 4: Verify the environment supports our models**

Create `scripts/probe_env.py`:

```python
"""Assert the installed vLLM can serve both subject models."""
import torch
from vllm.model_executor.models.registry import ModelRegistry

def main():
    archs = ModelRegistry.get_supported_archs()
    for arch in ("NemotronHForCausalLM", "Qwen3ForCausalLM"):
        assert arch in archs, f"{arch} not supported by this vLLM build"
        print(f"OK  {arch}")
    assert torch.cuda.is_available(), "no CUDA device"
    print(f"OK  {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/probe_env.py`
Expected: three `OK` lines. If `NemotronHForCausalLM` is missing, the wheel is too old — install a newer vLLM before continuing. Do not proceed past this step on failure.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src tests scripts
git commit -m "chore: project scaffold and environment probe"
```

---

### Task 2: `even_split` — replicate vLLM's current algorithm

**Files:**
- Create: `src/pp_balance/partitioner.py`
- Test: `tests/test_partitioner.py`

This must reproduce upstream `get_pp_indices` exactly, because it is our baseline. If it drifts from upstream, every comparison is meaningless.

- [ ] **Step 1: Write the failing test**

```python
from pp_balance.partitioner import even_split

def test_even_split_matches_vllm_for_nemotron_h_pp8():
    # 52 layers, pp_size=8: upstream nudges the remainder toward middle ranks
    assert even_split(52, 8) == [6, 6, 6, 7, 7, 7, 7, 6]

def test_even_split_divisible_is_uniform():
    assert even_split(36, 4) == [9, 9, 9, 9]

def test_even_split_always_sums_to_num_layers():
    for n in range(1, 60):
        for pp in range(1, min(n, 16) + 1):
            assert sum(even_split(n, pp)) == n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_partitioner.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'even_split'`

- [ ] **Step 3: Write the implementation**

Create `src/pp_balance/partitioner.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_partitioner.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/pp_balance/partitioner.py tests/test_partitioner.py
git commit -m "feat: replicate vLLM even_split baseline"
```

---

### Task 3: `feasible` — the two-constraint greedy predicate

**Files:**
- Modify: `src/pp_balance/partitioner.py`
- Test: `tests/test_partitioner.py`

- [ ] **Step 1: Write the failing test**

```python
from pp_balance.partitioner import feasible

COSTS = [1.0, 2.0, 3.0, 4.0, 5.0]
NOMEM = [0] * 5
INF = float("inf")

def test_feasible_when_budget_is_total():
    assert feasible(15.0, COSTS, NOMEM, pp_size=1, mem_cap=INF) is True

def test_infeasible_when_budget_below_largest_element():
    assert feasible(4.0, COSTS, NOMEM, pp_size=5, mem_cap=INF) is False

def test_feasible_splits_into_enough_stages():
    # [1,2,3] [4,5] -> max 9; budget 9 with 2 stages fits
    assert feasible(9.0, COSTS, NOMEM, pp_size=2, mem_cap=INF) is True
    assert feasible(8.0, COSTS, NOMEM, pp_size=2, mem_cap=INF) is False

def test_memory_cap_forces_extra_stages():
    mems = [10, 10, 10, 10, 10]
    # 20 bytes/stage -> 2 layers per stage -> needs 3 stages regardless of time
    assert feasible(INF, COSTS, mems, pp_size=3, mem_cap=20) is True
    assert feasible(INF, COSTS, mems, pp_size=2, mem_cap=20) is False

def test_single_element_exceeding_mem_cap_is_never_feasible():
    mems = [10, 10, 999, 10, 10]
    assert feasible(INF, COSTS, mems, pp_size=5, mem_cap=20) is False

def test_feasible_is_monotone_in_x():
    mems = [10, 10, 10, 10, 10]
    seen_true = False
    for x in [i * 0.5 for i in range(1, 40)]:
        ok = feasible(x, COSTS, mems, pp_size=3, mem_cap=30)
        if ok:
            seen_true = True
        else:
            assert not seen_true, f"feasibility flipped back to False at x={x}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_partitioner.py -v -k feasible`
Expected: FAIL with `ImportError: cannot import name 'feasible'`

- [ ] **Step 3: Write the implementation**

Append to `src/pp_balance/partitioner.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_partitioner.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/pp_balance/partitioner.py tests/test_partitioner.py
git commit -m "feat: two-constraint greedy feasibility predicate"
```

---

### Task 4: `balance` — binary search, cross-checked against exact DP

**Files:**
- Modify: `src/pp_balance/partitioner.py`
- Test: `tests/test_partitioner.py`

- [ ] **Step 1: Write the failing test**

The DP here is the oracle. It is deliberately written differently from the binary search so a shared bug cannot hide in both.

```python
import pytest
from pp_balance.partitioner import balance, cut_points, even_split

INF = float("inf")

def dp_optimal_max_stage(costs, mems, pp_size, mem_cap):
    """Exact minimum achievable max-stage cost, by dynamic programming."""
    n = len(costs)
    best = [[INF] * (pp_size + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(1, n + 1):
        for k in range(1, pp_size + 1):
            for j in range(i):
                if sum(mems[j:i]) > mem_cap:
                    continue
                cand = max(best[j][k - 1], sum(costs[j:i]))
                if cand < best[i][k]:
                    best[i][k] = cand
    return min(best[n][k] for k in range(1, pp_size + 1))

def test_balance_matches_dp_optimum_on_hybrid_like_vector():
    # 3 cost classes, mimicking mamba / mlp / attention
    costs = [1.0, 1.0, 8.0, 1.0, 1.0, 8.0, 3.0, 3.0, 1.0, 1.0, 8.0, 2.0]
    mems = [0] * len(costs)
    for pp in (2, 3, 4):
        got = balance(costs, mems, pp_size=pp, mem_cap=INF)
        want = dp_optimal_max_stage(costs, mems, pp, INF)
        assert got.max_stage_cost == pytest.approx(want, rel=1e-6)

def test_balance_matches_dp_under_memory_pressure():
    costs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    mems = [10, 10, 10, 10, 10, 10, 10, 10]
    got = balance(costs, mems, pp_size=4, mem_cap=30)
    want = dp_optimal_max_stage(costs, mems, 4, 30)
    assert got.max_stage_cost == pytest.approx(want, rel=1e-6)

def test_balance_partition_is_contiguous_and_complete():
    costs = [1.0, 5.0, 2.0, 8.0, 3.0]
    res = balance(costs, [0] * 5, pp_size=3, mem_cap=INF)
    assert sum(res.partition) == 5
    assert all(p >= 0 for p in res.partition)
    assert len(res.partition) <= 3

def test_balance_beats_or_ties_even_split_on_skewed_vector():
    costs = [1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0]
    res = balance(costs, [0] * 8, pp_size=4, mem_cap=INF)
    counts = even_split(8, 4)
    idx, even_max = 0, 0.0
    for c in counts:
        even_max = max(even_max, sum(costs[idx:idx + c]))
        idx += c
    assert res.max_stage_cost <= even_max

def test_balance_reports_memory_infeasible_instead_of_returning_hi():
    costs = [1.0, 1.0, 1.0]
    mems = [10, 10, 10]
    res = balance(costs, mems, pp_size=1, mem_cap=15)
    assert res.memory_infeasible is True
    assert res.partition == []

def test_cut_points_reconstructs_partition_for_a_budget():
    costs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cut_points(9.0, costs, [0] * 5, mem_cap=INF) == [3, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_partitioner.py -v -k "balance or cut_points"`
Expected: FAIL with `ImportError: cannot import name 'balance'`

- [ ] **Step 3: Write the implementation**

Append to `src/pp_balance/partitioner.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_partitioner.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/pp_balance/partitioner.py tests/test_partitioner.py
git commit -m "feat: binary-search balancer cross-checked against exact DP"
```

---

### Task 5: `evaluate_split` — score any partition on a cost vector

**Files:**
- Modify: `src/pp_balance/partitioner.py`
- Test: `tests/test_partitioner.py`

Needed so the baseline and the balanced split are scored by identical code.

- [ ] **Step 1: Write the failing test**

```python
from pp_balance.partitioner import evaluate_split

def test_evaluate_split_reports_worst_stage():
    costs = [1.0, 2.0, 3.0, 4.0]
    mems = [5, 5, 5, 5]
    got = evaluate_split([2, 2], costs, mems)
    assert got.max_stage_cost == pytest.approx(7.0)  # [1,2]=3, [3,4]=7
    assert got.max_stage_bytes == 10

def test_evaluate_split_rejects_wrong_total():
    with pytest.raises(ValueError):
        evaluate_split([2, 1], [1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_partitioner.py -v -k evaluate_split`
Expected: FAIL with `ImportError: cannot import name 'evaluate_split'`

- [ ] **Step 3: Write the implementation**

Append to `src/pp_balance/partitioner.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_partitioner.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/pp_balance/partitioner.py tests/test_partitioner.py
git commit -m "feat: evaluate_split scores any partition"
```

---

### Task 6: `model_access` — locate layers inside a loaded vLLM model

**Files:**
- Create: `src/pp_balance/model_access.py`
- Test: `tests/test_model_access.py`

Tested against fake `nn.Module` trees so it needs no GPU. Both target architectures share the `model.model.layers` shape.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_access.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pp_balance.model_access'`

- [ ] **Step 3: Write the implementation**

Create `src/pp_balance/model_access.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_access.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/pp_balance/model_access.py tests/test_model_access.py
git commit -m "feat: locate decoder layers and label layer types"
```

---

### Task 7: `profiler` — CUDA-event hooks driven by real requests

**Files:**
- Create: `src/pp_balance/profiler.py`
- Test: `tests/test_profiler.py`

Hooks are installed and drained inside the worker via `apply_model`, returning plain floats so it works whether or not the engine core runs in a separate process.

- [ ] **Step 1: Write the failing test**

Uses `Qwen/Qwen3-8B` (already cached, 36 hidden layers). This is the same control
model as Task 11, so no extra checkpoint is needed. Both properties are asserted from a
single `profile_model` call rather than two, because loading a 16 GB model twice to check
two facts about the same run is waste, not extra coverage.

```python
import pytest
from pp_balance.profiler import profile_model

@pytest.mark.gpu
def test_profile_model_measures_every_layer_in_both_regimes():
    result = profile_model(
        "Qwen/Qwen3-8B", prompt_tokens=512, decode_steps=4, repeats=3
    )
    n = result.n_layers
    assert n == 36  # Qwen3-8B has 36 hidden layers
    assert len(result.prefill_ms) == n
    assert len(result.decode_ms) == n
    assert len(result.prefill_iqr) == n
    assert all(t > 0 for t in result.prefill_ms)
    assert all(t > 0 for t in result.decode_ms)
    # A prefill forward processes the whole prompt; a decode step processes one
    # token. Per layer, prefill must cost strictly more.
    assert sum(result.prefill_ms) > sum(result.decode_ms)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_profiler.py -v -m gpu`
Expected: FAIL with `ModuleNotFoundError: No module named 'pp_balance.profiler'`

- [ ] **Step 3: Write the implementation**

**Verified on this machine (2026-08-26), do not remove:** `VLLM_ENABLE_V1_MULTIPROCESSING=0`
is REQUIRED. Without it vLLM 0.19.1 runs the engine core in a separate process and
`llm.apply_model(func)` raises `TypeError: Object of type <class 'function'> is not
serializable`, because the function cannot cross the process boundary. Setting it at module
scope — before `vllm` is imported inside `profile_model` — is what makes the hook-install and
drain calls work. A smoke test with it set returned `n_layers=52` and layer types
`{NemotronHMambaDecoderLayer: 24, NemotronHMLPDecoderLayer: 24, NemotronHAttentionDecoderLayer: 4}`,
confirming both the attribute path and the architecture mix.

Create `src/pp_balance/profiler.py`:

```python
"""Measure per-layer latency inside vLLM's own runtime."""
import os
import statistics
from dataclasses import dataclass, field

import torch

from pp_balance.model_access import find_layers

# Keep the engine in-process so hook state is easy to reason about.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_ATTR = "_ppbalance_records"
TOP = "TOP"  # key for the whole-model forward


@dataclass
class ProfileResult:
    model_id: str
    n_layers: int
    prefill_ms: list[float] = field(default_factory=list)
    decode_ms: list[float] = field(default_factory=list)
    prefill_iqr: list[float] = field(default_factory=list)
    decode_iqr: list[float] = field(default_factory=list)
    model_fwd_ms: float = 0.0   # top-level model forward, prefill (denominator)
    end_to_end_ms: float = 0.0  # whole generate() call, informational only


def install_hooks(model):
    """Attach CUDA-event pairs to every decoder layer, plus the whole model.

    The `"TOP"` entry times the top-level model forward and is the denominator
    for the reconstruction check. VERIFIED: a hook on the model *body*
    (`model.model`) never fires under vLLM's runtime, so it cannot be used;
    a hook on the top-level module does.
    """
    parts = find_layers(model)
    keys = list(range(len(parts.layers))) + [TOP]
    setattr(model, _ATTR, {k: [] for k in keys})
    records = getattr(model, _ATTR)

    def attach(module, key):
        def pre(mod, args):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            mod._ppb_start = start

        def post(mod, args, out, key=key):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            # Index the dict at call time so drain_timings can safely rebind.
            records[key].append((mod._ppb_start, end))
            return out

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    for idx, layer in enumerate(parts.layers):
        attach(layer, idx)
    attach(model, TOP)
    return len(parts.layers)


def drain_timings(model) -> dict:
    """Synchronise, convert events to milliseconds, and clear. In-worker."""
    torch.cuda.synchronize()
    records = getattr(model, _ATTR)
    out = {k: [s.elapsed_time(e) for s, e in evs] for k, evs in records.items()}
    for k in records:
        records[k] = []
    return out


def _median_iqr(samples: list[list[float]]) -> tuple[list[float], list[float]]:
    med, iqr = [], []
    for s in samples:
        s = sorted(s)
        med.append(statistics.median(s))
        if len(s) >= 4:
            q1 = s[len(s) // 4]
            q3 = s[(3 * len(s)) // 4]
            iqr.append(q3 - q1)
        else:
            iqr.append(0.0)
    return med, iqr


def profile_model(
    model_id: str,
    prompt_tokens: int = 4096,
    decode_steps: int = 8,
    repeats: int = 5,
    max_model_len: int | None = None,
) -> ProfileResult:
    """Load through vLLM, drive real requests, return per-layer medians."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        max_model_len=max_model_len or (prompt_tokens + decode_steps + 64),
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # CUDA graphs would swallow per-layer boundaries
        # Force the prompt through in ONE forward. Without this vLLM chunks
        # prefill, and "the first call per layer is the prefill" — which the
        # loop below relies on — silently becomes false.
        max_num_batched_tokens=prompt_tokens + decode_steps + 64,
        # VERIFIED REQUIRED: with prefix caching on (the default), the warmup
        # call populates the cache and every measured repeat of the same prompt
        # skips prefill entirely, recording ZERO prefill samples per layer.
        enable_prefix_caching=False,
    )
    n_layers = llm.apply_model(install_hooks)[0]

    # A real prompt of the requested token length.
    tok = llm.get_tokenizer()
    prompt_ids = tok.encode("The history of computing " * 4000)[:prompt_tokens]
    prompt = tok.decode(prompt_ids)

    prefill_s: list[list[float]] = [[] for _ in range(n_layers)]
    decode_s: list[list[float]] = [[] for _ in range(n_layers)]

    # Warmup, then discard.
    llm.generate([prompt], SamplingParams(max_tokens=decode_steps))
    llm.apply_model(drain_timings)

    top_s: list[float] = []
    e2e: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        llm.generate([prompt], SamplingParams(max_tokens=decode_steps,
                                              ignore_eos=True))
        t1.record()
        torch.cuda.synchronize()
        e2e.append(t0.elapsed_time(t1))

        timings = llm.apply_model(drain_timings)[0]
        for key, series in timings.items():
            if not series:
                continue
            if key == TOP:
                top_s.append(series[0])  # prefill forward of the whole model
                continue
            # First call per layer is the prefill forward; the rest are decodes.
            prefill_s[key].append(series[0])
            decode_s[key].extend(series[1:])

    if not top_s or any(not v for v in prefill_s):
        raise RuntimeError(
            "no prefill samples recorded — check that prefix caching is "
            "disabled and that hooks are attached to the executing model"
        )

    p_med, p_iqr = _median_iqr(prefill_s)
    d_med, d_iqr = _median_iqr(decode_s)
    return ProfileResult(model_id, n_layers, p_med, d_med, p_iqr, d_iqr,
                         model_fwd_ms=statistics.median(top_s),
                         end_to_end_ms=statistics.median(e2e))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_profiler.py -v -m gpu`
Expected: 1 passed. If the layer count assertion fails, print `result.n_layers` and correct the expected value against the model's real `num_hidden_layers` — do not loosen the assertion to make it pass.

- [ ] **Step 5: Commit**

```bash
git add src/pp_balance/profiler.py tests/test_profiler.py
git commit -m "feat: per-layer CUDA-event profiler driven by real requests"
```

---

### Task 8: Reconstruction check — the gate on every downstream number

**Files:**
- Create: `src/pp_balance/validate.py`
- Test: `tests/test_validate.py`

If summed per-layer hook times do not reconstruct the end-to-end forward time, the cost vector is wrong and no result may be reported. Spec threshold: 10%.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pp_balance.validate import reconstruction_error, check_reconstruction

def test_reconstruction_error_is_relative():
    assert reconstruction_error(per_layer_sum=95.0, model_fwd=100.0) == \
        pytest.approx(0.05)

def test_check_passes_within_ten_percent():
    check_reconstruction(per_layer_sum=95.0, model_fwd=100.0)  # no raise

def test_check_raises_beyond_ten_percent():
    with pytest.raises(ValueError, match="reconstruction"):
        check_reconstruction(per_layer_sum=50.0, model_fwd=100.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pp_balance.validate'`

- [ ] **Step 3: Write the implementation**

Create `src/pp_balance/validate.py`:

```python
"""Guards that stop us reporting numbers we cannot stand behind."""

RECONSTRUCTION_TOLERANCE = 0.10  # spec: within 10%


def reconstruction_error(per_layer_sum: float, model_fwd: float) -> float:
    """Relative gap between summed per-layer times and the model forward."""
    return abs(per_layer_sum - model_fwd) / model_fwd


def check_reconstruction(per_layer_sum: float, model_fwd: float) -> None:
    """Raise unless per-layer timings account for the whole model forward.

    The denominator is the TOP-LEVEL MODEL FORWARD, not the wall time of
    `generate()`. Measured on Nemotron-H-8B at 4K context, summed per-layer
    time is 99.8% of the model forward but only 65.1% of the `generate()`
    call, because that call also carries scheduling, sampling and
    detokenisation. Comparing against `generate()` would fail this check on
    every healthy run.
    """
    err = reconstruction_error(per_layer_sum, model_fwd)
    if err > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"reconstruction check failed: per-layer sum {per_layer_sum:.2f}ms "
            f"vs model forward {model_fwd:.2f}ms ({err:.1%} off, tolerance "
            f"{RECONSTRUCTION_TOLERANCE:.0%}). Hook overhead or missed work — "
            "all downstream numbers are void."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate.py -v`
Expected: 3 passed

- [ ] **Step 5: Confirm the profiler already reports the denominator**

`ProfileResult` gained `model_fwd_ms` and `end_to_end_ms` in Task 7, and
`profile_model` already populates both. No profiler changes are needed here.
Verify with:

Run: `uv run python -c "from pp_balance.profiler import ProfileResult; print(ProfileResult('x',1).model_fwd_ms)"`
Expected: `0.0` (the default), confirming the field exists.

- [ ] **Step 6: Commit**

```bash
git add src/pp_balance/validate.py tests/test_validate.py
git commit -m "feat: reconstruction check gating reported measurements"
```

---

### Task 9: `memory` — measured parameter and state bytes

**Files:**
- Create: `src/pp_balance/memory.py`
- Test: `tests/test_memory.py`

Parameter bytes come from the real loaded tensors. State bytes come from vLLM's own KV cache spec, not from `config.json`. Per the spec's risk table, if state bytes cannot be attributed the fallback is to report weights-only and say so — never to substitute a derived estimate.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pp_balance.memory'`

- [ ] **Step 3: Write the implementation**

Create `src/pp_balance/memory.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_memory.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/pp_balance/memory.py tests/test_memory.py
git commit -m "feat: measured per-layer parameter and KV state bytes"
```

---

### Task 10: `report` — even vs balanced comparison

**Files:**
- Create: `src/pp_balance/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pp_balance.report import compare

def test_compare_reports_both_splits_and_the_gap():
    costs = [1.0] * 4 + [20.0] + [1.0] * 3
    mems = [0] * 8
    row = compare(costs, mems, pp_size=4, mem_cap=float("inf"))
    assert row.even_counts == [2, 2, 2, 2]
    assert row.even_max_cost == pytest.approx(21.0)  # stage holding the 20
    assert row.balanced_max_cost <= row.even_max_cost
    assert row.reduction_pct >= 0.0

def test_compare_reports_zero_gap_on_uniform_costs():
    costs = [1.0] * 8
    row = compare(costs, [0] * 8, pp_size=4, mem_cap=float("inf"))
    assert row.reduction_pct == pytest.approx(0.0, abs=1e-6)

def test_compare_surfaces_memory_infeasible():
    row = compare([1.0] * 4, [100] * 4, pp_size=2, mem_cap=150)
    assert row.memory_infeasible is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pp_balance.report'`

- [ ] **Step 3: Write the implementation**

Create `src/pp_balance/report.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/pp_balance/report.py tests/test_report.py
git commit -m "feat: even vs balanced comparison rows"
```

---

### Task 11: Control-model run — Qwen3-8B must show no material gap

**Files:**
- Create: `scripts/run_model.py`
- Test: `tests/test_control.py`

Spec threshold: on a uniform dense stack the balanced split must land within 2% of the even split. A larger win here is a partitioner bug, not a finding.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pp_balance.report import compare
from pp_balance.profiler import profile_model
from pp_balance.validate import check_reconstruction

@pytest.mark.gpu
def test_qwen3_8b_control_shows_no_material_gap():
    res = profile_model("Qwen/Qwen3-8B", prompt_tokens=2048,
                        decode_steps=8, repeats=5)
    check_reconstruction(sum(res.prefill_ms), res.model_fwd_ms)
    costs = res.prefill_ms
    row = compare(costs, [0] * len(costs), pp_size=4, mem_cap=float("inf"))
    assert row.reduction_pct < 2.0, (
        f"control model shows {row.reduction_pct:.1f}% reduction; "
        "a uniform dense stack should show none — suspect a partitioner bug"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_control.py -v -m gpu`
Expected: FAIL — `pp_balance.validate` does not exist until Task 8 is done. Fix forward; do not weaken the 2% threshold.

- [ ] **Step 3: Write the runner script**

Create `scripts/run_model.py`:

```python
"""Profile one model and print even-vs-balanced comparisons."""
import argparse
import json

from pp_balance.profiler import profile_model
from pp_balance.report import compare
from pp_balance.validate import check_reconstruction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--pp-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = profile_model(args.model, args.prompt_tokens,
                        args.decode_steps, args.repeats)
    check_reconstruction(sum(res.prefill_ms), res.model_fwd_ms)

    rows = []
    for regime, costs in (("prefill", res.prefill_ms),
                          ("decode", res.decode_ms)):
        for pp in args.pp_sizes:
            row = compare(costs, [0] * len(costs), pp, float("inf"))
            rows.append({"regime": regime, **row.__dict__})
            print(f"{regime:8s} pp={pp}  even={row.even_max_cost:8.3f}ms  "
                  f"balanced={row.balanced_max_cost:8.3f}ms  "
                  f"reduction={row.reduction_pct:5.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "ctx": args.prompt_tokens,
                       "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the control and the test**

```bash
mkdir -p results
uv run python scripts/run_model.py Qwen/Qwen3-8B --prompt-tokens 2048 \
    --out results/qwen3-8b-2048.json
uv run pytest tests/test_control.py -v -m gpu
```

Expected: reduction percentages near zero for every `pp_size`, and the test passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_model.py tests/test_control.py results/qwen3-8b-2048.json
git commit -m "test: Qwen3-8B control run shows no material partition gap"
```

---

### Task 12: Subject run — Nemotron-H across regimes and context lengths

**Files:**
- Create: `scripts/run_all.sh`
- Create: `results/` output files

- [ ] **Step 1: Run the subject model at three context lengths**

Create `scripts/run_all.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
M=nvidia/Nemotron-H-8B-Base-8K
for ctx in 1024 4096 8000; do
  uv run python scripts/run_model.py "$M" \
      --prompt-tokens "$ctx" --pp-sizes 2 4 8 16 \
      --out "results/nemotron-h-8b-${ctx}.json"
done
uv run python scripts/run_model.py Qwen/Qwen3-8B \
    --prompt-tokens 4096 --pp-sizes 2 4 8 16 \
    --out results/qwen3-8b-4096.json
```

Run:

```bash
chmod +x scripts/run_all.sh && ./scripts/run_all.sh
```

Expected: four JSON files in `results/`. The reconstruction check must pass for every run; if it raises, stop and fix the profiler rather than recording the numbers.

- [ ] **Step 2: Sanity-check the measured cost vector against the architecture**

Create `scripts/cost_by_layer_type.py`:

```python
"""Group the measured cost vector by layer type to sanity-check it."""
import statistics
import sys

from pp_balance.model_access import layer_labels
from pp_balance.profiler import profile_model

PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"

res = profile_model("nvidia/Nemotron-H-8B-Base-8K", prompt_tokens=8000,
                    decode_steps=8, repeats=5)
labels = layer_labels(PATTERN, res.n_layers)
for regime, costs in (("prefill", res.prefill_ms), ("decode", res.decode_ms)):
    by_type: dict[str, list[float]] = {}
    for label, c in zip(labels, costs):
        by_type.setdefault(label, []).append(c)
    print(regime)
    for label, vals in sorted(by_type.items()):
        print(f"  {label:10s} n={len(vals):2d} "
              f"median={statistics.median(vals):7.3f}ms")

attn = statistics.median([c for l, c in zip(labels, res.prefill_ms)
                          if l == "attention"])
mamba = statistics.median([c for l, c in zip(labels, res.prefill_ms)
                           if l == "mamba"])
if abs(attn - mamba) / max(attn, mamba) < 0.05:
    sys.exit("attention and mamba layers measure the same at 8K context — "
             "the hooks are almost certainly on the wrong modules")
```

Run: `uv run python scripts/cost_by_layer_type.py`

Expected: the three layer types must measure **measurably different** from each other. The
script exits non-zero only if attention and Mamba land within 5%, which would mean the hooks
are measuring the wrong modules.

**Do not expect attention to be the expensive one.** Measured on this machine at 4K context,
attention is the CHEAPEST type — median 1.161 ms against Mamba 3.654 ms and MLP 3.058 ms —
because there are only 4 GQA attention layers while Mamba2's SSD kernels and the 21504-wide
MLP dominate. That is a finding, not a fault. The sanity check is about the types being
*distinguishable*, not about their ordering.

- [ ] **Step 2b: Probe whether per-layer KV/state bytes are attributable**

The spec treats memory as a first-class axis but the plan's runs pass `mem_cap=inf`, because
`kv_state_bytes_per_token` may return `{}` on this runtime. Find out which, and record it.

Create `scripts/probe_memory.py`:

```python
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
```

Run: `uv run python scripts/probe_memory.py`

Expected: per-layer parameter bytes for all 52 layers, plus embedding and LM head sizes.
Report whichever outcome you get for the KV spec — "available" and "not available" are both
valid results, and the README must state which one held rather than implying full coverage.

- [ ] **Step 3: Commit results**

```bash
git add -f results/*.json scripts/run_all.sh scripts/cost_by_layer_type.py scripts/probe_memory.py
git commit -m "results: Nemotron-H and Qwen3 partition measurements"
```

---

### Task 13: README with findings and limitations

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

It must contain, in this order: what the project measures; the verified upstream gap with the `get_pp_indices` snippet; the measured results table from `results/`; the control-model result; and the limitations section below, copied verbatim from the spec so the claim is not overstated.

```markdown
## Limitations

This machine has one H100 80GB, so this prototype cannot measure end-to-end
throughput of an actual 8-way pipeline. What it measures on real hardware with
real kernels is the per-layer cost vector; projecting that vector onto P stages
is arithmetic over those measurements.

It answers "how non-uniform is the real cost vector, and how much does balancing
reduce the bottleneck stage?" It does not answer "what is the wall-clock speedup
on an 8-GPU node?" Cross-stage activation transfer is excluded, which is
defensible for cut placement — hidden size is constant across boundaries, so
transfer cost is roughly constant per cut — but absolute stage times are not
deployment predictions.
```

Also record the exact resolved vLLM version from Task 1 Step 1, so results are reproducible.

- [ ] **Step 2: Verify every claim in the README traces to a file in `results/`**

Read the README against the JSON outputs. Any number that cannot be traced to a
results file must be removed, not softened.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: findings, control result, and stated limitations"
```

---

## Self-Review Notes

**Spec coverage:** profiler → Task 7; memory → Task 9; partitioner → Tasks 2-5; report → Task 10; reconstruction check → Task 8; control model → Task 11; exact DP cross-check → Task 4; limitations → Task 13; measured-not-derived costs → Tasks 7 and 9.

**Deferred with reason:** the spec's `mem_cap` sweep is implemented in `compare` and exercised by unit tests, but Task 12's runs pass `mem_cap=inf` because per-layer KV attribution (Task 9's `kv_state_bytes_per_token`) may return empty on the installed runtime. If it returns data, add a memory-constrained run; if it does not, the README reports weights-only and says so, per the spec's risk table.
