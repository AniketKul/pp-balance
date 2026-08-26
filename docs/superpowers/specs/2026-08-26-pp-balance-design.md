# Measured Pipeline-Parallel Partition Balancing

**Date:** 2026-08-26
**Status:** Approved design, not yet implemented
**Repo:** `~/aniket/pp-balance`

## Motivation

vLLM assigns transformer layers to pipeline-parallel stages by count, not by cost.
From upstream `vllm/distributed/utils.py` on `main` (verified 2026-08-26):

```python
layers_per_partition = num_hidden_layers // pp_size
partitions = [layers_per_partition for _ in range(pp_size)]
if remaining_layers := num_hidden_layers % pp_size:
    for i in range(2, remaining_layers + 2):
        partitions[-i] += 1
```

`get_pp_indices` is the only pipeline-parallel partitioning function in that file, and
no cost-, latency-, or memory-aware partitioning exists in it. Its docstring states two
objectives — "we are attempting to balance compute" and "we are attempting to reduce
maximum memory consumption across partitions" — and optimizes neither. It counts layers.
The only escape hatch is `VLLM_PP_LAYER_PARTITION`, an environment variable in which the
operator types a split by hand.

The uniform-cost assumption behind counting is false for hybrid architectures.
`nvidia/Nemotron-H-8B-Base-8K` has `hybrid_override_pattern` of length 52 composed of
**24 Mamba2 + 24 MLP + 4 attention** layers — three cost classes in one stack. Applying
vLLM's current algorithm at `pp_size=8` yields:

```
[6, 6, 6, 7, 7, 7, 7, 6]
  s0: 6L(M3,*0,-3)   s1: 6L(M3,*1,-2)   s2: 6L(M3,*0,-3)   s3: 7L(M3,*1,-3)
  s4: 7L(M3,*1,-3)   s5: 7L(M3,*0,-4)   s6: 7L(M3,*1,-3)   s7: 6L(M3,*0,-3)
```

Four of eight stages hold zero attention layers. Attention KV state is therefore
concentrated on stages 1, 3, 4 and 6 while stages 0, 2, 5 and 7 carry only Mamba
conv/SSM state. Because pipeline throughput is set by the slowest stage, and because
KV capacity is limited by the most constrained rank, a count-based split is structurally
unable to express the right answer here.

## Claim under test

> On real hybrid models, the measured per-layer cost vector is non-uniform enough that
> vLLM's count-based partition leaves the bottleneck stage materially more expensive —
> in time and in memory — than a cost-aware partition would.

**What would falsify it:** the measured cost vector comes out near-uniform, or the
balanced partition reduces the bottleneck stage by an amount within measurement noise.
Both are reportable results. This spec is designed to be able to return "no material
gap" without the prototype being a failure.

## Non-goals

- No analytic or config-derived cost model. Costs are measured on real checkpoints.
- No end-to-end multi-GPU pipeline benchmark (see Limitations).
- No vLLM patch or upstream PR. This prototype only decides whether one is warranted.
- No heterogeneous-GPU support (differing per-rank speed or capacity).
- No training-side pipeline concerns (bubbles, microbatch scheduling, 1F1B).

## Architecture

Four components with clean boundaries, each independently testable.

### 1. `profiler` — produces a measured cost vector

Loads a real checkpoint through **vLLM's own runtime** so that measured kernels are the
kernels a deployment would actually execute (Mamba2 SSD, FlashAttention, fused MLP).
HF `transformers` is explicitly rejected as the harness: it runs different kernels, so its
timings would be a proxy for the system under study rather than a measurement of it.

- Attaches `torch.cuda.Event` pairs to the real decoder-layer modules via forward hooks.
- Drives **real requests through the engine** — real prompts, real generated tokens —
  rather than synthesised tensors.
- Separates regimes: prefill costs from the first forward of a long prompt; decode costs
  from subsequent single-token steps.
- Sweeps context length at 1k / 4k / 8k (the model's 8K limit).
- Repeats after warmup; reports median per layer plus IQR so measurement noise is
  visible rather than hidden.

**Output:** `costs[regime][ctx_len] -> list[float]` (milliseconds per layer).

### 2. `memory` — produces a measured byte vector

- Per-layer parameter bytes read from the loaded modules' real tensors.
- Per-layer runtime state bytes read from vLLM's actual allocations: attention KV for the
  4 attention layers, conv + SSM state for the 24 Mamba layers. Not computed from
  `config.json`.
- Embedding and LM head recorded as explicit prologue/epilogue entries. At
  `vocab=131072 x hidden=4096` in bf16 these are ~1.07 GB each — large enough to move cut
  points on their own, and the reason upstream's docstring nudges the remainder toward
  middle ranks by hand.

**Output:** `bytes -> list[int]`, aligned index-wise with the cost vector.

### 3. `partitioner` — binary search on the bottleneck

The cost vector is extended to `n + 2` entries: `[embed, layer_0 .. layer_{n-1}, lm_head]`.
Because partitions must be contiguous, the prologue lands in the first stage and the
epilogue in the last stage automatically. No special-casing required.

```python
def feasible(x, costs, mems, pp_size, mem_cap):
    stages, t, m = 1, 0.0, 0
    for c, b in zip(costs, mems):
        if c > x or b > mem_cap:
            return False                 # single element cannot fit anywhere
        if t + c > x or m + b > mem_cap:
            stages += 1
            if stages > pp_size:
                return False
            t, m = c, b
        else:
            t, m = t + c, m + b
    return True
```

Binary search the smallest `x` for which `feasible(x)` holds, over
`lo = max(costs)`, `hi = sum(costs)`.

**Monotonicity.** `feasible` is monotone in `x` given non-negative costs. The greedy
places each cut as late as possible; raising `x` weakens only the time condition, so by
induction the k-th cut under the larger budget falls no earlier than under the smaller
one. The argument needs that a suffix of a feasible segment is feasible, which holds
because costs and bytes are non-negative. The memory condition is fixed with respect to
`x` and so cannot break monotonicity.

**Edge cases, handled explicitly:**
- A single layer exceeding `mem_cap` makes *every* `x` infeasible. Detect and report
  "memory-infeasible at this `pp_size`" rather than silently returning `hi`.
- `feasible(hi)` false for the same reason — check before searching.
- Greedy may produce fewer than `pp_size` stages; that is feasible and fine.

`mem_cap` is a parameter, swept across several values including unbounded, so the
compute-only effect can be isolated from the memory-constrained effect.

### 4. `report` — comparison output

Per model, per regime, per context length: the measured cost vector, vLLM's even split
vs the balanced split, the resulting max-stage time and max-stage bytes for each, and the
reduction in the bottleneck. Noise bounds carried through from the profiler.

## Validation

Three checks. This is where a prototype of this shape usually deceives itself.

1. **Reconstruction check (gates everything else).** Summed per-layer hook times must
   reconstruct the independently measured end-to-end forward time to within **10%**. If
   they do not, the hooks are either adding overhead or missing work, and every
   downstream number is void. Run first; abort on failure.

2. **Control model.** `Qwen3-8B` — 36 uniform dense layers. The balanced split must come
   out within **2%** of the even split on max-stage time. A larger reported win on the
   control is a bug in the partitioner, not a finding.

3. **Exact cross-check.** For `pp_size <= 4`, brute-force the true optimum over the
   52-element vector and assert the binary-search result matches. Cheap at this size, and
   it catches monotonicity errors that binary search would otherwise hide by converging
   confidently to a wrong answer.

## Limitations

Stated here and repeated in the README.

This machine has **one H100 80GB**. The prototype therefore cannot measure end-to-end
throughput of an actual 8-way pipeline. What it measures on real hardware with real
kernels is the per-layer cost vector; projecting that vector onto P stages is arithmetic
over those measurements.

The question this answers is *"how non-uniform is the real cost vector, and how much does
balancing reduce the bottleneck stage?"* It does **not** answer *"what is the wall-clock
speedup on an 8-GPU node?"* Cross-stage activation transfer is excluded, which is
defensible for cut *placement* — hidden size is constant across boundaries, so transfer
cost is roughly constant per cut — but it means absolute stage times are not deployment
predictions.

## Why binary search rather than DP

At `n=52` and `pp_size <= 16` an exact DP is also cheap, so binary search does not win on
asymptotics. It is chosen for two reasons that survive scrutiny:

- **Multiple simultaneous constraints.** Time and per-rank memory are checked in one
  greedy sweep with two accumulators. A DP would need a second state dimension.
- **Extensibility.** Heterogeneous per-rank speed and capacity slot into the same
  feasibility predicate without restructuring.

The exact cross-check in Validation exists precisely because binary search is the less
obviously-correct choice of the two.

## Repo layout

```
pp-balance/
  README.md               # findings + limitations, written last
  pyproject.toml          # uv-managed
  src/pp_balance/
    profiler.py           # component 1
    memory.py             # component 2
    partitioner.py        # component 3
    report.py             # component 4
  tests/
  docs/superpowers/specs/
  results/                # gitignored
```

Toolchain: `uv venv`, with a pinned released vLLM whose architecture support matches
upstream `main`. vLLM source questions are resolved against
`github.com/vllm-project/vllm`, not any local checkout.

## Risks

| Risk | Mitigation |
|---|---|
| Pinned vLLM wheel cannot load `NemotronHForCausalLM` | Verified supported on upstream `main`; confirm at env setup before any measurement work |
| Forward hooks perturb the timings they measure | Reconstruction check is designed to catch exactly this and gates all results |
| Decode-step timings dominated by launch overhead, not layer work | Report IQR; if noise swamps signal at decode, say so rather than reporting a fragile number |
| Per-layer state bytes not cleanly attributable inside vLLM's allocator | Fall back to reporting weights-only memory and state it, rather than substituting a config-derived estimate |
