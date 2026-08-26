# pp-balance

Does vLLM's count-based pipeline-parallel layer split leave the bottleneck stage more
expensive than a cost-aware split would? Measured on real checkpoints, through vLLM's own
runtime, on an H100.

**Short answer: yes, but only for genuinely heterogeneous models and only at `pp_size >= 8`.
And most of the effect you would naively measure on a single GPU is a hardware artifact that
has to be removed first.**

## The gap in vLLM

Upstream `vllm/distributed/utils.py::get_pp_indices` assigns layers to pipeline stages by
count:

```python
layers_per_partition = num_hidden_layers // pp_size
partitions = [layers_per_partition for _ in range(pp_size)]
if remaining_layers := num_hidden_layers % pp_size:
    for i in range(2, remaining_layers + 2):
        partitions[-i] += 1
```

It is the only pipeline-parallel partitioning function in that file, and no cost-, latency-
or memory-aware partitioning exists there. Its docstring states two goals — "we are
attempting to balance compute" and "we are attempting to reduce maximum memory consumption
across partitions" — and optimises neither. The only override is `VLLM_PP_LAYER_PARTITION`,
an environment variable the operator fills in by hand.

Verified against upstream `main` on 2026-08-26, and confirmed byte-identical in the pinned
`vllm==0.19.1` used here, so the baseline is not a strawman.

## Method

Pipeline throughput is set by the slowest stage, so the objective is to minimise the maximum
contiguous stage cost. That is solved by binary searching the bottleneck cost with a greedy
feasibility check — the "binary search on the answer" pattern — using two accumulators so
compute and per-rank memory are checked in one sweep.

Per-layer costs are **measured**, never derived from `config.json`: CUDA-event forward hooks
on the real decoder-layer modules, inside vLLM's own runtime, driven by real requests through
the engine.

## Headline result — Nemotron-H-8B (hybrid)

`nvidia/Nemotron-H-8B-Base-8K`, 52 layers, verified on the loaded model as 24
`NemotronHMambaDecoderLayer` + 24 `NemotronHMLPDecoderLayer` + 4 `NemotronHAttentionDecoderLayer`.

Reduction in max-stage cost, balanced vs vLLM's split, prefill:

| ctx | pp=2 | pp=4 | pp=8 | pp=16 |
|---|---|---|---|---|
| 1024 | 0.0% | 1.6% | 0.7% | 4.7% |
| 4096 | 0.0% | 0.0% | 7.8% | 14.1% |
| 8000 | 0.0% | 0.0% | 6.6% | 12.6% |

Decode is ~0% everywhere except `pp=16` (6.3–7.5%).

**Zero at `pp <= 4` is correct and structural.** With 4 attention layers, any `pp_size <= 4`
puts exactly one on every stage, so the count-based split is already optimal. The gap only
opens once `pp_size` exceeds the count of the rare layer type and stages start differing in
composition.

Measured per-layer medians:

| type | n | prefill @4096 | prefill @8000 | decode @8000 |
|---|---|---|---|---|
| mamba | 24 | 3.654 ms | 7.043 ms | 0.745 ms |
| mlp | 24 | 3.058 ms | 6.048 ms | 0.312 ms |
| attention | 4 | 1.161 ms | 2.836 ms | 0.466 ms |

Attention is the **cheapest** layer type, not the most expensive — there are only 4 GQA
layers, while Mamba2's SSD kernels and the 21504-wide MLP dominate. Note the ordering changes
between regimes: at prefill mamba > mlp > attention, at decode mamba > attention > mlp.
vLLM applies one static partition to both.

## The artifact you have to remove first

Measuring all layers sequentially on one GPU produces a **smooth monotonic cost ramp with
depth** that has nothing to do with the layers. On `Qwen/Qwen3-8B` — 36 identical dense
layers — measured cost rises steadily from 3.874 ms at L02 to 4.357 ms at L35, a 12% spread.

The cause is GPU clock decay under the power cap:

```
idle/boost      1755 MHz (max)   37 C   185 W
first quarter   1500 MHz
last quarter    1245 MHz         44 C   331 W   (H100 PCIe, 350 W TDP)
```

Temperature barely moves, so this is power-limited boost decay, not thermal throttling. A 17%
clock drop explains the 12% cost ramp.

**This does not transfer to a real pipeline deployment.** In an 8-way pipeline each GPU
executes only its own slice and sees its own boost curve; no GPU experiences "late in the
forward". A balancer fed raw single-GPU measurements partly optimises against this artifact.

Fitting `log(cost) ~ poly3(position) + layer_type` separates the two — the layer types
alternate regularly, so position and type are not confounded:

| model | fit | position span | conclusion |
|---|---|---|---|
| Nemotron-H | R²=0.9987 | 6.1% | type dominates: mamba ×3.167, mlp ×2.631 vs attention |
| Qwen3-8B | R²=0.9754 (position only) | 13.1% raw, 4.1% residual | **97.5% of log-variance is positional** |

Nemotron-H's result survives detrending unchanged:

| pp | raw | detrended | held-out |
|---|---|---|---|
| 2 | 0.00% | 0.00% | — |
| 4 | 0.00% | 0.00% | — |
| 8 | 7.86% | 8.15% | 7.51% |
| 16 | 14.36% | 14.07% | 14.05% |

The dense control's apparent 4.4%/10.4% is entirely artifact. **The control worked exactly as
designed** — a uniform stack has no architectural imbalance — but only once the hardware
effect is accounted for. Anyone repeating this measurement without detrending will conclude
that dense models benefit from cost-aware partitioning. They do not.

## Validation

Three gates, all passed:

1. **Reconstruction.** Summed per-layer hook times vs the top-level model forward: **0.08%–0.89%**
   across all runs, against a 10% tolerance. Note the denominator must be the model forward,
   not `generate()` wall time — summed layer time is 99.8% of the model forward but only 65.1%
   of the `generate()` call, which also carries scheduling, sampling and detokenisation.
2. **Control model.** Qwen3-8B shows 0.0% at `pp=4`. Its non-zero values at `pp>=8` are the
   positional artifact above, quantified rather than explained away.
3. **Exact cross-check.** The binary-search partitioner was checked against an independently
   written exhaustive brute force over **3000 randomised cases, 0 mismatches**, including 238
   correctly-detected memory-infeasible cases.

Held-out validation (fit the partition on one measurement, score it on an independent one)
confirms the effect is not in-sample overfitting: A/B per-layer agreement is 1.1%, and
held-out reduction matches in-sample.

## Memory

Measured per-layer parameter bytes range 83.9 MB to 352.3 MB (ratio 4.20); embeddings and LM
head are 1.074 GB each. **Per-layer KV/SSM byte attribution was not reachable** through
`apply_model` on this vLLM version, so all reported runs use `mem_cap=inf` and the memory
axis is **weights-only**. This is stated rather than filled in with a config-derived estimate.

The memory question therefore remains open, and may matter more than the compute one: at
`pp_size=8` the count-based split gives four stages an attention layer and four none, so
attention KV concentrates on half the ranks while vLLM sizes KV blocks by the most
constrained rank.

## Limitations

This machine has **one H100 80GB**, so this measures the per-layer cost vector on real
hardware with real kernels and computes what it implies for the bottleneck stage. Projecting
onto P stages is arithmetic over those measurements.

It answers *"how non-uniform is the real cost vector, and how much does balancing reduce the
bottleneck stage?"* It does **not** answer *"what is the wall-clock speedup on an 8-GPU
node?"* Cross-stage activation transfer is excluded — defensible for cut *placement*, since
hidden size is constant across boundaries and transfer cost is roughly constant per cut, but
it means absolute stage times are not deployment predictions.

## Reproducing

```bash
uv venv --python 3.10 && uv sync
uv run python scripts/probe_env.py          # asserts the vLLM build supports both models
uv run pytest tests/ -q                     # 33 tests
./scripts/run_all.sh                        # the measurement sweep
uv run python scripts/detrend_position.py   # separates clock artifact from layer type
uv run python scripts/clock_probe.py        # demonstrates the clock decay directly
uv run python scripts/heldout_validation.py nvidia/Nemotron-H-8B-Base-8K 4096
```

**`vllm` is pinned to `0.19.1`** and `uv.lock` is committed. Newer vLLM (>=0.20.0) pins
`torch>=2.11`, whose default wheels bundle CUDA 13 runtime libraries; the driver here
(570.211.01) supports at most CUDA 12.8 and cannot run them. `vllm 0.19.1` pins
`torch==2.10.0+cu128`. Its `get_pp_indices` is byte-identical to upstream `main`.

Two environment settings are required, both verified:
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` — otherwise `apply_model` cannot pass a function across
  the engine's process boundary.
- `enable_prefix_caching=False` — otherwise the warmup call populates the cache and every
  measured repeat skips prefill, recording zero prefill samples.

## Layout

```
src/pp_balance/
  partitioner.py    even_split (vLLM baseline), feasible, balance, evaluate_split
  profiler.py       CUDA-event hooks, real requests, per-layer medians
  model_access.py   locate decoder layers / embeddings / head; label layer types
  memory.py         measured parameter and KV state bytes
  report.py         even vs balanced comparison rows
  validate.py       reconstruction gate
scripts/            env probe, measurement runs, artifact analysis
results/            measured JSON + analysis-summary.json
docs/superpowers/   design spec and implementation plan
```

## Status

Research prototype. The finding is real but narrow: it applies to models with genuinely
heterogeneous layer types, at `pp_size >= 8`. Whether it justifies an upstream vLLM change
depends on the memory axis, which this prototype could not measure, and on validation with an
actual multi-GPU pipeline, which this hardware cannot provide.
