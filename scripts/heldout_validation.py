"""Decisive test: is the reduction real, or the balancer overfitting noise?

Fit the partition on measurement A, score it on INDEPENDENT measurement B.
"""
import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import statistics, torch
from vllm import LLM, SamplingParams
from pp_balance.partitioner import balance, even_split, evaluate_split

MODEL, CTX, REPS = sys.argv[1], int(sys.argv[2]), 16
llm = LLM(model=MODEL, max_model_len=CTX + 128, gpu_memory_utilization=0.85,
          enforce_eager=True, max_num_batched_tokens=CTX + 128,
          enable_prefix_caching=False)

def install(model):
    rec = {}; model._r = rec
    def attach(mod, key):
        rec[key] = []
        def pre(m, a):
            e = torch.cuda.Event(enable_timing=True); e.record(); m._s = e
        def post(m, a, o, key=key):
            e = torch.cuda.Event(enable_timing=True); e.record()
            rec[key].append((m._s, e)); return o
        mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)
    for i, l in enumerate(model.model.layers): attach(l, i)
    return len(model.model.layers)

def drain(model):
    torch.cuda.synchronize(); r = model._r
    out = {k: [s.elapsed_time(e) for s, e in v] for k, v in r.items()}
    for k in r: r[k] = []
    return out

n = llm.apply_model(install)[0]
tok = llm.get_tokenizer()
def mk(t): return tok.decode(tok.encode(f"Doc {t}. " + "The history of computing " * 3000)[:CTX])

llm.generate([mk("w")], SamplingParams(max_tokens=2, ignore_eos=True)); llm.apply_model(drain)
samples = [[] for _ in range(n)]
for rep in range(REPS):
    llm.generate([mk(rep)], SamplingParams(max_tokens=2, ignore_eos=True))
    d = llm.apply_model(drain)[0]
    for i in range(n):
        if d[i]: samples[i].append(d[i][0])

A = [statistics.median(s[0::2]) for s in samples]   # independent halves
B = [statistics.median(s[1::2]) for s in samples]
print(f"\n=== {MODEL}  ctx={CTX}  n_layers={n}  reps={REPS} ===")
print(f"A/B agreement: max per-layer |A-B|/B = "
      f"{max(abs(a-b)/b for a, b in zip(A, B)):.1%}")
print(f"{'pp':>3} {'in-sample':>10} {'HELD-OUT':>10}   verdict")
for pp in (2, 4, 8, 16):
    ins = balance(A, [0]*n, pp, float('inf'))
    ins_red = 100*(evaluate_split(even_split(n, pp), A, [0]*n).max_stage_cost
                   - ins.max_stage_cost) / evaluate_split(even_split(n, pp), A, [0]*n).max_stage_cost
    ev_B = evaluate_split(even_split(n, pp), B, [0]*n).max_stage_cost
    bal_B = evaluate_split(ins.partition, B, [0]*n).max_stage_cost
    ho_red = 100*(ev_B - bal_B)/ev_B
    verdict = "REAL" if ho_red > 2.0 else ("noise" if ins_red > 2.0 else "-")
    print(f"{pp:>3} {ins_red:9.2f}% {ho_red:9.2f}%   {verdict}")
