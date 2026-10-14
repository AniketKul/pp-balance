import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import statistics, torch
from vllm import LLM, SamplingParams
MODEL, CTX = sys.argv[1], 4096
llm = LLM(model=MODEL, max_model_len=CTX+128, gpu_memory_utilization=0.85,
          enforce_eager=True, max_num_batched_tokens=CTX+128, enable_prefix_caching=False)
def install(model):
    rec={}; model._r=rec
    for i,l in enumerate(model.model.layers):
        rec[i]=[]
        def pre(m,a):
            e=torch.cuda.Event(enable_timing=True); e.record(); m._s=e
        def post(m,a,o,k=i):
            e=torch.cuda.Event(enable_timing=True); e.record(); rec[k].append((m._s,e)); return o
        l.register_forward_pre_hook(pre); l.register_forward_hook(post)
    return len(model.model.layers)
def drain(model):
    torch.cuda.synchronize(); r=model._r
    out={k:[s.elapsed_time(e) for s,e in v] for k,v in r.items()}
    for k in r: r[k]=[]
    return out
n=llm.apply_model(install)[0]
tok=llm.get_tokenizer()
mk=lambda t: tok.decode(tok.encode(f"Doc {t}. "+"The history of computing "*3000)[:CTX])
llm.generate([mk("w")],SamplingParams(max_tokens=2,ignore_eos=True)); llm.apply_model(drain)
S=[[] for _ in range(n)]
for rep in range(12):
    llm.generate([mk(rep)],SamplingParams(max_tokens=2,ignore_eos=True))
    d=llm.apply_model(drain)[0]
    for i in range(n):
        if d[i]: S[i].append(d[i][0])
med=[statistics.median(s) for s in S]
mn=statistics.median(med)
print(f"\nSHAPE {MODEL}: median-of-layers={mn:.3f}ms  min={min(med):.3f} max={max(med):.3f} ratio={max(med)/min(med):.2f}")
for i,m in enumerate(med):
    bar="#"*int(round(40*m/max(med)))
    print(f"  L{i:02d} {m:7.3f}ms {(m/mn-1)*100:+6.1f}%  {bar}")
