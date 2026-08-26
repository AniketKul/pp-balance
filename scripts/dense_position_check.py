"""If Qwen3-8B's gap is purely the clock artifact, a position-only model explains ~all variance."""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
import statistics, numpy as np, torch
from vllm import LLM, SamplingParams
CTX=4096
llm=LLM(model="Qwen/Qwen3-8B",max_model_len=CTX+128,gpu_memory_utilization=0.85,
        enforce_eager=True,max_num_batched_tokens=CTX+128,enable_prefix_caching=False)
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
    o={k:[s.elapsed_time(e) for s,e in v] for k,v in r.items()}
    for k in r: r[k]=[]
    return o
n=llm.apply_model(install)[0]
tok=llm.get_tokenizer()
mk=lambda t: tok.decode(tok.encode(f"Doc {t}. "+"The history of computing "*3000)[:CTX])
llm.generate([mk("w")],SamplingParams(max_tokens=2,ignore_eos=True)); llm.apply_model(drain)
S=[[] for _ in range(n)]
for rep in range(14):
    llm.generate([mk(rep)],SamplingParams(max_tokens=2,ignore_eos=True))
    d=llm.apply_model(drain)[0]
    for i in range(n):
        if d[i]: S[i].append(d[i][0])
raw=np.array([statistics.median(s) for s in S]); x=np.arange(n)/(n-1)
X=np.column_stack([np.ones(n),x,x**2,x**3])
beta,*_=np.linalg.lstsq(X,np.log(raw),rcond=None)
resid=np.log(raw)-X@beta
r2=1-np.sum(resid**2)/np.sum((np.log(raw)-np.log(raw).mean())**2)
print(f"\nDENSE: position-only fit R^2={r2:.4f}")
print(f"DENSE: raw spread max/min = {raw.max()/raw.min():.3f}")
print(f"DENSE: residual spread after removing position = {np.exp(resid).max()/np.exp(resid).min():.3f}")
print(f"DENSE: => {100*r2:.1f}% of the log-variance in a UNIFORM stack is positional (clock decay)")
