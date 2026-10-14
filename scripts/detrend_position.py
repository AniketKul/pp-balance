"""Separate position (clock-decay) effect from layer-TYPE effect, then re-compare."""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import statistics, numpy as np, torch
from vllm import LLM, SamplingParams
from pp_balance.partitioner import balance, even_split, evaluate_split

PAT="M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
CTX=4096
llm=LLM(model="nvidia/Nemotron-H-8B-Base-8K",max_model_len=CTX+128,gpu_memory_utilization=0.85,
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
    out={k:[s.elapsed_time(e) for s,e in v] for k,v in r.items()}
    for k in r: r[k]=[]
    return out
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
raw=np.array([statistics.median(s) for s in S])
types=list(PAT)
x=np.arange(n)/ (n-1)

# log(cost) = poly3(position) + type effect   (attention '*' is the reference level)
X=np.column_stack([np.ones(n), x, x**2, x**3,
                   [1.0 if t=="M" else 0.0 for t in types],
                   [1.0 if t=="-" else 0.0 for t in types]])
beta,*_=np.linalg.lstsq(X,np.log(raw),rcond=None)
pred=X@beta
r2=1-np.sum((np.log(raw)-pred)**2)/np.sum((np.log(raw)-np.log(raw).mean())**2)
pos_only=np.exp(X[:,:4]@beta[:4]); pos_span=100*(pos_only.max()/pos_only.min()-1)
print(f"\nFIT: R^2={r2:.4f}   position effect spans {pos_span:.1f}% across depth")
print(f"FIT: type multipliers vs attention:  mamba x{np.exp(beta[4]):.3f}   mlp x{np.exp(beta[5]):.3f}")

# Detrended: hold position constant at the mean -> what a per-GPU slice would see
detr=np.exp(beta[0]+beta[1:4]@np.array([x.mean(),(x**2).mean(),(x**3).mean()])
            + X[:,4]*beta[4] + X[:,5]*beta[5])
print(f"\nRAW    per-layer: min={raw.min():.3f} max={raw.max():.3f} ratio={raw.max()/raw.min():.2f}")
print(f"DETREND per-layer: min={detr.min():.3f} max={detr.max():.3f} ratio={detr.max()/detr.min():.2f}")
print(f"\n{'pp':>3} {'RAW red%':>9} {'DETRENDED red%':>15}")
for pp in (2,4,8,16):
    out=[]
    for v in (raw,detr):
        v=list(map(float,v))
        ev=evaluate_split(even_split(n,pp),v,[0]*n).max_stage_cost
        ba=balance(v,[0]*n,pp,float('inf')).max_stage_cost
        out.append(100*(ev-ba)/ev)
    print(f"{pp:>3} {out[0]:8.2f}% {out[1]:14.2f}%")
