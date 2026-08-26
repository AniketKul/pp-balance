"""Is the per-layer ramp caused by GPU clock decay during the forward?"""
import os, subprocess, threading, time, statistics
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams

samples, stop = [], threading.Event()
def poll():
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,temperature.gpu,power.draw",
                                "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=2).stdout.strip()
            sm,tmp,pw = [float(x) for x in o.split(",")]
            samples.append((time.time(), sm, tmp, pw))
        except Exception: pass
        time.sleep(0.02)

CTX=4096
llm = LLM(model="Qwen/Qwen3-8B", max_model_len=CTX+128, gpu_memory_utilization=0.85,
          enforce_eager=True, max_num_batched_tokens=CTX+128, enable_prefix_caching=False)
tok = llm.get_tokenizer()
mk = lambda t: tok.decode(tok.encode(f"Doc {t}. "+"The history of computing "*3000)[:CTX])
llm.generate([mk("w")], SamplingParams(max_tokens=2, ignore_eos=True))

t = threading.Thread(target=poll, daemon=True); t.start()
time.sleep(0.5)
idle = [s for s in samples]
for rep in range(6):
    llm.generate([mk(rep)], SamplingParams(max_tokens=2, ignore_eos=True))
stop.set(); time.sleep(0.1)

if idle:
    print(f"CLOCK: idle-ish before load: sm={statistics.median([s[1] for s in idle]):.0f}MHz "
          f"temp={statistics.median([s[2] for s in idle]):.0f}C pw={statistics.median([s[3] for s in idle]):.0f}W")
busy = samples[len(idle):]
if busy:
    sm=[s[1] for s in busy]; tp=[s[2] for s in busy]; pw=[s[3] for s in busy]
    print(f"CLOCK: during runs  sm: min={min(sm):.0f} max={max(sm):.0f} median={statistics.median(sm):.0f}MHz")
    print(f"CLOCK: during runs temp: min={min(tp):.0f} max={max(tp):.0f}C   power: min={min(pw):.0f} max={max(pw):.0f}W")
    k=max(1,len(sm)//4)
    print(f"CLOCK: first-quarter sm median={statistics.median(sm[:k]):.0f}MHz  "
          f"last-quarter sm median={statistics.median(sm[-k:]):.0f}MHz  n={len(sm)}")
print(f"CLOCK: max clock supported: ", subprocess.run(["nvidia-smi","--query-gpu=clocks.max.sm","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip())
