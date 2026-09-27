"""Sustained fp16 GEMM for DUR seconds: per-call CUDA-event times vs the wall-clock total.
A median at full speed with a slower total means the GPU is taken away between/within calls
(time-slicing with another context); a slower median means the SMs themselves run slower."""
import os, sys, time, json, statistics, subprocess
import torch
DUR = float(os.environ.get("DUR", "40")); n = 8192
a = torch.randn(n, n, device="cuda", dtype=torch.float16); b = torch.randn_like(a); c = torch.empty_like(a)
for _ in range(10): torch.mm(a, b, out=c)
torch.cuda.synchronize()
ev = []; t0 = time.perf_counter()
while time.perf_counter() - t0 < DUR:
    batch = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(50)]
    for s, e in batch:
        s.record(); torch.mm(a, b, out=c); e.record()
    torch.cuda.synchronize(); ev += [s.elapsed_time(e) for s, e in batch]
wall = time.perf_counter() - t0
fl = 2 * n ** 3
med = statistics.median(ev); q = sorted(ev)
out = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "calls": len(ev), "median_ms": round(med, 3),
       "p99_ms": round(q[int(.99 * len(q))], 3), "max_ms": round(q[-1], 3),
       "tflops_median": round(fl / med / 1e9, 1), "tflops_sum_events": round(fl * len(ev) / sum(ev) / 1e9, 1),
       "tflops_wall": round(fl * len(ev) / wall / 1e12, 1), "load": open("/proc/loadavg").read().split()[:3]}
try:
    out["windows_gpu_procs"] = [l.strip() for l in subprocess.run(["/mnt/c/Windows/System32/nvidia-smi.exe", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout.splitlines()]
    out["windows_smi_tail"] = [l for l in subprocess.run(["/mnt/c/Windows/System32/nvidia-smi.exe"], capture_output=True, text=True, timeout=15).stdout.splitlines() if ".exe" in l]
except Exception as e:
    out["windows_gpu_procs"] = repr(e)
print(json.dumps(out))
