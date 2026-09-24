"""EXL3 MoE prefill (Ornith shapes, 256 experts, top-8, uniform random routing) at one 8K chunk:
sweep the MoE token sub-chunk (PREFILL_CHUNK_TOKENS) and the decoded-slab GEMM tiles.

    PYTHONPATH=python python tasks/ornith-exl3/perf/bench_moe_prefill.py [M] [--quick]

Prints the whole fused_experts_exl3 call (CUDA events, median) per config, then a per-kernel
profile of the baseline and of the best config, plus the peak CUDA memory each sub-chunk needs.
"""
import itertools
import sys
from collections import defaultdict

import torch
import triton.testing as tt

import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes

H, I, E, TOPK, BITS = 2048, 512, 256, 8, 5
M = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8192
QUICK = "--quick" in sys.argv
DEV = torch.device("cuda")


def banks():
    out = []
    for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
        if dtype == torch.int16:
            out.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
        else:
            out.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
    return tuple(out)


B = banks()
x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
run = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)


def setcfg(chunk, gemm):
    fe.PREFILL_CHUNK_TOKENS = chunk
    fe.PREFILL_GEMM = gemm


def timed():
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ms = tt.do_bench(run, warmup=2, rep=5 if QUICK else 12)
    return ms, (torch.cuda.max_memory_allocated() - base) / 2**20


def profile():
    run(); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        run(); torch.cuda.synchronize()
    per = defaultdict(float)
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            per[e.name[:60]] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    return sorted(per.items(), key=lambda kv: -kv[1])[:8]


OLD = dict(block_m=64, block_k=32, num_stages=2, num_warps=4)  # the pre-P1 tiles at 64 rows/expert
setcfg(2048, OLD)
ref = run().float()
ms, mem = timed()
print(f"baseline chunk 2048, old tiles {OLD}: {ms:8.2f} ms  transient {mem:7.0f} MiB", flush=True)
for name, us in profile():
    print(f"    {us / 1e3:8.2f} ms  {name}")
setcfg(8192, None)
y = run().float()
ms, mem = timed()
print(f"shipped defaults (chunk 8192, _prefill_gemm): {ms:8.2f} ms  transient {mem:7.0f} MiB  rel err {((y - ref).norm() / ref.norm()).item():.2e}", flush=True)
for name, us in profile():
    print(f"    {us / 1e3:8.2f} ms  {name}")
res = []
if QUICK:
    for name, us in profile():
        print(f"    {us / 1e3:8.2f} ms  {name}")
    sys.exit(0)
chunks = (2048, 4096, 8192)
grid = itertools.product((32, 64, 128), (32, 64), (2, 3, 4), (4, 8))
for chunk in chunks:
    for bm, bk, st, nw in grid if chunk == chunks[-1] else [(None, 32, 2, 4)]:
        cfg = dict(block_k=bk, num_stages=st, num_warps=nw)
        if bm:
            cfg["block_m"] = bm
        setcfg(chunk, cfg)
        try:
            y = run().float()
            err = ((y - ref).norm() / ref.norm()).item()
            ms, mem = timed()
        except Exception as ex:  # out of shared memory / resources for a tile shape
            print(f"chunk {chunk} {cfg}: {type(ex).__name__}", flush=True)
            continue
        res.append((ms, chunk, cfg, mem, err))
        print(f"chunk {chunk:5d} {str(cfg):70s} {ms:8.2f} ms  transient {mem:7.0f} MiB  rel err vs baseline {err:.2e}", flush=True)
    grid = itertools.product((32, 64, 128), (32, 64), (2, 3, 4), (4, 8))
res.sort(key=lambda r: r[0])
print("\nbest:")
for r in res[:6]:
    print(f"  {r[0]:8.2f} ms chunk {r[1]} {r[2]} transient {r[3]:.0f} MiB err {r[4]:.1e}")
setcfg(res[0][1], res[0][2])
print(f"profile of best ({res[0][1]}, {res[0][2]}):")
for name, us in profile():
    print(f"    {us / 1e3:8.2f} ms  {name}")
