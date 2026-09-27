#!/usr/bin/env python3
"""Saver prefill miss copy, pinned pool rows -> device buffer, Ornith EXL3 5.0bpw row layout
(6 banks, 1.89 MiB per expert): SM gather (fast_index_copy_multi, blocks_per_bank sweep) vs DMA
(cudaMemcpyBatchAsync, one entry per row per bank), alone and beside a compute-bound GEMM.
    bench_miss.py [N_MISS (default 135)] [POOL_ROWS (default 1024)]"""
import sys, time, statistics, json
import numpy as np, torch
from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.kernel.batch_memcpy import load_batch_memcpy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 135
R = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
E = 256
FEAT = [1310720, 8192, 2048, 655360, 1024, 4096]
dev = torch.device("cuda")
pool = [torch.empty((R, f), dtype=torch.uint8).pin_memory() for f in FEAT]
for p in pool: p.view(-1)[:: 4096].random_(0, 255)
buf = [torch.zeros((E, f), dtype=torch.uint8, device=dev) for f in FEAT]
ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
dst_ptrs, src_ptrs = ptr(buf), ptr(pool)
feat = torch.tensor(FEAT, dtype=torch.int64, device=dev)
rng = np.random.default_rng(0)
src_rows = rng.choice(R, N, replace=False).astype(np.int32)
dst_rows = rng.choice(E, N, replace=False).astype(np.int32)
di = torch.from_numpy(dst_rows).to(dev); si = torch.from_numpy(src_rows).to(dev)
cnt = torch.tensor([N], dtype=torch.int64, device=dev)
bm = load_batch_memcpy()
d_list, s_list, n_list = [], [], []
for b, f in enumerate(FEAT):
    for d, s in zip(dst_rows, src_rows):
        d_list.append(buf[b].data_ptr() + int(d) * f); s_list.append(pool[b].data_ptr() + int(s) * f); n_list.append(f)
D, S, NB = (torch.tensor(x, dtype=torch.int64) for x in (d_list, s_list, n_list))
nbytes = N * sum(FEAT)
copy_stream = torch.cuda.Stream()
a = torch.randn(8192, 8192, dtype=torch.float16, device=dev); bb = torch.randn_like(a)

def sm(bpb):
    return lambda: fast_index_copy_multi_jit(dst_ptrs, src_ptrs, feat, di, si, cnt, blocks_per_bank=bpb)
def dma():
    bm(D, S, NB, torch.cuda.current_stream().cuda_stream)

def check():
    for b in range(len(FEAT)):
        assert torch.equal(buf[b][di.long()].cpu(), pool[b][si.long().cpu()]), b

res = {}
def timeit(name, fn, reps=20, beside=False):
    ts = []
    for _ in range(reps + 3):
        torch.cuda.synchronize()
        if beside:
            for _ in range(3): torch.mm(a, bb)
        with torch.cuda.stream(copy_stream):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
        if beside:
            g0 = torch.cuda.Event(enable_timing=True); g1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts = ts[3:]; med = statistics.median(ts)
    res[name] = {"ms": round(med, 3), "GBps": round(nbytes / med / 1e6, 1)}
    print(f"{name:28s} {med:7.3f} ms  {nbytes / med / 1e6:6.1f} GB/s", flush=True)

for bpb in (8, 16, 32, 64):
    buf[0].zero_(); timeit(f"sm bpb{bpb}", sm(bpb)); check()
buf[0].zero_(); timeit("dma batch", dma); check()
# beside compute: GEMM stream busy, copy on the side; also time the GEMMs
for name, fn in (("sm bpb8", sm(8)), ("dma batch", dma)):
    gs = []
    for _ in range(10):
        torch.cuda.synchronize()
        g0 = torch.cuda.Event(enable_timing=True); g1 = torch.cuda.Event(enable_timing=True)
        g0.record()
        with torch.cuda.stream(copy_stream):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
        for _ in range(3): torch.mm(a, bb)
        g1.record(); torch.cuda.synchronize(); gs.append((g0.elapsed_time(g1), s.elapsed_time(e)))
    gm = statistics.median(x[0] for x in gs); cm = statistics.median(x[1] for x in gs)
    res[f"beside {name}"] = {"gemm3_ms": round(gm, 2), "copy_ms": round(cm, 3)}
    print(f"beside {name:20s} 3 GEMMs {gm:7.2f} ms, copy {cm:7.3f} ms", flush=True)
torch.cuda.synchronize(); g0 = torch.cuda.Event(enable_timing=True); g1 = torch.cuda.Event(enable_timing=True)
g0.record(); [torch.mm(a, bb) for _ in range(3)]; g1.record(); torch.cuda.synchronize()
print(f"3 GEMMs alone {g0.elapsed_time(g1):.2f} ms")
res["gemm3_alone_ms"] = round(g0.elapsed_time(g1), 2)
print(json.dumps({"n_miss": N, "bytes": nbytes, **res}))
