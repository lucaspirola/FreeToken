"""fp16-accumulate GEMM (gemm_f16acc) vs cuBLAS fp16 GEMM (fp32 accumulation) on the Ornith dense
prefill shapes at M rows, plus a tile sweep. P2b A/B (FREETOKEN_EXL3_F16ACC).

    PYTHONPATH=python python tasks/ornith-exl3/perf/bench_f16acc.py [M]
"""
import itertools
import sys

import torch
import triton.testing as tt

from freetoken.kernel.triton.exl3 import gemm_f16acc

M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
SHAPES = (("GDN in_proj", 2048, 12288), ("attn qkv", 2048, 9216), ("o_proj", 4096, 2048),
          ("shared gate|up", 2048, 1024), ("shared down", 512, 2048))
TILES = [dict(BM=bm, BN=bn, BK=bk, GROUP=8, num_warps=nw, num_stages=st)
         for bm, bn, bk, nw, st in itertools.product((64, 128), (128, 256), (32, 64), (4, 8), (3, 4))]
best_cfg = {}
for label, k, n in SHAPES:
    a = torch.randn(M, k, dtype=torch.float16, device="cuda") * 0.5
    b = torch.randn(k, n, dtype=torch.float16, device="cuda") * 0.05
    ref = a.float() @ b.float()
    fl = 2 * M * k * n
    ms = tt.do_bench(lambda: torch.mm(a, b), warmup=5, rep=40)
    print(f"{label:16s} K={k:5d} N={n:6d} cuBLAS fp32acc {ms * 1e3:8.1f} us {fl / ms / 1e9:6.1f} TF/s", flush=True)
    res = []
    for cfg in TILES:
        try:
            y = gemm_f16acc(a, b, **cfg)
            ms = tt.do_bench(lambda: gemm_f16acc(a, b, **cfg), warmup=5, rep=40)
        except Exception as ex:  # shared memory / registers
            continue
        err = ((y.float() - ref).norm() / ref.norm()).item()
        res.append((ms, cfg, err))
    res.sort(key=lambda r: r[0])
    for ms, cfg, err in res[:3]:
        print(f"    f16acc {ms * 1e3:8.1f} us {fl / ms / 1e9:6.1f} TF/s rel err {err:.2e} {cfg}", flush=True)
    print(f"    f16acc default {tt.do_bench(lambda: gemm_f16acc(a, b), warmup=5, rep=40) * 1e3:8.1f} us", flush=True)
