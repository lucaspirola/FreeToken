"""exllamav3's own kernels on the Ornith shapes of bench_roofline.py (run with the exllamav3 venv).

    /root/exl3-ref/bin/python tasks/ornith-exl3/perf/bench_roofline_exllamav3.py [M]

LinearEXL3.forward, random trellis, mul1 codebook: bsz 1 (input Hadamard + GEMV, exllamav3's decode
path) and M rows (> 144 rows: reconstruct, with both Hadamards folded in from 1024 rows, + cuBLAS hgemm,
exllamav3's prefill path). Fused projections run as one matrix of the summed width. Per-kernel device
times from torch.profiler; weights rotate over copies larger than L2 in the bsz-1 case.
"""
import sys
from collections import defaultdict

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3

MUL1 = 0x83DCD12D - (1 << 32)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
SHAPES = (("GDN in_proj qkvz", 2048, 12288, 5), ("attn q+gate|k|v", 2048, 9216, 5), ("o_proj / out_proj", 4096, 2048, 5),
          ("shared/expert gate|up", 2048, 1024, 5), ("shared/expert down", 512, 2048, 5), ("lm_head (6 bit)", 2048, 248320, 6))


def lin(k, n, bits):
    tr = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=torch.int32).to(torch.int16).cuda()
    return LinearEXL3(None, k, n, suh=torch.randn(k, dtype=torch.half, device="cuda").sign(),
                      svh=torch.randn(n, dtype=torch.half, device="cuda").sign(), trellis=tr,
                      mul1=torch.tensor(MUL1, dtype=torch.int32, device="cuda"))


def prof(fn, iters):
    fn(0); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for i in range(iters):
            fn(i)
        torch.cuda.synchronize()
    per = defaultdict(lambda: [0.0, 0])
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            us = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            per[e.name][0] += us / iters; per[e.name][1] += 1 / iters
    return per


for label, k, n, bits in SHAPES:
    wb = k * n * bits / 8
    copies = max(1, int(256e6 // wb) + 1)
    lins = [lin(k, n, bits) for _ in range(copies)]
    x1 = torch.randn(1, k, dtype=torch.half, device="cuda")
    per = prof(lambda i: lins[i % copies].forward(x1, {}), 50)
    tot = sum(v[0] for v in per.values())
    print(f"bsz1  {label:24s} K={k:5d} N={n:6d} total {tot:8.1f} us  ({wb / tot / 1e3:6.1f} GB/s of weights)")
    for name, (us, c) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"        {us:8.1f} us {c:4.1f}x {name[:90]}")
    if n > 65536:
        continue
    xm = torch.randn(M, k, dtype=torch.half, device="cuda")
    per = prof(lambda i: lins[0].forward(xm, {}), 5)
    tot = sum(v[0] for v in per.values())
    print(f"M={M} {label:24s} total {tot:8.1f} us  ({2 * M * k * n / tot / 1e6:6.1f} TF/s overall)")
    for name, (us, c) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"        {us:8.1f} us {c:4.1f}x {name[:90]}")
    del lins
    torch.cuda.empty_cache()
