"""D2: decode GEMV (exl3_gemv, production launcher path incl. in-kernel split-K reduce and the
prologue rotation) on the Ornith shapes, weights rotated over copies larger than L2, timed as CUDA-graph replays (device time), sweeping the
split count and GEMV_BANDS. Efficiency = weight bytes / time / 818 GB/s (box DRAM copy bound).

    PYTHONPATH=python python tasks/ornith-exl3/perf/bench_gemv_d2.py
"""
import inspect

import torch
import triton.testing as tt

from freetoken.kernel.triton import exl3 as K
from freetoken.layers.quantization.linear.exl3 import PREROT_MAX_NBLOCKS, pick_split_k

HAS_BANDS = "bands" in inspect.signature(K.exl3_gemv).parameters  # the pre-D2 kernel has no bands

BOUND = 818e9
dev = torch.device("cuda")
SHAPES = (("GDN in_proj", 2048, (12288,), 1, 0), ("attn q+gate|k|v", 2048, (8192, 512, 512), 1, 0),
          ("o_proj/out_proj", 4096, (2048,), 1, 0), ("shared gate|up", 2048, (512, 512), 1, 0),
          ("shared down", 512, (2048,), 1, 0), ("MoE gate|up x8", 2048, (512, 512), 8, 256),
          ("MoE down x8", 512, (2048,), 8, 256))
for label, k, sizes, rows, E in SHAPES:
    parts = K.Exl3Parts.build(k, sizes, 5, "mul1", "cuda")
    words = sum((k // 16) * (s // 16) * 40 for s in sizes)
    nexp = max(E, 1)
    wb_one = words * 4 * nexp
    copies = max(1, int(256e6 // wb_one) + 1)
    trs = [torch.randint(-(1 << 31), (1 << 31) - 1, (nexp, words), dtype=torch.int32, device=dev) for _ in range(copies)]
    svh = torch.ones(nexp, parts.n, dtype=torch.float16, device=dev)
    suh = torch.ones(nexp, parts.num_parts, k, dtype=torch.float16, device=dev)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=dev)
    ids = torch.randperm(E, device=dev)[:rows].to(torch.int32) if E else None
    out = torch.empty(rows, parts.n, dtype=torch.float32, device=dev)
    wb = (rows if E else 1) * words * 4
    pick = pick_split_k(rows, parts.n // 128, k, dev)

    # production routing: the prologue rotation up to PREROT_MAX_NBLOCKS output blocks (and for the
    # MoE gate|up), a separate had_rows launch otherwise (GDN in_proj, attn qkv; MoE down reads the
    # rotated activation that splitk_silu_had writes, so it is timed on a pre-rotated input)
    pre = parts.n // 128 <= PREROT_MAX_NBLOCKS and label != "MoE down x8"
    xh_moe = torch.randn(1, rows, k, dtype=torch.float16, device=dev)

    def one(c, split, bands):
        tr = trs[c] if E else trs[c][0]
        kw = dict(bands=bands) if HAS_BANDS else {}
        common = dict(out=out, experts=ids, tr_expert_stride=words if E else 0, svh_expert_stride=parts.n if E else 0,
                      split_k=split, **kw)
        if pre:
            K.exl3_gemv(None, tr, svh if E else svh[0], parts, x=x, suh=suh if E else suh[0],
                        suh_expert_stride=parts.num_parts * k if E else 0, src_div=rows if E else 1, **common)
        elif E:
            K.exl3_gemv(xh_moe, tr, svh, parts, **common)
        else:
            K.exl3_gemv(K.had_rows(x, suh[0], parts), tr, svh[0], parts, **common)

    def graph_us(split, bands):
        # device time only: one CUDA graph replays the GEMV over every weight copy (> L2) in turn
        for c in range(copies):
            one(c, split, bands)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for c in range(copies):
                one(c, split, bands)
        return tt.do_bench(g.replay, warmup=5, rep=100) * 1e3 / copies

    ref = None
    res = []
    for split in sorted({max(1, pick // 2), pick, pick * 2, pick * 4, pick * 8}):
        if k % (split * 16) or k // split < 32:
            continue
        for bands in ((1, 2, 4) if HAS_BANDS else (1,)):
            if (k // split) % (16 * bands):
                continue
            us = graph_us(split, bands)
            res.append((us, split, bands))
    res.sort()
    base = next(r for r in res if r[1] == pick and r[2] == 1)
    print(f"{label:18s} {wb / 1e6:6.2f} MB  launcher split {pick} bands 1: {base[0]:6.1f} us eff {wb / base[0] / 1e3 / (BOUND / 1e9):.2f} | best "
          + " | ".join(f"{u:6.1f} us eff {wb / u / 1e3 / (BOUND / 1e9):.2f} s{s} b{b}" for u, s, b in res[:3]), flush=True)
