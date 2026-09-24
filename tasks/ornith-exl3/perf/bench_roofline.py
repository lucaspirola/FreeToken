"""Roofline table of the Ornith EXL3 5.0 bpw hot kernels on this GPU (production launchers, random trellis).

    PYTHONPATH=python python tasks/ornith-exl3/perf/bench_roofline.py [--json out.jsonl]

Every op is the production call (``exl3_forward`` for dense projections, ``fused_experts_exl3`` for the
routed experts) timed kernel by kernel under torch.profiler. Each kernel gets its compulsory bytes and
FLOPs from the shapes (weights at their packed size, activations and outputs once, intermediate
scratch the kernel itself writes/reads), and its bound time = max(bytes / DRAM, FLOPs / tensor peak),
with both hardware numbers MEASURED here first (a 1 GiB device copy, an 8192^3 fp16 cuBLAS GEMM with
fp32 accumulation). "eff" = bound time / measured time.

Decode weights rotate over enough copies (> 4x the 64 MB L2) that every call reads them from DRAM,
as in a real step where 40 layers' weights pass through the cache once per token.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
import triton

from freetoken.kernel.triton import exl3 as K
from freetoken.layers.quantization.linear.exl3 import exl3_forward
from freetoken.models.exl3_banks import exl3_bank_shapes
from freetoken.moe.fused_exl3 import PREFILL_CHUNK_TOKENS, fused_experts_exl3

H, I, E, TOPK, BITS, HEAD_BITS = 2048, 512, 256, 8, 5, 6
DENSE = (  # label, K, parts, layers per token
    ("GDN in_proj qkvz", 2048, (2048, 2048, 4096, 4096), 30),
    ("attn q+gate|k|v", 2048, (8192, 512, 512), 10),
    ("o_proj / out_proj", 4096, (2048,), 40),
    ("shared gate|up", 2048, (512, 512), 40),
    ("shared down", 512, (2048,), 40),
)
LM_HEAD = ("lm_head (6 bit)", 2048, (248320,), 1)
DEV = torch.device("cuda")


def profile(fn, iters):
    fn(); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            fn(i)
        torch.cuda.synchronize()
    per = defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            us = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            per[e.name][0] += us / iters
            per[e.name][1] += 1 / iters
    return dict(per)


def hw_bounds():
    x = torch.empty(1 << 29, dtype=torch.float16, device=DEV)  # 1 GiB
    y = torch.empty_like(x)
    ms = triton.testing.do_bench(lambda: y.copy_(x), rep=300)
    dram = 2 * x.numel() * 2 / (ms * 1e-3)
    del x, y
    a = torch.randn(8192, 8192, dtype=torch.float16, device=DEV)
    b = torch.randn(8192, 8192, dtype=torch.float16, device=DEV)
    ms = triton.testing.do_bench(lambda: torch.mm(a, b), rep=300)
    fp16 = 2 * 8192 ** 3 / (ms * 1e-3)
    ab, bb = a.bfloat16(), b.bfloat16()
    ms = triton.testing.do_bench(lambda: torch.mm(ab, bb), rep=300)
    bf16 = 2 * 8192 ** 3 / (ms * 1e-3)
    return dram, fp16, bf16


def trellis(k, parts, bits, copies=1):
    words = sum((k // 16) * (n // 16) * 8 * bits for n in parts)
    return torch.randint(-(1 << 31), (1 << 31) - 1, (copies, words), dtype=torch.int32, device=DEV).view(torch.int16)


def rows_out(label, kern, us, nbytes, flops, bounds, calls, extra=None):
    dram, peak = bounds
    t_bw, t_fl = nbytes / dram * 1e6, flops / peak * 1e6
    bound = max(t_bw, t_fl)
    return {
        "op": label, "kernel": kern, "calls": round(calls, 2), "us": round(us, 2),
        "MB": round(nbytes / 1e6, 3), "GFLOP": round(flops / 1e9, 3),
        "GBps": round(nbytes / (us * 1e-6) / 1e9, 1) if us else None,
        "TFps": round(flops / (us * 1e-6) / 1e12, 2) if us else None,
        "bound": "DRAM" if t_bw >= t_fl else "TC", "bound_us": round(bound, 2),
        "eff": round(bound / us, 3) if us else None, **(extra or {}),
    }


def short(name):
    for key in ("_exl3_gemv_kernel", "_had_rows_kernel", "_splitk_silu_had_kernel", "_splitk_combine_kernel",
                "_had_cols_kernel", "_reconstruct_experts_kernel", "_reconstruct_kernel", "_exl3_gemm_kernel",
                "moe_align", "count_and_sort", "act_and_mul", "gemm", "gemv"):
        if key in name:
            return key
    return name[:60]


def dense_decode(label, k, parts, layers, bits, bounds):
    n = sum(parts)
    wbytes = k * n * bits / 8
    copies = max(1, int(256e6 // wbytes) + 1)
    tr = trellis(k, parts, bits, copies)
    suh = torch.randn(copies, len(parts), k, dtype=torch.float16, device=DEV)
    svh = torch.randn(copies, n, dtype=torch.float16, device=DEV)
    pt = K.Exl3Parts.build(k, parts, bits, "mul1", DEV)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=DEV)
    per = profile(lambda i=0: exl3_forward(x, tr[i % copies], suh[i % copies], svh[i % copies], pt, torch.bfloat16), 50)
    out = []
    for name, (us, c) in per.items():
        s = short(name)
        if s == "_exl3_gemv_kernel":
            nb, fl = wbytes + k * 2 * len(parts) + n * 2 + n * 2, 2 * k * n
        elif s == "_had_rows_kernel":
            nb, fl = k * 2 + len(parts) * k * 2, 0
        else:
            nb, fl = 0, 0
        out.append(rows_out(f"decode {label}", s, us, nb, fl, bounds, c, {"layers": layers}))
    return out


def dense_prefill(label, k, parts, layers, bits, m, bounds):
    n = sum(parts)
    tr = trellis(k, parts, bits)[0]
    suh = torch.randn(len(parts), k, dtype=torch.float16, device=DEV)
    svh = torch.randn(n, dtype=torch.float16, device=DEV)
    pt = K.Exl3Parts.build(k, parts, bits, "mul1", DEV)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
    per = profile(lambda i=0: exl3_forward(x, tr, suh, svh, pt, torch.bfloat16), 5)
    out = []
    for name, (us, c) in per.items():
        s = short(name)
        if s == "_reconstruct_kernel":
            nb, fl = k * n * bits / 8 + k * n * 2, 0
        elif s == "_had_rows_kernel":
            nb, fl = m * k * 2 + len(parts) * m * k * 2, 0
        elif s == "_had_cols_kernel":
            nb, fl = m * n * 4 + m * n * 2, 0
        elif "gemm" in name.lower() or "cutlass" in name.lower() or "sm80" in name or "s16816" in name:
            s = "cuBLAS mm (fp16, fp32 out)"
            nb, fl = len(parts) * m * k * 2 + k * n * 2 + m * n * 4, 2 * m * k * n
        else:
            nb, fl = 0, 0
        out.append(rows_out(f"prefill M={m} {label}", s, us, nb, fl, bounds, c, {"layers": layers}))
    return out


def moe_banks(n):
    banks = []
    for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
        if dtype == torch.int16:
            t = torch.randint(-(1 << 15), 1 << 15, (n, *shape), dtype=torch.int32, device=DEV).to(torch.int16)
        else:
            t = (torch.randn((n, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half()
        banks.append(t)
    return tuple(banks)


def moe_decode(bounds):
    banks = moe_banks(E)  # 256 slots, ~500 MB of trellis: random picks miss L2
    x = torch.randn(1, H, dtype=torch.bfloat16, device=DEV)
    w = torch.softmax(torch.randn(1, TOPK, device=DEV), -1)
    ids = [torch.randperm(E, device=DEV)[:TOPK].to(torch.int32).view(1, TOPK) for _ in range(64)]
    per = profile(lambda i=0: fused_experts_exl3(x, banks, w, ids[i % 64], bits=BITS, codebook="mul1", is_prefill=False), 50)
    gu_w, dn_w = H * 2 * I * BITS / 8, I * H * BITS / 8
    out = []
    for name, (us, c) in per.items():
        s = short(name)
        if s == "_exl3_gemv_kernel":
            # two launches per call: gate|up (8 x H x 2I) and down (8 x I x H); report them together
            nb = TOPK * (gu_w + dn_w) + TOPK * (H * 2 * 2 + 2 * I * 4 + I * 2 + H * 4 + 2 * I * 2 + H * 2)
            fl = 2 * TOPK * (H * 2 * I + I * H)
        elif s == "_splitk_silu_had_kernel":
            nb, fl = TOPK * (2 * I * 4 + I * 2), 0
        elif s == "_splitk_combine_kernel":
            nb, fl = TOPK * H * 4 + H * 2, 0
        elif s == "_had_rows_kernel":
            nb, fl = H * 2 + TOPK * 2 * H * 2, 0
        else:
            nb, fl = 0, 0
        out.append(rows_out("decode MoE (8 routes)", s, us, nb, fl, bounds, c, {"layers": 40}))
    return out


def moe_prefill(m, bounds):
    banks = moe_banks(E)
    x = torch.randn(m, H, dtype=torch.bfloat16, device=DEV)
    w = torch.softmax(torch.randn(m, TOPK, device=DEV), -1)
    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(m)]).to(torch.int32)
    per = profile(lambda i=0: fused_experts_exl3(x, banks, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E), 3)
    chunks = -(-m // PREFILL_CHUNK_TOKENS)
    routes = m * TOPK
    gu_w, dn_w = H * 2 * I, I * H  # elements per expert
    out = []
    for name, (us, c) in per.items():
        s = short(name)
        if s == "_reconstruct_experts_kernel":
            nb, fl = chunks * E * (gu_w + dn_w) * (BITS / 8 + 2), 0
        elif s == "_exl3_gemm_kernel":
            # compulsory: each chunk reads its decoded W_hat once, the rotated inputs once, writes outputs once
            nb = chunks * E * (gu_w + dn_w) * 2 + routes * (2 * H * 2 + 2 * I * 2 + I * 2 + H * 4)
            fl = 2 * routes * (gu_w + dn_w)
        elif s == "_had_rows_kernel":
            nb, fl = m * H * 2 + routes * 2 * H * 2 + routes * I * 2 * 2, 0
        else:
            nb, fl = 0, 0
        out.append(rows_out(f"prefill M={m} MoE", s, us, nb, fl, bounds, c, {"layers": 40}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--m", type=int, default=8192)
    args = ap.parse_args()
    dram, fp16, bf16 = hw_bounds()
    print(f"# measured bounds: DRAM copy {dram / 1e9:.0f} GB/s (read+write), fp16 GEMM {fp16 / 1e12:.1f} TF/s, "
          f"bf16 GEMM {bf16 / 1e12:.1f} TF/s (8192^3, fp32 accumulate)", flush=True)
    bounds = (dram, fp16)
    rows = []
    for label, k, parts, layers in DENSE:
        rows += dense_decode(label, k, parts, layers, BITS, bounds)
    rows += dense_decode(*LM_HEAD[:3], LM_HEAD[3], HEAD_BITS, bounds)
    rows += moe_decode(bounds)
    for label, k, parts, layers in DENSE:
        rows += dense_prefill(label, k, parts, layers, BITS, args.m, bounds)
    rows += moe_prefill(args.m, bounds)
    hdr = f"{'op':34s} {'kernel':30s} {'calls':>5s} {'us':>9s} {'MB':>9s} {'GFLOP':>8s} {'GB/s':>7s} {'TF/s':>6s} {'bound':>5s} {'bnd us':>8s} {'eff':>5s}"
    print(hdr)
    for r in rows:
        print(f"{r['op'][:34]:34s} {r['kernel'][:30]:30s} {r['calls']:5.1f} {r['us']:9.1f} {r['MB']:9.2f} {r['GFLOP']:8.2f} "
              f"{r['GBps'] or 0:7.0f} {r['TFps'] or 0:6.1f} {r['bound']:>5s} {r['bound_us']:8.1f} {r['eff'] or 0:5.2f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            f.write(json.dumps({"dram_Bps": dram, "fp16_flops": fp16, "bf16_flops": bf16}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
