#!/usr/bin/env python3
"""Prefill roofline for Ornith EXL3 5.0bpw on this GPU vs the traced request (nsys-prefill.sh).

    roofline_prefill.py PROFILE_DIR   -> markdown on stdout

Bounds: the GPU's own measured peaks in PROFILE_DIR/roofline.log (tasks/ornith-exl3/perf/
bench_roofline.py: fp16 GEMM with fp32 accumulate, 8192^3, and a 1 GiB device copy).
Work per request, from the model's shapes (H 2048, 40 layers: 30 GDN + 10 full attention with 16 q
heads x 256 and 2 KV heads; 256 routed experts, top-8, I 512; shared expert I 512):
  dense projections   2 * (2048*12288*30 + 2048*9216*10 + 4096*2048*40 + 2048*1024*40 + 512*2048*40) FLOP/token
  routed experts      2 * 8 * (2048*1024 + 512*2048) * 40 FLOP/token
  attention           4 * 16 * 256 * 10 * sum over 8192-token chunks of c * (prefix + c/2) FLOP (causal)
  routed-expert bytes 5-bit weights of all 256 experts per layer, once per chunk (the compulsory read)
Measured: kernel time on the compute stream (the stream with the most kernel time), bucketed by kernel
name; the saver's layer assembly runs on a side stream and is listed separately (not on the path)."""
import re, sqlite3, sys
from collections import defaultdict

P = sys.argv[1]
log = open(f"{P}/roofline.log").read()
m = re.search(r"DRAM copy (\d+) GB/s .*fp16 GEMM ([\d.]+) TF/s", log)
DRAM, TC = float(m[1]) * 1e9, float(m[2]) * 1e12
DENSE = 2 * (2048 * 12288 * 30 + 2048 * 9216 * 10 + 4096 * 2048 * 40 + 2048 * 1024 * 40 + 512 * 2048 * 40)
MOE = 2 * 8 * (2048 * 1024 + 512 * 2048) * 40
MOE_BYTES_LAYER = 256 * (2048 * 1024 + 512 * 2048) * 5 / 8
BUCKET = [
    ("attention (flashinfer + prefix dequant + merge + rope)", r"SinglePrefill|MergeState|_gather_dequant|Rotary"),
    ("routed experts (exl3 GEMM, reconstruct, Hadamard, combine, act, align)", r"_exl3_gemm|_reconstruct_experts|_had_rows|_splitk_combine|act_and_mul|moe_align|count_and_sort|_router"),
    ("dense projections (cuBLAS fp16 + weight reconstruct)", r"^Kernel2$|_reconstruct_folded|gemm|Gemm|cutlass"),
    ("copies/casts (torch direct_copy, elementwise)", r"elementwise"),
    ("GDN + conv + norms", r"chunk_|recompute_w_u|causal_conv1d|l2norm|_layer_norm|rmsnorm|RMSNorm|norm"),
]


def work(n):
    chunks, p, fa = [], 0, 0.0
    while p < n:
        c = min(8192, n - p); fa += c * (p + c / 2); chunks.append(c); p += c
    return {"dense": DENSE * n, "moe": MOE * n, "attn": 4 * 16 * 256 * 10 * fa, "moe_bytes": MOE_BYTES_LAYER * 40 * len(chunks)}


def measure(f):
    db = sqlite3.connect(f); names = dict(db.execute("select id, value from StringIds"))
    k = db.execute("select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    per_stream = defaultdict(float)
    for s, e, st, _ in k:
        per_stream[st] += e - s
    main = max(per_stream, key=per_stream.get)
    b = defaultdict(float); side = defaultdict(float)
    for s, e, st, n in k:
        name = names[n]
        if st != main:
            side[name] += e - s; continue
        for lab, rx in BUCKET:
            if re.search(rx, name):
                b[lab] += e - s; break
        else:
            b["other"] += e - s
    span = max(e for _, e, _, _ in k) - min(s for s, _, _, _ in k)
    return {kk: v / 1e9 for kk, v in b.items()}, sum(per_stream[main] for _ in [0]) / 1e9, span / 1e9, {kk: v / 1e9 for kk, v in side.items()}


print(f"Peaks measured on this GPU: fp16 tensor core {TC / 1e12:.1f} TF/s, DRAM {DRAM / 1e9:.0f} GB/s (`profile/roofline.log`).\n")
for res in ("saver", "whole"):
    for n_req in (32000, 80000):
        f = f"{P}/prefill-{res}-{n_req}.sqlite"
        try:
            b, main, span, side = measure(f)
        except Exception as exc:
            print(f"{f}: {exc}"); continue
        import json
        pr = json.loads(open(f"{P}/prefill-{res}-{n_req}-probe.jsonl").readline())
        n = pr["prompt_tokens"]; w = work(n)
        bnd = {"attention": w["attn"] / TC, "routed": max(w["moe"] / TC, w["moe_bytes"] / DRAM), "dense": w["dense"] / TC}
        tot_bound = sum(bnd.values())
        print(f"### {res}, {n} tokens (traced TTFT {pr['ttft_mono_s']:.2f} s = {n / pr['ttft_mono_s']:.0f} tok/s; "
              f"kernel span {span:.2f} s, compute stream {main:.2f} s)\n")
        print("| bucket (compute stream) | measured s | bound s | measured / bound | work |\n|---|---:|---:|---:|---|")
        for lab, _ in BUCKET + [("other", "")]:
            t = b.get(lab, 0.0)
            key = "attention" if lab.startswith("attention") else "routed" if lab.startswith("routed") else "dense" if lab.startswith("dense") else None
            if key:
                wk = {"attention": f"{w['attn'] / 1e12:.0f} TFLOP", "routed": f"{w['moe'] / 1e12:.0f} TFLOP, {w['moe_bytes'] / 1e9:.0f} GB 5-bit weights",
                      "dense": f"{w['dense'] / 1e12:.0f} TFLOP"}[key]
                print(f"| {lab} | {t:.2f} | {bnd[key]:.2f} | {t / bnd[key]:.2f} | {wk} |")
            else:
                print(f"| {lab} | {t:.2f} | – | – | (no FLOP bound counted) |")
        idle = span - main
        print(f"| idle / launch gaps on the compute stream | {max(idle, 0):.2f} | 0 | – | |")
        print(f"| **total** | **{span:.2f}** | **{tot_bound:.2f}** | **{span / tot_bound:.2f}** | roofline {n / tot_bound:.0f} tok/s |")
        if side:
            print(f"\nSide streams (overlapped): " + ", ".join(f"{k2[:30]} {v:.2f} s" for k2, v in sorted(side.items(), key=lambda x: -x[1])[:3]))
        print()
