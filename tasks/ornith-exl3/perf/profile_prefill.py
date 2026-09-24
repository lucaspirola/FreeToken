"""Where does one 8K prefill chunk's time go? (EXL3 dense GEMM vs EXL3 MoE vs GDN vs attention vs copies)

    python profile_prefill.py <out-prefix> <n_tokens> -- <ft serve flags>

Warms up on one prompt, then profiles a max_tokens=1 generate on a second prompt of n_tokens
(different prefix, no radix hit). Writes <out-prefix>.txt (kernel table + category totals) and
<out-prefix>.json (chrome trace).
"""
import dataclasses
import sys
import time
from collections import defaultdict

import torch

SENT = "The quick brown fox jumps over the lazy dog while the committee reviews the budget. "

CATS = [  # first match wins, on the lower-cased kernel name
    ("exl3 gemv (decode path)", ("exl3_gemv",)),
    ("exl3 gemm", ("exl3_gemm",)),
    ("hadamard", ("had_rows", "hadamard")),
    ("exl3 other", ("exl3",)),
    ("gdn / fla", ("chunk", "gated_delta", "fla", "wy_", "recompute_w_u", "solve_tril", "l2norm", "causal_conv", "conv1d")),
    ("attention", ("attn", "attention", "flash", "fwd_kernel", "extend")),
    ("moe routing/sort", ("topk", "sort", "align", "moe_", "scatter", "gather", "index")),
    ("memcpy HtoD", ("memcpy htod",)),
    ("memcpy DtoD", ("memcpy dtod",)),
    ("memcpy DtoH", ("memcpy dtoh",)),
    ("gemm (cublas/dense bf16)", ("gemm", "cutlass", "sm90", "sm100", "sm120", "ampere", "cublas", "nvjet")),
    ("norm/act/elementwise", ("norm", "silu", "act", "elementwise", "vectorized", "reduce", "softmax", "mul", "add")),
]


def cat(name):
    n = name.lower()
    for label, keys in CATS:
        if any(k in n for k in keys):
            return label
    return "other"


def main(prefix, n_tok, flags):
    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM
    from freetoken.scheduler import SchedulerConfig
    from freetoken.server.args import parse_args

    sa, _ = parse_args(flags)
    skip = {"model_path", "tp_info", "dtype", "offline_mode"}
    kwargs = {f.name: getattr(sa, f.name) for f in dataclasses.fields(SchedulerConfig) if f.init and f.name not in skip}
    llm = LLM(sa.model_path, dtype=sa.dtype, **kwargs)
    try:
        def ids(tag):
            text = f"{tag} " + SENT * (n_tok // 16 + 8)
            return llm.tokenizer.encode(text, add_special_tokens=False)[:n_tok]
        sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        for w in ("warm-a", "warm-b"):
            t = time.perf_counter(); llm.generate([ids(w)], sp); torch.cuda.synchronize()
            print(f"warmup {w}: {time.perf_counter() - t:.2f} s")
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=acts) as prof:
            t = time.perf_counter(); llm.generate([ids("measured")], sp); torch.cuda.synchronize()
            wall = time.perf_counter() - t
    finally:
        llm.shutdown()
    rows, tot = defaultdict(lambda: [0.0, 0]), defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            us = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            rows[e.name][0] += us; rows[e.name][1] += 1; tot[cat(e.name)] += us
    gpu = sum(tot.values())
    lines = [f"prefill {n_tok} tokens: wall {wall:.2f} s ({n_tok / wall:.0f} tok/s); GPU kernel+copy time {gpu / 1e6:.2f} s", "", "category totals:"]
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:28s} {v / 1e3:10.1f} ms  {100 * v / gpu:5.1f}%")
    lines += ["", "top kernels:"]
    for name, (us, n) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:40]:
        lines.append(f"  {us / 1e3:10.1f} ms  {n:6d}x  [{cat(name)}] {name[:150]}")
    open(prefix + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    prof.export_chrome_trace(prefix + ".json")


if __name__ == "__main__":
    sep = sys.argv.index("--")
    main(sys.argv[1], int(sys.argv[2]), sys.argv[sep + 1:])
