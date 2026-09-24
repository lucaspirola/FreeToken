"""Kernel-time breakdown of a long prefill (per 8K chunk) or of decode at a given context.

    python profile_step.py prefill <out-prefix> <n_tokens> -- <ft serve flags>
    python profile_step.py decode  <out-prefix> <n_ctx> <n_gen> -- <ft serve flags>

prefill: one generate of n_tokens (max_tokens=1) under torch.profiler; kernels are bucketed per
chunk by counting the 10 attention layers' extend kernels, so the first and last chunks show what
grows with context. decode: prefill n_ctx then n_gen tokens; only kernels after the last prefill-only
kernel count (decode window), and the report gives GPU busy time vs wall per token, i.e. how much of
a decode step is kernels, host->device expert copies (saver misses) and host-side gaps.
Writes <out-prefix>.txt.
"""
import dataclasses
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from profile_prefill import SENT, cat  # noqa: E402

ATTN_KERNEL = "_extend_attention_split_kernel"
PREFILL_ONLY = ("_extend_attention", "_had_cols_kernel", "_reconstruct_experts_kernel", "_exl3_gemm_kernel", "chunk_")


def build(flags):
    from freetoken.llm.llm import LLM
    from freetoken.scheduler import SchedulerConfig
    from freetoken.server.args import parse_args

    sa, _ = parse_args(flags)
    skip = {"model_path", "tp_info", "dtype", "offline_mode"}
    kw = {f.name: getattr(sa, f.name) for f in dataclasses.fields(SchedulerConfig) if f.init and f.name not in skip}
    return LLM(sa.model_path, dtype=sa.dtype, **kw)


def ids(llm, tag, n):
    return llm.tokenizer.encode(f"{tag} " + SENT * (n // 16 + 8), add_special_tokens=False)[:n]


def kernels(prof):
    out = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            us = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            out.append((e.time_range.start, e.time_range.end, e.name, us))
    out.sort()
    return out


def table(evs, title):
    tot, gpu = defaultdict(float), 0.0
    for _, _, name, us in evs:
        tot[cat(name)] += us
        gpu += us
    lines = [f"{title}: GPU {gpu / 1e3:.1f} ms"]
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:28s} {v / 1e3:10.1f} ms  {100 * v / max(gpu, 1):5.1f}%")
    return lines


def top(evs, n=25):
    rows = defaultdict(lambda: [0.0, 0])
    for _, _, name, us in evs:
        rows[name][0] += us
        rows[name][1] += 1
    return [f"  {us / 1e3:10.1f} ms {c:7d}x [{cat(k)}] {k[:140]}" for k, (us, c) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:n]]


def main(mode, prefix, args, flags):
    llm = build(flags)
    from freetoken.core import SamplingParams

    lines = []
    try:
        llm.generate([ids(llm, "warm", 8000)], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        if mode == "prefill":
            n = int(args[0])
            with torch.profiler.profile(activities=acts) as prof:
                t = time.perf_counter()
                llm.generate([ids(llm, "measured", n)], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
                torch.cuda.synchronize()
                wall = time.perf_counter() - t
            evs = kernels(prof)
            chunks, seen = defaultdict(list), 0
            for ev in evs:
                if ev[2] == ATTN_KERNEL:
                    seen += 1
                chunks[max(seen - 1, 0) // 10].append(ev)
            lines.append(f"prefill {n} tokens: wall {wall:.2f} s ({n / wall:.0f} tok/s), {len(chunks)} chunks")
            lines += table(evs, "whole prefill")
            for c in sorted(chunks):
                ev = chunks[c]
                span = (ev[-1][1] - ev[0][0]) / 1e3
                lines += table(ev, f"chunk {c} (span {span:.1f} ms)")
            lines += ["", "top kernels, last chunk:"] + top(chunks[max(chunks)])
        else:
            n_ctx, n_gen = int(args[0]), int(args[1])
            cache = llm.engine.moe_offload_cache

            def mstats():
                torch.cuda.synchronize()
                return cache.mirror_stats() if cache is not None else {}
            before = mstats()
            with torch.profiler.profile(activities=acts) as prof:
                t = time.perf_counter()
                res = llm.generate([ids(llm, "measured", n_ctx)], SamplingParams(temperature=0.0, max_tokens=n_gen, ignore_eos=True))
                torch.cuda.synchronize()
                wall = time.perf_counter() - t
            evs = kernels(prof)
            last_pf = max(i for i, e in enumerate(evs) if any(e[2].startswith(p) for p in PREFILL_ONLY))
            dec = evs[last_pf + 1:]
            span = (dec[-1][1] - dec[0][0]) / 1e6
            gen = len(res[0]["token_ids"])
            busy = sum(e[3] for e in dec) / 1e6
            lines.append(f"decode at ctx {n_ctx}: {gen} tokens, request wall {wall:.2f} s; decode window {span:.3f} s "
                         f"= {gen / span:.1f} tok/s, {1e3 * span / gen:.2f} ms/token; GPU busy {1e3 * busy / gen:.2f} ms/token "
                         f"({100 * busy / span:.0f}% of the window)")
            after = mstats()
            delta = {k: after[k] - before.get(k, 0) for k in after if isinstance(after[k], int)}
            lines.append(f"mirror counters over the request (prefill of {n_ctx} + {gen} decode tokens): {delta}"
                         + (f"; {delta.get('swaps', 0) / gen:.1f} swaps and {delta.get('writebacks', 0) / gen:.1f} writebacks per token" if gen else ""))
            lines += table(dec, "decode window")
            lines += ["", "top kernels, decode window:"] + top(dec, 30)
    finally:
        llm.shutdown()
    open(prefix + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sep = sys.argv.index("--")
    main(sys.argv[1], sys.argv[2], sys.argv[3:sep], sys.argv[sep + 1:])
