"""Decode timeline driver (any model): prefill N_CTX tokens, then decode N_GEN, through the in-process
LLM (the scheduler's overlap loop, as ft serve runs it).

    decode_timeline.py MODE N_CTX N_GEN -- <ft serve flags>

MODE nsys:  bracket the measured request with cudaProfilerStart/Stop for
            nsys --capture-range=cudaProfilerApi (job-timeline.sh).
MODE greedy: no instrumentation; prints the sha1 of the generated token ids (output identity A/B).
MODE sync:  torch.cuda.set_sync_debug_mode(1) during the measured request; every synchronizing torch
            op warns, and its Python stack (freetoken frames) is counted. Prints the sites that
            synchronize in (nearly) every decode step.
"""
import collections
import dataclasses
import sys
import time
import traceback
import warnings

import torch

SENT = "The quick brown fox jumps over the lazy dog while the river keeps flowing past the old mill. "


def build(flags):
    from freetoken.llm.llm import LLM
    from freetoken.scheduler import SchedulerConfig
    from freetoken.server.args import parse_args

    sa, _ = parse_args(flags)
    skip = {"model_path", "tp_info", "dtype", "offline_mode"}
    kw = {f.name: getattr(sa, f.name) for f in dataclasses.fields(SchedulerConfig) if f.init and f.name not in skip}
    return LLM(sa.model_path, dtype=sa.dtype, **kw)


def main(mode, n_ctx, n_gen, flags):
    from freetoken.core import SamplingParams

    llm = build(flags)
    tok = llm.tokenizer
    ids = lambda tag, n: tok.encode(f"{tag} " + SENT * (n // 16 + 8), add_special_tokens=False)[:n]  # noqa: E731
    sites = collections.Counter()
    try:
        llm.generate([ids("warm", 8000)], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
        torch.cuda.synchronize()
        if mode == "sync":
            def show(message, category, filename, lineno, file=None, line=None):
                frames = [f for f in traceback.extract_stack()[:-1] if "/freetoken/" in f.filename]
                key = " <- ".join(f"{f.filename.split('/freetoken/')[-1]}:{f.lineno} {f.name}" for f in reversed(frames[-6:]))
                sites[(str(message)[:60], key)] += 1
            warnings.showwarning = show
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode(1)
        elif mode == "nsys":
            torch.cuda.profiler.start()
        t = time.perf_counter()
        res = llm.generate([ids("measured", n_ctx)], SamplingParams(temperature=0.0, max_tokens=n_gen, ignore_eos=True))
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        if mode == "sync":
            torch.cuda.set_sync_debug_mode(0)
        elif mode == "nsys":
            torch.cuda.profiler.stop()
    finally:
        llm.shutdown()
    out = res[0]["token_ids"]
    print(f"{mode} window: ctx {n_ctx}, {len(out)} tokens, request wall {wall:.3f} s", flush=True)
    if mode == "greedy":
        import hashlib
        import json
        print(f"greedy sha1 {hashlib.sha1(json.dumps(out).encode()).hexdigest()} first ids {out[:12]}")
    if mode == "sync":
        print(f"synchronizing sites during the measured request (prefill {n_ctx} + {n_gen} decode steps):")
        for (msg, key), c in sites.most_common(40):
            print(f"{c:6d}x  {msg}\n         {key}")


if __name__ == "__main__":
    sep = sys.argv.index("--")
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[sep + 1:])
