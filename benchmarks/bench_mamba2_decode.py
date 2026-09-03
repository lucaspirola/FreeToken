"""Microbenchmark for the Mamba-2 decode step (task 2A3, Nemotron-3.5 Lightning).

Times one layer's `mamba2_decode` -- the selective state update only, not the
conv / projections / gated norm -- at the Lightning geometry (H=64, P=64, N=128,
G=8) in bf16 with an fp32 recurrent pool, for each available backend
(`FREETOKEN_MAMBA2_DECODE`: flashinfer, triton).

Timing is CUDA events, median of `--iters` after `--warmup`, with `out`
preallocated so the steady-state call allocates nothing. Two columns:

  eager  the whole call, launch overhead included. At bs=1 the kernel moves
         2 MiB and finishes in ~4 us, so this column is almost entirely the
         Python + launch path and is at the mercy of whatever else is on the
         CPU (median 27 us against a 4.4 us minimum on a busy host).
  graph  one CUDA-graph replay of the same call -- what the engine actually
         pays, since decode is captured. This is the column the gate uses.

Phase-2 targets (tasks/nemotron35-plan.md, 2A3), per graph replay:
  bs=1   <= 15 us/layer
  bs=16  <= 80 us/layer

Run (always under the GPU lock -- these are timings):
  CUDA_VISIBLE_DEVICES=0 scripts/gpu_lock.sh uv run python \
      benchmarks/bench_mamba2_decode.py \
      --json benchmarks/results/mamba2_decode_5080.jsonl --gate
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

# Nemotron-3.5 Lightning Mamba-2 geometry.
H, P, N, G = 64, 64, 128, 8
SLOTS = 32
BATCHES = (1, 8, 16)

# bs -> max us/layer
TARGETS = {1: 15.0, 16: 80.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, nargs="*", default=list(BATCHES))
    ap.add_argument(
        "--backend",
        nargs="*",
        default=None,
        help="subset of {flashinfer, triton}; default = every usable backend",
    )
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--gate", action="store_true", help="non-zero exit on a miss")
    return ap.parse_args(argv)


def _inputs(bs, device):
    import torch

    g = torch.Generator(device=device).manual_seed(bs)
    bf = {"device": device, "dtype": torch.bfloat16, "generator": g}
    f32 = {"device": device, "dtype": torch.float32, "generator": g}
    return dict(
        x=torch.randn(bs, H, P, **bf) * 0.5,
        dt=torch.randn(bs, H, **bf) * 0.5,
        B=torch.randn(bs, G, N, **bf) * 0.5,
        C=torch.randn(bs, G, N, **bf) * 0.5,
        A=-torch.exp(torch.rand(H, **f32) * 2.8),
        D=torch.randn(H, **f32),
        dt_bias=torch.rand(H, **f32) * 2.0 - 5.0,
        state=(torch.randn(SLOTS, H, P, N, **f32) * 0.1).contiguous(),
        indices=torch.randperm(SLOTS, device=device)[:bs].to(torch.int32),
        out=torch.empty(bs, H, P, device=device, dtype=torch.bfloat16),
    )


def _time_us(fn, iters, warmup):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        fn()
        e1[i].record()
    torch.cuda.synchronize()
    return sorted(e0[i].elapsed_time(e1[i]) * 1e3 for i in range(iters))


def run_case(backend, bs, args):
    import torch

    os.environ["FREETOKEN_MAMBA2_DECODE"] = backend
    from freetoken.kernel.triton.mamba2 import mamba2_decode, warm_mamba2_decode

    dev = torch.device("cuda", args.device)
    t = _inputs(bs, dev)

    t0 = time.perf_counter()
    resolved = warm_mamba2_decode(t["state"], bs, ngroups=G)
    warm_s = time.perf_counter() - t0
    assert resolved == backend, f"asked for {backend}, resolved to {resolved}"

    def _call():
        mamba2_decode(
            t["x"],
            t["dt"],
            t["B"],
            t["C"],
            A=t["A"],
            D=t["D"],
            dt_bias=t["dt_bias"],
            state_source=t["state"],
            indices=t["indices"],
            out=t["out"],
        )

    us = _time_us(_call, args.iters, args.warmup)

    # Graph replay: what the engine pays once decode is captured.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call()
    gus = _time_us(graph.replay, args.iters, args.warmup)

    eager = statistics.median(us)
    bytes_rw = 2 * bs * H * P * N * 4  # fp32 state read + write
    row = {
        "kernel": "mamba2_decode",
        "backend": backend,
        "bs": bs,
        "geometry": {"H": H, "P": P, "N": N, "G": G, "slots": SLOTS},
        "eager_us": round(eager, 3),
        "eager_min_us": round(us[0], 3),
        "eager_p90_us": round(us[int(0.9 * len(us))], 3),
        "graph_us": round(statistics.median(gus), 3),
        "state_gbps": round(bytes_rw / (eager * 1e-6) / 1e9, 1),
        "warm_s": round(warm_s, 3),
    }
    target = TARGETS.get(bs)
    if target is not None:
        row["target_us"] = target
        # Gated on the graph replay: eager at bs=1 is launch overhead, and the
        # engine captures the decode step.
        row["pass"] = row["graph_us"] <= target
        row["pass_eager"] = eager <= target
    return row


def main(argv=None):
    args = parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device", flush=True)
        return 2
    torch.cuda.set_device(args.device)

    from freetoken.kernel.triton.mamba2.selective_state_update import _flashinfer_ssu

    available = ["triton"]
    if _flashinfer_ssu() is not None:
        available.insert(0, "flashinfer")
    backends = available if args.backend is None else list(args.backend)

    print(f"device: {torch.cuda.get_device_name(args.device)}", flush=True)
    print(
        f"geometry: H={H} P={P} N={N} G={G} slots={SLOTS} dtype=bfloat16 state=fp32",
        flush=True,
    )
    print(f"backends: {backends}", flush=True)

    failed = False
    for backend in backends:
        for bs in args.batch:
            row = run_case(backend, bs, args)
            verdict = ""
            if "pass" in row:
                verdict = "  PASS" if row["pass"] else f"  FAIL (> {row['target_us']})"
                failed |= not row["pass"]
            print(
                f"{backend:<11} bs={bs:<3} eager={row['eager_us']:7.2f} us  "
                f"min={row['eager_min_us']:7.2f}  p90={row['eager_p90_us']:7.2f}  "
                f"graph={row['graph_us']:7.2f} us  "
                f"state={row['state_gbps']:6.1f} GB/s{verdict}",
                flush=True,
            )
            if args.json_out:
                with open(args.json_out, "a") as f:
                    f.write(json.dumps(row) + "\n")

    if args.gate and failed:
        print("gate: FAIL", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
