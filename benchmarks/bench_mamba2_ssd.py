"""Microbenchmark for the Triton Mamba-2 SSD prefill kernels (Nemotron-3.5).

Measures one Mamba-2 mixer layer's chunked scan -- the `mamba2_prefill` call
only, not the conv / projections / gated norm -- at the Lightning geometry
(H=64, P=64, N=128, G=8, chunk=128) in bf16, and compares it against the
pure-Torch reference the Phase-1 path uses today
(`transformers...nemotron_h.mamba2_chunk_scan`).

Cases: single-sequence prefill at 2048 / 8192 / 32768 tokens, and a varlen batch
of 4 x 8192. Timing is CUDA events, median of `--iters` after `--warmup`; the
transient allocation is the peak `torch.cuda.max_memory_allocated` delta across
one untimed call (the kernels materialise `[nchunks, H, P, N]` fp32 states and
`[nchunks, G, chunk, chunk]` fp32 CB, which is what the 700 MB budget at 32K is
about).

The reference is memory-bound, not compute-bound: HF's chunk scan materialises a
`[nchunks, chunk, H, P, N]` fp32 tensor, i.e. ~4.3 GB for a single 2048-token
call at this geometry. It is therefore run over `--ref-block` token blocks
carrying state forward -- which is also how the Phase-1 path actually runs it,
under `--max-prefill-length`. `--ref-max-tokens` skips it entirely for the long
cases so a sweep stays quick.

Phase-2 targets (plan tasks/nemotron35-plan.md, 2A2):
  8K   <= 1.5 ms/layer
  32K  <= 7 ms/layer
  >= 20x the torch reference
  < 700 MB transient at 32K

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python benchmarks/bench_mamba2_ssd.py
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python benchmarks/bench_mamba2_ssd.py \
      --json benchmarks/results/mamba2_ssd_5080.jsonl --gate
  # skip the slow reference entirely
  ... --ref-max-tokens 0
"""

from __future__ import annotations

import argparse
import json
import statistics

# Nemotron-3.5 Lightning Mamba-2 geometry.
H, P, N, G, CHUNK = 64, 64, 128, 8, 128

# (label, [sequence lengths])
CASES: list[tuple[str, list[int]]] = [
    ("single-2048", [2048]),
    ("single-8192", [8192]),
    ("single-32768", [32768]),
    ("varlen-4x8192", [8192] * 4),
]

# label -> (max ms/layer, min speedup vs torch, max transient MB)
TARGETS = {
    "single-8192": (1.5, 20.0, None),
    "single-32768": (7.0, 20.0, 700.0),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--case", nargs="+", default=None, help="subset of case labels")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ref-block",
        type=int,
        default=512,
        help="tokens per torch-reference call (its peak memory is quadratic-ish "
        "in this; 512 costs ~1 GB at H=64/P=64/N=128)",
    )
    p.add_argument(
        "--ref-max-tokens",
        type=int,
        default=8192,
        help="skip the torch reference for cases above this token count (0 = never run it)",
    )
    p.add_argument(
        "--gate", action="store_true", help="exit non-zero if a Phase-2 target misses"
    )
    p.add_argument(
        "--json", dest="json_out", default=None, help="append one JSON line per case here"
    )
    return p.parse_args(argv)


def _time_ms(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(10**7)  # settle clocks before the timed call
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _make_inputs(total: int, device, seed: int):
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    bf = {"device": device, "dtype": torch.bfloat16, "generator": g}
    f32 = {"device": device, "dtype": torch.float32, "generator": g}
    x = torch.randn(total, H, P, **bf)
    B = torch.randn(total, G, N, **bf) * 0.5
    C = torch.randn(total, G, N, **bf) * 0.5
    dt = torch.randn(total, H, **bf) * 0.3
    tgt = torch.rand(H, **f32) * 0.099 + 1e-3
    dt_bias = torch.log(torch.expm1(tgt))
    A = -torch.exp(torch.rand(H, **f32) * 2.77)
    D = torch.randn(H, **f32)
    return x, dt, B, C, A, D, dt_bias


def _torch_reference(x, dt, B, C, A, D, dt_bias, lens, block: int):
    """Phase-1 path: HF's fp32 chunk scan, run over `block`-token pieces."""
    import torch
    from transformers.models.nemotron_h.modeling_nemotron_h import mamba2_chunk_scan

    outs = []
    off = 0
    for length in lens:
        state = torch.zeros(1, H, P, N, device=x.device, dtype=torch.float32)
        pos = 0
        while pos < length:
            take = min(block, length - pos)
            sl = slice(off + pos, off + pos + take)
            out, state = mamba2_chunk_scan(
                x[sl][None].float(),
                dt[sl][None].float(),
                A,
                B[sl][None].float(),
                C[sl][None].float(),
                chunk_size=CHUNK,
                D=D,
                dt_bias=dt_bias,
                initial_states=state,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                return_final_states=True,
            )
            outs.append(out[0])
            pos += take
        off += length
    return torch.cat(outs, dim=0)


def run_case(label: str, lens: list[int], args) -> dict:
    import torch

    from freetoken.kernel.triton.mamba2 import build_mamba2_metadata, mamba2_prefill

    device = torch.device(f"cuda:{args.device}")
    total = sum(lens)
    x, dt, B, C, A, D, dt_bias = _make_inputs(total, device, args.seed)

    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    meta = build_mamba2_metadata(cu, CHUNK, device=device)
    pool = torch.zeros(len(lens), H, P, N, device=device, dtype=torch.float32)
    indices = torch.arange(len(lens), dtype=torch.int32, device=device)
    # Every call is a fresh prefill: without this the scan scatters its final
    # states into the pool and the *next* call picks them up as initial states,
    # so a repeated `call()` would drift away from the cold torch reference.
    has_init = torch.zeros(len(lens), dtype=torch.bool, device=device)
    out = torch.empty_like(x)

    def call():
        mamba2_prefill(
            x, dt, B, C,
            A=A, D=D, dt_bias=dt_bias, meta=meta, cu_seqlens=cu_seqlens,
            state_source=pool, indices=indices, has_initial_state=has_init,
            out=out,
        )

    call()  # trigger autotune before measuring memory
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    call()
    torch.cuda.synchronize()
    transient_mb = (torch.cuda.max_memory_allocated(device) - base) / 2**20

    triton_ms = _time_ms(call, args.warmup, args.iters)

    ref_ms = None
    max_abs = None
    if args.ref_max_tokens and total <= args.ref_max_tokens:
        ref_out = _torch_reference(x, dt, B, C, A, D, dt_bias, lens, args.ref_block)
        max_abs = (out.float() - ref_out.float()).abs().max().item()
        rel = (out.float() - ref_out.float()).square().mean().sqrt() / (
            ref_out.float().square().mean().sqrt() + 1e-8
        )
        if rel.item() > 5e-3:
            raise SystemExit(f"{label}: triton/torch rms mismatch {rel.item():.3e}")
        ref_ms = _time_ms(
            lambda: _torch_reference(
                x, dt, B, C, A, D, dt_bias, lens, args.ref_block
            ),
            max(1, args.warmup // 5),
            max(3, args.iters // 5),
        )
        del ref_out
        torch.cuda.empty_cache()

    row = {
        "case": label,
        "lens": lens,
        "tokens": total,
        "triton_ms": round(triton_ms, 4),
        "ref_ms": None if ref_ms is None else round(ref_ms, 4),
        "speedup": None if ref_ms is None else round(ref_ms / triton_ms, 2),
        "ref_block": args.ref_block if ref_ms is not None else None,
        "max_abs_err": None if max_abs is None else float(f"{max_abs:.3e}"),
        "transient_mb": round(transient_mb, 1),
        "num_chunks": meta.num_chunks,
    }

    target = TARGETS.get(label)
    if target is not None:
        max_ms, min_speedup, max_mb = target
        checks = {"latency": triton_ms <= max_ms}
        if row["speedup"] is not None:
            checks["speedup"] = row["speedup"] >= min_speedup
        if max_mb is not None:
            checks["transient"] = transient_mb <= max_mb
        row["target"] = {"max_ms": max_ms, "min_speedup": min_speedup, "max_mb": max_mb}
        row["pass"] = all(checks.values())
        row["checks"] = checks
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        print("no CUDA device", flush=True)
        return 2
    torch.cuda.set_device(args.device)
    print(f"device: {torch.cuda.get_device_name(args.device)}", flush=True)
    print(
        f"geometry: H={H} P={P} N={N} G={G} chunk={CHUNK} dtype=bfloat16 state=fp32",
        flush=True,
    )

    cases = CASES if args.case is None else [c for c in CASES if c[0] in set(args.case)]
    if not cases:
        print("no matching cases", flush=True)
        return 2

    failed = False
    for label, lens in cases:
        row = run_case(label, lens, args)
        verdict = ""
        if "pass" in row:
            verdict = "  PASS" if row["pass"] else f"  FAIL {row['checks']}"
            failed |= not row["pass"]
        speed = "n/a" if row["speedup"] is None else f"{row['speedup']:6.1f}x"
        ref = "n/a" if row["ref_ms"] is None else f"{row['ref_ms']:9.3f}"
        print(
            f"{label:<16} tokens={row['tokens']:>6} chunks={row['num_chunks']:>4}  "
            f"triton={row['triton_ms']:7.3f} ms  torch={ref} ms  "
            f"speedup={speed}  transient={row['transient_mb']:7.1f} MB{verdict}",
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
