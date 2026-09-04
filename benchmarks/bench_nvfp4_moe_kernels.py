"""Microbenchmark: NVFP4 routed-expert MoE kernels at the Nemotron-3.5-Lightning geometry.

One MoE layer's expert GEMMs only -- router, shared expert and the offload cache's PCIe
gather are out of scope -- over ungated ReLU^2 experts (H=2688, I=1856, E=128, top-6),
comparing the two GPU backends FreeToken can pick on sm_120 (``moe/nvfp4_backends.py``
``select_nvfp4_backend``):

  triton   FreeToken's inline-dequant kernels over the native ModelOpt rows: the
           Marlin-style int32 wide-load GEMV for decode, the grouped GEMM for prefill
           (the same split ``OffloadMoELayer._expert_gemm`` makes).
  b12x     flashinfer's SM12x CuTe-DSL W4A16 fused MoE over its prepared/tiled banks;
           one kernel for both regimes, with ``relu(x)**2`` fused into the epilogue
           (``SUPPORTED_MOE_ACTIVATIONS = {"silu", "relu2"}``).

Both read the SAME bytes: NVFP4 experts are ~5.36 MiB each, so a decode step at top-6
streams ~32 MiB of weights and is purely HBM-bound. The reported bandwidth counts the
bytes of the *distinct routed experts* for the step (what the kernel actually has to
read), so % of the RTX 5080's 960 GB/s is the meaningful figure of merit.

Each timing loop cycles ``--routings`` (default 8) independent routing draws, because
the 5080's 64 MiB L2 holds a whole M=1 step's 32 MiB of experts: replaying one routing
measures an L2-resident kernel and reports up to 120% of the roofline. ``--routings 1``
reproduces that warm number deliberately.

Weights are random (the layout, not the values, sets the speed); one layer of 128
experts is 686 MiB, and only one backend's layout is resident at a time.

Phase-2 targets (tasks/nemotron35-plan.md, 2B1):
  b12x >= 70% of the HBM roofline at M=1
  b12x >= 2x triton at M=8 and M=16

Measured on the RTX 5080 (2026-09-04, --routings 8; results jsonl in benchmarks/results):
neither decode target is met -- b12x is 0.72x Triton at M=1 (37% of roofline) and 1.6x at
M=8/16 -- while prefill is 5-6x Triton at 88% of roofline. The M=1 target was set from a
5090 L2-warm measurement; with one routing replayed (--routings 1) this bench reproduces
that regime and reports b12x at 85% of roofline at M=1. b12x stays the auto pick for the
ungated geometry on prefill and batched decode.

After 2B2 (ReLU^2 fused into gemm1's epilogue, arithmetic E2M1 dequant in the prefill
GEMM, and the tuned decode/prefill configs that ``benchmarks/tune_nvfp4_moe.py`` produces)
the Triton column moved a long way at the same --routings 8:

  decode  M=1  67.9 -> 57.7 us (52% -> 61% of roofline)   M=2   121 ->  94.5 us
          M=4   234 -> 168.3 us (55% -> 77%)              M=8   430 -> 299.3 us (54% -> 77%)
          M=16  723 -> 559.5 us (55% -> 71%)
  prefill M=256  4.49 -> 1.82 ms   M=2048  19.8 -> 8.47 ms   M=8192  74.6 -> 29.5 ms

Prefill is still ~2.3x off b12x (12.6 ms at M=8192) and no tile choice closes the rest: at
29.5 ms the pair runs at ~33 TFLOP/s, 15% of the 5080's ~225 TFLOP/s bf16-with-fp32-accum
tensor peak, against b12x's 78 TFLOP/s over a swizzled tensor-core layout. Note that the
2.5 ms/layer figure floated for 8K x 6 routes is below the hardware floor: the pair is
9.81e11 FLOPs, so 100% of that peak is 4.36 ms.

Run (always take the GPU lock for timing):
  CUDA_VISIBLE_DEVICES=0 scripts/gpu_lock.sh \
      uv run python benchmarks/bench_nvfp4_moe_kernels.py \
      --json benchmarks/results/nvfp4_moe_kernels_5080.jsonl --gate
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

# Nemotron-3.5-Lightning MoE geometry (ungated ReLU^2 experts: gate_up is [I, H]).
H, I, E, TOP_K = 2688, 1856, 128, 6  # noqa: E741 -- H/I/E is the MoE shape notation
ACTIVATION = "relu2"

DECODE_M = (1, 2, 4, 8, 16)
PREFILL_M = (256, 2048, 8192)

# RTX 5080 (GB203): 256-bit GDDR7 @ 30 Gbps -> 960 GB/s theoretical peak.
PEAK_GBPS = 960.0

# per-expert NVFP4 bytes: up [I, H] + down [H, I], each e2m1 codes + per-16 e4m3
# block scales + the fp16 per-row global (which folds into an alpha for b12x).
EXPERT_BYTES = I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--backend", nargs="+", default=["triton", "b12x"])
    p.add_argument("--decode-m", type=int, nargs="+", default=list(DECODE_M))
    p.add_argument("--prefill-m", type=int, nargs="+", default=list(PREFILL_M))
    p.add_argument("--experts", type=int, default=E)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--routings",
        type=int,
        default=8,
        help="distinct routing draws cycled through the timing loop (1 == L2-warm replay)",
    )
    p.add_argument("--gate", action="store_true", help="exit non-zero if a 2B1 target misses")
    p.add_argument("--json", dest="json_out", default=None, help="append one JSON line per row")
    return p.parse_args(argv)


def _time_us(calls, warmup: int, iters: int) -> float:
    """Median wall time of one call, cycling through ``calls`` (one per routing draw).

    The rotation is what keeps the measurement honest: the 5080's L2 is 64 MiB and a
    decode step touches only its routed experts (6 x 5.36 MiB == 32 MiB at M=1), so
    replaying ONE routing serves the whole working set out of L2 and reports above the
    HBM roofline (measured: 120% at M=2). Cycling several draws (--routings) walks
    ~R x that set past an LRU L2, which is what a real decode stream does -- consecutive
    steps route to different experts. ``--routings 1`` reproduces the L2-warm number."""
    import torch

    for _ in range(warmup):
        calls[0]()
    torch.cuda.synchronize()
    samples = []
    for i in range(iters):
        fn = calls[i % len(calls)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(10**7)  # settle clocks before the timed call
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def _native_sources(num_experts: int, seed: int):
    """One layer of random ungated ModelOpt NVFP4 banks, on the host (pageable).

    Same shapes ``models/nvfp4_banks._alloc_nvfp4_host_banks(gated=False)`` allocates for
    Nemotron; the b12x repack rewrites them in place, so this is the single 686 MiB copy.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    return {
        "gate_up_packed": [torch.randint(0, 256, (num_experts, I, H // 2), dtype=torch.uint8, generator=g)],
        "gate_up_scale": [
            ((torch.rand(num_experts, I, H // 16, generator=g) * 1.5 + 0.25)).to(torch.float8_e4m3fn)
        ],
        "gate_up_global": [torch.full((num_experts, I), 0.5, dtype=torch.float16)],
        "down_packed": [torch.randint(0, 256, (num_experts, H, I // 2), dtype=torch.uint8, generator=g)],
        "down_scale": [
            ((torch.rand(num_experts, H, I // 16, generator=g) * 1.5 + 0.25)).to(torch.float8_e4m3fn)
        ],
        "down_global": [torch.full((num_experts, H), 0.75, dtype=torch.float16)],
    }


def _routing(m: int, num_experts: int, device, seed: int):
    import torch

    g = torch.Generator(device=device).manual_seed(seed + m)
    ids = torch.randint(0, num_experts, (m, TOP_K), dtype=torch.int32, device=device, generator=g)
    weights = torch.rand(m, TOP_K, dtype=torch.float32, device=device, generator=g)
    return ids, weights


def _bench_backend(backend: str, args, device) -> list[dict]:
    import torch

    from freetoken.moe.nvfp4_backends import (
        _b12x_unusable_reason,
        b12x_fused_experts,
        b12x_repack_sources_inplace,
    )

    num_experts = args.experts
    sources = _native_sources(num_experts, args.seed)

    if backend == "b12x":
        reason = _b12x_unusable_reason(torch.cuda.get_device_capability(device))
        if reason is not None:
            print(f"skipping b12x: {reason}")
            return []
        import types

        cfg = types.SimpleNamespace(
            hidden_size=H, moe_intermediate_size=I, hidden_act=ACTIVATION, expert_gated=False
        )
        t0 = time.perf_counter()
        packed = b12x_repack_sources_inplace(sources, cfg, device, chunk=16)
        torch.cuda.synchronize()
        print(f"b12x repack of {num_experts} experts: {time.perf_counter() - t0:.1f} s")
        banks = tuple(
            packed[name][0].to(device)
            for name in ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale")
        )
        alphas = (packed["gate_up_alpha"], packed["down_alpha"])
        del packed, sources

        def call_factory(hidden, ids, weights, is_prefill):
            gu_p, gu_s, dn_p, dn_s = banks

            def call():
                return b12x_fused_experts(
                    hidden, gu_p, gu_s, alphas[0], dn_p, dn_s, alphas[1],
                    weights, ids, ACTIVATION, False,
                )

            return call
    else:
        from freetoken.moe.fused_nvfp4 import (
            fused_experts_decode_nvfp4_marlin,
            fused_experts_nvfp4,
        )

        banks = tuple(
            sources[name][0].to(device)
            for name in (
                "gate_up_packed", "gate_up_scale", "gate_up_global",
                "down_packed", "down_scale", "down_global",
            )
        )
        del sources

        def call_factory(hidden, ids, weights, is_prefill):
            # exactly the split OffloadMoELayer._expert_gemm makes for quant_format "nvfp4"
            if is_prefill:
                def call():
                    return fused_experts_nvfp4(
                        hidden, *banks, weights, ids, num_experts, ACTIVATION, False,
                    )
            else:
                def call():
                    return fused_experts_decode_nvfp4_marlin(
                        hidden, *banks, weights, ids, ACTIVATION, False,
                    )

            return call

    rows: list[dict] = []
    for regime, sizes in (("decode", args.decode_m), ("prefill", args.prefill_m)):
        for m in sizes:
            draws = [
                _routing(m, num_experts, device, args.seed + 1000 * k)
                for k in range(max(1, args.routings))
            ]
            hidden = torch.randn(m, H, dtype=torch.bfloat16, device=device) / 4
            calls = [
                call_factory(hidden, ids, weights, regime == "prefill")
                for ids, weights in draws
            ]
            calls[0]()
            torch.cuda.synchronize()
            us = _time_us(calls, args.warmup, args.iters)
            # bytes of the DISTINCT routed experts, averaged over the draws
            routed = sum(int(torch.unique(ids).numel()) for ids, _ in draws) / len(draws)
            gbps = routed * EXPERT_BYTES / (us * 1e-6) / 1e9
            rows.append({
                "backend": backend, "regime": regime, "m": m,
                "routed_experts": routed, "us": us,
                "expert_gbps": gbps, "pct_roofline": 100.0 * gbps / PEAK_GBPS,
            })
            print(
                f"  {backend:6s} {regime:7s} M={m:<5d} routed={routed:<6.1f} "
                f"{us:9.1f} us  {gbps:7.1f} GB/s  {100.0 * gbps / PEAK_GBPS:5.1f}%"
            )
            del hidden, draws, calls
            torch.cuda.empty_cache()
    # `banks` (686 MiB) dies with this frame; the caller empties the cache before the
    # next backend's layout is built, so only one layout is ever resident.
    return rows


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    print(
        f"NVFP4 MoE kernels @ H={H} I={I} E={args.experts} top-{TOP_K} act={ACTIVATION}; "
        f"{EXPERT_BYTES / 2**20:.2f} MiB/expert, roofline {PEAK_GBPS:.0f} GB/s"
    )

    rows: list[dict] = []
    for backend in args.backend:
        rows.extend(_bench_backend(backend, args, device))
        torch.cuda.empty_cache()

    by = {(r["backend"], r["regime"], r["m"]): r for r in rows}
    print("\n M      regime   triton us   b12x us   speedup   b12x GB/s   b12x %roof")
    for regime, sizes in (("decode", args.decode_m), ("prefill", args.prefill_m)):
        for m in sizes:
            t = by.get(("triton", regime, m))
            b = by.get(("b12x", regime, m))
            speed = (t["us"] / b["us"]) if (t and b) else float("nan")
            print(
                f" {m:<6d} {regime:7s} "
                f"{t['us'] if t else float('nan'):9.1f} {b['us'] if b else float('nan'):9.1f} "
                f"{speed:9.2f} {b['expert_gbps'] if b else float('nan'):11.1f} "
                f"{b['pct_roofline'] if b else float('nan'):11.1f}"
            )
            if t and b:
                b["speedup_vs_triton"] = t["us"] / b["us"]

    if args.json_out:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(args.json_out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"ts": stamp, "geometry": {"H": H, "I": I,
                                                              "E": args.experts,
                                                              "top_k": TOP_K,
                                                              "activation": ACTIVATION},
                                    "peak_gbps": PEAK_GBPS,
                                    "routings": args.routings, **r}) + "\n")
        print(f"\nappended {len(rows)} rows to {args.json_out}")

    failures = []
    m1 = by.get(("b12x", "decode", 1))
    if m1 is not None and m1["pct_roofline"] < 70.0:
        failures.append(f"b12x M=1 roofline {m1['pct_roofline']:.1f}% < 70%")
    for m in (8, 16):
        t, b = by.get(("triton", "decode", m)), by.get(("b12x", "decode", m))
        if t and b and t["us"] / b["us"] < 2.0:
            failures.append(f"b12x M={m} speedup {t['us'] / b['us']:.2f}x < 2x triton")
    for line in failures:
        print(f"MISS: {line}")
    return 1 if (args.gate and failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
