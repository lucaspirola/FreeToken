"""Offline tuner for the inline-dequant NVFP4 fused-MoE Triton kernels.

Sweeps the two tunable knobs sets that ``freetoken.moe.fused_nvfp4`` reads:

  ``--decode``  the Marlin-style decode GEMV (:func:`_decode_gemm_marlin`):
                ``BLOCK_SIZE_N`` x ``BLOCK_SIZE_KW`` x ``num_warps``. One config per
                ``(N, K, top_k, sm_count)`` -- decode is CUDA-graph captured, so there is
                no per-M dispatch; the pick minimizes the total over ``--decode-pick-m``
                (default M in {1, 2, 4}, the batch range where Triton beats b12x on
                sm_120). Prints a paste-ready ``_DECODE_MARLIN_CONFIGS`` literal.

  ``--prefill`` the grouped GEMM (:func:`_prefill_gemm`): ``BLOCK_SIZE_M`` x
                ``BLOCK_SIZE_N`` x ``BLOCK_SIZE_KB`` x ``GROUP_SIZE_M`` x ``num_warps`` x
                ``num_stages``, per M bucket. ``BLOCK_SIZE_M`` is host-coupled (one
                ``moe_align_block_size`` padding feeds both GEMMs), so it is chosen
                *jointly*: for each candidate ``BLOCK_SIZE_M`` both GEMMs are tuned
                independently and the pair's summed time decides. ``--write`` emits
                ``python/freetoken/moe/configs/triton_<ver>/nvfp4,E=..,N=..,K=..,
                device_name=...json`` for both GEMMs, which
                :func:`freetoken.moe.fused_nvfp4.nvfp4_moe_config` loads.

Timing follows ``benchmarks/bench_nvfp4_moe_kernels.py``: several independent routing
draws are cycled through the measurement loop so the 64 MiB L2 cannot serve a decode
step's whole 32 MiB working set (``--routings 1`` reproduces the L2-warm regime).

Run (always take the GPU lock):
  CUDA_VISIBLE_DEVICES=0 scripts/gpu_lock.sh \
      uv run python benchmarks/tune_nvfp4_moe.py --decode --prefill --write
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

# Nemotron-3.5-Lightning MoE geometry (ungated ReLU^2: gate_up is [I, H], down is [H, I]).
H, I, E, TOP_K = 2688, 1856, 128, 6  # noqa: E741 -- H/I/E is the MoE shape notation
ACTIVATION = "relu2"

DECODE_M = (1, 2, 4, 8, 16)
DECODE_PICK_M = (1, 2, 4)
PREFILL_M = (16, 64, 256, 1024, 4096, 8192)

DECODE_BLOCK_N = (8, 16, 32, 64)
DECODE_BLOCK_KW = (8, 16, 32)
DECODE_WARPS = (2, 4, 8)

PREFILL_BLOCK_M = (16, 32, 64, 128)
PREFILL_BLOCK_N = (32, 64, 128)
PREFILL_BLOCK_KB = (32, 64)
PREFILL_GROUP_M = (1, 8)
PREFILL_WARPS = (4, 8)
PREFILL_STAGES = (3, 4)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--decode", action="store_true")
    p.add_argument("--prefill", action="store_true")
    p.add_argument("--experts", type=int, default=E)
    p.add_argument("--decode-m", type=int, nargs="+", default=list(DECODE_M))
    p.add_argument("--decode-pick-m", type=int, nargs="+", default=list(DECODE_PICK_M))
    p.add_argument("--prefill-m", type=int, nargs="+", default=list(PREFILL_M))
    p.add_argument("--routings", type=int, default=8)
    p.add_argument(
        "--tie-margin",
        type=float,
        default=0.03,
        help="decode: configs within this fraction of the best pick-M score are ties, "
             "broken on the total over --decode-m",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--write", action="store_true", help="write the tuned prefill JSON configs in-tree"
    )
    p.add_argument("--config-dir", default=None, help="override the configs/ root for --write")
    return p.parse_args(argv)


def _time_us(calls, warmup: int, iters: int) -> float:
    """Median wall time of one call, cycling ``calls`` (one per routing draw) so the L2
    cannot serve the whole working set -- same methodology as bench_nvfp4_moe_kernels."""
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


def _try_time(calls, warmup: int, iters: int) -> float:
    """``inf`` for a config the compiler rejects (smem overflow, illegal tile)."""
    import torch

    try:
        return _time_us(calls, warmup, iters)
    except Exception as exc:  # noqa: BLE001 -- any compile/launch failure disqualifies it
        torch.cuda.synchronize()
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        return float("inf")


def _banks(num_experts: int, device, seed: int):
    """One layer of random ungated ModelOpt NVFP4 banks, resident on the GPU (~700 MiB)."""
    import torch

    g = torch.Generator().manual_seed(seed)

    def to(t):
        return t.to(device)

    gate_up = (
        to(torch.randint(0, 256, (num_experts, I, H // 2), dtype=torch.uint8, generator=g)),
        to((torch.rand(num_experts, I, H // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)),
        to(torch.full((num_experts, I), 0.5, dtype=torch.float16)),
    )
    down = (
        to(torch.randint(0, 256, (num_experts, H, I // 2), dtype=torch.uint8, generator=g)),
        to((torch.rand(num_experts, H, I // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)),
        to(torch.full((num_experts, H), 0.75, dtype=torch.float16)),
    )
    return gate_up, down


def _routing(m: int, num_experts: int, device, seed: int):
    import torch

    g = torch.Generator(device=device).manual_seed(seed + m)
    ids = torch.randint(0, num_experts, (m, TOP_K), dtype=torch.int32, device=device, generator=g)
    weights = torch.rand(m, TOP_K, dtype=torch.float32, device=device, generator=g)
    return ids, weights


# --------------------------------------------------------------------------- decode


def _tune_decode(args, device) -> dict:
    import torch

    from freetoken.moe.fused_nvfp4 import _decode_gemm_marlin

    gate_up, down = _banks(args.experts, device, args.seed)
    draws = {
        m: [_routing(m, args.experts, device, args.seed + 1000 * k) for k in range(args.routings)]
        for m in args.decode_m
    }
    gemms = (
        ("gate_up", gate_up, I, H, 1, False),  # ACT=1 (relu2 fused), no routed weight
        ("down", down, H, I, 0, True),
    )
    grid = list(itertools.product(DECODE_BLOCK_N, DECODE_BLOCK_KW, DECODE_WARPS))
    picks: dict[tuple, dict] = {}
    sm = torch.cuda.get_device_properties(device).multi_processor_count

    for name, banks, n, k, act, mul_w in gemms:
        # gate_up consumes the hidden state [M, H]; down consumes the activated
        # [M*top_k, I] (a_row_is_route), exactly as _fused_experts_decode_nvfp4 calls them.
        a_row_is_route = name == "down"
        times: dict[tuple, dict[int, float]] = {}
        for m in args.decode_m:
            rows = m * TOP_K if a_row_is_route else m
            a = torch.randn(rows, k, dtype=torch.bfloat16, device=device) / 4
            c = torch.empty(m, TOP_K, n, dtype=torch.bfloat16, device=device)
            for cfg_t in grid:
                cfg = dict(zip(("BLOCK_SIZE_N", "BLOCK_SIZE_KW", "num_warps"), cfg_t))
                calls = [
                    (lambda ids=ids, w=w, cfg=cfg, a=a, c=c: _decode_gemm_marlin(
                        a, *banks, c, w, ids, mul_w, a_row_is_route, act, cfg))
                    for ids, w in draws[m]
                ]
                times.setdefault(cfg_t, {})[m] = _try_time(calls, args.warmup, args.iters)
            del a, c
            torch.cuda.empty_cache()

        def score(cfg_t):
            return sum(times[cfg_t][m] for m in args.decode_pick_m)

        def total(cfg_t):
            return sum(times[cfg_t][m] for m in args.decode_m)

        # Ties inside the pick window are common (a few % apart, i.e. noise); break them
        # on the whole M range so a config that collapses at M=8/16 never wins on noise.
        floor = score(min(grid, key=score))
        best = min([c for c in grid if score(c) <= floor * (1.0 + args.tie_margin)], key=total)
        picks[(n, k, TOP_K, sm)] = dict(
            zip(("BLOCK_SIZE_N", "BLOCK_SIZE_KW", "num_warps"), best)
        )
        print(f"\n== decode {name} N={n} K={k} (ACT={act}) ==")
        print("  BLK_N BLK_KW warps " + " ".join(f"{f'M={m}':>10s}" for m in args.decode_m))
        for cfg_t in sorted(grid, key=score)[:8]:
            row = " ".join(f"{times[cfg_t][m]:10.1f}" for m in args.decode_m)
            mark = "  <-- pick" if cfg_t == best else ""
            print(f"  {cfg_t[0]:5d} {cfg_t[1]:6d} {cfg_t[2]:5d} {row}{mark}")
        baseline = (16, 16, 4)
        if baseline in times:
            print(
                "  baseline (16,16,4): "
                + " ".join(f"{times[baseline][m]:10.1f}" for m in args.decode_m)
            )

    print("\n_DECODE_MARLIN_CONFIGS entries:")
    for key, cfg in picks.items():
        print(f"    {key}: dict(BLOCK_SIZE_N={cfg['BLOCK_SIZE_N']}, "
              f"BLOCK_SIZE_KW={cfg['BLOCK_SIZE_KW']}, num_warps={cfg['num_warps']}),")
    del gate_up, down
    torch.cuda.empty_cache()
    return picks


# -------------------------------------------------------------------------- prefill


def _prefill_calls(gemm_args, cfg, act, draws, block_m, num_experts):
    """One timed call per routing draw for a single prefill GEMM under ``cfg``."""
    from freetoken.moe.fused import moe_align_block_size
    from freetoken.moe.fused_nvfp4 import _prefill_gemm

    a, banks, c, kernel_top_k, mul_w = gemm_args
    calls = []
    for ids, weights in draws:
        sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, block_m, num_experts)
        tw = weights.reshape(-1).contiguous()
        num_valid = ids.numel()
        calls.append(
            lambda s=sorted_ids, e=expert_ids, n=ntpp, tw=tw, v=num_valid: _prefill_gemm(
                a, *banks, c, tw, s, e, n, v, kernel_top_k, mul_w, cfg, act
            )
        )
    return calls


def _tune_prefill(args, device) -> dict:
    import torch

    gate_up, down = _banks(args.experts, device, args.seed)
    tables: dict[tuple, dict[int, dict]] = {(I, H): {}, (H, I): {}}
    grid = list(
        itertools.product(
            PREFILL_BLOCK_N, PREFILL_BLOCK_KB, PREFILL_GROUP_M, PREFILL_WARPS, PREFILL_STAGES
        )
    )
    keys = ("BLOCK_SIZE_N", "BLOCK_SIZE_KB", "GROUP_SIZE_M", "num_warps", "num_stages")

    for m in args.prefill_m:
        draws = [
            _routing(m, args.experts, device, args.seed + 1000 * k)
            for k in range(max(1, min(args.routings, 4)))
        ]
        hidden = torch.randn(m, H, dtype=torch.bfloat16, device=device) / 4
        ic1 = torch.empty(m, TOP_K, I, dtype=torch.bfloat16, device=device)
        ic3 = torch.empty(m, TOP_K, H, dtype=torch.bfloat16, device=device)
        gemms = (
            ((hidden, gate_up, ic1, TOP_K, False), 1, (I, H)),
            ((ic1.view(-1, I), down, ic3, 1, True), 0, (H, I)),
        )
        per_block_m: dict[int, dict] = {}
        for block_m in PREFILL_BLOCK_M:
            best_per_gemm = []
            for gemm_args, act, shape in gemms:
                best_t, best_cfg = float("inf"), None
                for cfg_t in grid:
                    cfg = dict(zip(keys, cfg_t)) | {"BLOCK_SIZE_M": block_m}
                    calls = _prefill_calls(gemm_args, cfg, act, draws, block_m, args.experts)
                    t = _try_time(calls, args.warmup, args.iters)
                    if t < best_t:
                        best_t, best_cfg = t, cfg
                best_per_gemm.append((best_t, best_cfg, shape))
            total = sum(t for t, _, _ in best_per_gemm)
            per_block_m[block_m] = {"total": total, "gemms": best_per_gemm}
            print(
                f"  M={m:<5d} BLOCK_M={block_m:<3d} gate_up {best_per_gemm[0][0]:9.1f} us  "
                f"down {best_per_gemm[1][0]:9.1f} us  total {total:9.1f} us"
            )
        best_block_m = min(per_block_m, key=lambda b: per_block_m[b]["total"])
        chosen = per_block_m[best_block_m]
        print(f"  M={m:<5d} -> BLOCK_M={best_block_m} ({chosen['total']:.1f} us/pair)")
        for _t, cfg, shape in chosen["gemms"]:
            tables[shape][m] = cfg
            print(f"      {shape}: {cfg}")
        del hidden, ic1, ic3, draws
        torch.cuda.empty_cache()

    del gate_up, down
    torch.cuda.empty_cache()
    return tables


def _write_configs(tables: dict, args, device) -> list[Path]:
    import torch
    import triton

    from freetoken.moe.fused_nvfp4 import PREFILL_CONFIG_KEYS, nvfp4_config_filename

    root = Path(args.config_dir) if args.config_dir else (
        Path(__file__).resolve().parents[1] / "python" / "freetoken" / "moe" / "configs"
    )
    out_dir = root / f"triton_{triton.__version__.replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(device).replace(" ", "_")
    written = []
    for (n, k), buckets in tables.items():
        if not buckets:
            continue
        path = out_dir / nvfp4_config_filename(args.experts, n, k, device_name)
        payload = {
            str(m): {key: int(cfg[key]) for key in PREFILL_CONFIG_KEYS}
            for m, cfg in sorted(buckets.items())
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(path)
        print(f"wrote {path}")
    return written


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    if not (args.decode or args.prefill):
        args.decode = args.prefill = True
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    print(
        f"tuning NVFP4 MoE kernels @ H={H} I={I} E={args.experts} top-{TOP_K} "
        f"act={ACTIVATION} on {torch.cuda.get_device_name(device)} "
        f"({torch.cuda.get_device_properties(device).multi_processor_count} SMs), "
        f"routings={args.routings}"
    )
    if args.decode:
        _tune_decode(args, device)
    if args.prefill:
        tables = _tune_prefill(args, device)
        if args.write:
            _write_configs(tables, args, device)
        else:
            print("\n(--write not given; configs not saved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
