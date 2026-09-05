"""Microbenchmark: the NVFP4 routed-expert **prefill** grouped GEMM, alone.

`bench_nvfp4_moe_kernels.py` times the whole `fused_experts_nvfp4` pair against b12x and
reports HBM roofline %, which is the right figure of merit for *decode* (weight-streaming
bound) and a misleading one for prefill at M=8192, where each expert is read once per
M-block and the kernel is arithmetic bound.  This script reports **TFLOP/s against the
card's measured GEMM ceiling** instead, times the two GEMMs separately, and sweeps the
tile space that `moe/configs/*.json` is keyed on.

Geometry is Nemotron-3.5-Lightning's (H=2688, I=1856, E=128, top-6, ungated ReLU^2), the
same as `bench_nvfp4_moe_kernels.py`.  At M=8192 the pair is 9.81e11 FLOPs; the RTX 5080's
*measured* ceilings (benchmarks/results/nemotron35_lightning_5080_prefill_q8_2026-09-05.md)
are 123.0 TFLOP/s for cuBLAS bf16 and 118.4 TFLOP/s for Triton's own `tl.dot`, so the
honest denominator for a Triton kernel is 118.

Routing matters: the padded row count `moe_align_block_size` produces is
`sum_e ceil(n_e / BLOCK_M) * BLOCK_M`, so a skewed routing pads more than a uniform one.
`--skew` draws expert probabilities from a Dirichlet (alpha -> inf is uniform) and
`--routing-file` replays a real capture (a saved int32 `[M, top_k]` tensor, e.g. from
`FREETOKEN_MOE_DUMP_TOPK`).

Run (always under the GPU lock, redirected to a file -- never piped):

    PYTHONPATH=python scripts/gpu_lock.sh .venv/bin/python \
        benchmarks/bench_moe_prefill_gemm.py --m 8192 --grid wide > sweep.log 2>&1
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

H, I, E, TOP_K = 2688, 1856, 128, 6  # noqa: E741 -- MoE shape notation
ACTIVATION = "relu2"

# Measured on this card (prefill_q8 write-up S1), not the 225 TFLOP/s spec sheet.
CUBLAS_BF16_TFLOPS = 123.0
TRITON_DOT_TFLOPS = 118.4

# The grid `benchmarks/tune_nvfp4_moe.py` swept for the shipped tables.
GRID_TUNER = dict(
    BLOCK_SIZE_M=(16, 32, 64, 128),
    BLOCK_SIZE_N=(32, 64, 128),
    BLOCK_SIZE_KB=(32, 64),
    GROUP_SIZE_M=(1, 8),
    num_warps=(4, 8),
    num_stages=(3, 4),
)
# Everything the tuner's grid excluded, at the large-M end.
GRID_WIDE = dict(
    BLOCK_SIZE_M=(64, 128, 256),
    BLOCK_SIZE_N=(64, 128, 256),
    BLOCK_SIZE_KB=(16, 32, 64, 128),
    GROUP_SIZE_M=(1, 4, 8),
    num_warps=(4, 8),
    num_stages=(2, 3, 4, 5),
)
GRIDS = {"tuner": GRID_TUNER, "wide": GRID_WIDE}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--m", type=int, nargs="+", default=[8192])
    p.add_argument("--experts", type=int, default=E)
    p.add_argument("--grid", choices=sorted(GRIDS) + ["shipped"], default="shipped")
    p.add_argument("--grid-json", default=None,
                   help='an explicit grid, e.g. \'{"BLOCK_SIZE_M":[64,128]}\' (missing keys '
                        "take the shipped value)")
    p.add_argument("--variant", nargs="+", default=["tree"],
                   help="kernel variant(s); 'tree' is the production kernel")
    p.add_argument("--skew", type=float, default=None,
                   help="Dirichlet alpha for the expert distribution (default: uniform)")
    p.add_argument("--routing-file", default=None,
                   help="a saved int32 [M, top_k] topk_ids capture to replay")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=15, help="rows of the sweep table to print")
    p.add_argument("--json", dest="json_out", default=None)
    p.add_argument("--verify", action="store_true",
                   help="check every configuration against the shipped one (rtol 2e-2)")
    return p.parse_args(argv)


def _banks(num_experts: int, seed: int, device):
    """One layer of random ungated ModelOpt NVFP4 banks, on the device (686 MiB)."""
    import torch

    g = torch.Generator().manual_seed(seed)

    def to(t):
        return t.to(device)

    return (
        to(torch.randint(0, 256, (num_experts, I, H // 2), dtype=torch.uint8, generator=g)),
        to((torch.rand(num_experts, I, H // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)),
        to(torch.full((num_experts, I), 0.5, dtype=torch.float16)),
        to(torch.randint(0, 256, (num_experts, H, I // 2), dtype=torch.uint8, generator=g)),
        to((torch.rand(num_experts, H, I // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)),
        to(torch.full((num_experts, H), 0.75, dtype=torch.float16)),
    )


def routing(m: int, num_experts: int, device, seed: int, skew: float | None,
            routing_file: str | None):
    import torch

    if routing_file:
        ids = torch.load(routing_file, map_location=device).to(torch.int32)
        assert ids.shape[1] == TOP_K, ids.shape
        ids = ids[:m] if ids.shape[0] >= m else ids.repeat((m + ids.shape[0] - 1) // ids.shape[0], 1)[:m]
    else:
        g = torch.Generator(device=device).manual_seed(seed)
        if skew is None:
            ids = torch.randint(0, num_experts, (m, TOP_K), dtype=torch.int32,
                                device=device, generator=g)
        else:
            # Dirichlet(alpha) expert probabilities, then top_k *distinct* draws per row.
            gam = torch._standard_gamma(
                torch.full((num_experts,), float(skew), device=device)
            )
            probs = (gam / gam.sum()).expand(m, num_experts)
            ids = torch.multinomial(probs, TOP_K, replacement=False).to(torch.int32)
    weights = torch.rand(m, TOP_K, dtype=torch.float32, device=device,
                         generator=torch.Generator(device=device).manual_seed(seed + 1))
    return ids, weights


def padded_rows(ids, num_experts: int, block_m: int) -> int:
    """What `moe_align_block_size` produces: sum_e ceil(n_e / BLOCK_M) * BLOCK_M."""
    import torch

    counts = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=num_experts)
    return int((((counts + block_m - 1) // block_m) * block_m).sum().item())


def _time_us(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(10**7)  # settle clocks
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def _pair_flops(m: int) -> float:
    """gate_up [I, H] plus down [H, I], over m * top_k routed rows."""
    return 2.0 * m * TOP_K * (H * I) * 2.0


def apply_cfg_env(cfg: dict | None) -> None:
    """Force (or clear) the prefill tile through the FREETOKEN_NVFP4_PREFILL_* overrides."""
    import os

    from freetoken.moe.fused_nvfp4 import _PREFILL_ENV_KEYS

    for key, var in _PREFILL_ENV_KEYS.items():
        if cfg is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = str(cfg[key])


def variant_runner(name: str):
    """-> a callable (hidden, banks, weights, ids, num_experts) -> out."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

    if name == "tree":
        def run(hidden, banks, weights, ids, num_experts):
            return fused_experts_nvfp4(
                hidden, *banks, weights, ids, num_experts, ACTIVATION, False,
            )
        return run
    raise SystemExit(f"unknown variant {name!r}")


def sweep_configs(grid_name: str, grid_json: str | None, m: int, device):
    from freetoken.moe.fused_nvfp4 import nvfp4_moe_config

    apply_cfg_env(None)
    shipped = nvfp4_moe_config(m, I, H, device, E)
    if grid_json:
        grid = {k: dict(shipped, **{})[k] if False else v for k, v in json.loads(grid_json).items()}
        grid = {k: tuple(v) for k, v in grid.items()}
        for k, v in shipped.items():
            grid.setdefault(k, (v,))
    elif grid_name == "shipped":
        return [dict(shipped)]
    else:
        grid = dict(GRIDS[grid_name])
    keys = list(shipped)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    from freetoken.moe.fused_nvfp4 import nvfp4_moe_config

    banks = _banks(args.experts, args.seed, device)
    print(
        f"NVFP4 prefill grouped GEMM @ H={H} I={I} E={args.experts} top-{TOP_K} "
        f"act={ACTIVATION}; ceilings {CUBLAS_BF16_TFLOPS:.0f} (cuBLAS) / "
        f"{TRITON_DOT_TFLOPS:.0f} (tl.dot) TFLOP/s"
    )
    rows: list[dict] = []
    for m in args.m:
        ids, weights = routing(m, args.experts, device, args.seed, args.skew, args.routing_file)
        hidden = torch.randn(m, H, dtype=torch.bfloat16, device=device) / 4
        distinct = int(torch.unique(ids).numel())
        flops = _pair_flops(m)
        shipped = nvfp4_moe_config(m, I, H, device, args.experts)
        pad = padded_rows(ids, args.experts, shipped["BLOCK_SIZE_M"])
        print(
            f"\nM={m}  routed rows={m * TOP_K}  distinct experts={distinct}  "
            f"padded rows @BLOCK_M={shipped['BLOCK_SIZE_M']}: {pad} "
            f"(+{100.0 * pad / (m * TOP_K) - 100.0:.1f}%)  pair {flops / 1e9:.0f} GFLOP"
        )
        ref = None
        for vname in args.variant:
            run = variant_runner(vname)
            for cfg in sweep_configs(args.grid, args.grid_json, m, device):
                if ref is None:
                    apply_cfg_env(None)
                    ref = run(hidden, banks, weights, ids, args.experts).clone()
                    torch.cuda.synchronize()
                apply_cfg_env(cfg)
                try:
                    out = run(hidden, banks, weights, ids, args.experts)
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001 -- a tile that does not compile
                    print(f"  {vname:12s} {cfg} SKIP {type(exc).__name__}: {str(exc)[:90]}")
                    apply_cfg_env(None)
                    torch.cuda.empty_cache()
                    continue
                if args.verify:
                    scale = ref.float().abs().max().item()
                    d = (out.float() - ref.float()).abs().max().item()
                    if d > 2e-2 * scale + 2e-2:
                        print(f"  {vname:12s} {cfg} MISMATCH max|d|={d:.3e} (ref max {scale:.3e})")
                        apply_cfg_env(None)
                        continue
                us = _time_us(
                    lambda: run(hidden, banks, weights, ids, args.experts),
                    args.warmup, args.iters,
                )
                apply_cfg_env(None)
                tflops = flops / (us * 1e-6) / 1e12
                rows.append({
                    "m": m, "variant": vname, "us": us, "tflops": tflops,
                    "pct_cublas": 100.0 * tflops / CUBLAS_BF16_TFLOPS,
                    "pct_tldot": 100.0 * tflops / TRITON_DOT_TFLOPS,
                    "distinct_experts": distinct, "padded_rows": pad,
                    **cfg,
                })
                del out
                torch.cuda.empty_cache()
        best = sorted([r for r in rows if r["m"] == m], key=lambda r: r["us"])
        print(f"  {'variant':12s} {'M/N/KB/G/w/s':22s} {'ms':>9s} {'TFLOP/s':>9s} {'%tl.dot':>8s}")
        for r in best[: args.top]:
            tile = (f"{r['BLOCK_SIZE_M']}/{r['BLOCK_SIZE_N']}/{r['BLOCK_SIZE_KB']}/"
                    f"{r['GROUP_SIZE_M']}/{r['num_warps']}/{r['num_stages']}")
            print(f"  {r['variant']:12s} {tile:22s} {r['us'] / 1000:9.3f} "
                  f"{r['tflops']:9.1f} {r['pct_tldot']:7.1f}%")
        hidden = ids = weights = ref = None
        torch.cuda.empty_cache()

    if args.json_out:
        with open(args.json_out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nappended {len(rows)} rows to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
