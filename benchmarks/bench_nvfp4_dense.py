"""Dense NVFP4 (W4A16) linear microbenchmark for the Nemotron-3.5-Lightning shapes.

Measures ``kernel/triton/nvfp4_linear.py`` on the three dense NVFP4 projections Lightning
actually runs -- shared-expert ``up`` (3712x2688), shared-expert ``down`` (2688x3712) and
``lm_head`` (131072x2688) -- across the M the serving loop produces (decode 1..16, the
lm_head last-token batch up to 64, and prefill 8192).

Method notes (both matter by more than the effect being measured on this host):

* **CUDA graphs.** A python-side triton launch costs ~20 us of CPU on this box, which is
  more than a whole shared-expert decode GEMV. Decode runs graph-captured in production,
  so the numbers here are taken from a captured graph of many launches divided by the
  launch count -- pure GPU time. This is not only a small-M concern: eager wall-clock at
  the M where the in-kernel and dequant-to-scratch paths cross (~100-500 us kernels) has a
  ~15% run-to-run noise floor even at min-of-150, enough to read that crossover several
  hundred M too high. The check that it is gone is to time two configs that dispatch to
  the *same* code path (any M past both crossovers) and see the ratio land on 1.00.
  Only points too large to capture (their outputs are live in the graph's private pool,
  see ``--graph-pool-bytes``) fall back to ordinary launches.
* **Cold L2.** The 5080 has 64 MB of L2 and a shared expert's packed weight is 5.9 MB, so a
  naive loop measures an L2 resident. In a real decode step ~200 MB of other weights stream
  between two calls of the same layer, so the bench rotates over enough independent weight
  replicas to overflow L2 (``--replica-bytes``). Warm-L2 numbers come out ~2x optimistic.

``--tuning both`` (default) runs every point twice -- once with the per-arch launch config
table and once forced onto the ``_DEFAULT_TUNING`` fallback (the pre-2B3 H100 constants) --
and prints the before/after table.

Run:
    CUDA_VISIBLE_DEVICES=0 scripts/gpu_lock.sh uv run python benchmarks/bench_nvfp4_dense.py \
        --json benchmarks/results/nvfp4_dense_5080.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch  # noqa: E402

from freetoken.kernel.triton import nvfp4_linear as NL  # noqa: E402

# Lightning's dense NVFP4 projections: (N, K).
SHAPES = {
    "shared_up": (3712, 2688),
    "shared_down": (2688, 3712),
    "lm_head": (131072, 2688),
}
# Decode (1..16), the lm_head last-token batch (64), the in-kernel/scratch crossover
# region (128..512) and prefill (8192).
DEFAULT_MS = [1, 2, 4, 8, 16, 64, 128, 256, 512, 8192]
ROOFLINE_GBS = 960.0  # RTX 5080: 256-bit GDDR7 @ 30 Gbps


def weight_bytes(n: int, k: int) -> int:
    """Resident bytes of one NVFP4 weight: packed codes + fp8 block scales + fp16 globals."""
    return n * k // 2 + n * k // 16 + n * 2


def make_weights(n: int, k: int, seed: int):
    """Random weight in the K-major resident layout (what ``nvfp4_transpose_resident`` makes)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed_t = torch.randint(-(2**31), 2**31 - 1, (k // 8, n), dtype=torch.int32,
                             device="cuda", generator=g)
    scale_t = (torch.rand((k // 16, n), device="cuda", generator=g) * 0.5 + 0.5).to(NL.FP8)
    gscale = torch.full((n,), 0.03, dtype=torch.float16, device="cuda")
    return packed_t, scale_t, gscale


def make_replicas(n: int, k: int, target_bytes: int, cap: int = 32):
    reps = max(1, min(cap, -(-target_bytes // weight_bytes(n, k))))
    return [make_weights(n, k, seed=i) for i in range(reps)]


def bench_graph(make_run, reps: int, iters: int, warmup: int = 8) -> float:
    """us per launch, from a captured graph of ``iters`` launches rotating over replicas."""
    runs = [make_run(i) for i in range(reps)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            runs[i % reps]()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(iters):
            runs[i % reps]()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    best = None
    for _ in range(5):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        t = start.elapsed_time(end) * 1e3 / iters
        best = t if best is None else min(best, t)
    del graph
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # release the graph's private allocation pool
    return float(best)


def graph_iters(n: int, k: int, m: int, args) -> int:
    """How many launches fit in one capture. Every captured launch's output -- and, past
    the in-kernel crossover, its bf16 weight scratch -- stays live in the graph's private
    pool for the graph's lifetime, so the launch count is bounded by bytes, not by taste.
    Returns 0 when even two launches will not fit; the caller falls back to eager."""
    t = NL._tuning(torch.device("cuda", 0))
    wide = n >= t.gemm_wide_n
    in_kernel = m <= (t.gemm_max_inkernel_m_wide if wide else t.gemm_max_inkernel_m)
    scratch = 0 if (m == 1 or in_kernel) else min(n * k * 2, t.scratch_chunk_bytes)
    per = int((m * n * 2 + scratch + m * k * 2) * 1.5)  # 1.5x covers the split-K partials
    return min(args.graph_iters, int(args.graph_pool_bytes // max(per, 1)))


def bench_best(make_run, reps: int, n: int, k: int, m: int, args) -> tuple[float, str]:
    """Graph-timed when the capture fits, eager otherwise. Returns (us, method)."""
    iters = graph_iters(n, k, m, args)
    if iters >= 2:
        try:
            return bench_graph(make_run, reps, iters=iters), "graph"
        except (RuntimeError, torch.OutOfMemoryError):  # capture refused / pool too small
            torch.cuda.empty_cache()
    return bench_eager(make_run, reps, iters=args.eager_iters), "eager"


def bench_eager(make_run, reps: int, iters: int, warmup: int = 3) -> float:
    """us per call for prefill-size work (kernels are ms; python launch cost is noise)."""
    runs = [make_run(i) for i in range(reps)]
    for i in range(warmup):
        runs[i % reps]()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        runs[i % reps]()
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end) * 1e3)
    return float(statistics.median(ts))


def _free_gib() -> float:
    return torch.cuda.mem_get_info()[0] / 1024**3


def _fits(name: str, n: int, k: int, m: int, headroom_gib: float) -> bool:
    """Output-dominated points (lm_head at prefill M) need more VRAM than this host has to
    spare on a shared GPU: 8192 x 131072 bf16 is 2.1 GB of logits alone."""
    need = (m * n * 2 + m * k * 2 + weight_bytes(n, k) * 2) / 1024**3
    return need + 0.7 <= headroom_gib


def run_point(name: str, n: int, k: int, m: int, reps, args) -> dict:
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05
    a = x[0] if m == 1 else x

    def mk(i):
        pk, sc, g = reps[i]
        return lambda: NL.nvfp4_dense_linear_t(a, pk, sc, g)

    us, method = bench_best(mk, len(reps), n, k, m, args)
    del x
    torch.cuda.empty_cache()
    wb = weight_bytes(n, k)
    return {
        "shape": name, "N": n, "K": k, "M": m,
        "us": round(us, 2), "method": method,
        "weight_gbs": round(wb / us * 1e-3, 1),
        "roof_pct": round(wb / us * 1e-3 / args.roofline * 100, 1),
    }


def relu2_rows(name: str, n: int, k: int, ms, reps, args, props, stamp) -> list[dict]:
    """The shared expert is ``down(relu(up(x))**2)``; ``act="relu2"`` folds that into the
    up-projection's epilogue. Measured against the eager two-kernel form it replaces."""
    rows = []
    for m in ms:
        if not _fits(name, n, k, m, args.headroom):
            continue
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05
        a = x[0] if m == 1 else x

        def mk_fused(i):
            pk, sc, g = reps[i]
            return lambda: NL.nvfp4_dense_linear_t(a, pk, sc, g, act="relu2")

        def mk_eager(i):
            pk, sc, g = reps[i]
            return lambda: torch.relu(NL.nvfp4_dense_linear_t(a, pk, sc, g)).square()

        fused, _ = bench_best(mk_fused, len(reps), n, k, m, args)
        eager, _ = bench_best(mk_eager, len(reps), n, k, m, args)
        print(f"{name:11s} M={m:<5d} relu2   fused {fused:8.2f} us  eager {eager:8.2f} us  "
              f"saved {eager - fused:6.2f} us ({(1 - fused / eager) * 100:.0f}%)")
        for mode, us in (("relu2_fused", fused), ("relu2_eager", eager)):
            rows.append({"shape": name, "N": n, "K": k, "M": m, "us": round(us, 2),
                         "mode": mode, "device": props.name, "timestamp": stamp})
        del x
        torch.cuda.empty_cache()
    return rows


def cublas_bf16_us(n: int, k: int, m: int, iters: int) -> float:
    """torch.matmul on a dequantized bf16 copy of the weight -- the cuBLAS number the
    prefill path is judged against (the FP4 path must stay within ~1.3x of it)."""
    buf = [
        torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02,
        torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05,
        torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    ]

    def run():
        torch.matmul(buf[1], buf[0].t(), out=buf[2])

    us = bench_eager(lambda _i: run, 1, iters=iters)
    buf.clear()
    torch.cuda.empty_cache()
    return us


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default=",".join(SHAPES),
                    help="comma-separated subset of %s" % ",".join(SHAPES))
    ap.add_argument("--m", default=",".join(str(m) for m in DEFAULT_MS))
    ap.add_argument("--tuning", choices=["both", "arch", "default"], default="both",
                    help="'both' prints the before/after table (default = pre-2B3 constants)")
    ap.add_argument("--replica-bytes", type=int, default=134 << 20,
                    help="weight replicas per shape, in bytes, to defeat the 64 MB L2")
    ap.add_argument("--graph-iters", type=int, default=64)
    ap.add_argument("--graph-pool-bytes", type=int, default=512 << 20,
                    help="byte budget for one capture's private allocation pool; points "
                         "whose outputs will not fit two launches fall back to eager")
    ap.add_argument("--eager-iters", type=int, default=20)
    ap.add_argument("--roofline", type=float, default=ROOFLINE_GBS)
    ap.add_argument("--headroom-gib", type=float, default=0.0,
                    help="VRAM this run may use (default: 80%% of what is free)")
    ap.add_argument("--json", default="", help="append one JSON object per point to this file")
    ap.add_argument("--cublas", action="store_true", default=True,
                    help="also time cuBLAS bf16 at the largest M (default on)")
    ap.add_argument("--no-cublas", dest="cublas", action="store_false")
    ap.add_argument("--relu2", action="store_true", default=True,
                    help="also measure the fused relu^2 epilogue on the up projection")
    ap.add_argument("--no-relu2", dest="relu2", action="store_false")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    props = torch.cuda.get_device_properties(0)
    headroom = args.headroom_gib or _free_gib() * 0.8
    args.headroom = headroom
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    ms = [int(v) for v in args.m.split(",") if v.strip()]
    modes = {"both": ["default", "arch"], "arch": ["arch"], "default": ["default"]}[args.tuning]

    tuned = NL._ARCH_TUNING.get((props.major, props.minor))
    print(f"# {props.name} sm_{props.major}{props.minor} {props.multi_processor_count} SMs, "
          f"free {_free_gib():.1f} GiB, budget {headroom:.1f} GiB, roof {args.roofline:g} GB/s")
    print(f"# arch tuning: {'present' if tuned else 'MISSING (falls back to default)'}")

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_rows: dict[tuple, dict] = {}
    for name in shapes:
        n, k = SHAPES[name]
        reps = make_replicas(n, k, args.replica_bytes)
        for mode in modes:
            NL._TUNING_CACHE.clear()
            if mode == "default":
                NL._TUNING_CACHE[0] = NL._resolve_tuning(0, NL._DEFAULT_TUNING)
            for m in ms:
                if not _fits(name, n, k, m, headroom):
                    print(f"{name:11s} M={m:<5d} skipped (needs more VRAM than the budget)")
                    continue
                row = run_point(name, n, k, m, reps, args)
                row.update(mode=mode, device=props.name, sm=f"{props.major}.{props.minor}",
                           timestamp=stamp, replicas=len(reps))
                out_rows[(name, m, mode)] = row
                print(f"{name:11s} M={m:<5d} {mode:7s} {row['us']:9.2f} us "
                      f"{row['weight_gbs']:7.1f} GB/s  ({row['roof_pct']:.0f}% of roof)"
                      f"  [{row['method']}]")
            NL._TUNING_CACHE.clear()
        if args.relu2 and name == "shared_up":
            for row in relu2_rows(name, n, k, ms, reps, args, props, stamp):
                out_rows[(name, row["M"], row["mode"])] = row
        del reps
        torch.cuda.empty_cache()
        if args.cublas:
            big = max([m for m in ms if _fits(name, n, k, m, headroom)], default=0)
            if big > 64:
                us = cublas_bf16_us(n, k, big, args.eager_iters)
                row = {"shape": name, "N": n, "K": k, "M": big, "us": round(us, 2),
                       "mode": "cublas_bf16", "device": props.name, "timestamp": stamp}
                out_rows[(name, big, "cublas_bf16")] = row
                print(f"{name:11s} M={big:<5d} cublas  {us:9.2f} us  (bf16 dequantized copy)")

    if "default" in modes and "arch" in modes:
        print("\n=== before/after (us; ratio < 1 = tuned config faster) ===")
        print(f"{'shape':12s}{'M':>7s}{'before':>11s}{'after':>11s}{'ratio':>8s}{'GB/s':>9s}")
        for (name, m, mode), row in out_rows.items():
            if mode != "arch":
                continue
            before = out_rows.get((name, m, "default"))
            if not before:
                continue
            print(f"{name:12s}{m:>7d}{before['us']:>11.2f}{row['us']:>11.2f}"
                  f"{row['us'] / before['us']:>8.2f}{row['weight_gbs']:>9.1f}")
    for (name, m, mode), row in out_rows.items():
        if mode not in ("arch", "default"):
            continue
        cub = out_rows.get((name, m, "cublas_bf16"))
        if cub:
            print(f"\n{name} M={m} {mode}: {row['us'] / cub['us']:.2f}x cuBLAS bf16 "
                  f"({row['us']:.1f} vs {cub['us']:.1f} us)")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            for row in out_rows.values():
                fh.write(json.dumps(row) + "\n")
        print(f"\nappended {len(out_rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
