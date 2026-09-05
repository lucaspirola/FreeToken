"""Microbenchmark for the Triton extend/prefill attention launch configuration.

The prefill twin of ``bench_decode_launch.py``. It drives the production
``extend_paged_attention`` entry point -- no server, no weights -- for one
chunked-prefill step: ``--chunk`` new bf16 tokens attending over a ``--prefix``
of quantized paged KV, at an arbitrary head shape.

This is the kernel that makes whole-prompt prefill quadratic: the grid is
``(batch, num_q_heads, cdiv(chunk, BLOCK_M))`` with **no** KV-split dimension, so
every one of the ``cdiv(chunk, BLOCK_M)`` q-blocks walks the entire prefix
serially. Occupancy is therefore never the problem (2048 CTAs at the Nemotron
shape); the per-CTA KV stream is. The knobs that matter are the tile
(``BLOCK_M`` x ``BLOCK_N``), the warp count and -- because the inner loop is a
load/dequantize/``tl.dot`` chain -- the number of software-pipeline stages.

Reported per configuration:
  * ``ms``            -- median time for one attention layer's chunk
  * ``TFLOP/s``       -- ``2 * 2 * chunk * (prefix + chunk/2) * q_heads * head_dim``
  * ``KV GB/s``       -- prefix K+V bytes *as each q-block sees them*, i.e.
                         multiplied by ``cdiv(chunk, BLOCK_M)``; this is the
                         traffic the kernel actually issues, and it is what the
                         64 MB L2 has to absorb.
  * ``uniq GB/s``     -- the same over unique prefix bytes (the DRAM floor if L2
                         captured every re-read perfectly)
  * ``prompt s``      -- projected seconds for a whole cold prompt of
                         ``--project-prompt`` tokens at this ms curve, summed
                         over ``--layers`` attention layers and all chunks.

Every configuration is verified against the tree's current default launch
(``rtol=atol=2e-2``, the kernel's own regression tolerance) before it is timed;
tile/warp/stage changes reorder the flash accumulation so bitwise equality is
neither expected nor required.

Run (Nemotron 3.5 Lightning: 6 attention layers, 32Q / 2KV / D128, q8_0 KV):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_prefill_attention.py \
      --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 --layers 6 \
      --prefix-lens 0 131072 262144 524288 1048576 --chunk 8192 \
      --block-m 64 128 --block-n 32 64 128 --warps 4 8 --stages 1 2 \
      --json benchmarks/results/prefill_attention_5080.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics

QUANT_CHOICES = ("bf16", "q8_0", "int4", "q8_q6", "q6_q5", "fp8_e4m3")
_QUANT_ALIAS = {"q4_0": "int4"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--quant", default="q8_0", choices=QUANT_CHOICES)
    p.add_argument("--prefix-lens", type=int, nargs="+", default=[0, 131072, 262144])
    p.add_argument("--chunk", type=int, default=8192, help="new tokens per prefill step")
    p.add_argument("--block-m", type=int, nargs="+", default=[0], help="0 = tree default")
    p.add_argument("--block-n", type=int, nargs="+", default=[0])
    p.add_argument("--warps", type=int, nargs="+", default=[0])
    p.add_argument("--stages", type=int, nargs="+", default=[0])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--layers", type=int, default=1, help="attention layers per token (scales projections)"
    )
    p.add_argument(
        "--project-prompt",
        type=int,
        default=1048576,
        help="prompt length whose whole-prefill attention cost is projected",
    )
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--json", dest="json_out", default=None)
    return p.parse_args(argv)


def _quant_specs(name: str):
    from freetoken.kvcache.quant import resolve_kv_quant

    name = _QUANT_ALIAS.get(name, name)
    if name == "bf16":
        return resolve_kv_quant(None), resolve_kv_quant(None)
    if name == "q8_q6":
        return resolve_kv_quant("q8_0"), resolve_kv_quant("q6_0")
    if name == "q6_q5":
        return resolve_kv_quant("q6_0"), resolve_kv_quant("q5_0")
    return resolve_kv_quant(name), resolve_kv_quant(name)


def _build_prefix_pool(slots, kv_heads, head_dim, device, seed, k_spec, v_spec):
    """Quantized paged K/V holding ``slots`` prefix tokens (``slots`` may be 0)."""
    import torch
    from freetoken.kernel.triton.kv_quant import store_kv_quant
    from freetoken.kvcache.quant import BLOCK

    slots = max(slots, 1)  # the kernel still wants a valid base pointer at prefix 0
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(slots, kv_heads, head_dim, generator=g, device=device, dtype=torch.bfloat16)
    v = torch.randn(slots, kv_heads, head_dim, generator=g, device=device, dtype=torch.bfloat16)
    if not k_spec.enabled and not v_spec.enabled:
        return k, None, v, None

    kq = torch.zeros(
        slots, kv_heads, k_spec.storage_dim(head_dim), device=device, dtype=k_spec.storage_dtype
    )
    vq = torch.zeros(
        slots, kv_heads, v_spec.storage_dim(head_dim), device=device, dtype=v_spec.storage_dtype
    )
    ks = torch.zeros(slots, kv_heads, head_dim // BLOCK, device=device, dtype=torch.float16)
    vs = torch.zeros_like(ks)
    indices = torch.arange(slots, device=device, dtype=torch.int32)
    store_kv_quant(kq, ks, vq, vs, indices, k, v, k_spec, v_spec)
    del k, v
    torch.cuda.empty_cache()
    return kq, ks, vq, vs


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


def _projected_prompt_seconds(ms_at, prompt: int, chunk: int, layers: int) -> float:
    """Whole-prompt attention seconds, interpolating the measured ms(prefix) curve."""
    xs = sorted(ms_at)
    if len(xs) < 2:
        return float("nan")

    def ms_of(pos: int) -> float:
        """Piecewise-linear in prefix, extrapolating past the last measured point.

        Cost is linear in prefix (every q-block walks the whole prefix), so a flat
        clamp beyond the last sample would understate a 1M projection built from a
        131K measurement by several fold.
        """
        lo = max([x for x in xs if x <= pos], default=None)
        hi = min([x for x in xs if x >= pos], default=None)
        if lo is not None and hi is not None and hi != lo:
            t = (pos - lo) / (hi - lo)
            return ms_at[lo] + t * (ms_at[hi] - ms_at[lo])
        if lo is None:  # below the first sample
            a, b = xs[0], xs[1]
        elif hi is None:  # above the last sample
            a, b = xs[-2], xs[-1]
        else:
            return ms_at[lo]
        slope = (ms_at[b] - ms_at[a]) / (b - a)
        return max(ms_at[a] + slope * (pos - a), 0.0)

    total_ms = 0.0
    pos = 0
    while pos < prompt:
        c = min(chunk, prompt - pos)
        total_ms += ms_of(pos) * layers * (c / chunk)
        pos += c
    return total_ms / 1000.0


def _sweep_prefix(args, attn_mod, device, k_spec, v_spec, prefix, out_fh) -> list[dict]:
    """Every (block_m, block_n, warps, stages) point for one prefix length."""
    import os

    import torch

    chunk = args.chunk
    kq, ks, vq, vs = _build_prefix_pool(
        prefix, args.kv_heads, args.head_dim, device, args.seed, k_spec, v_spec
    )
    g = torch.Generator(device=device).manual_seed(args.seed + 1)
    q = torch.randn(
        chunk, args.q_heads, args.head_dim, generator=g, device=device, dtype=torch.bfloat16
    )
    k_ext = torch.randn(
        chunk, args.kv_heads, args.head_dim, generator=g, device=device, dtype=torch.bfloat16
    )
    v_ext = torch.randn_like(k_ext)
    qo_indptr = torch.tensor([0, chunk], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device=device)
    kv_indices = torch.arange(max(prefix, 1), dtype=torch.int32, device=device)[:prefix]
    prefix_lens = torch.tensor([prefix], dtype=torch.int32, device=device)
    sm_scale = args.head_dim**-0.5

    def call():
        return attn_mod.extend_paged_attention(
            q,
            kq,
            vq,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            prefix_lens=prefix_lens,
            max_q_len=chunk,
            sm_scale=sm_scale,
            k_extend=k_ext,
            v_extend=v_ext,
            k_scale=ks,
            v_scale=vs,
        )

    env_names = (
        "FREETOKEN_EXTEND_BLOCK_M",
        "FREETOKEN_EXTEND_BLOCK_N",
        "FREETOKEN_EXTEND_NUM_WARPS",
        "FREETOKEN_EXTEND_NUM_STAGES",
    )
    saved = {n: os.environ.get(n) for n in env_names}

    def force(values):
        for name, val in zip(env_names, values):
            if val:
                os.environ[name] = str(val)
            else:
                os.environ.pop(name, None)
        attn_mod._extend_launch_env_override.cache_clear()

    rows: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    try:
        force((0, 0, 0, 0))
        baseline = call()
        default_cfg = attn_mod.extend_launch_config(
            head_dim=args.head_dim,
            block_d=1 << (args.head_dim - 1).bit_length(),
            smem_optin=attn_mod._optin_smem_bytes(device.index),
            capability=torch.cuda.get_device_capability(device),
        )
        for block_m in args.block_m:
            for block_n in args.block_n:
                for warps in args.warps:
                    for stages in args.stages:
                        force((block_m, block_n, warps, stages))
                        cfg = attn_mod.extend_launch_config(
                            head_dim=args.head_dim,
                            block_d=1 << (args.head_dim - 1).bit_length(),
                            smem_optin=attn_mod._optin_smem_bytes(device.index),
                            capability=torch.cuda.get_device_capability(device),
                        )
                        if cfg in seen:
                            continue
                        seen.add(cfg)
                        label = "%d/%d/%d/%d" % cfg
                        try:
                            if not args.skip_verify:
                                torch.testing.assert_close(
                                    call(), baseline, rtol=2e-2, atol=2e-2
                                )
                            ms = _time_ms(call, args.warmup, args.iters)
                        except Exception as exc:  # OOM / smem overflow / mismatch
                            print(
                                f"  prefix {prefix:>9,}  {label:<14} SKIP {type(exc).__name__}: "
                                f"{str(exc).splitlines()[0][:90]}",
                                flush=True,
                            )
                            torch.cuda.empty_cache()
                            continue
                        q_blocks = -(-chunk // cfg[0])
                        flops = (
                            2 * 2 * chunk * (prefix + chunk / 2) * args.q_heads * args.head_dim
                        )
                        bpe = k_spec.bytes_per_element(torch.bfloat16) + v_spec.bytes_per_element(
                            torch.bfloat16
                        )
                        uniq_bytes = prefix * args.kv_heads * args.head_dim * bpe
                        row = {
                            "prefix": prefix,
                            "chunk": chunk,
                            "block_m": cfg[0],
                            "block_n": cfg[1],
                            "warps": cfg[2],
                            "stages": cfg[3],
                            "is_default": cfg == default_cfg,
                            "ctas": args.q_heads * q_blocks,
                            "ms": ms,
                            "tflops": flops / (ms * 1e-3) / 1e12,
                            "kv_gbps": uniq_bytes * q_blocks / (ms * 1e6),
                            "uniq_gbps": uniq_bytes / (ms * 1e6),
                        }
                        rows.append(row)
                        print(
                            f"  prefix {prefix:>9,}  {label:<14} ctas {row['ctas']:>5}  "
                            f"{ms:8.3f} ms  {row['tflops']:6.1f} TFLOP/s  "
                            f"{row['kv_gbps']:8.1f} GB/s issued  "
                            f"{row['uniq_gbps']:6.1f} GB/s unique"
                            + ("  <- tree default" if row["is_default"] else ""),
                            flush=True,
                        )
                        if out_fh is not None:
                            out_fh.write(json.dumps(row) + "\n")
                            out_fh.flush()
    finally:
        for name, val in saved.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val
        attn_mod._extend_launch_env_override.cache_clear()
    # The pool tensors die with this frame; the caller empties the allocator cache
    # before building the next (up to 1M-slot) pool.
    torch.cuda.empty_cache()
    return rows


def _report(rows, args) -> None:
    if not rows:
        return
    by_cfg: dict[tuple, dict[int, float]] = {}
    for r in rows:
        by_cfg.setdefault((r["block_m"], r["block_n"], r["warps"], r["stages"]), {})[
            r["prefix"]
        ] = r["ms"]
    prefixes = sorted({r["prefix"] for r in rows})
    default = next(
        (
            (r["block_m"], r["block_n"], r["warps"], r["stages"])
            for r in rows
            if r["is_default"]
        ),
        None,
    )
    head = "  ".join(f"{p:>10,}" for p in prefixes)
    print(f"\nms per attention layer, chunk {args.chunk}")
    print(f"{'m/n/warps/stages':<18}  {head}   prompt {args.project_prompt:,} s")
    for cfg, ms_at in sorted(by_cfg.items(), key=lambda kv: sum(kv[1].values())):
        cells = "  ".join(
            f"{ms_at[p]:10.3f}" if p in ms_at else f"{'-':>10}" for p in prefixes
        )
        proj = _projected_prompt_seconds(
            ms_at, args.project_prompt, args.chunk, args.layers
        )
        tag = "  <- tree default" if cfg == default else ""
        print(f"{'%d/%d/%d/%d' % cfg:<18}  {cells}   {proj:10.1f}{tag}")
    if default is not None and default in by_cfg:
        base = by_cfg[default]
        best = min(
            (c for c in by_cfg if len(by_cfg[c]) == len(base)),
            key=lambda c: sum(by_cfg[c].values()),
        )
        if best != default:
            print(
                f"\nbest {best} vs tree default {default}: "
                + ", ".join(
                    f"{p:,}: {base[p] / by_cfg[best][p]:.2f}x" for p in sorted(base)
                )
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch

    from freetoken.kernel.triton import attention as attn_mod

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print(
        f"gpu {props.name}  sms {props.multi_processor_count}  "
        f"l2 {props.L2_cache_size / 2**20:.0f} MiB  "
        f"geometry q={args.q_heads} kv={args.kv_heads} d={args.head_dim} "
        f"quant={args.quant} chunk={args.chunk}",
        flush=True,
    )
    print(
        "extend grid = (batch, q_heads, cdiv(chunk, BLOCK_M)) -- no KV split dimension: "
        "every q-block walks the whole prefix",
        flush=True,
    )

    k_spec, v_spec = _quant_specs(args.quant)
    rows: list[dict] = []
    with contextlib.ExitStack() as stack:
        out_fh = stack.enter_context(open(args.json_out, "a")) if args.json_out else None
        for prefix in args.prefix_lens:
            rows += _sweep_prefix(args, attn_mod, device, k_spec, v_spec, prefix, out_fh)
            torch.cuda.empty_cache()
    _report(rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
