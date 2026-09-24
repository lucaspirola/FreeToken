"""Decode-attention launch-configuration sweep for an arbitrary GQA head shape.

``kernel/triton/attention.py::decode_launch_config`` returns ``(kv_splits, block_n,
num_warps)`` for the split-K decode kernel. Stage 1 launches
``batch * cdiv(num_q_heads, min(16, group)) * kv_splits`` CTAs, so on a head shape with
no measured branch the historical fallback of 8 splits put 16 CTAs on an 84-SM RTX 5080
and made single-stream decode slow down linearly with context. This script sweeps the
three knobs against the SAME production entry point (``decode_paged_attention``, no
server, no weights) for a caller-supplied geometry, and reports per-layer ms plus the
KV bytes/s the configuration actually achieves.

Two correctness gates run before anything is timed (unless ``--skip-verify``):

  baseline  -- every configuration's output is diffed against the 8-split baseline at the
               same context. Split-K changes the ORDER of the log-sum-exp reduction, so
               the outputs are not bit-identical; the gate is bf16-scale agreement
               (rtol/atol 2e-2, the kernel's own regression tolerance) plus a reported
               max-abs delta.
  oracle    -- for a quantized pool, the production call is diffed against the same
               kernel fed the pool's DEQUANTIZED values (skipped above
               ``--oracle-max-ctx`` where the extra bf16 copy no longer fits).

Run:
    PYTHONPATH=python python benchmarks/bench_decode_launch.py \
        --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 \
        --ctx-lens 131072 262144 524288 1048576 --splits 8 16 32 64 128 --json out.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
from dataclasses import dataclass

QUANT_CHOICES = ("bf16", "q8_0", "int4", "q8_q6", "q6_q5", "fp8_e4m3")
_QUANT_ALIAS = {"q4_0": "int4"}


@dataclass(frozen=True)
class Case:
    ctx_len: int
    batch: int
    splits: int
    block_n: int
    warps: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--quant", default="q8_0", choices=QUANT_CHOICES)
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[131072, 262144, 524288])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--splits", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    p.add_argument("--block-n", type=int, nargs="+", default=[32])
    p.add_argument("--warps", type=int, nargs="+", default=[4])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, default=1, help="attention layers per token (scales the reported per-token ms)")
    p.add_argument("--oracle-max-ctx", type=int, default=131072)
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


def _build_pool(slots: int, kv_heads: int, head_dim: int, device, seed: int, k_spec, v_spec, want_oracle: bool):
    """Quantized K/V pool of ``slots`` slots, plus dequantized oracles when asked."""
    import torch
    from freetoken.kernel.triton.kv_quant import store_kv_quant
    from freetoken.kvcache.quant import BLOCK

    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(slots, kv_heads, head_dim, generator=g, device=device, dtype=torch.bfloat16)
    v = torch.randn(slots, kv_heads, head_dim, generator=g, device=device, dtype=torch.bfloat16)
    if not k_spec.enabled and not v_spec.enabled:
        return k, None, v, None, None, None

    kq = torch.zeros(slots, kv_heads, k_spec.storage_dim(head_dim), device=device, dtype=k_spec.storage_dtype)
    vq = torch.zeros(slots, kv_heads, v_spec.storage_dim(head_dim), device=device, dtype=v_spec.storage_dtype)
    ks = torch.zeros(slots, kv_heads, head_dim // BLOCK, device=device, dtype=torch.float16)
    vs = torch.zeros_like(ks)
    indices = torch.arange(slots, device=device, dtype=torch.int32)
    store_kv_quant(kq, ks, vq, vs, indices, k, v, k_spec, v_spec)
    k_oracle = v_oracle = None
    if want_oracle:
        k_oracle = k_spec.dequantize(kq.float(), ks).to(torch.bfloat16).reshape(slots, kv_heads, head_dim)
        v_oracle = v_spec.dequantize(vq.float(), vs).to(torch.bfloat16).reshape(slots, kv_heads, head_dim)
    del k, v
    torch.cuda.empty_cache()
    return kq, ks, vq, vs, k_oracle, v_oracle


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


def _sweep_case(args, attn_mod, real_config, device, k_spec, v_spec, ctx_len, batch,
                head_blocks, sm_count, out_fh) -> list[dict]:
    """Every (splits, block_n, warps) point for one (context, batch) pool."""
    import torch

    slots = ctx_len * batch
    want_oracle = k_spec.enabled and not args.skip_verify and ctx_len <= args.oracle_max_ctx
    k_cache, k_scale, v_cache, v_scale, k_oracle, v_oracle = _build_pool(
        slots, args.kv_heads, args.head_dim, device, args.seed, k_spec, v_spec, want_oracle
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 7)
    q = torch.randn(
        batch, args.q_heads, args.head_dim, generator=generator, device=device,
        dtype=torch.bfloat16,
    )
    indptr = torch.arange(0, slots + 1, ctx_len, device=device, dtype=torch.int32)
    indices = torch.arange(slots, device=device, dtype=torch.int32)
    q_pos = torch.full((batch,), ctx_len - 1, device=device, dtype=torch.int32)
    sm_scale = args.head_dim**-0.5

    def call(splits: int, block_n: int, warps: int, k_c, v_c, k_s, v_s):
        # decode_paged_attention picks its own launch config; force the swept one.
        attn_mod.decode_launch_config = lambda **kwargs: (splits, block_n, warps)
        try:
            logits = torch.empty(
                batch, args.q_heads, splits, args.head_dim, device=device, dtype=torch.float32
            )
            lse = torch.empty(batch, args.q_heads, splits, device=device, dtype=torch.float32)
            nsplits = torch.full((batch,), splits, device=device, dtype=torch.int32)
            return attn_mod.decode_paged_attention(
                q, k_c, v_c, indptr, indices, q_pos, logits, lse, nsplits, splits,
                sm_scale, k_scale=k_s, v_scale=v_s,
            )
        finally:
            attn_mod.decode_launch_config = real_config

    baseline = None
    if not args.skip_verify:
        baseline = call(8, 32, 4, k_cache, v_cache, k_scale, v_scale).float()

    kv_bytes = slots * args.kv_heads * args.head_dim * (
        k_spec.bytes_per_element(torch.bfloat16) + v_spec.bytes_per_element(torch.bfloat16)
    )
    rows: list[dict] = []
    for block_n in args.block_n:
        for splits in args.splits:
            for warps in args.warps:
                label = f"ctx={ctx_len} bs={batch} splits={splits} bn={block_n} w={warps}"
                got = call(splits, block_n, warps, k_cache, v_cache, k_scale, v_scale)
                base_err = oracle_err = None
                if baseline is not None:
                    base_err = (got.float() - baseline).abs().max().item()
                    torch.testing.assert_close(
                        got.float(), baseline, rtol=2e-2, atol=2e-2,
                        msg=lambda m, label=label: f"{label} vs 8-split baseline: {m}",
                    )
                if k_oracle is not None:
                    want = call(splits, block_n, warps, k_oracle, v_oracle, None, None)
                    oracle_err = (got.float() - want.float()).abs().max().item()
                    torch.testing.assert_close(
                        got.float(), want.float(), rtol=2e-2, atol=2e-2,
                        msg=lambda m, label=label: f"{label} vs dequantized oracle: {m}",
                    )
                    del want
                time_ms = _time_ms(
                    lambda s=splits, bn=block_n, w=warps: call(
                        s, bn, w, k_cache, v_cache, k_scale, v_scale
                    ),
                    args.warmup, args.iters,
                )
                row = {
                    "q_heads": args.q_heads, "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim, "quant": args.quant, "ctx_len": ctx_len,
                    "batch": batch, "splits": splits, "block_n": block_n, "warps": warps,
                    "ctas": batch * head_blocks * splits, "sms": sm_count, "ms": time_ms,
                    "ms_per_token": time_ms * args.layers,
                    "gbps": kv_bytes / (time_ms * 1e6),
                    "max_abs_vs_8split": base_err, "max_abs_vs_oracle": oracle_err,
                }
                rows.append(row)
                print(
                    f"ctx={ctx_len:>8} bs={batch:>2} splits={splits:>3} bn={block_n:>3} "
                    f"w={warps} ctas={row['ctas']:>5} {time_ms:8.3f} ms "
                    f"{row['gbps']:7.1f} GB/s"
                    + (f"  d8={base_err:.2e}" if base_err is not None else "")
                    + (f"  dq={oracle_err:.2e}" if oracle_err is not None else ""),
                    flush=True,
                )
                if out_fh:
                    out_fh.write(json.dumps(row) + "\n")
                    out_fh.flush()
    return rows


def _report(rows: list[dict]) -> None:
    print("\nbest per (ctx, batch):", flush=True)
    for key in sorted({(r["ctx_len"], r["batch"]) for r in rows}):
        group = [r for r in rows if (r["ctx_len"], r["batch"]) == key]
        best = min(group, key=lambda r: r["ms"])
        base = next(
            (r for r in group if (r["splits"], r["block_n"], r["warps"]) == (8, 32, 4)), None
        )
        speedup = f"{base['ms'] / best['ms']:.2f}x vs 8/32/4" if base else ""
        print(
            f"  ctx={key[0]:>8} bs={key[1]:>2} -> splits={best['splits']} "
            f"bn={best['block_n']} w={best['warps']}  {best['ms']:.3f} ms  {speedup}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    from freetoken.kernel.triton import attention as attn_mod

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    print(
        f"gpu {props.name}  sms {sm_count}  geometry q={args.q_heads} kv={args.kv_heads} "
        f"d={args.head_dim} quant={args.quant}",
        flush=True,
    )
    group = args.q_heads // args.kv_heads
    head_blocks = -(-args.q_heads // min(16, group))
    print(
        f"stage-1 head blocks per request: {head_blocks} "
        f"(CTAs = batch * {head_blocks} * splits)",
        flush=True,
    )

    k_spec, v_spec = _quant_specs(args.quant)
    real_config = attn_mod.decode_launch_config
    rows: list[dict] = []
    with contextlib.ExitStack() as stack:
        out_fh = stack.enter_context(open(args.json_out, "a")) if args.json_out else None
        for ctx_len in args.ctx_lens:
            for batch in args.batch_sizes:
                rows += _sweep_case(
                    args, attn_mod, real_config, device, k_spec, v_spec, ctx_len, batch,
                    head_blocks, sm_count, out_fh,
                )
                torch.cuda.empty_cache()
    if rows:
        _report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
