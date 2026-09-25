"""Per-kernel breakdown of the production decode-attention call (stage 1 vs stage-2 merge).

Builds a synthetic q8_0 (or other) pool with ``bench_decode_launch._build_pool`` and calls
``decode_paged_attention`` exactly as the Triton backend does -- its own launch config, no
override -- under ``torch.profiler``. Reports, per context length, the median wall time
of the whole call (CUDA events), the per-kernel device time (stage 1 / stage 2 / other)
and the KV bytes/s stage 1 achieves. No server, no weights.

    PYTHONPATH=python python benchmarks/bench_decode_attn_breakdown.py \
        --q-heads 32 --kv-heads 2 --head-dim 128 --quant q8_0 \
        --ctx-lens 8192 81920 262144 1048576 --layers 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--quant", default="q8_0")
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[8192, 81920, 262144, 1048576])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--json", dest="json_out", default=None)
    args = p.parse_args(argv)

    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_decode_launch as bdl
    from freetoken.kernel.triton import attention as attn

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    k_spec, v_spec = bdl._quant_specs(args.quant)
    out = open(args.json_out, "a") if args.json_out else None
    for ctx in args.ctx_lens:
        slots = ctx * args.batch
        kc, ks, vc, vs, _, _ = bdl._build_pool(
            slots, args.kv_heads, args.head_dim, device, 0, k_spec, v_spec, False
        )
        g = torch.Generator(device=device).manual_seed(7)
        q = torch.randn(args.batch, args.q_heads, args.head_dim, generator=g,
                        device=device, dtype=torch.bfloat16)
        indptr = torch.arange(0, slots + 1, ctx, device=device, dtype=torch.int32)
        indices = torch.arange(slots, device=device, dtype=torch.int32)
        q_pos = torch.full((args.batch,), ctx - 1, device=device, dtype=torch.int32)
        cap = torch.cuda.get_device_capability(device)
        quant_name = {"q8_0": "q8_0"}.get(args.quant, args.quant)
        splits, bn, warps = attn.decode_launch_config(
            quant_name=quant_name if args.quant != "bf16" else None,
            head_dim=args.head_dim, num_q_heads=args.q_heads, num_kv_heads=args.kv_heads,
            compute_capability=cap, sm_count=attn._sm_count(0),
        )
        logits = torch.empty(args.batch, args.q_heads, splits, args.head_dim,
                             device=device, dtype=torch.float32)
        lse = torch.empty(args.batch, args.q_heads, splits, device=device, dtype=torch.float32)
        ns = torch.full((args.batch,), splits, device=device, dtype=torch.int32)

        def call():
            return attn.decode_paged_attention(
                q, kc, vc, indptr, indices, q_pos, logits, lse, ns, splits,
                args.head_dim ** -0.5, k_scale=ks, v_scale=vs,
            )

        for _ in range(5):
            call()
        torch.cuda.synchronize()
        wall = []
        for _ in range(args.iters):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); call(); b.record(); b.synchronize()
            wall.append(a.elapsed_time(b))
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                call()
            torch.cuda.synchronize()
        agg = defaultdict(float)
        for ev in prof.events():
            if ev.device_type.name == "CUDA":
                agg[ev.name] += ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
        per = {k: v / args.iters for k, v in agg.items()}  # us per call
        s1 = sum(v for k, v in per.items() if "stage1" in k)
        s2 = sum(v for k, v in per.items() if "stage2" in k)
        kv_bytes = slots * args.kv_heads * args.head_dim * (
            k_spec.bytes_per_element(torch.bfloat16) + v_spec.bytes_per_element(torch.bfloat16))
        row = {
            "ctx": ctx, "batch": args.batch, "quant": args.quant,
            "geom": f"{args.q_heads}Q/{args.kv_heads}KV/D{args.head_dim}",
            "launch": [splits, bn, warps], "wall_ms_median": statistics.median(wall),
            "stage1_us": s1, "stage2_us": s2,
            "other_us": sum(per.values()) - s1 - s2,
            "stage1_GBps": kv_bytes / (s1 * 1e-6) / 1e9 if s1 else None,
            "kv_MB": kv_bytes / 1e6,
            "per_token_ms_all_layers": statistics.median(wall) * args.layers,
            "kernels": {k[:80]: round(v, 2) for k, v in per.items()},
        }
        print(json.dumps(row), flush=True)
        if out:
            out.write(json.dumps(row) + "\n"); out.flush()
        del kc, ks, vc, vs, logits, lse
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
