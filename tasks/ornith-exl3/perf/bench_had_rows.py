"""had_rows tile sweep on the Ornith MoE prefill shapes (8192 tokens x top-8 routes):
gate_up input (K=2048, 2 parts, src_div 8, fp32->fp16 from bf16 x) and down input (K=512, 1 part).

    PYTHONPATH=python python tasks/ornith-exl3/perf/bench_had_rows.py

Prints us and effective GB/s (bytes read + written, x counted once) per (block_m, num_warps, k_blocks).
"""
import torch
import triton.testing as tt

from freetoken.kernel.triton.exl3 import had_rows
from freetoken.moe.fused_exl3 import expert_parts

T, TOPK, E, H, I = 8192, 8, 256, 2048, 512
dev = torch.device("cuda")
gu, dn = expert_parts(H, I, 5, "mul1", dev)
P = T * TOPK
ids = torch.randint(0, E, (P,), dtype=torch.int32, device=dev)
x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
a = torch.randn(P, I, dtype=torch.float16, device=dev)
gsuh = torch.randn(E, 2, H, device=dev).half()
dsuh = torch.randn(E, I, device=dev).half()
cases = (("gate_up", lambda **k: had_rows(x, gsuh, gu, src_div=TOPK, experts=ids, suh_expert_stride=gsuh.stride(0), **k),
          T * H * 2 + 2 * P * H * 2),
         ("down", lambda **k: had_rows(a, dsuh, dn, experts=ids, suh_expert_stride=dsuh.stride(0), **k),
          2 * P * I * 2))
for name, fn, nbytes in cases:
    ref = fn(block_m=16, num_warps=4, k_blocks=1)
    for bm in (16, 32, 64, 128):
        for nw in (2, 4, 8):
            for kb in (1, 2, 4, 16):
                y = fn(block_m=bm, num_warps=nw, k_blocks=kb)
                assert torch.equal(y, ref), (name, bm, nw, kb)
                ms = tt.do_bench(lambda: fn(block_m=bm, num_warps=nw, k_blocks=kb), warmup=5, rep=50)
                print(f"{name:8s} bm {bm:4d} warps {nw} kb {kb:2d}: {ms * 1e3:8.1f} us  {nbytes / ms / 1e6:7.1f} GB/s", flush=True)
