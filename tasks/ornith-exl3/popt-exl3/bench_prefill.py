"""Median device time (CUDA events, 30 reps after warm-up) of the EXL3 prefill paths at Ornith shapes:
fused MoE prefill (256 experts, top-8, 5-bit mul1) at 8192 / 512 / 9 tokens and dense exl3_forward
(2048 -> 1536) at 8192 rows. Usage: python bench_prefill.py TAG >> out.tsv"""
import sys

import torch

import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
from freetoken.layers.quantization.linear.exl3 import exl3_forward
from freetoken.models.exl3_banks import exl3_bank_shapes

DEV = torch.device("cuda")
H, I, E, TOPK, BITS = 2048, 512, 256, 8, 5
torch.manual_seed(0)
B = []
for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
    if dtype == torch.int16:
        B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
    else:
        B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
B = tuple(B)


def med(fn, reps=30):
    fn(); fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


row = [sys.argv[1]]
for M in (8192, 512, 9):
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    row.append(f"moe{M}={med(lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook='mul1', is_prefill=True, num_experts=E)):.4f}")
parts = k3.Exl3Parts.build(2048, (1024, 512), BITS, "mul1", DEV)
tr = torch.randint(-(1 << 15), 1 << 15, (parts.words * 2,), dtype=torch.int32, device=DEV).to(torch.int16)
suh = (torch.randn(2 * 2048, device=DEV) * 0.5).half()
svh = (torch.randn(1536, device=DEV) * 0.05).half()
for rows in (8192, 64):
    x = torch.randn(rows, 2048, dtype=torch.bfloat16, device=DEV)
    row.append(f"dense{rows}={med(lambda: exl3_forward(x, tr, suh, svh, parts, torch.bfloat16)):.4f}")
print("\t".join(row), flush=True)
