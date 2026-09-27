"""torch.profiler per-kernel CUDA time of one fused MoE prefill (Ornith shapes, 5-bit mul1) at M tokens.
Usage: python prof_prefill.py M"""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes

DEV = torch.device("cuda")
H, I, E, TOPK, BITS, M = 2048, 512, 256, 8, 5, int(sys.argv[1])
torch.manual_seed(0)
B = []
for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
    if dtype == torch.int16:
        B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
    else:
        B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
B = tuple(B)
x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
run = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)
run(); run(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(5):
        run()
    torch.cuda.synchronize()
print(f"M={M} (5 calls)")
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))
