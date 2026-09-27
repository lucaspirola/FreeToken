"""Fused MoE prefill: in-kernel-decode GEMM vs decoded-scratch path (PREFILL_DECODED_MIN_TOKENS) at
Ornith shapes, per token count: median ms of each and whether the outputs are bitwise equal (both
feed the same W_hat values to 16-wide mma k-steps in the same order). Also with the MoE input fold."""
import torch

import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes
from sweep_had_rows import med

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
import sys
MS = tuple(int(v) for v in sys.argv[1].split(',')) if len(sys.argv) > 1 else (16, 128, 256, 384, 512, 768, 1024, 2048, 4096)
BM32 = len(sys.argv) > 2
if BM32:
    fe._prefill_block_m = lambda routes, num_experts: 16 if routes / num_experts < 16 else 32
for M in MS:
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    run = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)
    fe.PREFILL_DECODED_MIN_TOKENS = 1 << 30
    yi = run(); ti = med(run, 10)
    fe.PREFILL_DECODED_MIN_TOKENS = 1
    yd = run(); td = med(run, 10)
    yf, tf = yd, float('nan')
    print(f"M{M:5d} inline {ti:8.4f}  decoded {td:8.4f} {'EQ' if torch.equal(yi.view(torch.int16), yd.view(torch.int16)) else 'DIFF'}"
          f"  decoded+fold {tf:8.4f} {'EQ' if torch.equal(yi.view(torch.int16), yf.view(torch.int16)) else 'DIFF'}", flush=True)
