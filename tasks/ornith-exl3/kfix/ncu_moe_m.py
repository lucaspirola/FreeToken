"""One production routed-expert call (argv: M tokens, prefill|decode) (Ornith EXL3 shapes: 256 experts, top-8, H 2048, I 512, 5 bit,
one token, uniform random routing, default env: fused epilogues + GEMV pre-rotation) inside an NVTX range:
    ncu --nvtx --nvtx-include "moe/" ... python ncu_decode.py
Two warm-up calls first (triton compile), then the measured call."""
import sys

import torch

import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes

H, I, E, TOPK, BITS = 2048, 512, 256, 8, 5
M, PRE = int(sys.argv[1]), sys.argv[2] == "prefill"
DEV = torch.device("cuda")
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
run = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=PRE, num_experts=E)
run(); run(); torch.cuda.synchronize()
torch.cuda.nvtx.range_push("moe")
run()
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("done")
