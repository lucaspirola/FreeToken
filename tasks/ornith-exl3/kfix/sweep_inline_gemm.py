"""Launcher configs of the in-kernel-decode grouped GEMM (fused MoE prefill below
PREFILL_DECODED_MIN_ROUTES_PER_EXPERT) at Ornith shapes: block_k / num_warps / num_stages keep every output's
16-wide mma k-step sequence, so outputs must be bitwise equal to the shipped config. Median ms of the
whole fused prefill (CUDA events) per config and token count."""
import functools
import itertools

import torch

import freetoken.kernel.triton.exl3 as k3
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
orig = k3.exl3_gemm
for M in (9, 64, 200):
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    run = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)
    fe.exl3_gemm = orig
    ref = run()
    print(f"M{M} shipped (bk32 w4 s2): {med(run, 10):.4f}", flush=True)
    for bk, nw, ns in itertools.product((16, 32, 64, 128), (4, 8), (1, 2, 3)):
        fe.exl3_gemm = functools.partial(orig, block_k=bk, num_warps=nw, num_stages=ns)
        try:
            y = run(); torch.cuda.synchronize()
        except Exception as ex:
            print(f"M{M} bk{bk} w{nw} s{ns}: ERR {type(ex).__name__} {str(ex)[:80]}", flush=True)
            continue
        eq = torch.equal(y.view(torch.int16), ref.view(torch.int16))
        print(f"M{M} bk{bk} w{nw} s{ns}: {med(run, 10):.4f} {'EQ' if eq else 'DIFF'}", flush=True)
