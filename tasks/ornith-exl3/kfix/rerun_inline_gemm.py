"""Interleaved rounds of the best in-kernel-decode GEMM launcher configs (sweep_inline_gemm.py) at
9 / 64 / 200 tokens, fused MoE prefill, plus the dense exl3_forward at 12 rows (the dense in-kernel
GEMM range). Run on each tree; each config's output must be bitwise the shipped one."""
import functools

import torch

import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
import freetoken.layers.quantization.linear.exl3 as lx
from freetoken.models.exl3_banks import exl3_bank_shapes
from sweep_had_rows import med

DEV = torch.device("cuda")
H, I, E, TOPK, BITS, ROUNDS = 2048, 512, 256, 8, 5, 3
torch.manual_seed(0)
B = []
for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
    if dtype == torch.int16:
        B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
    else:
        B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
B = tuple(B)
orig = k3.exl3_gemm
CFGS = [None, (16, 4, 1), (32, 4, 1), (32, 8, 1), (16, 8, 1)]
parts = k3.Exl3Parts.build(2048, (4096, 1024, 1024), BITS, "mul1", DEV)
tr = torch.randint(-(1 << 15), 1 << 15, (parts.words * 2,), dtype=torch.int32, device=DEV).to(torch.int16)
suh = (torch.randn(3 * 2048, device=DEV) * 0.5).half()
svh = (torch.randn(parts.n, device=DEV) * 0.05).half()
cases = []
for M in (9, 64, 200):
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    cases.append((f"moe{M}", lambda x=x, w=w, ids=ids: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)))
xd = torch.randn(12, 2048, dtype=torch.bfloat16, device=DEV)
cases.append(("dense12x6144", lambda: lx.exl3_forward(xd, tr, suh, svh, parts, torch.bfloat16)))
for name, run in cases:
    fe.exl3_gemm = k3.exl3_gemm = orig
    ref = run()
    ts = {c: [] for c in CFGS}
    for r in range(ROUNDS):
        for c in CFGS:
            f = orig if c is None else functools.partial(orig, block_k=c[0], num_warps=c[1], num_stages=c[2])
            fe.exl3_gemm = k3.exl3_gemm = f
            y = run()
            assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), (name, c)
            ts[c].append(med(run, 10))
    for c, t in ts.items():
        print(f"{name:14s} {'shipped' if c is None else 'bk%d w%d s%d' % c:12s} median {sorted(t)[ROUNDS // 2]:.4f}  " + " ".join(f"{v:.3f}" for v in t), flush=True)
