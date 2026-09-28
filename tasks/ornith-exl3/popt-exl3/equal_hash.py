"""sha1 of every EXL3 kernel output that goes through ``_exl3_decode``, on seeded inputs, for a byte
comparison of two trees (run the same file with PYTHONPATH=<tree>/python; diff the outputs):

* dense ``exl3_forward`` at 1 / 4 rows (GEMV, control), 9 rows (exl3_gemm in-kernel decode),
  64 rows (reconstruct + cuBLAS), 2048 rows (reconstruct_folded), bits 2..8 x 3 codebooks;
* ``fused_experts_exl3`` prefill at 9 tokens (in-kernel decode GEMM), 512 and 8192 tokens
  (reconstruct_experts + decoded GEMM; again with the MoE input fold = reconstruct_folded), plus
  the f16acc GEMM variant, and decode (GEMV, control).
Usage: python equal_hash.py OUT.txt"""
import hashlib
import sys

import torch

import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
from freetoken.layers.quantization.linear.exl3 import exl3_forward
from freetoken.models.exl3_banks import exl3_bank_shapes

DEV = torch.device("cuda")
lines = []


def h(tag, t):
    torch.cuda.synchronize()
    lines.append(f"{tag} {hashlib.sha1(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}")


def dense(bits, cb):
    K, sizes = 2048, (1024, 512)
    g = torch.Generator(device="cpu").manual_seed(1000 * bits + len(cb))
    parts = k3.Exl3Parts.build(K, sizes, bits, cb, DEV)
    words = sum((K // 16) * (s // 16) * 16 * bits for s in sizes)
    tr = torch.randint(-(1 << 15), 1 << 15, (words,), generator=g, dtype=torch.int32).to(torch.int16).to(DEV)
    suh = (torch.randn(len(sizes) * K, generator=g) * 0.5).half().to(DEV)
    svh = (torch.randn(sum(sizes), generator=g) * 0.05).half().to(DEV)
    for rows in (1, 4, 9, 64, 2048):
        x = torch.randn(rows, K, generator=g).to(torch.bfloat16).to(DEV)
        h(f"dense b{bits} {cb} rows{rows}", exl3_forward(x, tr, suh, svh, parts, torch.bfloat16))


def moe(bits, cb, E, Ms):
    H, I, TOPK = 2048, 512, 8
    torch.manual_seed(bits * 7 + len(cb))
    B = []
    for name, (shape, dtype) in exl3_bank_shapes(H, I, bits).items():
        if dtype == torch.int16:
            B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
        else:
            B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
    B = tuple(B)
    for M in Ms:
        x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
        w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
        ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
        run = lambda pre: fe.fused_experts_exl3(x, B, w, ids, bits=bits, codebook=cb, is_prefill=pre, num_experts=E)
        if M == 1:
            h(f"moe b{bits} {cb} E{E} decode", run(False))
            continue
        for fold in (False, True):
            fe.PREFILL_FOLD_INPUT = fold
            h(f"moe b{bits} {cb} E{E} M{M} fold{int(fold)}", run(True))
        fe.PREFILL_FOLD_INPUT = False


for bits in range(2, 9):
    for cb in k3.CODEBOOKS:
        dense(bits, cb)
for bits in (2, 3, 4, 5, 6, 8):
    for cb in k3.CODEBOOKS:
        moe(bits, cb, 32, (1, 9, 512))
moe(5, "mul1", 256, (1, 9, 512, 8192))
k3.f16acc_enabled = fe.f16acc_enabled = lambda: True
moe(5, "mul1", 256, (512, 8192))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "hashes")
