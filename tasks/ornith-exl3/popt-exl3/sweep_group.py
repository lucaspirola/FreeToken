"""Fused MoE prefill (8192 tokens, Ornith shapes) vs PREFILL_DECODE_GROUP: a smaller decoded scratch
may stay L2-resident between reconstruct and GEMM. Outputs bitwise vs group 32. Interleaved reps."""
import torch
import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes

DEV = torch.device("cuda")
H, I, E, TOPK, BITS, M = 2048, 512, 256, 8, 5, 8192
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
fe.PREFILL_DECODE_GROUP = 32
ref = run(); torch.cuda.synchronize()
GS = (4, 8, 12, 16, 24, 32, 64)
ts = {g: [] for g in GS}
for rep in range(12):
    for g in GS:
        fe.PREFILL_DECODE_GROUP = g
        run(); torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); o = run(); b.record(); b.synchronize(); ts[g].append(a.elapsed_time(b))
        assert torch.equal(o, ref), g
for g in GS:
    v = sorted(ts[g]); print(f"group {g:3d}: median {v[len(v)//2]:.3f} ms  min {v[0]:.3f}")
