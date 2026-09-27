"""Mid-size prefill (the e2e 300 / 1000-token probes) at Ornith shapes: fused MoE prefill with uniform
and skewed (Zipf s=1.2 over experts, like repetitive text) routing, and had_rows alone at the gate_up
input of 8136 routes under the shipped launcher. Run on each tree (rerun-trees.sh)."""
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
zipf = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** 1.2
perm = torch.randperm(E)
for M in (300, 1017):
    for kind in ("uniform", "zipf"):
        x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
        w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
        if kind == "uniform":
            ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
        else:
            ids = torch.stack([perm[torch.multinomial(zipf, TOPK, replacement=False)] for _ in range(M)]).to(torch.int32).to(DEV)
        run = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)
        n_exp = ids.unique().numel()
        print(f"moe M{M} {kind:7s} experts={n_exp:3d} max_routes={int(torch.bincount(ids.flatten().long(), minlength=E).max())}: {med(run, 20):.4f} ms", flush=True)
for rows in (2400, 8136, 16384):
    parts = k3.Exl3Parts.build(2048, (1024,), BITS, "mul1", DEV)
    x = torch.randn(rows // TOPK, 2048, device=DEV).half()
    suh = (torch.randn(E, 2048, device=DEV) * 0.5).half()
    ids = torch.randint(0, E, (rows,), device=DEV, dtype=torch.int32)
    print(f"had_rows gate_up rows={rows}: {med(lambda: k3.had_rows(x, suh, parts, src_div=TOPK, experts=ids, suh_expert_stride=suh.stride(0)), 30):.4f} ms", flush=True)
