"""Cost of the decoded GEMM's early-exit blocks: one launch whose expert_range holds no expert (every
block returns at once) vs one full group launch, 8192 tokens, Ornith gate_up / down shapes."""
import torch
import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
from freetoken.moe.fused import moe_align_block_size
DEV = torch.device("cuda")
H, I, E, TOPK, BITS, M, G = 2048, 512, 256, 8, 5, 8192, 32
torch.manual_seed(0)
gu, dn = fe.expert_parts(H, I, BITS, "mul1", DEV)
ids2 = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
routes = ids2.numel()
s, e, n = moe_align_block_size(ids2, 32, E)
def med(fn, reps=20):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]
for nm, parts, K in (("gu", gu, H), ("dn", dn, I)):
    tr = torch.zeros((E, parts.words * 2), dtype=torch.int16, device=DEV)
    svh = torch.zeros((E, parts.n), dtype=torch.float16, device=DEV)
    xin = torch.randn(parts.num_parts, routes, K, device=DEV).half()
    out = torch.empty((routes, parts.n), dtype=torch.float16, device=DEV)
    w = torch.randn((G, parts.k, parts.n), device=DEV).half()
    cfg = dict(block_m=32, block_k=64, num_stages=3, num_warps=8)
    f = lambda lo, hi: k3.exl3_gemm(xin, tr, svh, parts, out=out, decoded=w, expert_range=(lo, hi), tr_expert_stride=tr.stride(0) // 2,
                                    svh_expert_stride=svh.stride(0), sorted_ids=s, expert_ids=e, num_post_pad=n, **cfg)
    print(nm, f"empty launch {med(lambda: f(0, 0)) * 1000:.1f} us, one group {med(lambda: f(0, G)) * 1000:.1f} us")
