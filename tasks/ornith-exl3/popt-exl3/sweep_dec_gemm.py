"""Tile sweep of the decoded-slab grouped GEMM (exl3_gemm with ``decoded``) at Ornith shapes,
8192 tokens, per projection, W_hat pre-decoded per 32-expert group (reconstruct not timed).
Every config's output must be bitwise the production config's (tiles keep the k16 step order).
Usage: python sweep_dec_gemm.py [gu|dn]"""
import itertools, sys
import torch
import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
from freetoken.moe.fused import moe_align_block_size
from freetoken.models.exl3_banks import exl3_bank_shapes

DEV = torch.device("cuda")
H, I, E, TOPK, BITS, M, G = 2048, 512, 256, 8, 5, 8192, 32
torch.manual_seed(0)
B = []
for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
    if dtype == torch.int16:
        B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
    else:
        B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
gu_tr, gu_suh, gu_svh, dn_tr, dn_suh, dn_svh = B
gu, dn = fe.expert_parts(H, I, BITS, "mul1", DEV)
ids2 = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
ids = ids2.reshape(-1).contiguous(); routes = ids.numel()
which = sys.argv[1] if len(sys.argv) > 1 else "gu"
if which == "gu":
    tr, svh, parts = gu_tr, gu_svh, gu
    xin = torch.randn(2, routes, H, device=DEV).half()
else:
    tr, svh, parts = dn_tr, dn_svh, dn
    xin = torch.randn(1, routes, I, device=DEV).half()
Ws = [k3.reconstruct_experts(tr, parts, lo, lo + G) for lo in range(0, E, G)]
odt = torch.float16 if which == "gu" else torch.float32


def run(cfg, out):
    sorted_ids, expert_ids, npad = SORT[cfg["block_m"]]
    for i, lo in enumerate(range(0, E, G)):
        k3.exl3_gemm(xin, tr, svh, parts, out=out, decoded=Ws[i], expert_range=(lo, lo + G),
                     tr_expert_stride=tr.stride(0) // 2, svh_expert_stride=svh.stride(0),
                     sorted_ids=sorted_ids, expert_ids=expert_ids, num_post_pad=npad, **cfg)


SORT = {bm: moe_align_block_size(ids2, bm, E) for bm in (16, 32, 64, 128)}


def med(fn, reps=15):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


prod = fe._prefill_gemm(routes, E)
ref = torch.empty((routes, parts.n), dtype=odt, device=DEV)
run(prod, ref); torch.cuda.synchronize()
print(which, "prod", prod, f"{med(lambda: run(prod, ref)):.3f} ms", flush=True)
res = []
for bm, bk, st, nw in itertools.product((32, 64, 128), (32, 64, 128), (2, 3, 4), (4, 8)):
    cfg = dict(block_m=bm, block_k=bk, num_stages=st, num_warps=nw)
    out = torch.empty_like(ref)
    try:
        t = med(lambda: run(cfg, out))
    except Exception as e:  # out of shared memory etc.
        print(which, cfg, "fail", type(e).__name__, flush=True); continue
    eq = torch.equal(out, ref)
    res.append((t, cfg, eq)); print(which, cfg, f"{t:.3f} ms", "equal" if eq else "DIFF", flush=True)
res.sort(key=lambda r: r[0])
print("best:", *[f"{t:.3f} {c} {e}" for t, c, e in res[:6]], sep="\n")
