"""In-kernel-decode grouped GEMM (exl3_gemm without ``decoded``) at 8192 tokens, Ornith shapes, per
projection: can a big-BM in-kernel tile beat reconstruct + decoded GEMM (the production path)?
Outputs must be bitwise the decoded path's. Usage: python sweep_inl_8k.py [gu|dn]"""
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
tr, svh, parts, xin = (gu_tr, gu_svh, gu, torch.randn(2, routes, H, device=DEV).half()) if which == "gu" else \
    (dn_tr, dn_svh, dn, torch.randn(1, routes, I, device=DEV).half())
odt = torch.float16 if which == "gu" else torch.float32
SORT = {bm: moe_align_block_size(ids2, bm, E) for bm in (32, 64, 128)}
common = dict(tr_expert_stride=tr.stride(0) // 2, svh_expert_stride=svh.stride(0))


def prod(out):
    cfg = fe._prefill_gemm(routes, E)
    s, e, n = SORT[cfg["block_m"]]
    w = torch.empty((G, parts.k, parts.n), dtype=torch.float16, device=DEV)
    for lo in range(0, E, G):
        k3.reconstruct_experts(tr, parts, lo, lo + G, out=w)
        k3.exl3_gemm(xin, tr, svh, parts, out=out, decoded=w, expert_range=(lo, lo + G), sorted_ids=s, expert_ids=e, num_post_pad=n, **common, **cfg)


def inl(cfg, out):
    s, e, n = SORT[cfg["block_m"]]
    k3.exl3_gemm(xin, tr, svh, parts, out=out, sorted_ids=s, expert_ids=e, num_post_pad=n, **common, **cfg)


def med(fn, reps=10):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


ref = torch.empty((routes, parts.n), dtype=odt, device=DEV)
prod(ref); torch.cuda.synchronize()
print(which, "production reconstruct+decoded", f"{med(lambda: prod(ref)):.3f} ms", flush=True)
for bm, bk, st, nw in itertools.product((64, 128), (16, 32, 64), (1, 2), (4, 8)):
    cfg = dict(block_m=bm, block_k=bk, num_stages=st, num_warps=nw)
    out = torch.empty_like(ref)
    try:
        t = med(lambda: inl(cfg, out))
    except Exception as ex:
        print(which, cfg, "fail", type(ex).__name__, flush=True); continue
    print(which, cfg, f"{t:.3f} ms", "equal" if torch.equal(out, ref) else "DIFF", flush=True)
