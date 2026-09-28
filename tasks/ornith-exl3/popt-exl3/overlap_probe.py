"""Can _reconstruct_experts_kernel (ALU/issue-bound) hide under the decoded GEMM (tensor-bound)?
Times, at Ornith shapes (8192 tokens, 256 experts, top-8, 5-bit mul1), the gate_up half of the
decoded prefill: (a) serial reconstruct+GEMM per group (production), (b) reconstruct of group g+1 on a
side stream while the GEMM of group g runs (double-buffered scratch), for group sizes 8/16/32, and
checks (b)'s output is bitwise (a)'s."""
import torch
import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
from freetoken.moe.fused import moe_align_block_size
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
gu_tr, gu_suh, gu_svh, dn_tr, dn_suh, dn_svh = B
gu, dn = fe.expert_parts(H, I, BITS, "mul1", DEV)
x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
ids2 = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
ids = ids2.reshape(-1).contiguous()
routes = ids.numel()
cfg = fe._prefill_gemm(routes, E)
sorted_ids, expert_ids, npad = moe_align_block_size(ids2, cfg["block_m"], E)
sort = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_post_pad=npad, **cfg)
xh = k3.had_rows(x, gu_suh, gu, src_div=TOPK, experts=ids, suh_expert_stride=gu_suh.stride(0))
gu_args = dict(tr_expert_stride=gu_tr.stride(0) // 2, svh_expert_stride=gu_svh.stride(0), **sort)
side = torch.cuda.Stream()


def serial(group, out):
    w = torch.empty((group, gu.k, gu.n), dtype=torch.float16, device=DEV)
    for lo in range(0, E, group):
        k3.reconstruct_experts(gu_tr, gu, lo, lo + group, out=w)
        k3.exl3_gemm(xh, gu_tr, gu_svh, gu, out=out, decoded=w, expert_range=(lo, lo + group), **gu_args)


def overlapped(group, out):
    ws = [torch.empty((group, gu.k, gu.n), dtype=torch.float16, device=DEV) for _ in range(2)]
    main = torch.cuda.current_stream()
    ready = [torch.cuda.Event() for _ in range(2)]
    free = [torch.cuda.Event() for _ in range(2)]
    n = E // group
    side.wait_stream(main)
    with torch.cuda.stream(side):
        k3.reconstruct_experts(gu_tr, gu, 0, group, out=ws[0]); ready[0].record(side)
    for i in range(n):
        if i + 1 < n:
            with torch.cuda.stream(side):
                if i >= 1:
                    side.wait_event(free[(i + 1) % 2])
                lo = (i + 1) * group
                k3.reconstruct_experts(gu_tr, gu, lo, lo + group, out=ws[(i + 1) % 2]); ready[(i + 1) % 2].record(side)
        main.wait_event(ready[i % 2])
        lo = i * group
        k3.exl3_gemm(xh, gu_tr, gu_svh, gu, out=out, decoded=ws[i % 2], expert_range=(lo, lo + group), **gu_args)
        free[i % 2].record(main)


def med(fn, reps=20):
    fn(); fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


ref = torch.empty((routes, gu.n), dtype=torch.float16, device=DEV)
serial(32, ref)
w = torch.empty((32, gu.k, gu.n), dtype=torch.float16, device=DEV)
k3.reconstruct_experts(gu_tr, gu, 0, 32, out=w)
print(f"gemm-only 32 experts x8 (no reconstruct): {8 * med(lambda: k3.exl3_gemm(xh, gu_tr, gu_svh, gu, out=ref.clone(), decoded=w, expert_range=(0, 32), **gu_args)):.3f} ms")
print(f"reconstruct-only 32 x8: {8 * med(lambda: k3.reconstruct_experts(gu_tr, gu, 0, 32, out=w)):.3f} ms")
for group in (8, 16, 32):
    o1 = torch.empty_like(ref); o2 = torch.empty_like(ref)
    t1 = med(lambda: serial(group, o1)); t2 = med(lambda: overlapped(group, o2))
    serial(group, o1); overlapped(group, o2); torch.cuda.synchronize()
    print(f"group {group}: serial {t1:.3f} ms  overlapped {t2:.3f} ms  equal={torch.equal(o1, ref) and torch.equal(o2, ref)}")
