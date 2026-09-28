"""Is a Triton tl.dot GEMM (fp16 in, fp32 accumulate over sequential 16-wide k steps, one rounding to
fp16) bitwise torch.mm (cuBLAS) on the dense EXL3 folded shapes? And how fast, with the bf16 -> fp16
input cast in the prologue and the fp16 -> bf16 output cast in the epilogue (both fused)?"""
import itertools
import torch, triton, triton.language as tl


@triton.jit
def _mm_kernel(x_ptr, w_ptr, o_ptr, M, N, K, sxm, swk, som, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP: tl.constexpr, OUT_BF16: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    gid = pid // (GROUP * npn); fm = gid * GROUP; gs = min(npm - fm, GROUP)
    pm = fm + (pid % (GROUP * npn)) % gs; pn = (pid % (GROUP * npn)) // gs
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + rm[:, None].to(tl.int64) * sxm + rk[None, :], mask=rm[:, None] < M, other=0.0).to(tl.float16)
        b = tl.load(w_ptr + rk[:, None].to(tl.int64) * swk + rn[None, :])
        acc = tl.dot(a, b, acc)
    y = acc.to(tl.float16)
    if OUT_BF16:
        y = y.to(tl.bfloat16)
    tl.store(o_ptr + rm[:, None].to(tl.int64) * som + rn[None, :], y, mask=rm[:, None] < M)


def mm(x, w, out, BM, BN, BK, st, nw, G=8):
    M, K = x.shape; N = w.shape[1]
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _mm_kernel[grid](x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0), BM=BM, BN=BN, BK=BK, GROUP=G,
                     OUT_BF16=out.dtype == torch.bfloat16, num_stages=st, num_warps=nw)
    return out


def med(fn, reps=20):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


if __name__ == "__main__":
    torch.manual_seed(0)
    for M, K, N in ((8192, 2048, 2048), (8192, 4096, 2048), (8192, 2048, 1024), (8192, 512, 2048), (3000, 2048, 2048)):
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        w = (torch.randn(K, N, device="cuda") * 0.03).half()
        xf = x.to(torch.float16)
        ref16 = torch.mm(xf, w)
        ref = ref16.to(torch.bfloat16)
        t_ref = med(lambda: torch.mm(x.to(torch.float16), w).to(torch.bfloat16))
        t_mm = med(lambda: torch.mm(xf, w))
        best = None
        for BM, BN, BK, st, nw in itertools.product((128, 64), (128, 256, 64), (32, 64), (3, 4, 5), (4, 8)):
            out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            try:
                t = med(lambda: mm(x, w, out, BM, BN, BK, st, nw), reps=10)
            except Exception:
                continue
            eq = torch.equal(out.view(torch.int16), ref.view(torch.int16))
            if best is None or (eq and t < best[0]) or (not best[2] and eq):
                best = (t, (BM, BN, BK, st, nw), eq)
            if not eq:
                print(f"  DIFF {M}x{K}x{N} cfg {(BM, BN, BK, st, nw)} maxdiff {(out.float() - ref.float()).abs().max().item():.3g}", flush=True)
        print(f"M{M} K{K} N{N}: cast+cuBLAS+cast {t_ref:.3f} ms (cuBLAS alone {t_mm:.3f}); best fused Triton {best[0]:.3f} ms {best[1]} equal={best[2]}", flush=True)
