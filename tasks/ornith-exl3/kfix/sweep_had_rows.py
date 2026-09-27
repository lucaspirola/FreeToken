"""Bit-exact tiling variants of _had_rows_kernel at the MoE prefill shapes (8192 tokens x top-8: gate_up
input 65536 x 2048 from 8192 source rows, down input 65536 x 512). Each variant keeps every output
element's K=128 dot unchanged (same fp16 hi/lo operands, same H, separate hi and lo dots, one fp32
add), so outputs must be bitwise equal to had_rows; only M tiling, warps, the number of N slices of H
and stacking [hi; lo] into one M=2BM dot vary. Prints median ms (CUDA events) per variant."""
import itertools
import sys

import torch
import triton
import triton.language as tl

import freetoken.kernel.triton.exl3 as k3


@triton.jit
def _had_rows_v(
    x_ptr, stride_xm, out_ptr, stride_opart, stride_om, suh_ptr, suh_expert_stride, suh_part_stride,
    expert_ptr, had_ptr, P,
    SRC_DIV: tl.constexpr, HAS_EXPERT: tl.constexpr, BM: tl.constexpr, KB: tl.constexpr,
    NS: tl.constexpr, STACK: tl.constexpr, HOIST: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    part = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < P
    src = rows // SRC_DIV
    if HAS_EXPERT:
        e = tl.load(expert_ptr + rows, mask=rmask, other=0).to(tl.int64)
    else:
        e = tl.zeros([BM], dtype=tl.int64)
    idx = tl.arange(0, 128)
    NW: tl.constexpr = 128 // NS
    jn = tl.arange(0, NW)
    if HOIST:  # NS == 2: both H column halves loaded once per program
        h0 = tl.load(had_ptr + idx[:, None] * 128 + jn[None, :])
        h1 = tl.load(had_ptr + idx[:, None] * 128 + NW + jn[None, :])
    for i in tl.static_range(KB):
        cols = (pid_k * KB + i) * 128 + idx
        x = tl.load(x_ptr + src[:, None].to(tl.int64) * stride_xm + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        s = tl.load(suh_ptr + e[:, None] * suh_expert_stride + part * suh_part_stride + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        xs = x * s
        hi = xs.to(tl.float16)
        lo = (xs - hi.to(tl.float32)).to(tl.float16)
        if STACK:
            both = tl.reshape(tl.permute(tl.join(hi, lo), (2, 0, 1)), (2 * BM, 128))
        for n in tl.static_range(NS):
            if HOIST:
                h = h0 if n == 0 else h1
            else:
                h = tl.load(had_ptr + idx[:, None] * 128 + n * NW + jn[None, :])
            if STACK:
                d = tl.reshape(tl.dot(both, h), (2, BM, NW))
                dh, dl = tl.split(tl.permute(d, (1, 2, 0)))
                y = (dh + dl) * 0.08838834764831843
            else:
                y = (tl.dot(hi, h) + tl.dot(lo, h)) * 0.08838834764831843
            oc = (pid_k * KB + i) * 128 + n * NW + jn
            tl.store(out_ptr + part * stride_opart + rows[:, None].to(tl.int64) * stride_om + oc[None, :], y.to(tl.float16), mask=rmask[:, None])


def run_v(x, suh, parts, ids, src_div, out, bm, nw, kb, ns, stack, hoist=False):
    k = parts.k
    while (k // 128) % kb:
        kb //= 2
    rows = out.shape[1]
    grid = (triton.cdiv(rows, bm), k // 128 // kb, parts.num_parts)
    _had_rows_v[grid](x, x.stride(0), out, out.stride(0), out.stride(1), suh, suh.stride(0), parts.suh_part_stride,
                      ids, k3.hadamard_pm1(x.device), rows, SRC_DIV=src_div, HAS_EXPERT=True, BM=bm, KB=kb,
                      NS=ns, STACK=stack, HOIST=hoist, num_warps=nw)


def med(fn, reps=20):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


def main():
    DEV = torch.device("cuda")
    torch.manual_seed(0)
    T, TOPK, E = 8192, 8, 256
    ids = torch.randint(0, E, (T * TOPK,), device=DEV, dtype=torch.int32)
    cases = []
    for K, N, src_div, xrows in ((2048, 1024, TOPK, T), (512, 2048, 1, T * TOPK)):
        parts = k3.Exl3Parts.build(K, (N,), 5, "mul1", DEV)
        x = torch.randn(xrows, K, device=DEV).half()
        suh = (torch.randn(E, K, device=DEV) * 0.5).half()
        ref = k3.had_rows(x, suh, parts, src_div=src_div, experts=ids, suh_expert_stride=suh.stride(0))
        base = med(lambda: k3.had_rows(x, suh, parts, src_div=src_div, experts=ids, suh_expert_stride=suh.stride(0), out=ref))
        print(f"K{K} base bm{k3.HAD_ROWS_BM} w{k3.HAD_ROWS_WARPS} kb{k3.HAD_ROWS_KB}: {base:.4f} ms", flush=True)
        cases.append((K, parts, x, suh, src_div, ref))
    grid = list(itertools.product((16, 32, 64), (4, 8), (1, 2, 4), (False, True), (16, 4)))
    for bm, nw, ns, stack, kb in grid:
        res = []
        for K, parts, x, suh, src_div, ref in cases:
            out = torch.empty_like(ref)
            try:
                run_v(x, suh, parts, ids, src_div, out, bm, nw, kb, ns, stack)
                torch.cuda.synchronize()
            except Exception as ex:  # compile/resource failures: report and go on
                res.append(f"K{K} ERR {type(ex).__name__}")
                continue
            eq = torch.equal(out.view(torch.int16), ref.view(torch.int16))
            t = med(lambda: run_v(x, suh, parts, ids, src_div, out, bm, nw, kb, ns, stack))
            res.append(f"K{K} {t:.4f} {'EQ' if eq else 'DIFF'}")
        print(f"bm{bm} w{nw} ns{ns} stack{int(stack)} kb{kb}: " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()
