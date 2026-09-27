"""Bitstream-order GEMV (_exl3_gemv_kernel) on Ornith shapes: warps x split sweep, and what the launcher picks."""
import torch
import triton.testing as tt
from freetoken.kernel.triton import exl3 as K
from freetoken.layers.quantization.linear.exl3 import pick_split_k


def launch(kern, xh, dst, tr, svh, parts, ids, k, split, nw, bits, cb, **kw):
    kern[(xh.shape[1], parts.n // 128, split)](
        xh, xh.stride(0), xh.stride(1), dst, dst.stride(-2), dst.stride(0), tr, tr.stride(0), svh, svh.stride(0),
        K.hadamard_pm1(xh.device), parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart,
        ids if ids is not None else parts.part_of_nb, k // split, BITS=bits, CB=K.CODEBOOKS[cb],
        HAS_EXPERT=ids is not None, num_warps=nw, **kw)


for label, k, sizes, rows, E in (("GDN in_proj", 2048, (12288,), 1, 0), ("attn q+gate|k|v", 2048, (8192, 512, 512), 1, 0),
                                 ("o_proj/out_proj", 4096, (2048,), 1, 0), ("shared gate|up", 2048, (512, 512), 1, 0),
                                 ("shared down", 512, (2048,), 1, 0), ("MoE gate|up x8", 2048, (512, 512), 8, 256),
                                 ("MoE down x8", 512, (2048,), 8, 256)):
    parts = K.Exl3Parts.build(k, sizes, 5, "mul1", "cuda")
    words = sum((k // 16) * (s // 16) * 40 for s in sizes)
    tr = torch.randint(-(1 << 31), (1 << 31) - 1, (max(E, 1), words), dtype=torch.int32, device="cuda")
    svh = torch.ones(max(E, 1), parts.n, dtype=torch.float16, device="cuda")
    xh = torch.randn(parts.num_parts, rows, k, device="cuda", dtype=torch.float16)
    ids = torch.randperm(E, device="cuda")[:rows].to(torch.int32) if E else None
    wb = (rows if E else 1) * words * 4
    res = []
    for nw in (1, 2, 4):
        for split in (1, 2, 4, 8, 16, 32):
            if k // split < 16:
                continue
            dst = torch.empty((split, rows, parts.n), dtype=torch.float32, device="cuda")
            res.append((tt.do_bench(lambda: launch(K._exl3_gemv_kernel, xh, dst, tr, svh, parts, ids, k, split, nw, 5, "mul1"), rep=100) * 1e3, nw, split))
    res.sort()
    pick = pick_split_k(rows, parts.n // 128, k, torch.device("cuda"))
    dst = torch.empty((pick, rows, parts.n), dtype=torch.float32, device="cuda")
    tp = tt.do_bench(lambda: launch(K._exl3_gemv_kernel, xh, dst, tr, svh, parts, ids, k, pick, 1, 5, "mul1"), rep=100) * 1e3
    print(f"{label:18s} launcher split {pick}: {tp:.1f}us ({wb / tp / 1e3:.0f} GB/s)")
    print(f"{label:18s} {wb / 1e6:6.2f} MB  v4 " + " | ".join(f"{t:.1f}us ({wb / t / 1e3:.0f} GB/s) w{a} s{b}" for t, a, b in res[:3]))
