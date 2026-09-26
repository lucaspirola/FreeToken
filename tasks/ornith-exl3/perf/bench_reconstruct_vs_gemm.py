"""Is EXL3 prefill decode-bound? Time exl3_gemm against reconstruct-once + cuBLAS fp16 on Ornith shapes.

    python bench_reconstruct_vs_gemm.py            (random trellis: decode cost does not depend on content)
"""
import torch
import triton.testing as tt

from freetoken.kernel.triton.exl3 import Exl3Parts, exl3_gemm, reconstruct

torch.manual_seed(0)
dev = "cuda"
BITS, CB = 5, "mul1"


def trellis(k, n):
    return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * BITS), dtype=torch.int16, device=dev)


def dense(m, k, n, label):
    tr = trellis(k, n)
    parts = Exl3Parts.build(k, (n,), BITS, CB, dev)
    xh = torch.randn(1, m, k, device=dev, dtype=torch.float16)
    svh = torch.ones(n, device=dev, dtype=torch.float16)
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    bm = 16 if m <= 32 else (32 if m <= 128 else 64)
    t_gemm = tt.do_bench(lambda: exl3_gemm(xh, tr.reshape(-1), svh, parts, out=out, block_m=bm), rep=200)
    t_rec = tt.do_bench(lambda: reconstruct(tr, CB), rep=200)
    w = reconstruct(tr, CB)
    t_mm = tt.do_bench(lambda: torch.matmul(xh[0], w), rep=200)
    fl = 2 * m * k * n
    print(f"{label:28s} M={m:5d} K={k:5d} N={n:6d}  exl3_gemm {t_gemm:7.2f} ms ({fl / t_gemm / 1e9:6.1f} TF/s)  "
          f"reconstruct {t_rec:6.2f} ms + cuBLAS fp16 {t_mm:6.2f} ms = {t_rec + t_mm:6.2f} ms  x{t_gemm / (t_rec + t_mm):.1f}")


def experts(n_exp, k, n, label):
    tr = trellis(k, n)
    t_rec = tt.do_bench(lambda: reconstruct(tr, CB), rep=100)
    print(f"{label:28s} reconstruct one expert [{k}x{n}] {t_rec * 1e3:7.1f} us -> {n_exp} experts {t_rec * n_exp:6.2f} ms")


for m in (1024, 8000):
    dense(m, 2048, 12288, "GDN in_proj (30 layers)")
    dense(m, 2048, 9216, "attn qkv+gate (10 layers)")
    dense(m, 4096, 2048, "attn o_proj / gdn out")
    dense(m, 2048, 1024, "shared expert gate_up")
experts(256, 2048, 1024, "MoE gate_up per layer")
experts(256, 512, 2048, "MoE down per layer")
