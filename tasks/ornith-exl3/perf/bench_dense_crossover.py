"""Row count where reconstruct + cuBLAS beats the tl.dot GEMM for a dense EXL3 linear (sets
RECONSTRUCT_MIN_ROWS). Ornith shapes, 5-bit mul1, random trellis."""
import torch
import triton.testing as tt

from freetoken.kernel.triton.exl3 import Exl3Parts, had_rows
from freetoken.layers.quantization.linear import exl3 as lin

dev = "cuda"
for k, sizes in ((2048, (12288,)), (2048, (8192, 512, 512)), (4096, (2048,)), (2048, (512, 512))):
    parts = Exl3Parts.build(k, sizes, 5, "mul1", dev)
    words = sum((k // 16) * (s // 16) * 80 for s in sizes)
    tr = torch.randint(-32768, 32767, (words,), dtype=torch.int16, device=dev)
    suh = torch.ones(len(sizes), k, dtype=torch.float16, device=dev)
    svh = torch.ones(parts.n, dtype=torch.float16, device=dev)
    line = []
    for m in (16, 32, 64, 128, 192, 256, 384, 512, 1024):
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        lin.RECONSTRUCT_MIN_ROWS = 1 << 30
        tg = tt.do_bench(lambda: lin.exl3_forward(x, tr, suh, svh, parts, torch.bfloat16), rep=100)
        lin.RECONSTRUCT_MIN_ROWS = 1
        trc = tt.do_bench(lambda: lin.exl3_forward(x, tr, suh, svh, parts, torch.bfloat16), rep=100)
        line.append(f"M={m}: gemm {tg:.3f} / recon {trc:.3f} ms")
    print(f"K={k} parts={sizes}\n  " + "\n  ".join(line))
