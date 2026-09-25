"""Reference: exllamav3's own bsz-1 LinearEXL3.forward on the Ornith dense shapes (run in the exllamav3 venv).
Times the whole forward (input Hadamard + GEMV), random trellis, 5-bit mul1."""
import torch
import triton.testing as tt
from exllamav3.modules.quant.exl3 import LinearEXL3

MUL1 = 0x83DCD12D - (1 << 32)
for label, k, n in (("GDN in_proj", 2048, 12288), ("attn q+gate|k|v", 2048, 9216), ("o_proj/out_proj", 4096, 2048),
                    ("shared/expert gate|up", 2048, 1024), ("shared/expert down", 512, 2048)):
    tr = torch.randint(-32768, 32768, (k // 16, n // 16, 80), dtype=torch.int32).to(torch.int16).cuda()
    lin = LinearEXL3(None, k, n, suh=torch.ones(k, dtype=torch.half, device="cuda"), svh=torch.ones(n, dtype=torch.half, device="cuda"),
                     trellis=tr, mul1=torch.tensor(MUL1, dtype=torch.int32, device="cuda"))
    x = torch.randn(1, k, dtype=torch.half, device="cuda")
    t = tt.do_bench(lambda: lin.forward(x, {}), rep=200) * 1e3
    wb = k * n * 5 / 8
    print(f"{label:24s} K={k} N={n}: {t:7.1f} us  ({wb / t / 1e3:6.1f} GB/s of weights)")
