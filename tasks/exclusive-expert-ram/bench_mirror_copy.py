"""Mirror copy bandwidth on this host: admission (pinned pool row -> GPU slot) and writeback
(GPU slot -> pool row) of n expert rows, through fast_index_copy_multi_jit (SM loads/stores on
mapped pinned memory) and through the copy engine (cudaMemcpyAsync per bank row), for the row
layout given on the command line.

    python bench_mirror_copy.py [bank_bytes,bank_bytes,...]    (default: two synthetic layouts)
"""
import sys

import torch
import triton.testing as tt

from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.kernel.pinned import device_ptr

LAYOUTS = {
    "5.36 MiB/row, 4 banks": [2_621_440, 163_840, 2_621_440, 163_840],
    "1.98 MB/row, 6 banks": [1_310_720, 8_192, 2_048, 655_360, 1_024, 4_096],
}
if len(sys.argv) > 1:
    LAYOUTS = {"argv": [int(v) for v in sys.argv[1].split(",")]}

for name, rows in LAYOUTS.items():
    E, S = 32, 16
    host = [torch.randint(0, 255, (E, v), dtype=torch.uint8).pin_memory() for v in rows]
    gpu = [torch.randint(0, 255, (S, v), dtype=torch.uint8, device="cuda") for v in rows]
    hp = torch.tensor([device_ptr(h) for h in host], dtype=torch.int64, device="cuda")
    gp = torch.tensor([g.data_ptr() for g in gpu], dtype=torch.int64, device="cuda")
    fb = torch.tensor(rows, dtype=torch.int64, device="cuda")
    per = sum(rows)
    print(f"{name}: {per / 2**20:.2f} MiB per row")
    for n in (1, 2, 4, 8):
        di = torch.zeros(8, dtype=torch.int32, device="cuda"); di[:n] = torch.arange(n)
        si = torch.zeros(8, dtype=torch.int32, device="cuda"); si[:n] = torch.arange(3, 3 + n)
        cnt = torch.tensor([n], dtype=torch.int64, device="cuda")
        sm_h2d = tt.do_bench(lambda: fast_index_copy_multi_jit(gp, hp, fb, di, si, cnt), rep=100) * 1e3
        sm_d2h = tt.do_bench(lambda: fast_index_copy_multi_jit(hp, gp, fb, si, di, cnt), rep=100) * 1e3

        def dma_d2h():
            for h, g in zip(host, gpu):
                for j in range(n):
                    h[3 + j].copy_(g[j], non_blocking=True)
        ce_d2h = tt.do_bench(dma_d2h, rep=100) * 1e3
        gbs = lambda t: n * per / t / 1e3
        print(f"  n={n}: SM H2D {sm_h2d:7.1f} us ({gbs(sm_h2d):5.1f} GB/s)  SM D2H {sm_d2h:7.1f} us ({gbs(sm_d2h):5.1f} GB/s)"
              f"  copy-engine D2H {ce_d2h:7.1f} us ({gbs(ce_d2h):5.1f} GB/s)")
