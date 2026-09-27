"""Pinned host <-> device copy bandwidth (DMA), 256 MiB, median of 20: the PCIe path the expert
misses / the saver's pool rows use. python pcie_bw.py -> one line."""
import torch, triton.testing as tt
n = 256 << 20
h = torch.empty(n, dtype=torch.uint8, pin_memory=True); d = torch.empty(n, dtype=torch.uint8, device="cuda")
h2d = tt.do_bench(lambda: d.copy_(h, non_blocking=True), rep=400, return_mode="median")
d2h = tt.do_bench(lambda: h.copy_(d, non_blocking=True), rep=400, return_mode="median")
print(f"H2D {n / h2d / 1e6:.1f} GB/s  D2H {n / d2h / 1e6:.1f} GB/s")
