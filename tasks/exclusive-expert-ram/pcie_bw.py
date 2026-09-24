#!/usr/bin/env python3
"""Pinned host<->device copy bandwidth (the mirror's H2D admissions and D2H write-backs).

    python3 pcie_bw.py            # prints GB/s for H2D and D2H at 8 MiB and 256 MiB

Run with the GPU otherwise idle. Used to tell a PCIe-bound pool arm from a CPU-bound one
when two hosts' decode differ (ck4: rented box PCIe Gen4 x16 vs owner Gen5 x16).
"""
import torch

def bw(n_bytes, direction, reps=20):
    h = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    src, dst = (h, d) if direction == "H2D" else (d, h)
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        dst.copy_(src, non_blocking=True)
    e.record()
    torch.cuda.synchronize()
    return n_bytes * reps / (s.elapsed_time(e) / 1e3) / 1e9

print(torch.cuda.get_device_name(), torch.version.cuda)
for mib in (8, 256):
    for dirn in ("H2D", "D2H"):
        print(f"{dirn} {mib:4d} MiB  {bw(mib << 20, dirn):6.1f} GB/s")
