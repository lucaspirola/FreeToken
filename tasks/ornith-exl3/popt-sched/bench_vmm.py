#!/usr/bin/env python3
"""Expert-arena grow/shrink cost: 6 VMM banks (Ornith EXL3 row bytes), a 600-slot grow in 24-slot
ladder chunks (the ~154 per-chunk mappings the headroom release commits), timed commit+uncommit."""
import time, statistics, json, sys
import torch
from freetoken.kernel.vmm import VMMTensor, allocation_granularity
dev = torch.device("cuda")
g = allocation_granularity(dev)
ROW = [1310720, 8192, 2048, 655360, 1024, 4096]; STEP = 24; CHUNKS = 25; BASE = 4848
def ranges(row):
    b = [((BASE + k * STEP) * row + g - 1) // g * g for k in range(CHUNKS + 1)]
    return [(s, e - s) for s, e in zip(b, b[1:]) if e > s]
banks = []
for row in ROW:
    rs = ranges(row)
    if not rs: continue
    total = rs[-1][0] + rs[-1][1]
    first = (BASE * row + g - 1) // g * g
    banks.append((VMMTensor((total,), dtype=torch.uint8, device=dev, reserved_bytes=total, initial_ranges=[(0, first)]), rs))
n = sum(len(r) for _, r in banks)
c, u = [], []
for _ in range(12):
    torch.cuda.synchronize(); t = time.perf_counter()
    for a, rs in banks: a.commit_ranges(rs)
    c.append((time.perf_counter() - t) * 1e3)
    for a, rs in banks: a.tensor[rs[0][0]: rs[-1][0] + rs[-1][1]].fill_(1)
    torch.cuda.synchronize(); t = time.perf_counter()
    for a, rs in banks: a.uncommit_ranges(rs)
    u.append((time.perf_counter() - t) * 1e3)
r = {"ranges": n, "commit_ms": round(statistics.median(c[2:]), 2), "uncommit_ms": round(statistics.median(u[2:]), 2)}
print(sys.argv[1] if len(sys.argv) > 1 else "", json.dumps(r))
