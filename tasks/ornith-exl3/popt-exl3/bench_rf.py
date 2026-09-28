"""reconstruct_folded alone (dense Ornith shapes, 5-bit mul1, one slab of <= 2048 columns): time + sha1."""
import hashlib, sys, torch
import freetoken.kernel.triton.exl3 as k3
from bench_moe import med, sha
DEV = torch.device("cuda")
g = torch.Generator().manual_seed(3)
row = [sys.argv[1]]
for K, sizes in ((2048, (1024, 512)), (2048, (2048,) * 6), (4096, (2048,)), (512, (2048,)), (2048, (512, 512))):
    parts = k3.Exl3Parts.build(K, sizes, 5, "mul1", DEV)
    tr = torch.randint(-(1 << 15), 1 << 15, (parts.words * 2,), generator=g, dtype=torch.int32).to(torch.int16).to(DEV)
    suh = (torch.randn(len(sizes) * K, generator=g) * 0.5).half().to(DEV)
    svh = (torch.randn(sum(sizes), generator=g) * 0.05).half().to(DEV)
    f = lambda: k3.reconstruct_folded(tr, suh, svh, parts, cols=(0, min(2048, parts.n)))
    row.append(f"rf{K}x{sum(sizes)}={med(f) * 1000:.1f}:{sha(f())}")
print("\t".join(row), flush=True)
