"""Fine map, Ornith dense slab shapes (N 2048/1024 at K 2048, K 4096 -> 2048, K 512 -> 2048): every
M in [1024, 2048] and every 7th M up to 8192 -- where is cuBLAS (torch.mm fp16) not bitwise the
sequential-k Triton GEMM?"""
import torch
from tri_vs_cublas import mm
torch.manual_seed(0)
for K, N in ((2048, 2048), (2048, 1024), (4096, 2048), (512, 2048)):
    w = (torch.randn(K, N, device="cuda") * 0.03).half()
    xall = torch.randn(8192, K, device="cuda").to(torch.bfloat16)
    diff = []
    for M in list(range(1024, 2049)) + list(range(2055, 8193, 7)) + [8192]:
        x = xall[:M]
        ref = torch.mm(x.to(torch.float16), w)
        out = torch.empty(M, N, dtype=torch.float16, device="cuda")
        mm(x, w, out, 64, 128, 32, 3, 4)
        if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
            diff.append(M)
    runs = []
    for m in diff:
        if runs and m <= runs[-1][1] + 7:
            runs[-1][1] = m
        else:
            runs.append([m, m])
    print(f"K{K} N{N}: {len(diff)} differing M, ranges {runs}", flush=True)
