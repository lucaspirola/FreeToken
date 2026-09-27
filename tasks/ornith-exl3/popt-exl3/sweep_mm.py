"""Config sweep of the fused-cast Triton GEMM (tri_vs_cublas._mm_kernel) over the dense EXL3 folded
shapes at 1024 / 4096 / 8192 rows; every config's output must be bitwise torch.mm's (cuBLAS).
Prints per shape the cuBLAS-path time and each config's time, then the configs ranked by the sum of
their slowdowns vs each shape's best."""
import itertools, collections
import torch
from tri_vs_cublas import mm, med

torch.manual_seed(0)
CFGS = list(itertools.product((64, 128), (64, 128, 256), (32, 64), (3, 4), (4, 8)))
SHAPES = [(M, K, N) for M in (1152, 2048, 4096, 8192) for K, N in ((2048, 2048), (4096, 2048), (2048, 1024), (512, 2048))]
res = collections.defaultdict(dict)
for M, K, N in SHAPES:
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    w = (torch.randn(K, N, device="cuda") * 0.03).half()
    xf = x.to(torch.float16)
    ref = torch.mm(xf, w).to(torch.bfloat16)
    t_ref = med(lambda: torch.mm(xf, w).to(torch.bfloat16))
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    for c in CFGS:
        try:
            t = med(lambda: mm(x, w, out, *c), reps=10)
        except Exception:
            continue
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (M, K, N, c)
        res[(M, K, N)][c] = t
    b = min(res[(M, K, N)].values())
    print(f"M{M} K{K} N{N}: cuBLAS+outcast {t_ref:.3f} ms, best Triton {b:.3f} ms {min(res[(M, K, N)], key=res[(M, K, N)].get)}", flush=True)
score = collections.Counter()
for s, d in res.items():
    b = min(d.values())
    for c in CFGS:
        score[c] += d.get(c, 99) / b - 1
print("configs by total slowdown vs per-shape best:")
for c, v in sorted(score.items(), key=lambda kv: kv[1])[:10]:
    print(f"  {c}: {v:.3f}  " + " ".join(f"{res[s].get(c, 0):.3f}" for s in SHAPES))
