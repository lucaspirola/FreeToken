"""Interleaved re-timing of the plausible bit-exact had_rows tilings from sweep_had_rows.py against
the shipped launcher: ROUNDS rounds, each timing base then every candidate (median of 50 CUDA-event
reps), so a slow window hits all arms alike. Prints per-arm median over rounds and each round."""
import torch

import freetoken.kernel.triton.exl3 as k3
from sweep_had_rows import med, run_v

DEV = torch.device("cuda")
torch.manual_seed(0)
T, TOPK, E, ROUNDS = 8192, 8, 256, 7
ids = torch.randint(0, E, (T * TOPK,), device=DEV, dtype=torch.int32)
CANDS = [(64, 4, 2, 4, 0), (64, 4, 2, 2, 0), (64, 4, 2, 8, 0), (128, 4, 2, 4, 0), (128, 8, 2, 4, 0), (64, 8, 2, 4, 0),
         (64, 4, 2, 4, 1), (64, 4, 2, 16, 1), (128, 8, 2, 4, 1), (32, 4, 2, 4, 1)]
for K, N, src_div, xrows in ((2048, 1024, TOPK, T), (512, 2048, 1, T * TOPK)):
    parts = k3.Exl3Parts.build(K, (N,), 5, "mul1", DEV)
    x = torch.randn(xrows, K, device=DEV).half()
    suh = (torch.randn(E, K, device=DEV) * 0.5).half()
    ref = k3.had_rows(x, suh, parts, src_div=src_div, experts=ids, suh_expert_stride=suh.stride(0))
    out = torch.empty_like(ref)
    times = {a: [] for a in ["base"] + CANDS}
    for r in range(ROUNDS):
        times["base"].append(med(lambda: k3.had_rows(x, suh, parts, src_div=src_div, experts=ids, suh_expert_stride=suh.stride(0), out=ref), 50))
        for c in CANDS:
            bm, nw, ns, kb, ho = c
            run_v(x, suh, parts, ids, src_div, out, bm, nw, kb, ns, False, bool(ho))
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), c
            times[c].append(med(lambda: run_v(x, suh, parts, ids, src_div, out, bm, nw, kb, ns, False, bool(ho)), 50))
    for a, ts in times.items():
        print(f"K{K} {str(a):16s} median {sorted(ts)[ROUNDS // 2]:.4f}  rounds " + " ".join(f"{t:.3f}" for t in ts), flush=True)
