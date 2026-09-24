# Where EXL3 prefill time goes — BOX NUMBERS (ft-dev Vast RTX 5080), 2026-09-24

`profile_prefill.py` (torch.profiler, one 8000-token chunk after two warmups, ratio 1.00, saver on,
scratch/exl3-headroom code): wall 9.64 s (830 tok/s), GPU time 10.18 s —
`profile_prefill-8k-box-2026-09-24.txt`, trace `.trace.json.gz`.

| share | what |
|---|---|
| 89.6% (9.12 s) | `_exl3_gemm_kernel`, 800 launches |
|   of which ~5.6 s | MoE experts (8 sub-chunks of 1024 tok x 40 layers, gate_up + down, BM=16) |
|   of which ~3.5 s | dense projections (GDN in_proj 1.84 s, attn qkv+gate 0.46 s, o/out_proj 0.99 s, shared expert 0.21 s) |
| 5.8% (0.59 s) | `fast_index_copy_multi` (116x, mirror prefill assembly) |
| 1.1% | Hadamard rows |
| 0.9% / 0.8% | attention / GDN FLA kernels |

`bench_reconstruct_vs_gemm.py` (`...-box-2026-09-24.txt`), same shapes, random trellis:
exl3_gemm runs at 5-6.6 TF/s on every dense shape, while `reconstruct` (decode W_hat to fp16 once:
0.10 ms for 2048x12288) + cuBLAS fp16 is 13-18x faster (GDN in_proj M=8000: 61.1 ms -> 3.5 ms).
Decoding is cheap; the kernel re-decodes each [K,128] weight column once per M-block (125x for
M=8000 at BM=64, ~16x per expert per 8K for the MoE) and its MMA pipeline (BK=32, 2 stages, decode
on the critical path) is far from tensor-core peak. All 256 experts of a layer reconstruct in 4.5 ms.

Projection (not measured): dense 3.5 s -> ~0.2 s via reconstruct+cuBLAS above a row threshold;
MoE 5.6 s -> ~0.5 s via per-layer grouped reconstruct into an fp16 scratch + the existing grouped
fp16/bf16 MoE GEMM; the 0.59 s mirror assembly copy then becomes the next item. ~1.7 s per 8K chunk
~= 4-5K tok/s, the NVFP4 range.
