# Decode GEMV rewrite (5d6f143) on Ornith EXL3 5.0bpw + saver — BOX NUMBERS (ft-dev Vast RTX 5080), 2026-09-24

Box code: scratch/exl3-headroom + EXL3 files at 5d6f143 (dense reconstruct, MoE scratch, new GEMV).

## Serving probe, ratio 1.00 (`step5.sh` TAG=v3), pass 2 of record (pass 1)

| prompt | prefill tok/s | decode tok/s | decode before (v2) |
|---|---|---|---|
| 8K   | 4253 (4210) | 118.9 (40.5) | 43.8 |
| 32K  | 3779 (3773) | 115.8 (58.7) | 43.4 |
| 80K  | 2967 (2888) | 76.6 (55.8)  | 35.7 |
| 128K | 2386 (2388) | 81.5 (58.7)  | 36.8 |

Zero OOM, peak VRAM 15734 MiB, both chats coherent (544 / 924 tokens, finish=stop). Pass 1 decode
(40-59) is the cold mirror: writebacks at 2.3 GB/s on this box (../../perf/README.md).

## Correctness
- saver vs whole: ids equal, logits bit-equal, short and long (`v3-scores.txt`).
- Free-running greedy ids leave the v2 run at token 17 (short) / 6 (long): the new kernel sums in a
  different fp32 order. Teacher-forced on the v2 ids (`ft_logits.py` FT_FORCE, `force-scores.txt`):

| same ids | top-1 | mean abs dlogit | KL mean |
|---|---|---|---|
| short: new vs exllamav3 | 30/31 | 0.215 | 1.6e-2 |
| short: v2 vs exllamav3  | 29/31 | 0.228 | 1.9e-2 |
| long: new vs exllamav3  | 30/31 | 0.236 | 1.1e-2 |
| long: v2 vs exllamav3   | 30/31 | 0.221 | 7.5e-3 |
| short / long: new vs v2 | 30/31, 29/31 | 0.213, 0.180 | 1.6e-2, 6.5e-3 |

New and old GEMV differ from each other about as much as each differs from exllamav3, and as much as
exllamav3 differs from itself with a different CPU-offload split (0.17-0.18, step 2): summation-order
noise amplified through 40 bf16 layers, not a precision loss. The free-running comparison
(`v3-scores.txt`, new ids vs exllamav3: 28/31 and 29/31) compares different sequences.
