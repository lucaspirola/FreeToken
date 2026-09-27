# EXL3 prefill v2 (dba11a2 dense reconstruct+cuBLAS, c90a28e MoE decoded scratch) — BOX NUMBERS (ft-dev Vast RTX 5080), 2026-09-24

Code on the box: scratch/exl3-headroom + dba11a2/c90a28e files. Chain: `v2chain.sh`.

## Correctness on the real Ornith 5.0bpw (`v2-scores.txt`, logs `v2-*.log`)
- saver (mirror) vs whole: ids equal, logits bit-equal, short (149 tok) and long (5141 tok).
- FT new vs FT before the change (s2 runs): max|dlogit| 0.0 — bit-identical 32-token logits on both prompts.
- FT new vs exllamav3 (24 layers CPU-offloaded): top-1 29/31 short, 30/31 long, mean|dlogit| 0.228 / 0.221,
  KL mean 1.9e-2 / 7.5e-3 (step 2 envelope unchanged).

## Serving probe, ratio 1.00 with reorg-headroom (`step5.sh` TAG=v2), pass 2 of record (pass 1)

| prompt | TTFT s | prefill tok/s | decode tok/s | before (step4-headroom) prefill |
|---|---|---|---|---|
| 8K   | 1.90  | 4223 (4148) | 43.8 (24.1) | 775 |
| 32K  | 8.51  | 3765 (3762) | 43.4 (30.4) | 767 |
| 80K  | 26.97 | 2968 (2880) | 35.7 (30.0) | 735 |
| 128K | 53.70 | 2384 (2385) | 36.8 (33.3) | 699 |

5.4x / 4.9x / 4.0x / 3.4x prefill; decode unchanged (GEMV path untouched). Zero OOM, peak VRAM 15734 /
16303 MiB, 2 chats coherent (542 / 930 tokens, finish=stop). Prefill now falls with context (4.2K -> 2.4K)
and decode is 36-44 tok/s: both profiled next (`../../perf/profile_step.py`).
