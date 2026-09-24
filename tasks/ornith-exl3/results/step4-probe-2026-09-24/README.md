# Step 4 — serving performance, Ornith EXL3 5.0bpw-hq, RAM saver on — BOX NUMBERS (ft-dev Vast RTX 5080, 48 vCPU, 256 GB, native Linux container), 2026-09-24

Not the owner's WSL2 5080 host; not comparable to numbers under C-EMPTY-GPU protocol there.
Server: `../step3-serve-2026-09-24/step3.sh` with RATIO=0.85 (0.95 and 0.90 OOM in GDN prefill, see below),
code d1c6e39, q8_0 KV growable to 262144, mirror pool auto: 4845 GPU expert slots, 7472 host rows (13.79 GiB pinned).
Probe `scripts/probe_decode.py 8000 32000 80000`, PROBE_PASSES=2, 127 generated tokens; pass 2 of record.

| prompt | pass 2 TTFT s | pass 2 prefill tok/s | pass 2 decode tok/s | (pass 1 prefill / decode) |
|---|---|---|---|---|
| 8K  (8011)  | 10.34 | 775 | 35.9 | 773 / 22.4 |
| 32K (32024) | 41.80 | 766 | 39.0 | 766 / 28.6 |
| 80K (80025) | 108.85 | 735 | 37.8 | 735 / 34.9 |

Prefill is flat at ~750 tok/s, far below NVFP4 Nemotron (~5,200 tok/s on 5080): the EXL3 prefill path
(dense GEMM + MoE prefill) is the next thing to profile, before any kernel change.

Ratio attempts: 0.95 died on the first 8K request (FLA `wy_fast` 64 MiB, 28 MiB free); 0.90 served 8K
(647 tok/s, 24.3 dec) and 32K (766, 30.5) then died at 80K right after "Committed growable KV through
131072 tokens ... MoE slots 5255 -> 5255" (FLA `chunk_delta_h` 128 MiB, 42 MiB free). Shared post-grow
prefill headroom, addressed on branch reorg-headroom (82207c8…), being tested next.
