# Ornith RTX 5080 growable-KV validation

Date: 2026-08-29  
Host: NVIDIA GeForce RTX 5080 (sm_120), 16 GiB, WSL  
Stack: Torch 2.11, CUDA 13.0, Triton 3.6

The opt-in `--kv-grow-step-tokens 65536` path reserves a stable virtual address
for the maximum KV cache, maps physical HBM in 64K-token increments, and shrinks
the MoE expert cache at each safe boundary. Scheduler/forward overlap is disabled
in this single-request mode; MoE's layer-copy/compute prefill overlap remains on.
Decode graphs are destroyed at the first pointer-changing expert resize and
recaptured once after the final prompt chunk.

Both full gates used a 261,800-token deterministic needle prompt, 128 requested
output tokens, an exact 262,144-token ceiling, LFU, 8,192-token chunks, and
`--memory-ratio 0.97`.

| Gate | Q4_K_M + INT4 KV | Q6_K + Q8_0 KV |
|---|---:|---:|
| KV physical at 262,144 | 1.48 GiB | 2.73 GiB |
| Expert slots, 64K → 256K | 6,175 → 5,581 | 4,412 → 3,583 |
| Average prefill | 705.88 tok/s | 325.24 tok/s |
| Last full-chunk instant prefill | 555.41 tok/s | 331.59 tok/s |
| Decode | 91.09 tok/s | 59.41 tok/s |
| Needle / coherent answer | PASS | PASS |

The Q4 result improves full-prompt prefill over the static-pool gate (678.38 to
705.88 tok/s, +4.1%) while reducing final decode (96.37 to 91.09 tok/s, -5.5%)
because 44 fewer expert slots remain after VMM granularity is charged. Q6 is a
correctness pass but not a performance recommendation at full context: it trails
the static 393.21 tok/s prefill and 82.92 tok/s decode results. Its 64K–128K
interval was the dominant regression; later full chunks recovered into the
332–432 tok/s range. Keep growable Q8 opt-in pending a better physical-layout or
expert-resize implementation.

Additional transition gates crossed one boundary at 70,000 tokens for both pairs
and two boundaries at 140,000 tokens for Q4. Every retained run recovered
`5663623` and returned a coherent explicit explanation.
