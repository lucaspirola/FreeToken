# Ornith RTX 5080 growable-KV validation and Q8 step tuning

Date: 2026-08-29  
Host: NVIDIA GeForce RTX 5080 (sm_120), 16 GiB, WSL  
Stack: Torch 2.11, CUDA 13.0, Triton 3.6

The opt-in growable path reserves a stable virtual address for the maximum KV
cache, maps physical HBM in fixed token increments, and shrinks the MoE expert
cache at each safe boundary. Scheduler/forward overlap is disabled in this
single-request mode; MoE's layer-copy/compute prefill overlap remains on. Decode
graphs are destroyed at the first pointer-changing expert resize and recaptured
once after the final prompt chunk.

Both full gates used a 261,800-token deterministic needle prompt, 128 requested
output tokens, an exact 262,144-token ceiling, LFU, 8,192-token chunks, and
`--memory-ratio 0.97`.

| 64K-step gate | Q4_K_M + INT4 KV | Q6_K + Q8_0 KV |
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
correctness pass but is not the recommended Q8 step: it trails
the static 393.21 tok/s prefill and 82.92 tok/s decode results. Its 64K–128K
interval was the dominant regression; later full chunks recovered into the
332–432 tok/s range.

## Q6_K + Q8_0: accepted 128K growth step

The first Q8 resize, rather than the 4,135-slot geometry itself, caused the 64K
step's bad interval. A 140,000-token screen that started directly at 4,135 slots
ran normally, recovered `5663623` coherently, and reached 570.83 tok/s average
prefill. The retained full gate therefore uses `--kv-grow-step-tokens 131072`:

| Q6_K + Q8_0 gate | Static KV | Growable, 64K step | Growable, 128K step |
|---|---:|---:|---:|
| Initial → final expert slots | 3,615 | 4,412 → 3,583 | 4,135 → 3,583 |
| Average prefill | 393.21 tok/s | 325.24 tok/s | **527.36 tok/s** |
| Last full-chunk instant prefill | 296.10 tok/s | 331.59 tok/s | **416.57 tok/s** |
| Final remainder instant prefill | — | 234.15 tok/s | **397.40 tok/s** |
| Decode | 82.92 tok/s | 59.41 tok/s | **78.40 tok/s** |
| Needle / coherent answer | PASS | PASS | **PASS** |

Against the rejected 64K step, the 128K step improves average prefill by 62.1%,
last-full-chunk prefill by 25.6%, and decode by 32.0%. Against static KV it
improves average prefill by 34.1% and the last full chunk by 40.7%, with a 5.5%
decode tradeoff. Because this 262K test contains only one 128K resize, it is the
controlled performance comparison, not sufficient by itself to validate repeated
growth.

### Repeated-growth gate at 512K

The multi-step Q6_K + Q8_0 gate used a 524,000-token prompt, 128 requested output
tokens, a 524,288-token ceiling (exactly four 131,072-token segments), static YaRN
factor 2 from the native 262,144-token context, and otherwise the same LFU,
8,192-token-chunk, and 0.97-memory-ratio settings. It exercised three live growth
events:

| Event | Physically committed KV | Expert slots | Boundary / first post-growth prefill |
|---|---:|---:|---:|
| Startup | 131,072 tokens | 4,096 | — |
| Growth 1 | 262,144 tokens | 3,544 | 293.98 instant / 624.28 average; next 548.53 instant |
| Growth 2 | 393,216 tokens | 2,991 | 408.27 instant / 539.92 average; next 400.16 instant |
| Growth 3 | 524,288 tokens | 2,438 | 324.36 instant / 463.22 average; next 316.09 instant |

Every transition preserved the old KV and resumed without a sustained throughput
collapse. The completed run reached 402.10 tok/s average prefill, 268.71 tok/s on
the last full 8,192-token chunk, 228.51 tok/s on the final 7,904-token remainder,
and 62.28 tok/s decode. It recovered `5663623` exactly and produced a coherent,
explicit answer. This is the repeated-growth validation supporting the 128K Q8_0
recommendation. Use 64K growth for Q4_0 KV and 128K growth for Q8_0 KV on this
RTX 5080 host.

Additional transition gates crossed one boundary at 70,000 tokens for both pairs
and two boundaries at 140,000 tokens for Q4. Every retained run recovered
`5663623` and returned a coherent explicit explanation.
