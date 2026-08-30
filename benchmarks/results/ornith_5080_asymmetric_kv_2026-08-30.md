# Ornith RTX 5080 asymmetric KV validation

Date: 2026-08-30  
Host: NVIDIA GeForce RTX 5080 (sm_120), 16 GiB, WSL  
Model: Ornith-1.5-35B-A3B GGUF Q6_K

FreeToken now supports independently selected key/value cache formats for its
full-attention MHA pool. The first deliberately narrow production lane is Q8_0 keys
plus Q6_0 values. Keys retain the established Q8 precision; values use GGML Q6_0's
quantizer and a cache-native contiguous bit-plane layout (6 payload bits plus one fp16
scale per 32 values). The existing symmetric Q8/Q8 and Q4/Q4 paths are unchanged.

The implementation includes separate K/V static and CUDA-VMM growable slabs, exact
cost/admission accounting, copy/rebuild/decommit support, independent attention loads,
and config-time rejection of unvalidated asymmetric pairs. The production host RAM
reserve remained 3 GiB throughout validation.

## Correctness gates

- Reference Q6 quantize/dequantize, GPU store, paged attention, decode attention,
  split and non-split extend attention all matched their dequantized-tensor oracles.
- Static and growable asymmetric pools round-tripped values and grew/shrank cleanly.
- 118 focused kernel/pool/config tests passed after the final layout and launch tuning.
- A live 8,192-token deterministic needle gate recovered passcode `5663623` and gave a
  coherent explicit answer after every material kernel-layout revision.
- The final exact 65,536-token capacity gate used a 65,408-token prompt plus 128 output
  allowance. It crossed the 32K CUDA-VMM boundary, recovered `5663623`, and remained
  coherent.

## Measurements

| Gate | Q8-K/Q8-V baseline | Q8-K/Q6-V |
|---|---:|---:|
| 64K logical KV | 0.66 GiB | **0.59 GiB** |
| 64K growable physical ceiling | 0.74 GiB | **0.66 GiB** |
| Planned final expert slots | 4,407 | **4,440** |
| 8K decode | 51.24 tok/s | **51.43 tok/s** |
| Exact 64K average prefill | — | **515.91 tok/s** |
| Exact 64K final-chunk instant prefill | — | **693.56 tok/s** |
| Exact 64K harness prefill (includes TTFT) | — | **526.08 tok/s** |
| Exact 64K decode | — | **47.06 tok/s** |
| Coherent retrieval | PASS | **PASS** |

The first correct Q6 layout was rejected for performance: at a synthetic 262K prefix
it took 0.655 ms per full-attention layer. Contiguous low/high planes plus the measured
sm_120 mixed launch (64 splits, 32-token tile, 8 warps) reduced this to 0.492 ms/layer.
Q8/Q8 remains faster in isolated long-prefix attention (0.347 ms/layer), so Q8/Q6 is a
capacity/residency option rather than a universal replacement. Its expected 1M KV is
about 9.44 GiB versus the measured 10.70 GiB Q8/Q8 pool, but that extrapolation has not
been presented as a measured 1M result.

## Command shape

```bash
FREETOKEN_PIN_BUDGET_GB=20 ft serve \
  --model /path/to/Ornith-1.5-35B-Q6_K.gguf \
  --attention-backend triton --moe-backend offload --moe-cache-auto \
  --moe-pageable-gpu --host-ram-reserve-gb 3 \
  --kv-cache-dtype-k q8_0 --kv-cache-dtype-v q6_0 \
  --num-tokens 262144 --kv-grow-step-tokens 131072
```

