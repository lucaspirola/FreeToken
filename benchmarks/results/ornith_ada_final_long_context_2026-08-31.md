# Ornith RTX 2000 Ada final long-context validation

Date: 2026-08-31

Host: NVIDIA RTX 2000 Ada Generation (sm_89), 16 GiB, 70 W, WSL 2

Stack: Windows driver 595.95, CUDA toolkit 13.1, PyTorch 2.11.0+cu130,
Triton 3.6.0

This is the final end-to-end gate for the Ornith machinery imported from the RTX
5080 fork and adapted to Ada. Each test issued one cold synthetic-needle prompt,
requested 128 output tokens with greedy decoding and `ignore_eos`, and accepted
the run only when passcode `5663623` appeared in a coherent answer. There was no
full-prompt warm pass. The sm_89 Qwen3.5-MoE GGUF policy automatically selected
4,096-token prefill chunks.

## Results

| Model / KV | Context ceiling | Prompt | Growth step | Cold prefill | Engine average | Final chunk | Full-tail decode | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q4_K_M / INT4 | 262,144 | 261,800 | 65,536 | 357.06 tok/s | 356.38 tok/s | 181.32 tok/s | 28.76 tok/s | PASS |
| Q6_K / Q8_0 | 262,144 | 261,800 | 131,072 | 404.05 tok/s | 403.32 tok/s | 211.92 tok/s | 22.92 tok/s | PASS |
| Q6_K / Q8_0, YaRN 2x | 524,288 | 524,000 | 131,072 | 222.01 tok/s | 221.87 tok/s | 111.19 tok/s | 15.34 tok/s | PASS |

Against the earlier static-pool Ada reference at commit `770b572`, the retained
Q4/INT4 configuration improves 262K cold prefill from 250.72 to 357.06 tok/s
(+42.4%); its 28.76 tok/s decode is within 1.7% of the old 29.27 tok/s result.
Q6/Q8 improves cold prefill from 237.17 to 404.05 tok/s (+70.4%) and decode from
22.28 to 22.92 tok/s (+2.9%). All comparisons use the same prompt length and
128-token output allowance, although the old reference used 8K chunks and a
static full KV pool.

At 262K, Q6/Q8 prefills 13.2% faster than Q4/INT4 but decodes 20.3% slower. This
is a model/KV quality and phase tradeoff, not evidence that one tier dominates
the other. Both remain production choices.

## Growth and residency

Q4/INT4 exercised three 64K commits. Each needed 0.35 GiB plus the permanent
0.25 GiB guard and saw 1.01 GiB driver-visible free before commit:

| Physical KV | Resident expert slots |
|---:|---:|
| 65,536 tokens | 6,367 |
| 131,072 tokens | 6,028 |
| 196,608 tokens | 5,830 |
| 262,144 tokens / 1.48 GiB | 5,631 |

The 262K Q6/Q8 run committed 1.33 GiB at its only live boundary, with 1.99 GiB
free versus 1.58 GiB required. It moved from 131,072 tokens and 4,277 expert
slots to 262,144 tokens, 2.73 GiB physical KV, and 3,620 expert slots.

The 524K Q6/Q8 YaRN gate proved repeated growth on Ada:

| Event | Physical KV | Resident expert slots |
|---|---:|---:|
| Startup | 131,072 tokens | 4,238 |
| Growth 1 | 262,144 tokens / 2.73 GiB | 3,581 |
| Growth 2 | 393,216 tokens / 4.06 GiB | 3,028 |
| Growth 3 | 524,288 tokens / 5.39 GiB | 2,476 |

Every Q6 commit saw 1.99 GiB live free, committed 1.33 GiB, and retained the
0.25 GiB guard. Prefill resumed normally after each expert-cache rebuild. Final
batch-one graph capture left 0.66 GiB free in all three gates.

## Commands

~~~bash
.venv/bin/python benchmarks/bench_long_context.py \
  --model /path/to/Ornith-1.5-35B-Q4_K_M.gguf \
  --synthetic-needle --target-prompt-tokens 261800 --decode 128 \
  --max-context 262144 --kv-cache-dtype int4 \
  --kv-grow-step-tokens 65536 --mem-ratio 0.97

.venv/bin/python benchmarks/bench_long_context.py \
  --model /path/to/Ornith-1.5-35B-Q6_K.gguf \
  --synthetic-needle --target-prompt-tokens 261800 --decode 128 \
  --max-context 262144 --kv-cache-dtype q8_0 \
  --kv-grow-step-tokens 131072 --mem-ratio 0.97

.venv/bin/python benchmarks/bench_long_context.py \
  --model /path/to/Ornith-1.5-35B-Q6_K.gguf \
  --synthetic-needle --target-prompt-tokens 524000 --decode 128 \
  --max-context 524288 --kv-cache-dtype q8_0 \
  --kv-grow-step-tokens 131072 --mem-ratio 0.97 \
  --rope-yarn-factor 2 --rope-yarn-original-context 262144
~~~

## Decision

The 5080 growable-VMM KV, exact graph recapture, adaptive scheduler, cold-session,
and elastic-agent machinery is accepted on RTX 2000 Ada. The architecture-specific
changes are the sm_89 quantized-attention launch tables, the 1,536-token automatic
multi-prompt grouping crossover, and the 4K automatic prefill chunk. The 5080
low-RAM pageable-expert ordering is intentionally not enabled: this 103 GiB WSL
host pins all Q6 expert banks, so staging those layers would add a copy without
relieving a constraint.

Focused cache-budget, parser, scheduler, benchmark, and Triton-attention tests:
233 passed, 2 unrelated generic FlashInfer-dependent tests deselected.
