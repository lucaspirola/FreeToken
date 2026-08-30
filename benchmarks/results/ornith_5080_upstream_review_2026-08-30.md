# Ornith RTX 5080 upstream optimization review

Date: 2026-08-30  
Host: NVIDIA GeForce RTX 5080 (sm_120), 16 GiB  
Production host-RAM reserve: 3 GiB

This pass reviewed every repository currently published by
[High-Performance-AI-Lab](https://github.com/High-Performance-AI-Lab) and the current
[BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) tree. Source revisions were
`kvpack@88895d0`, `muser@bfddc73`, `muser-book@15b6dcb`, and
`beellama.cpp@f8cd4e6`.

## Accepted transfers

- `kvpack`'s bounded parallel restore model maps cleanly onto FreeToken's process-local
  cold agent sessions. Disk restore now preloads exactly one chunk on one worker while
  the caller installs the preceding chunk on the GPU. It retains bounded streaming,
  the 3 GiB reserve, exact token/layout validation, and recompute-on-failure behavior.
- Muser's benchmark ladder treats correctness and fixed pre-registered performance
  thresholds as one acceptance decision. `bench_long_context.py` now exposes the final
  engine instant/average prefill pair in JSON and can compare a candidate with the last
  row of `--baseline-json`, using independent prefill/decode regression limits or
  absolute throughput floors. Missing needle retrieval still fails the same command.
- `bench_ornith_attention.py` now directly benchmarks the production asymmetric
  Q8-key/Q6-value format, including its dequantized-oracle check. Previously the script
  could measure only symmetric Q4 or Q8 even though the server supported Q8/Q6.

## Live disk-restore gates

Both models were forced through physical VMM shrink and disk restore with the RAM spill
budget set to zero. The 16K Q6 growth step is validation-only; production remains 128K.

| Pair | First turn | Restored prefix | Resume wall | Answer | Physical transition |
|---|---:|---:|---:|---|---|
| Q4_K_M + INT4 KV | 70,021 input / 207.74 s | 70,028 | 1.73 s | exact `5663623` | 131K -> 65K -> 131K |
| Q6_K + Q8-K/Q6-V | 20,021 input / 57.95 s | 20,028 | 4.15 s | exact `5663623` | 32K -> 16K -> 32K |

Q4 is effectively tied with the earlier 1.70-second sequential restore result. Q6 is
31% below the earlier 6.02-second recorded restore path, but those runs differ in cache
warmth and should not be presented as a strict paired speedup. The important acceptance
facts are that look-ahead stayed bounded, both actual disk imports completed, the full
prefixes were reused, and both model answers remained coherent.

## Measured and rejected: BF16 precision tail

Bee's recent-token precision tail was prototyped in FreeToken's production Triton decode
entry point. The first implementation selected exact rows inside every quantized tile;
it was mathematically correct but exceeded the 5080's shared-memory limit at two pipeline
stages. At one stage, Q4 attention slowed from 0.3464 to 0.6536 ms per layer.

A second implementation computed the quantized body and 1,024-row BF16 tail as separate
partials and merged their `(output, log-sum-exp)` state through the existing final
reduction. This retains one exact FP32 online softmax and avoids materializing history.

| 262,144-token decode attention | No tail | 1,024 BF16 tail | Toll | Oracle max error |
|---|---:|---:|---:|---:|
| Q4/Q4 | 0.3465 ms | 0.3854 ms | +11.2% | 0.0001 |
| Q8-K/Q6-V | 0.4939 ms | 0.5493 ms | +11.2% | 0.0001 |

The feature adds VRAM and has no Ornith-specific quality evidence, so it is not an
inference optimization for either requested pair. The runtime prototype was removed;
only the reusable asymmetric benchmark support remains.

## Reviewed but not transferred

- **KVarN:** attractive when cache capacity is the primary limit, but it needs a new
  128-token rotated/normalized record store plus dedicated prefill, decode, exact-suffix,
  checkpoint, compaction, and graph paths. KVarN4's body is only 2.8% smaller than Q4_0,
  before its mandatory exact suffix and staging; Bee's own request-zero 64K measurement
  used 8.4% more peak KV memory than Q4_0. KVarN6/KVarN5 could save Q6-pair memory, but
  changes the requested cache format and has no Ornith quality result. It is not safe or
  evidence-backed enough to merge.
- **Other Bee low-bit caches:** Q3/Q2 and alternative K/V widths define new quality
  tiers, not optimizations of Q4/INT4 or Q6/Q8. Bee's own standard-cache ladder shows a
  clear quality cliff below Q4. Independent Q8/Q6 selection, the useful conservative
  standard pair, is already implemented here.
- **DFlash/adaptive speculation:** intentionally excluded with the user's no-MTP/no-risky
  speculative boundary.
- **Muser disaggregated prefill:** requires a second accelerator host and a cache handoff;
  it does not apply to this single RTX 5080 deployment. Its cross-model cache composition
  notes explicitly require quality reconciliation rather than exact reuse.
- **Persistent kvpack publication:** crash-safe immutable packs, encryption, cross-process
  catalogs, and Merkle validation are valuable for a durable distributed cache, but the
  current FreeToken tier is deliberately process-local and deleted on shutdown. Adding
  those costs would not improve inference speed or the present lifecycle contract.
- `muser-console`, `muser-book`, the organization profile, and both static websites contain
  operations, documentation, or presentation layers rather than transferable CUDA/runtime
  kernels.

## Regression validation

- The asymmetric benchmark exercised decode, fresh prefill, and cached-prefix extend on
  RTX 5080; all three matched their dequantized oracles (maximum errors 0.0005, 0.0156,
  and 0.0005 respectively).
- The scheduler/KV/server/benchmark suite passed 947 tests with one skip.
- The complete suite reached 1,712 passed and nine skipped. Its one CUDA batch-copy probe
  failure and six global-TP setup-order errors are the same unrelated environment/order
  failures recorded before this pass.
- Ruff and `git diff --check` passed.
