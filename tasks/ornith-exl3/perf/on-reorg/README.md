# EXL3 re-merged onto exp/reorg (round 2): ft-dev suite, greedy check, short (a)/(b) table (BOX NUMBERS, 2026-09-25)

What was tested:
- **Branch:** exp/exl3-on-reorg 2fb1026 = exp/reorg 3b19f4a (round 2) + exp/ornith-exl3 f980722 (EXL3 P2b/D2/D3a/D3c).
  Not included: overlap-sync (round 3) and the exact-decode experiment (rejected).
- **Box tree:** /root/FT-exl3p11, built from a git archive of 2fb1026. The C++ extensions were copied from FT-os, since their sources are unchanged from 01548e7 to 2fb1026.
- **Chain:** /root/K/chain54.sh (copied here).
- **Box:** ft-dev, Vast RTX 5080, PCIe gen4.

## Merge

exp/ornith-exl3 carried 17 commits that exp/reorg also has, as cherry-picks with different SHAs:
- mirror DMA writebacks;
- the measured prefill headroom;
- extend_flashinfer;
- exclusive-expert-ram scripts.

Those 56 files take exp/reorg's version, so scheduler/ and engine/ are byte-identical to exp/reorg.
residency.py keeps exp/reorg's version plus the one EXL3 hunk (sourced mirror formats gguf + exl3).
Everything else is the EXL3 lane's own files.

## Tests (p11, under /root/gpu.lock)

| suite | result |
|---|---|
| tests/moe | 420 passed, 5 skipped |
| tests/engine | 259 passed, 2 skipped |
| tests/scheduler | 412 passed, 1 skipped |
| tests/kernels | 896 passed, 7 skipped |

1987 passed, 0 failed.

## Greedy output (114K-token haystack prompt, 1023 greedy tokens, 2 reps; `greedy-compare.txt`)

| tree | rep 0 | reps agree | quoted facts |
|---|---|---|---|
| p9 (exp/ornith-exl3 + overlap-sync) | a43622edc2, 1023 tok | yes | 3/3 |
| p10 (p9 + exact decode) | a43622edc2, 1023 tok | yes | 3/3 |
| **p11 (this merge)** | 7b01f7c887, 925 tok | **no** (rep 1: c7d8a5e0c0, 1023 tok) | 3/3 in both reps |

- **p11 vs p9.** The first divergence is at generated token 28, and it is formatting only (`\n` vs `\n\n` between records). The facts are identical.
  - Round 2 changed decode attention's stage 2: it combines 32 split partials per step against the chunk max, where before it folded one split per step. That is a different fp32 summation order, inherited from exp/reorg and not from EXL3.
- **Rep 0 vs rep 1 on p11.** The server log shows the cause: rep 1's first prefill batch has `#cached-token: 24512`.
  - Round 2's cross-conversation prefix reuse restored a snapshot left by rep 0.
  - The remaining chunks then start 64 tokens off the 8192 grid, so the text diverges numerically. Both reps quote all three facts correctly.
  - Any greedy A/B on round 2+ must either disable prefix reuse or compare only the first request after a start. Otherwise the second rep is not comparable.

## (a) Saver / (b) whole, decode tok/s, pass 2 of record (`fuse/job-e2e.sh`, PREROT=1, ratio 1.00)

A = p8 (exp/ornith-exl3 52020bc content), B = p11, bracketed A1 / B / A2:

| residency | ctx | A1 | **B** | A2 | B vs mean(A) |
|---|---:|---:|---:|---:|---:|
| saver | 8K | 152.0 | **154.2** | 151.9 | +1.5% |
| saver | 32K | 144.6 | **145.2** | 144.5 | +0.5% |
| saver | 80K | 122.1 | **128.6** | 122.0 | +5.4% |
| whole | 8K | 165.1 | **163.6** | 165.1 | -0.9% |
| whole | 32K | 154.1 | **155.1** | 154.2 | +0.6% |
| whole | 80K | 138.6 | **140.9** | 138.7 | +1.6% |

Prefill B (saver) is 7289 / 6643 / 5378 tok/s and (whole) 7839 / 7025 / 5643 tok/s, within 1.5% of A.

Saver counters, B vs A:
- swaps 42824 vs 54800;
- ring_full_fallbacks 235 vs 761;
- stage_redirects 545 vs 332.

Round 2 returns the prefill transient to the arena during decode. Coverage faults and starved writebacks are 0 in every run, with one capture and 0 tracebacks everywhere.

The whole-model 8K is 0.9% slower on B, outside the A spread (165.1 / 165.1). Round 2's triton decode launch
does not explain it: Ornith keeps its tuned 64/64/4. It is left as an observation.

**Owner's-machine baseline:** B above is exp/reorg + EXL3 without overlap-sync. With overlap-sync (measured on
exp/ornith-exl3 today, `exl3e2e-os-orn-B`), saver was 159.2 / 151.1 / 126.8. That fix is +4-5% on every model
and stacks on top once round 3 lands in exp/reorg.

Results: `/root/K/results/{tests-p11-*,exl3greedy-f16acc-onreorg,exl3greedy-onreorg-compare,exl3e2e-onreorg-*}`,
archived to `~/ai/box-archive/ft-dev-exl3/results` (md5 manifest `chain54.md5`).
