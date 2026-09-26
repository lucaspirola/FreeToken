# Step 2 — real Ornith-1.5-35B-A3B EXL3 5.0bpw-hq, whole-model logits (ft-dev RTX 5080 box, 2026-09-24)

Code: 742b364 (deterministic split-K GEMV). Prompts from `compare/ft_logits.py`:
short = FT_PROMPT_REPEAT=1 (149 prompt tokens), long = FT_PROMPT_REPEAT=40 (5141 prompt tokens,
chunked prefill at 8192 max-prefill -> one chunk; MoE prefill path at thousands of tokens).
31 decode positions each, greedy. Commands: `step2.sh` (FreeToken), `step2b.sh` / `step2c.sh`
(exllamav3 1.5.1 reference, teacher-forced on FreeToken's ids, MoE layers CPU-offloaded
because the 35B does not fit 16 GB: EXL3_MOE_CPU_OFFLOAD=24 and =36). Raw logits in `raw/`
(local only, gitignored, 30 MB each).

## RAM saver (mirror, auto pool) vs whole-model residency

| prompt | ids | logits |
|---|---|---|
| short (149 tok) | identical 31/31 | bit-equal, max\|d\| 0.0 |
| long (5141 tok) | identical 31/31 | bit-equal, max\|d\| 0.0 |

Before 742b364 the saver and whole runs had identical step-0 logits but decode drifted
(max\|d\| 0.7, tokens diverged at 15): split-K atomic_add order. Now fixed-order plane sum.

## FreeToken (saver) vs exllamav3

| pair | prompt | top-1 | mean\|dlogit\| | max\|dlogit\| | KL mean | KL max |
|---|---|---|---|---|---|---|
| FT vs exl3(off24) | short | 28/31 | 0.244 | 4.04 | 1.62e-2 | 1.19e-1 |
| FT vs exl3(off24) | long  | 29/31 | 0.247 | 3.98 | 8.49e-3 | 5.16e-2 |
| FT vs exl3(off36) | short | 29/31 | 0.234 | 4.72 | 1.81e-2 | 1.87e-1 |
| FT vs exl3(off36) | long  | 29/31 | 0.231 | 3.10 | 8.44e-3 | 3.45e-2 |
| **exl3(off24) vs exl3(off36)** (reference self-spread) | short | 30/31 | 0.173 | 2.52 | 1.28e-2 | 1.05e-1 |
| **exl3(off24) vs exl3(off36)** | long  | 29/31 | 0.183 | 2.88 | 6.79e-3 | 5.22e-2 |

Reading: moving 12 MoE layers between exllamav3's own CPU and GPU paths moves its logits by
~0.18 mean / ~2.7 max; FreeToken sits at ~1.3x that spread. Layer-level (layer0-real-2026-09-24.txt)
FreeToken is the closer one to the exact fp32 result (MoE 3.2e-3 vs 9.5e-3, lm_head 2.2e-4 vs 1.3e-2),
so the residual is shared quantization/accumulation noise, not a FreeToken error.

Every top-1 disagreement is a near-tie (probabilities of the two candidates):

    short pos 3   FT 0.202/0.202   exl3 0.370/0.159   ("given" vs "written")
    short pos 7   FT 0.478/0.478   exl3 0.548/0.414   ("that" vs "with")
    short pos 17  FT 0.573/0.348   exl3 0.505/0.406
    long  pos 7   FT 0.331/0.258   exl3 0.309/0.241   ("amount" vs "block")
    long  pos 24  FT 0.509/0.350   exl3 0.433/0.426

Text (FreeToken greedy):
- short: "The user has given me a prompt that starts with a lot of repeated text (the same sentence repeated many times), and then asks a genuine question at the"
- long: "The user has pasted a huge amount of repeated text (the \"quick brown fox\" sentence repeated many times), but the actual request is at the very"

Gap: no exact fp32 whole-model oracle (the emulation kernel over 40 layers + 256 experts was not run);
the verdict rests on the reference self-spread plus the per-layer exact comparison.
