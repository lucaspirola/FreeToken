# Ornith EXL3 5.0bpw with the RAM saver at long context (BOX NUMBERS, ft-dev, 2026-09-25/26)

Box: ft-dev (Vast 52296107), RTX 5080 16 GB, PCIe gen4, 251 GB RAM. Every number below comes from that box.
Model: `/root/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq`. Saver: `--expert-residency mirror --moe-mirror-host-rows -1`.
Common flags (see `job-reach.sh`): triton attention, `--kv-grow-step-tokens 65536`, `--memory-ratio 1.00`,
`--max-prefill-length 8192`, `--max-running-requests 1`. 384K adds `--rope-yarn-factor 2 --max-seq-len-override 393216`.
The driver is `longctx_decode2.py`, run in-process through the scheduler's overlap loop. It runs an 8K warm-up, then:
- pass1 and pass2: two probe prompts;
- pass3: timing only;
- pass4: natural text from a pinned corpus (`LONGCTX_CORPUS`).

Each pass generates 128 greedy tokens. `osha` is the sha1 of the output token ids and `psha` the sha1 of the prompt ids.

Trees:
- p11 = exp/exl3-on-reorg 2fb1026.
- p12 = p11 + a refuted "return idle blocks" fix. That fix never landed, and its branch was reset.
- p13 = exp/exl3-longctx 843f62d: the record.
- p14 = exp/exl3-longctx 2ad5dc2: record + prediction bound.

## 1. Reach on exp/exl3-on-reorg (p11), before any fix

| KV | ceiling | result |
|---|---|---|
| q8_0 | 262144 | **refused** on the last KV commit: "need 1.78 GiB free, have 1.78 GiB (coverage floor of 4272 slots for a 6737-row mirror)" |
| q8_0 | 393216 (YaRN 2) | runs 384000-token prompts. Pass 2: prefill 2447 tok/s, decode 84.6 tok/s. Pass 4 (natural): decode 44.5 tok/s. 0 coverage faults, 0 starved writebacks, one capture. The last commit had 0.03 GiB to spare. |
| q4_0 | 262144 | **refused**: "need 1.47 GiB, have 1.45 (floor 4960, 6050-row mirror)" |
| q4_0 | 393216 (YaRN 2) | pass 1 and pass 2 run (prefill 611 tok/s, decode 86.5 tok/s), then **refused** on a later pass: "need 1.47, have 1.47 (floor 4536)" |

KV size per token (10 full-attention layers, 2 KV heads, head_dim 256):
- q8_0 is 10,880 B/token: 0.664 GiB per 65536-token grow step, 2.73 GiB physical at 262144 and 4.06 GiB at 393216.
- q4_0 is 5,760 B/token: 0.35 GiB per step, 1.48 GiB at 262144 and 2.19 GiB at 393216.

**Where the arena floor comes from.** The floor is the bounded mirror's coverage minimum,
`min_gpu_slots = total experts (40 x 256 = 10240) - pool rows + reserve rows (3 x 256 = 768)`, rounded up to the
8-slot arena step. For example, 10240 − 6737 + 768 = 4271, so the floor is 4272. It is not pinned buffers, and it is not
the prefill-overlap floor (2 x 256 = 512). The prefill transient enters indirectly, in two places:
- the pool is sized from a ceiling plan that subtracts one prefill chunk's transient;
- every KV commit must leave `commit + transient` free.

The reserve rows are load-bearing (starved writebacks without them, `mirror_pool.default_reserve_rows`), so they stay.

**Exact VRAM ledger at the refusal** (`ledger_probe.py`, run `reach-ledger-p11-q8-256k`, MiB):

| point | slots | free | torch alloc | torch reserved | arena | KV | outside torch |
|---|---:|---:|---:|---:|---:|---:|---:|
| startup geometry | 5288 | 1286.5 | 3322.8 | 3360.0 | 9998.0 | 760.0 | 437.2 |
| 3rd grow, before | 4560 | 150.5 (chunk in flight) | 3327.3 | 4512.0 | 8620.0 | 2120.0 | 439.2 |
| at floor 4272 (computed) | 4272 | ≈1822 | | 3384 | ≈8076 | 2120 | 439 |

- required = 680 MiB commit + 1146 MiB measured transient ≈ 1827 MiB, so the commit falls about 5 MiB short.
- The live budget measured at startup is 12044.5 MiB. The ceiling plan (the budget minus the transient) had placed the final arena at 4280, 8 slots (15 MiB) above the floor.
- After startup, torch's reserved pool grew by 24 MiB and the usage outside torch by 2 MiB. Of the 24 MiB, only 4.5 MiB is new allocations; the rest is pages held by small live blocks.
- 26 MiB of drift is more than 15 MiB of slack.
- q4_0 (`reach-ledger-p11-q4-256k`) shows the same drift (+24 MiB reserved, +2 outside torch) against 0 slots of slack.

**Why the slack was that thin.** The pool is sized before any forward pass can measure the transient. It used
the per-token estimate, 128 KiB/token = 1024 MiB. Ornith measures 1146 MiB on this box (1166 MiB on the owner's machine
at q6_0/q5_0), so the estimate is 122 MiB low. That error consumed the sizing margin.

**Refuted hypothesis (p12).** "The check double-counts the allocator's idle cached blocks." The fix returned them
with `empty_cache` before refusing, and it returned **0.00 GiB**: `_sync_get_memory` already empties the cache before the
grow. p12 was still refused (`reach-p12-q8-256k`, `reach-p12-q4-256k`). The commit was reset.

**Discriminating test.** `FREETOKEN_PREFILL_TRANSIENT_MB=1146` sizes the pool for the measured transient. The floor drops
4272 → 4208 (q8_0), and 256K runs at q8_0 and q4_0 on p11 unmodified (`reach-p11-q8-256k-tmb`, `reach-p11-q4-256k-tmb`).

## 2. Fix (model-agnostic; exp/exl3-longctx, same commits on exp/kv-transient-record from exp/reorg)

- **843f62d** records the measured transient per setup (model, chunk, attention backend, KV formats, residency, TP, GPU)
  in `~/.cache/freetoken/prefill-transient.json`. `FREETOKEN_PREFILL_TRANSIENT_RECORD` sets another path, or `off`.
  The pool is priced with max(estimate, record). If the pool was priced below the measurement and the slack is thin,
  the load fails with the reason, instead of the request failing at the ceiling.
- **2ad5dc2** makes the first start price the pool with `memory_prediction.transient_upper_bound`: prediction / (1 − 0.25).
  This is the largest measurement the engine accepts without its own "prediction vs measurement" WARN.
  - Ornith: 0.876 GiB prediction, 1196 MiB bound, which is at least the measured 1146 and 1166.
  - Nemotron: 0.90 GiB bound, below the 1.00 GiB estimate, so it sizes as before. Its measured transient on record is 0.59–0.65 GiB.

  The validation now requires `PREFILL_HEADROOM_MARGIN_BYTES` (128 MiB) of plan above the floor. This covers the measured
  26 MiB of drift 5x. It names the fix: restart, or `--moe-mirror-host-rows` to at least N.

Outputs cannot change: only the host pool's row count and the VRAM ledger move. The arena never goes below its floor,
and no kernel or numeric path changes. The whole-model residency (floor 0) is never refused.

Cost: 256K q8_0 pool 6737 → 6830 rows, 12.43 → 12.60 GiB pinned host RAM.

## 3. Results after the fix

First start with an empty record (p14):

| run | pool | floor / plan | result | outputs |
|---|---|---|---|---|
| q8_0 256K (262144) | 6830 rows, 12.60 GiB | 4184 / 4280 | rc 0. Pass 2: prefill 3184, decode 100.2. Pass 4: decode 62.9. 0 cov faults, 0 starved, 1 capture | = `-tmb` and p13 start2 (ac23a93b1c / 48179f17fa / 77d5bc9021) |
| q4_0 256K (262144) | | 4866 / 4960 | rc 0. Pass 2: prefill 909, decode 101.6. Pass 4: decode 78.7. 0 / 0 / 1 | = `-tmb` (b7e10f03a8 / abc3de1768 / 56a37d7d10) |
| q8_0 384K (393216, YaRN 2) | 7605 rows, 14.03 GiB | 3403 / 3528 | rc 0. Pass 2: prefill 2444, decode 84.5. Pass 4: decode 43.0. 0 / 0 / 1 | = A1..A3 (4d3171e231 / 8c1a3b4464 / 2504971632) |

q4_0 384K on p13 (record): rc 0. Pass 2: prefill 610, decode 86.3. Pass 4: decode 61.0. 0 coverage faults, 0 starved, 1 capture.
p11 had been refused on a later pass of the same run.

p13 (record only):
- first start: refused at load in 2m20s with "...recorded..., so starting again sizes the pool for it" (q8_0 and q4_0);
- second start: runs, with outputs identical to the `-tmb` runs.

Decode before/after at 128K/256K/384K, q8_0, 393216 ceiling, YaRN 2, pass 2 = probe, pass 4 = natural (tok/s):
- A = p11;
- B = p12, which behaves the same as p11 here, since its floor path never triggered;
- C = p13 with the record.

| ctx | A1 probe / natural | B | A2 | C | A3 |
|---|---|---|---|---|---|
| 128K | 120.2 / 87.2 | 120.2 / 87.2 | 120.3 / 87.2 | 118.2 / 86.5 | 119.8 / 87.3 |
| 256K | 99.8 / 65.2 | 99.8 / 65.9 | 99.8 / 66.0 | 99.9 / 65.8 | 99.8 / 65.9 |
| 384K | 84.6 / 44.4 | 84.7 / 44.5 | 84.6 / 44.5 | 84.5 / 43.0 | 84.7 / 44.4 |

Pass 1 at 384K: A 73.8 / 73.6 / 73.8, C 75.0, p14 75.1.

Output identity: every arm gives the same `osha` per pass and context as A1 (`summaries.txt`).

**The larger pool moves the 384K natural pass by −3.3%** (44.4–44.5 → 43.0, both C and p14; all within the 9% gate).
The counters are deterministic per configuration (A1 = A2, C ≈ p14). With the larger pool, pass 4 at 384K has:
- free evictions 5352 → 5264;
- writebacks 3690 → 3767;
- ring-full fallbacks 699 → 755;
- retained rows 4936 → 4792.

Probe pass 1 improves +1.8% (ring-full fallbacks 63 → 38). Pass 2 is flat. The 128K C arm, the first job after the box
restart, is −1.5% on pass 2. The staging ring (66 rows) and the arena (5256 slots) are identical across arms; only the
pool's row count differs (7511 → 7605 at 384K). Why the extra rows shift the eviction trajectory is not explained yet.
**Open.**

## 4. The −0.9% whole-model 8K (exp/exl3-on-reorg vs exp/ornith-exl3)

This was a repeat of `job-e2e.sh` in whole residency at 8K. The arms alternated A B A B A, with A = p8 (exp/ornith-exl3
content) and B = p11. Pass 2 decode:
- A: 148.2, 148.2, 148.1;
- B: 154.4, 152.9.

B is **+3.2..+4.2%**, so the −0.9% (165.1/163.6/165.1 in chain54) does not reproduce and is dismissed as noise.
The absolute level moved about 10% between the two sessions, which is box drift larger than the effect.
`exl3e2e-w8k-*`.

## 5. Other findings

- **q4_0 KV prefill is 3.5x slower than q8_0** at long prefix: 909 vs 3184 tok/s at 256K, and 611 vs 2447 at 384K.
  Decode is a little faster (pass 2: 101.9 vs 100.3 at 256K). The cost is the long-prefix extend kernel's Q4_0
  nibble unpack (`kernel/triton/attention.py`, FORMAT 2, "register-bound" per its comment) on a 32x32 tile at
  head_dim 256. This is not addressed here; it matters for the planned KV-speed eval.

## Files

- `job-reach.sh`, `job-reach2.sh` (output hashes), `job-ledger.sh` + `ledger_probe.py`, `longctx_decode2.py`, chain60..66.
  chain61 carried the refuted p12 arm; chain64 was killed by the 2026-09-25 stop, and chain65 re-ran its arms.
- `summaries.txt`: every run's summary and the p12/p13/p14 test totals. The raw logs are in `~/ai/box-archive/ft-dev-exl3/results/`,
  with the md5 manifest `chain60-66.md5` (220 files, verified).

## Tests (ft-dev, under /root/gpu.lock)

| tree | tests/engine | tests/scheduler | tests/moe |
|---|---|---|---|
| p13 (843f62d) | 268 passed, 2 skipped | 412 passed, 1 skipped | 420 passed, 5 skipped |
| p14 (2ad5dc2) | 270 passed, 2 skipped | 412 passed, 1 skipped | not rerun: the change since p13 is engine and memory_prediction only |

p12 (refuted, reset): 2 failed in tests/engine. One was its own negative test; the other was
`test_prefill_headroom.py`, whose stub has no `empty_cache`.
