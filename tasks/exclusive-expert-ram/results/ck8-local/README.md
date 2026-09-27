# Round 5: the saver's pass-1 gap, cause and fix (Ornith EXL3 5.0bpw, 262144 tokens, local RTX 5080, 2026-09-27)

**Verdict: every gate item passes. The local gate (ck8o) ran on exp/reorg-round5 4b34dd4, and the
suite on 7e1df41 (4b34dd4 + a test-stub fix). exp/reorg-round5 is merged into exp/reorg (local,
not pushed).** ft-dev answered `resources_unavailable` to every start from 03:13 on
(`_orch/r5/box-start.log`), so the orchestrator stopped the retry loop and the suite ran locally
with the model unloaded.

Setup is the same as ck6o/ck7o (`ck7-local/chain-r5.sh`, `CK_LABEL=ck8o CK_OUT=ck8-local/gate`):
* Saver = FT_ROWS=-1 with the pool's default reserve (3E = 768 rows). Whole = FT_ROWS=0.
* `FREETOKEN_PIN_BUDGET_GB=20`, q8_0 KV, ratio 1.00.
* LFU halving period 64 (unchanged, as decided).
* Both arms run from the same worktree and commit.

## The cause: not a cold-start storm, but DUP_BAND at LFU 64
Round 4 blamed pass 1 on writebacks after the arena regrows from the coverage floor. The direct
evidence says otherwise:
* `ck7-local/diag/`: the saver's 8K pass 1 had **4163 swaps and 0 writebacks** (176.4 tok/s).
  Pass 2 had the same swap count (183.2 tok/s).
* The swaps were the problem. Duplicate-aware eviction (DUP_BAND, e70849e) evicts the coldest
  duplicate among candidates with count ≤ min + max(1, √min). At LFU 64 the counts are small, so
  the band covers most candidates, and it evicts experts the next steps route to again.
* `ck7-local/dupband/`, 8K p1:

  | arm | swaps | tok/s |
  |---|---:|---:|
  | band | 4131 | 175.2 |
  | DUP_BAND=0 | 2767 | 193.2 |
  | pure LFU | 2296 | 184.6 |
  | whole | – | 203.4 |

* `ck7-local/policy/poltab.txt` (`--moe-collect-stats`), 8K p1 hit rate: band 0.954, tie-break
  only (tb1) 0.972, whole 0.982.
* nsys (`ck7-local/nsys/steps-graph-level.txt`, `kernels-node.txt`), graph GPU time per step:
  * saver 5397 µs vs whole 4742 µs;
  * fast_index_copy_kinds 554 µs vs 224 µs;
  * ensure kernel 227 vs 126 µs.

**Fix 1 (e472d20):** DUP_BAND is off by default (`FREETOKEN_MIRROR_DUP_BAND=1` restores it).
* This gave ck7o (`ck7-local/gate/`). 8K p1 and 256K p1 still failed.
* The rest came from writebacks: with the tie-break, the residents LFU evicts next mostly have no
  host copy, so pass 1 paid ~640 writebacks at 8K and ~900 at 80K (`ck7-local/seed/sdtab.txt`).
* The swap kernel only keeps copies of experts it has just admitted, and those are the hot ones.

**Fix 2 (fc66bf1 + 5c4dcd1, merged in 4b34dd4):** `MirrorResidency.idle_seed`.
* When to run: at each idle boundary (`Scheduler.run_when_idle`, single rank).
* Which experts: the residents coldest by the kernel's own key (LFU count + recency bonus, then
  last use) that have no pool row.
* Where their row comes from:
  * a free row above the reserve first;
  * otherwise, the row of a duplicate held by a strictly hotter resident. That resident stays on
    the GPU, so coverage is unchanged.
* The bytes are the resident's own GPU slot, copied to the pool: the path the arena shrink already
  uses. Outputs cannot change, and the pool keeps its size and reserve (no new RAM).
* Model-agnostic: it uses only the residency's maps and the policy key.
* Prefill-buffer slots are left alone.
* Never delays a request:
  * it checks for a waiting request before any work and after every 16 rows (~1 ms; the largest
    seed took 120 ms in total);
  * TTFT is unchanged (8K p1 0.33 s saver vs 0.40 s whole in ck8o-256k).
* `FREETOKEN_MIRROR_IDLE_SEED=0` disables it. `/v1/stats` reports `mirror.idle_seed_rows`.
* Unit tests are in `tests/moe/test_mirror_retention.py`:
  * the seeded bytes match the golden copy, and coverage and reserve hold;
  * it stops for a waiting request;
  * it yields between chunks with consistent maps;
  * the next decode pays fewer writebacks.

## Idle-seed A/B (`ck7-local/seed/`, `--moe-collect-stats`, arms alternated)
| arm | 8K p1 | 8K p2 | 80K p1 | 80K p2 | natural | 8K p1 wb / hit |
|---|---:|---:|---:|---:|---:|---:|
| tb1 (ck7o code) | 92.2 / 91.2% | 88.7 / 94.2% | 94.6 / 93.8% | 94.3 / 93.7% | 91.8% | 641 / 0.972 |
| seed (tie-break + seeding) | 96.1 / 94.8% | 94.9 / 94.9% | 97.0 / 95.8% | 93.8 / 94.6% | 91.6% | 0 / 0.987 |
| seed0 (pure LFU + seeding) | 94.7 / 95.1% | 86.8 / 96.5% | 94.7 / 96.5% | 96.8 / 90.9% | **88.4%** | 0 / 0.984 |

* At 256K (one pair, `sd-256k-compare.txt`) the seed arm scored 103.7 / 94.6 / **96.0** / 97.1%,
  with 0 writebacks.
* Pure LFU loses natural text, so the tie-break stays on.
* Seeding does not help natural text: its writebacks happen inside long continuous decodes
  (~132K per arm), where there is no idle boundary.

## Gate ck8o on 4b34dd4 (`ck8-local/gate/`)
| item | result | verdict |
|---|---|---|
| 8K/80K alternated x3, median, bar 91% (`ck8o-compare.txt`) | 8K p1 **95.3%** (93.4/96.2/95.3), 8K p2 95.1%, 80K p1 95.9%, 80K p2 93.4%; lowest single run 93.3% | PASS |
| 256K same commit (`ck8o-256k-compare.txt`) | 8K p1 97.9%, 8K p2 98.1%, **256K p1 98.3%** (116.3 vs 118.3), 256K p2 93.7% | PASS |
| needles/recall 256K (`ck8o-needles-compare.txt`) | 0 differences in 17 items | PASS |
| coverage faults / starved / captures | 0 / 0 in every saver arm; captures=1, 0 tracebacks in all 11 arms (`*-acceptance-R3.txt`) | PASS |
| natural text x2 (`ck8o-nat-compare.txt`) | whole 168.0 / 167.5, saver 154.2 / 154.8 tok/s = 91.8% / 92.1%; md5 5/5 in all arms | PASS |
| /clear replay (`ck8o-replay-compare.txt`) | cached counts identical to ck7o (and ck7o to ck6o, `ck7-local/gate/ck7o-replay-compare.txt`) for all 15 requests | PASS |
| RAM (256K arm) | whole ram_gib 21.94 (RSS 22.74); saver ram_gib 18.69, RSS 17.58, anon 16.77 GiB (ck7o: 16.64 / 17.59 / 16.84) | recorded |
| suite, 8 dirs, local RTX 5080 with the model unloaded (`suite/`) | on 5cb4d78 (orchestrator run, `suite/orch-local-5cb4d78-*`): 4 failed, 3432 passed, all 4 in `test_moe_stats_logging.py`. Cause: AttributeError, the SimpleNamespace scheduler stub lacked the new `_mirror_idle_seed`. **Fixed** in 7e1df41 (a no-op on the stub, like its `_maybe_retune_pageable_layers`). On 7e1df41: rc 0, **3436 passed**, 27 skipped, 0 failed, 0 illegal access | PASS |

* The saver's ram_gib (a MemAvailable difference) reads 2 GiB higher than in ck7o, but RSS and
  anon memory match ck7o within 0.1 GiB. The pool is pinned once at startup and idle seeding
  allocates nothing. So the difference is host page-cache noise, not the fix.
* The 1-min host load at the arm starts was 1.4–8.1 (`gate/chain-status.txt`). It was 2.1–10.6
  during the 8K/80K part (`gate/load5s.txt`), so the three pairs there ran on a loaded host; the
  lowest single pair still scored 93.3%. The 256K saver arm started at load 2.2 (quiet poll 0).

## Pass 1 before and after (saver / whole, same commit)
| point | ck6o (round 4) | ck7o (DUP_BAND off) | ck8o (+ idle seed) |
|---|---:|---:|---:|
| 8K p1 (x3 median) | 88.3% | 90.0% | **95.3%** |
| 80K p1 (x3 median) | 88.6% | 94.7% | **95.9%** |
| 256K p1 | 87.4% | 88.0% | **98.3%** |
| 8K p1 in the 256K arm | 89.8% | 92.8% | **97.9%** |
| natural text | 90.5 / 92.7% | 92.3 / 93.3% | 91.8 / 92.1% |

## Open
* Natural text sits ~1% above the bar. Its residual is the in-decode writeback cost: ~132K
  writebacks per arm, 0.18 per decode layer-call (`ck7-local/seed/`, sd-nat stats). Idle seeding
  cannot reach it, because there is no idle boundary inside a long decode.
* The suite has not run on ft-dev (Blackwell box) for the seed code; it passed on the local RTX 5080
  (sm_120 as well).
