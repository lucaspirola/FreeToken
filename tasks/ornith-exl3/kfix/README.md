# EXL3 expert kernels, bit-exact speed-up (Ornith EXL3 5.0bpw, local RTX 5080, 2026-09-27)

**Branch `exp/exl3-kfix` from exp/reorg b967140.** Every change keeps the bits: 206 kernel and
fused outputs are byte-identical to b967140 (`results/hash-diff.txt`). The end-to-end A/B matches
too: `out_sha1` in every probe arm, and md5 of the natural-text arms.

Summary:
* Decode is untouched; it had nothing to fix (step 1).
* Prefill MoE per layer:
  * 8192 tokens: −3%;
  * extends under 256 tokens: 4–5× faster.
  * EXL3 extends always run the fused prefill, because cached extends are NVFP4-only.
* In the server:
  * saver TTFT: 64-token extends −33%, 150-token −40%;
  * whole TTFT: 150-token extends −20%;
  * prefill at 32K–256K: +0.5 to +1.6%;
  * decode: unchanged.
* The gate (ck9k) and the suite pass. Merged into exp/reorg.

## Step 1: the decode path has no I2F.U64 / FP64
* **SASS of every cached cubin** (cuobjdump, 249 `_exl3_gemv_kernel` variants): 0 I2F.U64, 0 DADD/DMUL/DFMA.
  * The GEMV already decodes with the 32-bit `_decode_words`.
  * I2F.U64 was only in the prefill kernels:
    * `_exl3_gemm_kernel`, 66 of 310 variants (the in-kernel-decode ones);
    * `_reconstruct_experts_kernel`, 16 of 48.
* **ncu** of one decode MoE call (`ncu_decode.py`, `profile/ncu-decode-base-details.txt`):

  | kernel | time | limiter |
  |---|---:|---|
  | gate/up GEMV | 22.2 µs | DRAM 50% (478 GB/s), 0.51 waves of single-warp blocks, stalls mostly long scoreboard |
  | down GEMV | 13.3 µs | DRAM 42% |
  | splitk_silu_had | 5.8 µs | |
  | combine | 2.2 µs | |

* Latency/ramp-bound, as found in D2 (`tasks/ornith-exl3/perf/d2-gemv/README.md`). The only
  bit-exact knob, BANDS, was already swept there. Changing the split changes the fp32 order, so it
  is not bit-exact.
* **Nothing changed on decode.** The e2e decode tok/s moves within ±2%, noise.

## Step 2: 32-bit trellis decode (`_exl3_decode` → `_decode_words`)
**Cause.** `_exl3_decode` built the 16-bit window as uint64 (`(a << 32 | b) >> shift`). That made
the mul1 byte sum convert u64→fp32 (I2F.U64), which runs on the FP64 pipe on GeForce.

**Why the 32-bit path is bit-exact:**
* shift is in [0, 31], so a 32-bit funnel shift gives the same window.
* The multiplies only ever kept the low 32 bits.
* s + 1024 ≤ 2044 converts exactly from either width. So the fp32 FMA and the fp16 rounding are the
  same.

**Tests** (`tests/kernels/test_exl3_bitexact.py`):
* exhaustive: every 16-bit state at every shift 0..31, all 3 codebooks, random surrounding bits;
* compared with the old uint64 formula (kept verbatim in the test) and with the torch `_decode_codes`;
* bitwise equality is required.

**Result:**
* new cubins have 0 I2F.U64 and 0 FP64 ops;
* `_reconstruct_experts_kernel` 2.45 → 2.28 ms per 8192-token call, −7% (`results/prof-*.txt`).

## Step 3: `_had_rows_kernel`, H in column slices
**What changed.** From 4096 rows on, H is multiplied in two 64-column slices, each loaded per block
(`NS=2`), with 64-row tiles (`HAD_ROWS_BIG`).

**Bit-exact:** every output element keeps the same K=128 hi and lo dots, so any BM / NS / KB / warps
gives the same bits. The sweep asserted this for every variant, and `test_had_rows_slices_equal_held_h`
covers tails that don't divide 64.

**Registers:** 255 → 184 per thread.

**Sweep** (`sweep_had_rows.py`, then interleaved re-timing `results/rerun2-had-rows.txt`):

| input | base | new |
|---|---:|---:|
| gate_up | 0.744 ms | 0.673 ms (−9.6%) |
| down | 0.244 ms | 0.201 ms (−17.6%) |

**Rejected variants:**
* holding both H slices spills (×2.5);
* stacking [hi; lo] into one dot changed bits;
* BM 128 and 8 warps lose.

## Step 4: `_exl3_gemm_kernel`: in-kernel-decode tile (a bit-exact gain was evident)
**Symptom.** A 9-token MoE prefill cost 3.8 ms in this GEMM alone. The decode rewrite did not move it.

**ncu** (`profile/ncu-gemm9-*`):
* L1-bound: L1/TEX 95% of peak, DRAM 28 GB/s, issue slots 6.5%, 255 registers;
* MIO stalls.

**Two causes.**
1. **Gathers span too many cache lines.** In the [BK, 128] layout, a warp's decode gathers span 8
   tiles.
   * **Fix:** decode tile-major ([BK/16, 8, 16, 16]), then permute to [BK, 128].
   * Same values into the same dot, and a warp now reads one tile's 40 words.
2. **Pipelining routes the gathers through shared memory.** Two stages software-pipeline the
   gathers via shared memory.
   * **Fix:** in-kernel-decode default bk 16 / 1 stage (the decoded path passes its own config).
   * Tile shape and stages keep each output's 16-wide mma k-step sequence, so outputs are bitwise
     the same (sweep asserts it: `results/sweep-inline-gemm.txt`).

**Fused MoE prefill per layer** (`results/rerun-inline-gemm.txt`, interleaved medians):

| tokens | b967140 | new |
|---:|---:|---:|
| 9 | 3.54 ms | 0.75 ms |
| 64 | 12.54 ms | 2.42 ms |
| 200 | 14.41 ms | 2.86 ms |
| dense 12 rows | 0.46 ms | 0.13 ms |

**The in-kernel and decoded paths are bitwise equal** at every M, on both trees
(`results/crossover*.txt`); the test now asserts `torch.equal`.
* The switch point (`PREFILL_DECODED_MIN_TOKENS`, 256) is therefore a pure speed knob.
* Offline, the new in-kernel GEMM wins up to ~1100 tokens.
* In the server the move did not pay:
  * the saver's 1000-token TTFT pass 2 was 0.416 / 0.423 s new vs 0.385 / 0.360 s base (`e2e/`);
  * one nsys-traced request each (`nsys-mid/`) showed less MoE kernel time (300 tokens: 133 vs 181 ms);
  * at that size the layer stream (`fast_index_copy_multi`, ~270 ms of SM-driven copies) dominates.
* **Reverted to 256** (4fe9bc3). Only demonstrated gains are kept.

## Kernel-level A/B (`bench_prefill.py`, alternating base/new, `results/bench-*.tsv`)
| path | b967140 | kfix |
|---|---:|---:|
| MoE prefill 8192 tokens | 11.68 ms | 11.33 ms (−3.0%) |
| MoE prefill 9 tokens | 4.07 ms | 0.89 ms |
| dense exl3_forward 8192 / 64 rows | 0.64 / 0.16 ms | same (noise) |

Per 8192-token call (`results/prof-had.txt`):
* had_rows 1.64 → 1.45 ms;
* reconstruct 2.45 → 2.28 ms;
* GEMM unchanged (decoded path: its config is untouched).

## Step 5: end-to-end A/B vs b967140 (`e2e/`, `e2e_table.py`)
**Setup:**
* kbase (b967140) vs this tree, arms from each tree's `final/arm.sh` on :1920;
* ratio 1.00, q8_0, 262144 ceiling;
* ABBA per mode;
* probe 300/1000/8K/32K/80K/256K × 2 passes, 512 tokens;
* natural text one pair per mode.

**Outputs:**
* `out_sha1` identical in all 4 arms per mode × every size and pass;
* natural md5 5/5 in both modes;
* captures=1, 0 tracebacks, 0 coverage faults / starved writebacks in all 12 arms.

**Speed.** Prefill at 32K–256K is +0.5 to +1.6% (MoE is part of prefill and overlaps the layer
stream). Decode is within ±2%.

| mode | 256K p1 prefill tok/s | 80K p2 | 8K p2 decode | natural decode tok/s |
|---|---:|---:|---:|---:|
| saver base / new | 3320 / 3320 | 5861 / 5774 | 185.2 / 185.4 | 154.4 / 152.3 |
| whole base / new | 3365 / 3396 | 5931 / 6025 | 194.1 / 192.6 | 168.3 / 167.8 |

That table ran on 69938fe (switch point at ~1152 tokens). The final code has the base switch point.
The two paths are bitwise equal, so the outputs stand for both.

### Short extends on the final code (4fe9bc3, `e2e2/`)
**Setup:** probe 64/150/300/1000/8000 × 2 passes, 128 tokens, ABBA per mode.

**Outputs:** `out_sha1` identical in all arms at every size and pass; captures=1, 0 tracebacks,
0 coverage faults. EXL3 extends always take the fused prefill (cached extends are NVFP4-only), so
extends under 256 tokens run the new in-kernel GEMM.

TTFT, median of two arms per tree:

| mode | 64 p1 | 64 p2 | 150 p1 | 150 p2 | 300 p2 | 1000 p2 |
|---|---:|---:|---:|---:|---:|---:|
| saver base | 0.543 | 0.495 | 1.173 | 0.608 | 0.374 | 0.367 |
| saver new | **0.364** | **0.334** | **0.444** | **0.362** | 0.351 | 0.358 |
| whole base | 0.392 | 0.418 | 0.524 | 0.508 | 0.415 | 0.417 |
| whole new | 0.395 | 0.403 | **0.448** | **0.404** | 0.409 | 0.412 |

**Reading the table:**
* **Saver:** 64-token extends are −33%, 150-token −40%.
* **Whole:** 150-token extends are −20%. At 64 tokens whole TTFT stays at ~0.40 s, where the
  layer stream sets the floor.
* **Unchanged at 300 and 1000 tokens**, because both trees take the decoded path there.

**First-request outliers** (excluded from the reading above):
* 1.746 s at base-saver-1 150 p1;
* 0.622 s at new-saver-1 300 p1.
Each is the first request of that size in a fresh server, and each appears in only one arm.

**One decode outlier:** new-whole-1 300 p1 at 43.4 tok/s, from one 2.55 s gap.
* The journal shows the server decoding at 204–211 tok/s and idle by 17:11:06.
* The 1-min host load was 9–11 (other processes), so the stall was outside the GPU work.
* The second new arm decoded at 192 tok/s.

## Step 6: gate ck9k and suite (`gate-chain.sh` = ck8o's `chain-r5.sh` on this tree, `gate/`)
Code 4fe9bc3; e90ac87 differs from it only in tasks/ files. Ornith EXL3 5.0bpw at 262144, q8_0,
ratio 1.00. Saver = mirror with the pool's default reserve; whole = pin budget 20 GiB.

| item | result | verdict |
|---|---|---|
| 8K/80K alternated x3, bar 91% (`ck9k-compare.txt`) | 8K p1 **94.3%**, 8K p2 **91.1%** (93.7 / 91.1 / 90.8), 80K p1 95.9%, 80K p2 101.2% | PASS (8K p2 thin, see below) |
| 256K same commit (`ck9k-256k-compare.txt`) | 8K p1 97.1%, 8K p2 96.5%, 256K p1 96.9% (112.1 vs 115.7), 256K p2 97.6% | PASS |
| needles/recall 256K (`ck9k-needles-compare.txt`) | 0 differences | PASS |
| coverage faults / starved / captures | 0 / 0 in every saver arm; captures=1, 0 tracebacks in all 15 arms (`*-acceptance-R3.txt`) | PASS |
| natural text x2 (`ck9k-nat-compare.txt`) | whole 167.6 / 168.1, saver 154.6 / 154.4 = 92.2% / 92.1%; md5 5/5 in every arm, also 5/5 against ck8o's whole arm | PASS |
| /clear replay (`ck9k-replay-compare.txt`) | cached counts identical to ck8o for all 15 requests. The 2 "cold" lines differ by one prompt token, as ck8o vs ck7o did | PASS |
| RAM (256K arm) | whole ram_gib 21.74 (RSS 22.51); saver ram_gib 14.36, RSS 17.59, anon 16.82 GiB (ck8o: 18.69 / 17.58 / 16.77) | recorded |
| suite, 8 dirs, model unloaded (`suite/`) | see below | |

**8K p2 at 91.1%.** The saver side is lower than in ck8o: mirror 185.1 / 184.8 / 187.4 tok/s vs
195.0 / 192.4 / 192.9. Whole is about the same, 197.5–206.3 vs 202.5–204.7.
* This does not come from this branch: the decode kernels are unchanged, and outputs are byte-equal.
* The same-day e2e A/B measured the saver's 8K p2 decode at 185.2 on b967140 vs 185.4 on this tree.
* Two differences from ck8o's setup:
  * ck8o ran on the round5 worktree with the 09-25 builds of `_pinned_tensor` / `_pageable_stage` /
    `_cpu_moe`;
  * this gate, and both e2e trees, ran with the 09-06 builds copied from the main tree (see the suite).
* **Same-day control** (`control-8k.sh`, `gate-control/`): the identical recheck, 3 alternated pairs,
  on b967140 (ck9b) and then on this tree (ck9n), back to back, with the same builds:

  | point | b967140 ck9b | kfix ck9n |
  |---|---:|---:|
  | 8K p1 | 95.1% | 96.4% |
  | 8K p2 | 93.2% | 94.2% |
  | 80K p1 | 92.8% | 94.6% |
  | 80K p2 | 95.7% | 96.6% |

  kfix is at or above the base on every point. The gate's 91.1% was run-to-run spread of the saver
  arms, not this branch.

**Suite.**
* **First run** (`suite/`, same builds as the gate): 1 failed, 3441 passed.
  * The failure: `test_cpu_extension_supports_swiglu_clamp`, "compiled _cpu_moe extension is stale".
  * Cause: the untracked extension builds had been copied from the main tree (built 09-06), which
    predate ACT_SWIGLU_CLAMP.
  * Fix: installed the exp/reorg worktree's builds (09-25: `_cpu_moe`, `_pageable_stage`,
    `_pinned_tensor`, `_ple_store`). No code change.
* **Re-run** (`suite2/`, md5s in `suite2/status`): rc 0, **3442 passed**, 27 skipped, 0 failed,
  0 illegal memory access.
* The gate, the e2e A/B (both trees) and the control ran with the 09-06 builds. The live server's
  main tree has the same ones.

## Verdict
Faster, identical outputs, suite and gate pass. Merged into exp/reorg (local, not pushed).

**Kept:**
* the 32-bit decode;
* sliced-H had_rows;
* the in-kernel-decode GEMM tile (tile-major, bk 16, 1 stage).

**Reverted:** the routes-per-expert switch point, which was not faster in the server.

**Open items:**
* The decode GEMV is latency/ramp-bound. Nothing bit-exact is left there; D3 (fusing the shared
  expert) is the known lever.
* At 300–1000 tokens, TTFT is set by the layer stream (`fast_index_copy_multi`), not the MoE kernels.
* The main tree's extension builds (09-06) are stale for `test_swiglu_clamp`. Rebuilding them in the
  owner's tree is the owner's call.

## Files
* `ncu_decode.py`, `ncu-decode.sh`, `ncu_moe_m.py`, `ncu-m.sh`: ncu drivers.
* `equal_hash.py`, `equal-hash.sh`: byte A/B of two trees.
* `bench_prefill.py`, `bench-ab.sh`, `prof_prefill.py`, `prof-ab.sh`, `bench_mid.py`: timing.
* `sweep_had_rows.py`, `rerun_had_rows.py`, `sweep_inline_gemm.py`, `rerun_inline_gemm.py`,
  `crossover*.py`: tile sweeps.
  * The crossover results were recorded under the routes-per-expert name. The scripts now set
    `PREFILL_DECODED_MIN_TOKENS`.
* `e2e-chain.sh`, `e2e2-chain.sh`, `e2e_table.py`, `nsys-mid.sh`: server A/B.
* `pytest-unit.sh`: tests under the lock.
