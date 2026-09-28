# popt-sched: prefill scheduling / movement optimizations (owner-approved 2026-09-27)

Branch `exp/popt-sched`, from exp/reorg `757caee`. Ornith EXL3 5.0bpw, q8_0 KV, 262144 ceiling,
ratio 1.00, RTX 5080 (local). Saver = `--expert-residency mirror`, pool auto; whole = pin budget
20 GiB. All A/B arms are ABBA, back to back, on :1920 under the GPU host lock
(`arm.sh`, `ab1.sh`, `ab2b.sh`, `ab3.sh`); base tree = `/home/lucas/ai/FreeToken-wt/popt-base`
(detached 757caee, same extension builds). Every arm serves the owner's shapes:
probe 300/1000/8K/32K/80K (pass 2 reported), file-read extends (`extend_probe.py`: 10K and 30K of
code appended to a 100K context, 30K at 200K; prefix cached, reported per pass), 8K/80K/256K
decode, natural text (5 tasks, 3500 tokens).

## What the profile said (757caee, `profile/base/`, traces in `_orch/popt/profile-base/`)

| cause | evidence | size |
|---|---|---|
| Triton recompiles `_shared_expert_add_kernel` for every new prompt length (`n_elements: tl.constexpr`) — every file-read extend has a new length | `profile/base/biggaps-saver-d100000-a10000.txt`: one `cuModuleLoadData` per extend trace, gaps 178.5 / 183.1 / 276.7 ms | 180–280 ms per extend |
| Dynamic prefill headroom always reserves the full 8192-chunk transient (1.10 GiB), even for a 300-token prefill; the release before the first token re-maps it | journals `results/ab1/b-s1-journal.txt` ("Prefill headroom reserved/released", 32–40 ms each) | ~70 ms around every short prefill (saver) |
| Whole mode re-copies all 256 experts of every layer over PCIe, including the ones already resident | `results/ab1/compare-whole.txt` (`--moe-prefill-hit-d2d` arms) | 300/1000-token TTFT 0.46/0.39 s |
| Layer 0's expert copy is issued by layer 0's MoE block (after its GDN/attention and router); its GEMMs wait for the whole copy | `_orch/popt/profile-d91a342/ext-whole-d100000-a10000` timeline: copy 29.67→38.1 ms, `moe_align` at 38.69 | ~8 ms per chunk forward |
| Saver misses copied by an SM gather instead of DMA | `results/bench-miss.txt`: 51.7 GB/s (any blocks_per_bank) vs `cudaMemcpyBatchAsync` 54.8 GB/s; neither slows a concurrent GEMM | ≤0.3 ms/layer — not worth it |
| Attention at depth | roofline (`roofline_ext.py`): flashinfer prefill at ~94% of peak at 200K | no headroom |
| Copies/casts around EXL3 dense GEMMs and GDN (8.4% of fresh 80K) | `profile/base/neighbours-saver-fresh-80000.txt` | popt-exl3's files → `_orch/popt/handoff-popt-sched.md` |

"Idle" decode gaps in `gaps.py`/`biggaps.py` (8–12 ms `memcpy4 -> reduce_kernel`) are the decode CUDA
graph (traced as graph activity, not kernels) — a tool artifact, not idle time.

## Changes kept (all generic: no model-, bpw- or arch-specific case)

1. **Sized prefill headroom** (`engine/growable_kv.py`, `engine/engine.py`, `scheduler/scheduler.py`).
   Startup also measures the transient of a small chunk (`FREETOKEN_SMALL_PREFILL_TOKENS`, default 1024;
   Ornith: 0.30 GiB vs 1.10 GiB). A prefill batch of ≤ that many tokens reserves only that; a larger
   batch after a small reserve re-reserves the full chunk. Reserve 32–40 → 11–16 ms, release
   33–40 → 10–15 ms around short prefills (`results/ab1/n-s1-journal.txt`); full releases per arm
   17 → 11 (`results/ab2b`). Tests: `tests/engine/test_dynamic_prefill_headroom.py`.
2. **One shared-expert epilogue build per model** (`kernel/triton/shared_expert.py`): `n_elements` is a
   runtime argument in both kernels. Byte-identical to the per-length build and one compiled variant
   for any length (`tests/kernels/test_shared_expert_epilogue.py`; the old kernel fails it with 6 and 4
   builds, `results/ktest2-old.txt`). First call on a new length: 25–26 ms → 0.06–0.24 ms in isolation
   (`results/bench-jit.txt`); in the server the stall was 178–277 ms and is gone from the new traces
   (`_orch/popt/profile-d91a342/`). Not visible in the A/B tables because the probe lengths were
   already in the on-disk Triton cache; the owner's real file reads always have new lengths.
3. **Hit/miss split for short chunks** (`moe/offload_cache.py`, `layers/moe.py`, `engine/config.py`,
   `server/args.py`, `docs/cli.md`): chunks of ≤ `--moe-prefill-hit-d2d-tokens` (default 2048) gather
   resident experts device-side and DMA only the misses, without `--moe-prefill-hit-d2d`. Longer
   chunks keep the full-layer copy: there the device-side gather is a compute-stream cost, which is
   what `docs/models.md` recorded as slower on the Ada stack. The 2048 threshold is [agent practice]
   (measured gains at 300/1000, neutral at 8K). Tests: `tests/moe/test_prefill_hit_d2d_auto.py`.
   No effect on the saver (its residency assembles prefill layers itself).
4. **Layer 0 primed before the forward** (`engine.forward_batch` → `OffloadMoeCache.prime_prefill`,
   consumed by layer 0's `take_primed_prefill`). Skipped when the setup would wait on the host while
   the GPU is still busy (saver snapshot sync, short-chunk split sync on a continuation chunk): the
   first version drained the GPU at every saver chunk boundary (saver 80K −2%,
   `results/ab2/compare-saver.txt`), fixed in daf82a4.

Reverted: batched `cuMemSetAccess` over contiguous VMM runs (c756c4e). The release made 151
SetAccess calls (15.85 ms, nsys API sum), the microbench gained 1–3 ms (`results/bench-vmm.txt`),
but full releases in the server were 35.8–48.1 ms base vs 36.6–46.4 ms new — no measurable gain.
The round-2b "new" arms still had it; it is behavior-neutral (same mappings).

## Results (final code; `compare.py` tables, cells TTFT s / prefill tok/s / decode tok/s)

Every output identical to 757caee in every arm: probe/extend `out_sha1` (column "identical": yes on
every row of every table), natural md5 5/5.

**Whole** (`results/ab2b/compare-whole.txt`, b-w1+b-w2 vs n-w1+n-w2):

| shape | 757caee | popt-sched | Δ |
|---|---:|---:|---:|
| TTFT 300 | 0.427 s | 0.291 s | **−32%** |
| TTFT 1000 | 0.392 s | 0.291 s | **−26%** |
| 8K / 32K / 80K prefill tok/s | 8662 / 8044 / 6284 | 8745 / 7944 / 6327 | +1.0% / −1.2% / +0.7% |
| ext 10K@100K p1 / p2 | 3.865 / 3.931 s | 3.882 / 4.002 s | −0.4% / −1.8% |
| ext 30K@100K p1 / p2 | 12.721 / 12.871 s | 12.512 / 12.602 s | +1.7% / +2.1% |
| ext 30K@200K p1 / p2 | 19.836 / 19.227 s | 18.821 / 19.013 s | +5.4% / +1.1% |
| fresh ctx 200K p1 / p2 | 52.33 / 50.58 s | 49.60 / 49.67 s | +5.5% / +1.8% |

**Saver** (`results/ab2b/compare-saver.txt`, b-s3+b-s4 vs n-s3+n-s4):

| shape | 757caee | popt-sched | Δ |
|---|---:|---:|---:|
| TTFT 300 | 0.362 s | 0.350 s | −3% |
| TTFT 1000 | 0.343 s | 0.314 s | **−8%** |
| 8K / 32K / 80K prefill tok/s | 8434 / 7738 / 6110 | 8343 / 7806 / 6060 | −1.1% / +0.9% / −0.8% |
| ext 10K@100K p1 / p2 | 4.005 / 4.039 s | 3.990 / 3.967 s | +0.4% / +1.8% |
| ext 30K@100K p1 / p2 | 13.113 / 12.742 s | 12.806 / 12.742 s | +2.4% / 0 |
| ext 30K@200K p1 / p2 | 19.527 / 19.024 s | 19.889 / 19.094 s | −1.8% / −0.4% |

Per-arm spread on the saver is 2–4% (see the per-arm columns); only the short-prompt TTFT moves
beyond it. Round 1 (`results/ab1/`, lever A + JIT fix + whole `--moe-prefill-hit-d2d` for every
chunk) showed saver 32K/80K +4.5%/+3.3%; round 2b did not reproduce it, so it is not claimed.
The round-2 n-s1 300-token pass-2 TTFT of 1.114 s is the harness: it arrived while the previous
80K session was still being torn down ("KV shrank … after agent teardown", `results/ab2/n-s1-journal.txt`).

**Decode and natural text** (`results/ab3/compare-*.txt`), bar ~2%:

| | 757caee | popt-sched | Δ |
|---|---:|---:|---:|
| saver decode 8K / 80K / 256K | 179.6 / 155.1 / 108.7 | 181.1 / 155.1 / 114.2 | +0.8% / 0 / +5.1% |
| whole decode 8K / 80K / 256K | 186.7 / 164.0 / 120.8 | 183.9 / 164.4 / 121.1 | −1.5% / +0.2% / +0.2% |
| saver natural (tok/s, md5) | 155.0, 154.9 | 150.9, 155.2 | −1.2%, 5/5 |
| whole natural (tok/s, md5) | 167.5, 167.7 | 169.9, 169.0 | +1.2%, 5/5 |

## Gate ck9s and suite (`gate-chain.sh` = kfix's gate chain on this tree, `gate/`, `suite/`)

Code c756c4e (4a1bb45 differs only in tasks/). Saver = mirror, pool auto; whole = pin budget 20 GiB.

| item | result | verdict |
|---|---|---|
| 8K/80K alternated x3, bar 91% (`gate/recheck-8k.log`) | 8K p1 **92.9%**, 8K p2 **94.4%**, 80K p1 **93.9%**, 80K p2 **94.7%** | PASS |
| 256K same commit (`gate/ck9s-256k-compare.txt`) | 8K p1 92.8%, 8K p2 97.6%, **256K p1 88.3%** (107.1 vs 121.3), 256K p2 96.1% | one point below, see control |
| needles/recall 256K (`gate/ck9s-needles-compare.txt`) | 0 differences (recall 120000 / 240000 SAME) | PASS |
| coverage faults / starved / captures | 0 / 0 in every saver arm; captures=1, 0 tracebacks in every arm (`gate/*-acceptance-R3.txt`) | PASS |
| natural text x2 (`gate/ck9s-nat-compare.txt`) | whole 168.7 / 169.0, saver 154.9 / 155.5 = 91.8% / 92.2% (ck9k: 92.2 / 92.1); md5 5/5 in every arm, also 5/5 vs ck8o | PASS |
| /clear replay (`gate/ck9s-replay-compare.txt`) | cached counts identical to ck8o for all 15 requests; the 2 "cold" lines differ by 1 and 5 prompt tokens, as in ck9k | PASS |
| RAM (256K arms) | whole ram_gib 22.01 (RSS 22.93); saver ram_gib 16.64, RSS 17.61, anon 16.78 (ck9k: 14.36 / 17.59 / 16.82) | recorded |
| suite, 8 dirs, model unloaded (`suite/status`) | rc 0, **3458 passed**, 27 skipped, 0 failed, 0 illegal memory access (builds md5-identical to exp/reorg's) | PASS |

**256K p1 at 88.3%.** One saver reading (107.1) below every other one of this code:
* ab3 (same code path for decode, same probe with an 80K step): saver 256K p1 118.3 / 117.9 on this
  tree vs 113.6 / 115.6 on 757caee; whole 117.0-118.8 (`results/ab3/*-probe.jsonl`).
* **Same-day control** (`control-256k.sh`, `results/control-256k/compare.txt`), the gate's shape
  ("8000 256000"), alternated: this tree saver 113.8 / 117.4 vs whole 120.8 / 120.8 = **95.7%**
  (p2 96.7%); 757caee saver 116.6 / 112.9 vs whole 119.3 / 121.8 = 95.2% (p2 94.2%). This branch is at or above the base.
* Nothing in this branch runs in decode (headroom, prefill copy, shared-expert build, layer-0 prime
  are prefill-side), and every output is byte-identical. The gate's point is run-to-run spread of a
  single saver arm; not fixed, nothing to revert.
* Control artifact: `cn-w1` 8K p2 reads 259.6 tok/s (127 tokens in 0.49 s after a 1.16 s TTFT: the
  first-token wait absorbed part of the decode), which makes the control's 8K p2 ratio meaningless;
  the gate's own 8K p2 is 94.4%.

## Handoff

`/home/lucas/ai/FreeToken-wt/_orch/popt/handoff-popt-sched.md`: copies/casts next to cuBLAS `Kernel2`
and GDN (1136 ms of 13.44 s at fresh 80K), routed experts at 3.2–3.4× their roofline bound at depth,
and the constexpr-length pattern to check in the EXL3 kernels.

## Files

Harness: `arm.sh` (one arm; per-arm `--session-spill-dir` so arms cannot restore each other's
sessions from disk), `extend_probe.py`, `compare.py`, `natcmp.py`, `nsys-extend.sh`,
`roofline_ext.py`, `neighbours.py`, `gaps.py`, `biggaps.py`, `bench_miss.py`, `bench_jit.py`,
`bench_vmm.py`, `lockrun.sh`, `gate-chain.sh`, `gate_compare.py`, `close-chain.sh`.
Large traces: `_orch/popt/profile-base/` (757caee), `_orch/popt/profile-d91a342/`.

## Verdict

Faster on the owner's short-prompt and file-read shapes (whole TTFT −26–32%, saver 1000-token TTFT
−8%, whole extends at depth +1–5%, the per-length Triton compile stall of 180–280 ms per file read
gone), no decode regression beyond ~2%, byte-identical outputs everywhere. Gate ck9s passes (the one
256K p1 reading below 91% did not reproduce in the same-day control, where this branch is at or
above 757caee); suite 3458 passed. Merged into exp/reorg (local, not pushed).
