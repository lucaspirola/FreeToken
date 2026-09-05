# Nemotron 3.5 Lightning 30B-A3B-NVFP4 on RTX 5080 (Switchyard) — 2026-09-03

Full plan: tasks/nemotron35-plan.md

## Phase 1 — bring-up
- [x] 1A model package
- [x] 1B engine/CLI/AOT/docs/preflight
- [x] 1C gates (parity, invariance, prefix, elastic, tool, 64K needle) — [x] 128K needle root cause
      (benchmark bug in `trim_filler`, not the engine; 131 072 needle now exact at 4K and 8K chunks)
- [ ] Focused tests + ruff + commit

## Phase 2 — kernels
- [x] T0 flashinfer SSU probe (usable)
- [x] 2A1 layout/metadata/wiring (state_layout kv|mamba2, track_chunk_size 64|128, FLAMetadata.mamba2)  - [x] 2A2 SSD prefill (kernels validated)  - [x] 2A3 decode SSU + gated norm
- [x] 2B1 b12x relu2 (auto default; decode M≤4 to be settled by 2B4)  - [x] 2B3 dense NVFP4 tuning
- [x] 2A4 integrate (P1 smoke + all 4 P2 serving gates + 32K needle + A/B vs FREETOKEN_MAMBA2_REF=1;
      fixed a CUDA-graph use-after-free in the decode out-buffer; 131K needle at 8K chunks still open --
      benchmarks/results/nemotron35_lightning_5080_mamba2_2026-09-04.md)
  - [x] 2B2 triton fallback tuning  - [x] 2B4 cache sizing study (triton default, LFU for 16-way, hybrid rejected)  - [x] 1M single-session spill gate (all 4 criteria PASS, 2026-09-04; restore NVMe 1.32 GiB/s vs RAM 5-8 GiB/s)

## Phase 3 — Switchyard
- [x] 3A wire/errors  - [x] 3B JSON mode  - [x] 3C sessions+parsers  - [x] 3D soak run (**PASSES at `4a99e34`, 2026-09-05**: stage 470 req / 0 err / 0 STALLED, passthrough 1,600 / 0 err / 1 STALLED, 0 finishability-invariant warnings, 0 fatals, 1 s trailing silence, decode @16 +19 %/+10 % and prefill median +13 %/+23 % vs §R4 -- soak results §"Run against 4a99e34" and handover item 2)  - [x] 3E residency: spill on demand + capacity/age retention + restart-persistent checkpoints  - [x] 3F prefetch next queued checkpoint to RAM  - [x] 3G partial-prefix restore + stray `</think>` + prefill-time boundary capture

## Phase 3H — hidden-state export (Switchyard probe contract)
- [x] 3H implement (1f2de67)  - [x] 3H GPU parity check (all 52 layers cosine >= 0.998840)

## Long-context recall
- [x] 1M multi-needle, one prefill, 8 graded questions (5/8; depths 0.05/0.75/0.95 pass, 0.25/0.50/0.60 fail,
      control denied, combined question recovers the depth-0.25 needle) -- `benchmarks/bench_multi_needle.py`,
      `benchmarks/results/nemotron35_lightning_5080_1m_multineedle_2026-09-04.md`

## Phase 4 — MTP (time-boxed, flagged)
- [x] NO-GO (2026-09-04 cache study: verify step 1.63× cost, projected 0.96× gain; flag not built)

---
# Ornith RTX 5080 full-context optimization

## Phase 3 (2026-08-29): post-Ada SM120 revalidation

- [x] Rebuilt the native GGUF extensions as real sm_120 cubins on RTX 5080,
  Torch 2.11/CUDA 13/Triton 3.6, and re-captured Q4+INT4/Q6+Q8 baselines.
- [x] Enabled four output-row warps for fused routed+shared GGUF decode on sm_120.
  Exact 128-token live A/B: Q4 164.51 -> 168.08 tok/s (+2.17%), Q6 125.64 ->
  128.52 tok/s (+2.29%); both retained their exact pre-change output hashes and
  produced coherent derivations.
- [x] Lowered the sm_120 grouped-MMA crossover 320 -> 272 after all three routed
  projections won at 272 and still lost at 256.
- [x] Added exact Q6_K dense bands: the 8K cold-prefill live A/B improved 528.14
  -> 623.31 tok/s (+18.0%). The analogous Q4 bands regressed 1259.52 -> 1149.97
  tok/s under real H2D/GEMM overlap and were removed despite microbenchmark wins.
- [x] Added a portable `--synthetic-needle` serving gate and fixed benchmark
  quantized-KV backend selection plus adjacent distributed-port allocation.
- [x] After each accepted inference change, both model/KV pairs recovered passcode
  5663623 from 4K/8K prompts and returned coherent explicit answers.
- [x] Final validation: 147 focused GGUF/model/benchmark tests passed; ruff and
  `git diff --check` passed. The full non-slow suite reached 1,622 passed, 9
  skipped, with 8 unrelated failures and 6 setup errors in the same documented
  clean-main failure families.
- [x] Rejected further attention changes: 128 splits, alternate tiles/warps, and
  Q8 native-score variants did not beat the existing 64-split sm_120 launches.

- [x] Add reproducible Ornith Q4_0 attention and serving benchmark controls.
- [x] Record unchanged-main RTX 5080 synthetic baselines.
- [x] Sweep and implement numerically safe sm_120 attention launch geometry.
- [x] Sweep Ornith Q4_K/Q6_K dense and routed-expert GGUF dispatch on sm_120.
- [x] Implement only repeatable sm_120 wins with safe non-sm_120 fallbacks.
- [x] Verify focused tests, full non-slow tests, and lint.
- [x] Validate a live near-262K request and long-context retrieval.
- [x] Document the final RTX 5080 launch command and measured before/after results.

## Review (2026-08-25, RTX 5080 sm_120, Torch 2.11/CUDA 13/Triton 3.6, WSL)

### Implemented
- `decode_launch_config` is architecture-aware (`compute_capability` param). Ornith
  Q4_0 decode on sm_120 uses (kv_splits=64, block_n=64, warps=8): 0.356 ms vs
  0.822 ms per 262K full-attention layer with the sm_89 tuple (2.31x). sm_89 and
  the conservative fallback are unchanged; scratch/CUDA-graph capacity follows.
- `extend_paged_attention` uses num_warps=4 on sm_120 (8 elsewhere): 1.12x on the
  production long-Q4-prefix split kernel, 2.01x on the fused cold-prefill kernel.
  BLOCK_N=16 was reconfirmed to silently corrupt the packed loader on sm_120 in
  BOTH extend kernels and remains excluded.
- `benchmarks/bench_ornith_attention.py` (oracle-gated decode/prefill/extend
  sweeps at the exact Ornith geometry) + full-context flags on
  `bench_decode_moe.py` (`--max-context`, `--kv-cache-dtype`, `--prefill-chunk`,
  `--prefill-hit-d2d`); GPU-free unit tests under `tests/benchmarks/`.
- `docs/models.md`: sm_120 full-262K launch command (explicit
  `--attention-backend triton`; sm_120 auto-resolves to FlashInfer, which cannot
  read the quantized KV pool).

### Measured but intentionally not changed
- The vendored GGUF DP4A kernels predate llama.cpp's int8-tensor-core MMQ
  rewrite (as do vLLM/SGLang's copies); porting that is phase 2b (below).

## Phase 2a (2026-08-25): arch-aware GGUF dispatch thresholds

- [x] `layers/gguf.py`: `dequant_gemm_min_rows(cc)` — 24 on sm_120, 32
  elsewhere (Q4_K attn shapes cross at 24: dequant 0.0645 vs MMQ 0.0778 ms;
  16 would regress the Q6_K lm_head where MMQ still wins at 16).
- [x] `moe/fused_gguf.py`: `mmq_min_tokens(cc)` — 16 on sm_120, 32 elsewhere
  (grouped MMQ 0.314 vs vec 0.324 ms @16; 0.382 vs 0.475 @24).
- [x] `_MMVQ_SAFE` left at 6 (per-shape ambiguous on sm_120).
- [x] Tests: `tests/kernels/test_gguf_dispatch.py` — pure threshold tests for
  both archs + CUDA dispatch-branch tests with faked capability (18 passed).
- [x] Live re-verify on real Ornith blk.3.attn_q (Q4_K): 24 rows now 0.0690 ms
  via dequant vs 0.0806 ms with the old threshold; 16/32 rows unchanged.
- [x] Kernel+benchmark suites 123 passed; full non-slow suite failures A/B
  identical to clean main (7 failures; laguna errors are suite-order artifacts
  present in the clean-main baseline too); ruff clean.

## Phase 2b (2026-08-25): upstream int8-MMA MMQ port (Q4_K/Q6_K)

- [x] Vendored llama.cpp master `eab8ee41` CUDA MMQ verbatim into
  `python/freetoken/kernel/csrc/gguf_mmq/` (mmq/mma/load-tiles/vec-dot/configs/
  quantize/mmid + the ggml headers). `mmq_ext.cu` is the only hand-written
  file: backend shims (device info, torch-allocator pool, abort/error) + torch
  bindings; only Q4_K/Q6_K `mul_mat_q` cases are instantiated.
- [x] Dense: `ggml_mul_mat_a8_mma` wired into `fused_mul_mat_gguf` on sm_120
  for rows > `_MMVQ_SAFE` (replaces BOTH the DP4A-MMQ band and dequant+cuBLAS).
  Measured (real Ornith tensors, embedder stopped): Q4_K attn_q 8192 rows
  1.79 ms vs dequant 2.40 vs DP4A 22.9; Q6_K lm_head @2048 tokens 17.5 vs 19.2
  vs 262; wins at every rows >= 4. Build failure falls back to the old path.
- [x] MoE: `ggml_moe_a8_mma` (upstream ids path: mm_ids_helper +
  scatter/gather q8_1_mmq quantize + expert_bounds mul_mat_q) wired into
  `_moe_matmul` for 320 <= tokens <= 16384 on sm_120; broadcast (gate/up) and
  per-slot (down) forms; padded flat slots addressed in whole blocks (stride
  must divide by block size -- true for Ornith banks). Ornith geometry
  (E=256 top-8): 8192 tokens gate_up 4.16 ms vs DP4A 23.2, down 4.90 vs 15.3;
  DP4A keeps 16..319 (crossover ~288), MMVQ keeps decode.
- [x] Numerics: MMA matches dequant reference within activation-quant noise on
  real Ornith tensors (rel <= 0.013, on par with DP4A) and on random-safe
  packed bytes vs gguf-py (`tests/kernels/test_gguf_mma.py`); MoE broadcast +
  gather verified vs per-expert dense reference and vs the vec kernel
  (end-to-end `fused_experts_gguf` rel ~1e-3).
- [x] Tests: dispatch-branch tests extended (MMA seams, `_use_mma_moe` gate);
  `test_dense_gguf_prefill_uses_dequantized_cublas_result` pinned to the
  dequant branch it validates. Kernels+benchmarks: 336 passed, 1 skipped.
  Full non-slow suite: same 7 pre-existing failures as clean main.
- [x] Live A/B at the production 262K config (2026-08-26, hostile prompt:
  28K words seeded non-repetitive text, ~50K tokens / 6+ chunks, three distinct
  needles at 10/50/90% depth, greedy, 320-token decode):
  - MMA: 13.88 s wall, 3/3 needles exact, 0 tracebacks (prefill ~4,700 tok/s).
  - Fallback (FREETOKEN_GGUF_DISABLE_MMA=1, same box/prompt minutes apart):
    21.94 s wall, 3/3 needles exact, 0 tracebacks (prefill ~2,700 tok/s).
  - Net: ~1.75x prefill, 1.58x end-to-end TTFT+decode; identical answers.
  - Path evidence: the MMA leg's worker warmup demonstrably blocked polling the
    freetoken_gguf_mmq JIT lock (only `_mma_module()` touches it) and ran at
    the faster wall time; the fallback leg never touched that cache. The
    in-log `int8-MMA MMQ ACTIVE` INFO line is swallowed by the server's log
    handler -- known instrumentation gap, not a dispatch gap.
- OPS HAZARD found: `torch.utils.cpp_extension.load` leaves a stale `lock` in
  ~/.cache/torch_extensions/ when a building/loading process is SIGKILLed; the
  next server then hangs in warmup FOREVER, sleep-polling it (looks like a
  startup hang). Fix on sight: `rm ~/.cache/torch_extensions/py312_cu130/*/lock`.
  Also: 3 host OOMs during testing were the ~20 GiB shmem expert banks + agent
  processes on the 29 GiB WSL box; mitigated with a 12 GiB swapfile
  (/swapfile-claude, left enabled) + oom_score_adj (serve 800, terminal -600).
  A killed `ft serve` leaves `multiprocessing.spawn` workers holding the banks:
  `pkill -9 -f "FreeToken/.venv/bin/python3"` and check `free -g`.
- Flaky pre-existing `test_reference_roundtrip_error_is_within_the_scheme_envelope[int4]`
  (~5% failure, unseeded randn) now seeded.

### Live 262K validation (Ornith-1.5-35B-Q4_K_M, one RTX 5080 16 GB)
Command: `ft serve --model ~/ai/models/Ornith-1.5-35B-Q4_K_M.gguf
--attention-backend triton --kv-cache-dtype q4_0 --num-tokens 262144
--kv-reserve-tokens 262144 --max-seq-len-override 262144
--max-running-requests 1 --moe-backend offload --moe-cache-auto
--max-prefill-length 8192`
- Auto-sizing: 4,835 expert slots + 262,263 KV pages, prefill overlap on.
- Cold ~259,400-token prefill: 210–220 s (~1,230 tok/s sustained).
- Decode at ~259K context: 99–104 tok/s (Ada baseline: 33.67 tok/s at 170K).
- NIAH 3/3 exact: passcode `7391-ALPHA` recovered at 10%/50%/90% depth
  (greedy, through the model's `<think>` block).
- Radix-cached TTFT for a repeated full-context prompt: 4.9 s.
- No crash, OOM, or worker restart across the whole run.

### Test/lint status
- `tests/kernels/test_triton_attention.py` + `test_kv_quant.py`: 75 passed.
- `tests/benchmarks`: 8 passed; ruff clean on all changed files.
- Full non-slow suite: 9 failures + 6 errors pre-exist on clean main
  (`moe_pageable_gpu` config-test drift), byte-identical A/B vs baseline.

## RTX 2000 Ada port (2026-08-26, sm_89, 16 GB, 70 W, Torch 2.11/CUDA 13)

- [x] Confirm the sm_120 int8-MMA extension compiles and passes numeric tests on
  sm_89; keep the Blackwell dispatch unchanged.
- [x] Sweep dense Q4_K/Q6_K rows and output geometries. Ada uses exact measured
  shapes, not the sm_120 all-row rule: Q4_K out=8192 rows 8--512; Q6_K out=8192
  rows 8--448 and out=2048 rows 8--64. Larger dense batches return to
  dequant+cuBLAS; unmeasured output sizes (including the large lm_head) retain
  their prior path.
- [x] Sweep both exact Ornith routed projections. The MMA crossover is 272 tokens
  for Q4_K gate/up and Q6_K down; DP4A remains selected through 256. At 8192
  tokens, gate/up improved 103.78 -> 16.52 ms (6.28x) and down improved
  70.17 -> 17.80 ms (3.94x).
- [x] Keep the existing Ada Q4_0 attention launch (32 splits, BLOCK_N=32,
  4 warps). The sm_120 64/64/8 tuple was slower at full 262K context.
- [x] Extend `bench_gguf_gemm.py` with exact gate/up and down projections,
  activation scaling, and GPU-free case-construction tests.
- [x] Focused dispatch/MMA/benchmark tests: 61 passed. Kernels+benchmarks after
  updating the dispatcher test double: 356 passed, 1 skipped; two unrelated,
  reproducible FP8 failures remain on this Ada/Torch build (`blk_aq_y` strict
  native/emu equality and 0.01090 error against a 0.01000 W8A8 tolerance).
- [x] Cold live server A/B at the production 262K/Q4_0 configuration, identical
  96,026-token prompt, fresh process per leg: fallback TTFT 176.912 s, Ada MMA
  TTFT 125.265 s (29.2% lower / 1.41x faster). Both completed 15-token coherent
  responses without a crash. Worker environments proved the fallback leg had
  `FREETOKEN_GGUF_DISABLE_MMA=1` and the optimized leg did not.
- tests/moe/test_prefill_hit_d2d.py::test_batch_memcpy_roundtrip is a pre-existing order-dependent flake (dst produced on the default stream, probe syncs its own stream). Ticket it; not Nemotron work.
  ROOT CAUSE (2026-09-05, CUDA-only — cannot be reproduced or fixed without a GPU): the raiser is the
  JIT loader's probe, not the test body. `kernel/batch_memcpy.py:28-39` zeroes `dst` on the *ambient*
  stream (line 29), runs the batch copy on a fresh private stream (31-36), then `stream.synchronize()`
  (37) waits on that private stream only. Nothing orders the memset ahead of the copy, so once earlier
  tests have left work queued on the ambient stream the zero-fill retires *after* the copy and clobbers
  it -> `RuntimeError: cudaMemcpyBatchAsync probe copied wrong bytes` (line 39), surfacing at
  tests/moe/test_prefill_hit_d2d.py:75. Deterministic when the whole tests/moe dir runs; passes alone.
  Fix: `stream.wait_stream(torch.cuda.current_stream())` before the copy and `dst.record_stream(stream)`
  after it (or allocate/zero `dst` inside `with torch.cuda.stream(stream)`); the same two lines belong in
  test_batch_memcpy_roundtrip. Wider impact worth ticketing: `_probe` runs on the caller's ambient stream,
  so a busy stream at first resolve can make `moe/offload_cache.py:1194-1205` latch `_batch_memcpy=False`
  and silently disable prefill hit-D2D process-wide. The production copy path (offload_cache.py:1294-1300)
  is event-ordered and unaffected.
- Ticket: `--kv-grow-step-tokens` + `--nvfp4-backend flashinfer` crashes at init (b12x banks include an int32 bank that `kernel/vmm.py` `_DTYPE_NAMES` lacks). Not blocking (triton is the default).
- 1M gate must retest the 262K/524K needle through the chat endpoint (cache study saw misses on the raw completions probe, the known artifact).

## Task 3F — Switchyard soak scheduler crash (2026-09-04)

- [x] `_cache_req_hybrid` reserves the replacement ping-pong slot before donating the frozen
  Mamba snapshot; when no slot can be reserved the chunk commit is skipped (debug-logged once)
  instead of raising `LinearStatePool exhausted` (`scheduler/cache.py:725-780`).
- [x] One escalating reclaim path (`CacheManager.reserve_mamba_slots` / `acquire_mamba_slot`):
  free-list -> LRU snapshot eviction -> on-demand spill of the LRU idle session lease
  (`Scheduler._reclaim_soft_sessions_for_state_slot`). Wired into the chunk commit, the cold
  session restore and admission (`scheduler/prefill.py:71-77`), covering both R=16 and the
  1M R=1 / `--linear-state-slots 5` case.
- [x] `/health` consults the backend process handles and answers 503 with a reason when a
  worker is gone (`server/control_api.py:19-40,80-85`).
- [x] Shutdown bounded: `timeout_graceful_shutdown` on both uvicorn entry points, terminate +
  reap (join/SIGKILL, whole-set budget) on every stop path, plus a hard-exit backstop
  (`server/api_server.py:61-71,99-155,~410`).
- [x] Tests: `tests/scheduler/test_hybrid_pool_exhaustion.py` (11),
  `tests/server/test_health_liveness_and_shutdown.py` (12). Both files fail at HEAD in a
  detached worktree with the byte-identical soak stack trace.
- [ ] Follow-up ticket: `_maybe_shrink_growable_kv` calls `evict_all_unlocked_prefixes()`
  *before* computing whether a shrink is possible, so once KV is above its initial step every
  idle moment wipes the whole prefix cache and often shrinks nothing (`server.gen1.log`
  09:44-09:47 shows dozens of "teardown evicted N ... keep 262144 tokens committed").
- [x] **262K recall ROOT CAUSE (2026-09-04) -- the Mamba-2 dt floor.** `dt_limit` was
  `(config.time_step_min, inf)` = `(1e-3, inf)` in the prefill scan; `time_step_min` is HF's
  *initializer* range for `dt_bias`, not a runtime bound, and the floor caps every head's
  memory horizon at `1/(|A|*1e-3)` tokens. `dt_limit=(0.0, inf)` (vLLM's value; llama.cpp and
  FreeToken's own decode kernel never clamped) turns 147,456 and 262,144 @ depth 0.52 from FAIL
  to PASS at identical TTFT. Write-up:
  `benchmarks/results/nemotron35_lightning_5080_262k_rootcause_2026-09-04.md`; fix in
  `models/nemotron_h/config.py::_dt_floor` (+ `FREETOKEN_NEMOTRON_DT_MIN` A/B hatch) with three
  tests in `tests/models/test_nemotron_h.py`. **This retracts the bisect entry below and its
  "gate mid-depth recall at 131K only" recommendation.**
- [x] ~~262K recall bisect (2026-09-04) -- **no FreeToken bug**; the limit is the checkpoint's.~~
  **RETRACTED** by the root-cause entry above (and, before it, by the cross-engine check):
  all eight variants shared the dt floor, so the matrix proved only that the fault was common
  to every cell. The exonerations it does establish (growable-vs-static KV bit-identical, KV
  dtype, attention backend, chunk size, dense dequant) still stand.
  Write-up: `benchmarks/results/nemotron35_lightning_5080_262k_bisect_2026-09-04.md`.
  One fixed 262,144-token prompt (needle depth 0.52) + the same prompt at 131,072, fresh server
  per row, chat endpoint, greedy, thinking off. All eight variants fail at 262K and pass at 131K:
  q8_0 / fp8_e4m3 / bf16 KV, Triton / FlashInfer attention, SSD-kernel / `FREETOKEN_MAMBA2_REF=1`,
  4096 / 8192 prefill chunks, growable / static KV, NVFP4 / `FREETOKEN_NEMOTRON_DENSE_DEQUANT=1`
  dense. Growable-vs-static is **bit-identical** (0.000e+00 state and logits at both lengths), and
  the q8_0-vs-bf16 state divergence is *larger* at the passing 131K than at the failing 262K.
  The predictor is the needle's absolute position: at 262,144 tokens depth 0.057 recalls exactly
  while 0.267 / 0.519 / 0.761 / 0.947 all miss, and the length sweep is non-monotonic (147,456
  FAIL, 163,840 PASS, 180,224 PASS, 196,608 FAIL), so nothing keyed on 2^18 can explain it.
  Actions: gate mid-depth recall at 131K only; at >=196,608 gate a depth<=0.1 needle plus
  capacity/coherence. `bench_long_context.py` gained `--needle-depth` (+ tests).
- [x] Follow-up from that bisect (perf, not correctness): `decode_launch_config` had no
  context-length key and every tuned branch needed the Ornith head shape, so Nemotron always took
  the `(kv_splits=8, block_n=32, warps=4)` fallback -- 16 CTAs on 84 SMs at 262K KV. **DONE
  2026-09-05**: `_grid_filling_splits` sizes the split count to the SM count for untuned shapes
  ((64, 64, 8) here), tuned branches pinned unchanged, `FREETOKEN_DECODE_KV_SPLITS/_BLOCK_N/
  _NUM_WARPS` for A/B. Decode 82.8 -> 145.3 (131K) / 58.7 -> 132.4 (262K) / 35.4 -> 113.6 (524K)
  tok/s, prefill unchanged;
  `benchmarks/results/nemotron35_lightning_5080_decode_launch_2026-09-04.md`. Still open from the
  same note: `num_kv_splits_ptr` is passed to both decode kernels and dereferenced in neither
  (dead argument; the split count is a constexpr) -- delete it or use it.
- [x] Follow-up from that bisect (1M blocker): Triton KV loaders widen slot ids to int64 on store
  (`kv_quant.py:47,168`) but not on load. **DONE 2026-09-05**: compile-time `SLOT_I64` constexpr
  in all four gather kernels, set from the pools' `numel()` by `_slot_offsets_need_int64`, so the
  64-bit address math costs nothing on geometries inside the int32 ceiling.

## Queue (2026-09-04 evening, agreed with user)
- [ ] Scheduler admission fix -- **attempt 2 (`ea7ed7c`, `admissible_size`) also FAILS the live
      soak (2026-09-05)**, worse than `81ab30e`: permanent deadlock among 14 concurrently
      chunked prefills whose summed footprint is 1.76x the pool. The gate is applied only at
      admission, against capacity that later admissions spend again. Next attempt: make the
      admit's remaining footprint a STANDING reservation for the life of the prefill (or cap
      concurrent chunked prefills) -- soak write-up §T8, handover item 2.
- [x] Combined GPU session: soak + one 1M prefill with needles at 6 depths, a control, and
      combined/reverse probes -- superseded by `benchmarks/oracle_cross_engine.py`, run
      2026-09-05 (oracle results file).
- [x] decode_launch_config Nemotron head-shape branch (bisect ticket) -- done 2026-09-05
- [ ] Q4: push the 101 unpushed commits to a branch; CI for CPU-only checks (scheduler replay,
      scheduler tests, parity probe CPU half)
  - [x] CI half: `.github/workflows/cpu-checks.yml` (push + PR, any branch, hosted runner,
        no GPU) running `ruff check`, the scheduler/server/kvcache/dsv4/Nemotron-H unit tests,
        and `benchmarks/scheduler_replay.py --gate` (the bisect stage-replay harness promoted
        out of scratch, with floors stage >=6.5M tok / 350 done, pressure >=8.5M / 85).
        Docs: `docs/cpu-checks.md`. Measured at 81ab30e in a torch-2.11-CPU venv:
        1,239 passed / 51 skipped in 14.2 s; gate 3.8 s wall, 428 MB RSS on two cores.
  - [ ] Re-run the full unit-test step once no GPU job is live and record the wall time
        (the 14.2 s above was measured before the current GPU session; the re-run was
        deferred to keep host RSS free). Nothing else about the workflow is unverified.
  - [ ] Optional follow-up: fold `tests/moe` + `tests/kernels` into the CI test list. Only
        `tests/moe/test_offload.py::test_adjust_config_converts_moe_cache_rate_to_cache_size`
        blocks it (it asks for the `fi` backend and errors instead of skipping when
        flashinfer is absent); the other 154 tests in those two directories pass on CPU.
  - [ ] Optional follow-up: the `[tool.ruff.lint] ignore` list in `pyproject.toml` is the
        pre-existing violation set (E741 x79, E702 x47, E731 x18, F401 x17, F841 x13, ...).
        Clearing any of them lets that line be deleted and the rule re-enabled.
- [x] Q5: profile the prefill curve — **DONE 2026-09-05, root cause found and fixed**
      (`benchmarks/results/nemotron35_lightning_5080_prefill_profile_2026-09-05.md`).
      The whole superlinear term is the extend/prefill attention kernel, and it was a launch
      configuration: `_select_extend_tile`'s `head_dim<=128` arm returned a hard-coded
      `BLOCK_M=128` while sm_120 takes 4 warps, so the fp32 accumulator spilled (**396 spill
      slots vs 14** at `BLOCK_M=64`) — 28.6 TFLOP/s instead of 70.4. `extend_launch_config`
      now caps `BLOCK_M` by the accumulator's register budget; only that never-measured arm
      moves, every `head_dim>=256` branch and every 8-warp device is byte-identical.
      Paired A/B, needle recalled every run: **131,088 3,230 -> 5,288 tok/s (1.64x);
      262,160 1,965 -> 3,683 (1.87x); 1,040,016 573-576 -> 1,307 (2.28x), TTFT 1,810-1,824 s
      -> 795.8 s.** SSD scan exonerated (flat in position, 0.2 % of a 1M prefill); engine /
      KV grow / page-index build exonerated (non-attention position-dependent slope solves to
      **0 ± 0.2e-3 ms/token = ±13 s of a 1M prefill**); the flat term is **79 % MoE expert
      GEMMs**. New: `benchmarks/bench_prefill_attention.py`, `FREETOKEN_EXTEND_*` overrides,
      a `Triton extend launch:` startup line, 4 tests. Follow-ups filed in §9 of the
      write-up: (1) the extend kernel is still at 31 % of peak and dequantizes q8_0 in the
      inner loop with no native-Q8 QK path — another 1.5-2x, 1M TTFT ~470-530 s;
      (2) MoE prefill is now the flat term at 33 TFLOP/s and `b12x` was not measurable at
      M=8192; (3) sweep the extend tile on an sm_89/sm_90 part (`128/64/8/1` still spills 148
      slots) before extending the cap above 4 warps.
- [x] Q5.1 (follow-up 1 above): native-Q8 extend attention — **measured 2026-09-05, CLOSED
      NEGATIVE, kernel unchanged**
      (`benchmarks/results/nemotron35_lightning_5080_prefill_q8_2026-09-05.md`).
      The ticket's premise fails on three independent measurements: (a) **225 TFLOP/s is the
      spec sheet, not the part** — cuBLAS does 123.0 TFLOP/s bf16 and Triton's own `tl.dot`
      118.4, so the extend kernel is at **57-60 % of achievable**, not 31 %; (b) **int8 is not
      faster than bf16 here** — `torch._int_mm` 128.0 TOP/s = 1.04x bf16, so a native int8 dot
      cannot speed up the QK; (c) **the whole q8_0 dequant is worth 1.206x** — the same kernel,
      same launch, over a bf16 KV pool, flat to ±0.4 % at 131K/262K/524K/1M. Desk result: the
      q8_0 scale is per 32 `head_dim` elements, so folding it after the dot costs
      `BLOCK_M*BLOCK_N*D/QBLOCK` multiplies vs `BLOCK_D*BLOCK_N` to dequantize in place —
      cheaper only when **`BLOCK_M < QBLOCK`** (decode's `_Q8_NATIVE_QK` has `BLOCK_M=16`
      heads and pays; extend's 64-token tile does not), and the V scale sits on the PV
      *reduction* axis where no fold exists at all. `num_stages>1` re-swept with the masks
      removed (the pipeliner's precondition): still loses at all 18 tiles. Best combination
      found (`split`+`exp2`+`bf16deq`+`qfold`, §5 of the write-up) is 1.14x and **fails the
      accuracy gate** — 7.6e-4 vs the fp32 oracle against the tree's 3.7e-4, because q8_0's
      fp16 scale does not fit bf16's 7 mantissa bits. Accuracy-neutral subset is 1.05x. A
      *perfect* native-Q8 kernel would put 1M TTFT at 680 s, not the ticketed 470-530 s.
      **Next prefill work is follow-up (2), the MoE: 33 TFLOP/s of the same measured 123 =
      3.7x off, against attention's 1.7x.**
- [x] Q1: standing cross-engine oracle — tool landed in `5f7c0d6`, **first live sweep run
      2026-09-05 at 262K** (`benchmarks/results/nemotron35_lightning_5080_oracle_2026-09-05.md`):
      FreeToken 19/24 vs llama.cpp 17/24, 2 `freetoken-only-miss` (both composition, not
      retrieval), **0 retention failures on either engine**. Top-k logit comparison still
      blocked on FreeToken having no logprobs in `SamplingParams` (see docs/oracle.md).
      **1M FreeToken leg also run** (no llama.cpp leg): 7/19 turns, 0 `retention`, 5/6 needles
      recovered by leak-free reverse probes — interference, not retention.
- [x] Q1a: **settle the 1M direct-addressing question — DONE 2026-09-05 at `2a139ad`, verdict
      "no FreeToken engine defect"** (same results file, §§10–12).
      - The llama.cpp 1M leg is **impossible on this card**, not merely slow: `-c 1052672`
        reserves all 16 GiB at every `--n-cpu-moe` (14/20/23 all show 18–22 MiB free), the
        first 4,096-token chunk costs 27.3 / 11.6 / 3.87 s against 2.05 s at `-c 270336`, and
        even at the `--n-cpu-moe 23` floor (685.1 MiB × 23 expert blocks — nothing left to
        offload) the written KV outgrows residency at ~570K tokens and chunk cost then climbs
        +11.5 s per 4,096 tokens → **≈20 h of remaining prefill against a 4 h lock cap.**
        Killed at 622,592 tokens. `docs/oracle.md` budget table and host rules corrected.
      - Fell back to **524,288 on both engines** (byte-identical prompt, sha `72683f24c68885d1`).
        Pass rate by shape — FreeToken 1/6 direct, 0/6 combined, **6/6 reverse**; llama.cpp
        2/6, 1/6, **6/6**. 8 `agree` / **9 `both-miss`** / 2 `freetoken-only-miss` / 0
        `llamacpp-only-miss`; **0 `retention`, 0 `selection` on either engine.**
      - Four of the six direct probes return the same key's near-duplicate `register` code
        **byte-for-byte in both engines** (quarry 1607392, cavern 3518470, meadow 8043961,
        thicket 5290638). The `key → code` collapse between 262K and 524K is a **model**
        property; `code → key` is intact at 6/6 on both engines. 1M's 1/6-direct-5/6-reverse
        shape is the same phenomenon one rung up. **Close as model-limited — no kernel bug.**
      - Cost note: 524K FreeToken prefill is now **2,297 tok/s** (228 s) against 1,064 tok/s on
        2026-09-04 — `2a139ad` visible on a real workload — and 2.4x llama.cpp on the same
        prompt, where at 262K the two were within 12 %.
- [x] **`cached_tokens: 0` on every FreeToken turn — NOT a regression, a missing flag.**
      `docs/oracle.md`'s Phase-A serve line omitted `--enable-cache-report`;
      `openai_api.py:937-939` then returns 0 and `:955` omits `prompt_tokens_details`, so flag
      off / genuine zero / field absent are indistinguishable on the wire. The scheduler was
      hitting the cache all along (`#new-token: 55, #cached-token: 524287` on turn 2). Nothing
      on that code path changed in `acc91e9..2a139ad`. **Fixed: the flag is now in
      `docs/oracle.md`'s Phase-A line, and the runbook's re-prefill check says to corroborate
      with TTFT and the server log.**
- [x] **Made the omission impossible to misread -- DONE.** The flag gates *reporting*, never
      work (`PromptAdmittedMsg.cached_tokens` is populated for every admission regardless), so
      the fix is a presence rule rather than a value: `prompt_tokens_details` is emitted
      **whenever reporting is on**, carrying an explicit `cached_tokens: 0` for a genuine
      miss, and is **absent entirely** when it is off. `_reported_cached` now returns
      `int | None` and `_usage` keys off `is not None`. `/v1/messages` follows the same rule
      (`cache_read_input_tokens = cached if cache_report else None`), keeping the flag's
      `input_tokens`-excludes-the-prefix billing semantics untouched. `/v1/responses` is the
      one exception and is documented as such: `usage.input_tokens_details` is **mandatory**
      in the Responses schema, so it cannot express "not reported" by omission and a gated 0
      would be the same lie -- that route now always reports the true hit and the flag no
      longer gates it (its dead `cache_report` plumbing is gone). `docs/oracle.md` carries the
      three-row wire table; `docs/switchyard.md` and the `--enable-cache-report` help updated.
- [ ] Small lead from 524K: `direct:harbour` is the one leak-free direct probe llama.cpp holds
      and FreeToken loses (returned the *orchard* code — `interference-cross`, and
      `reverse:harbour` recovers it two turns later). It is also **turn 2, the one turn whose
      TTFT was 50.0 s against 2.4 s for turns 3–19** — a partial prefix re-prefill. Cheap
      re-probe: re-run 524K with `--filler-cursor 65` and see whether the pairing repeats.
- [ ] Remaining oracle rung: 131K on both engines (cheap, ~10 min total).

## Backlog
- [x] Prompt-lookup (n-gram) speculative decoding for agent-session decode — **NO-GO 2026-09-05**
      (`benchmarks/results/nemotron35_lightning_5080_ngram_spec_2026-09-05.md`). Acceptance is
      fine (n=8 drafter: copy-heavy agent tool output λ=3.615 at 92.6% per-token accept; code and
      prose within ±0.5% of neutral because it almost never fires), and Mamba-2 rollback is cheap
      (never advance the live state — verify into the ping-pong scratch slot, cache ~5 MiB of scan
      inputs, commit with one varlen SSD scan over the accepted j: ~0.3 ms). The blocker is that
      **the extend path costs 290 ms host per forward, flat 1→32 tokens, of which 267 ms is the
      23 MoE layers** (11.6 ms/layer/forward regardless of token count) against 33.9 ms for a
      1-token decode forward — 36–42x a graphed decode step, so break-even needs λ≈40 vs a ceiling
      of k+1≤17.
- [ ] **Extend-path MoE: reuse the decode expert cache** (blocker above; worth more than spec
      decoding — it also means prefill chunks below ~3K tokens go host-bound). Repro:
      `benchmarks/probe_ngram_spec.py --layer-profile`. Then: a graph-captured fixed-width verify
      forward, then `--speculative ngram` (n=8, k=8, greedy). With the cache fix alone the
      copy class projects 1.52x; with the captured forward, 1.74x.
- [ ] Replay real Switchyard traces as the benchmark instead of synthetic soaks/needles
- [ ] NOTE 2026-09-04: fork/main has 14 Ada/Ornith commits (cefa4bd..62f5a66) not in local main;
      local main pushed as fork/nemotron35 (58e5f04). Merge/rebase decision pending (user).

## Scheduler admission gate, redo after the 81ab30e revert (worktree agent-a88d6be62b77b9bee)

Background: `81ab30e` passed the CPU replay gate but stalled the live 16-way soak 52% of
the wall clock (soak report §S). Its finishability gate charged
`max_size - inflight_prefill_size`, ignoring KV held by decoding requests and by
retained/locked session prefixes. Reverted in `5bf0bcc`.

- [x] 1. Make the replay reproduce the live failure
  - [x] session residency leases: `retain_prefix` on finish, locked across turns,
        released only on idle expiry or demand reclaim
  - [x] model `Scheduler._reclaim_for_blocked_prefill` (LRU idle lease) and the
        600 s client timeout that abandons a starved request
  - [x] `switchyard-stage` profile: 16 sessions, sticky scenario, growing prefixes
  - [x] stall metrics (`stall_seconds` / `stall_frac` / episodes) + `timeouts`
  - [x] wall-clock proxy: `match_req` calls/tokens per pass
  - [x] acceptance: 81ab30e stalls on the new profile, 68c54e7 does not
- [x] 2. Corrected admission gate
  - [x] charge finishability against obtainable pages (`available_size`), seeding
        the adder with queued continuations' remaining footprints
  - [x] cache the prefix match per pending req so refused passes stop re-walking
  - [x] keep R2 continue-past-refusals, completing-chunk decode reservation,
        `spill.valid`; drop `max_size` / `inflight_prefill_size` / `prefill_footprint`
- [x] 3. Verify: tests/scheduler green, replay gate on all profiles, ruff,
      `git diff --check`; floors raised to achieved minus 5%

### Review

**What the replay was missing.** It ran with `session_id=None` throughout, so no turn ever
reached `_free_req_resources(retain_session=True)` -> `retain_prefix`, and no page was ever
*locked*. Everything a request finished with went back to the tree as evictable, so
`available_size` never diverged from "the pool minus what is running" -- exactly the
divergence 81ab30e's gate mis-read. It also had no idle expiry, no demand reclaim, no client
timeout, and no wall-clock accounting for passes that emit no batch, so a stall was invisible
(the old loop called it a `LIVELOCK` fatal and stopped).

**The corrected gate.** Charge a fresh admit's whole remaining footprint against
`CacheManager.admissible_size` = `available_size` + tokens held by *idle reclaimable* session
leases. That is the pool's three-way split done honestly: free/evictable, locked-but-buyable,
locked-and-not-buyable (decoding requests, and leases of sessions with a request in flight).
`available_size` alone under-counts the second bucket; `max_size` over-counts the third, which
is the soak's stall. The adder is additionally seeded with the *unforwarded tail* of every
prompt already mid-prefill, which is the anti-livelock invariant and is stable under progress:
forwarding a chunk drops both the budget and the charge by the same tokens, so an advancing
prefill never manufactures admission room. Kept from 81ab30e: continue-past-refusals, the
completing-chunk decode reservation, the oversize skip, `spill.valid`. Dropped: `max_size` as
the budget, `inflight_prefill_size`, `ChunkedReq.prefill_footprint`, the chunk-only charge.
Replaced: strict FIFO among fresh admits became *aging* (`admission_patience`), because
unconditional strict FIFO turns one unaffordable prompt into a dead scheduler.

**Numbers** (seed 7, 20,000 forwards): see the report and the floors in
`benchmarks/scheduler_replay.py`. The load-bearing result is that `81ab30e` PASSES the two
residency-free profiles (it out-throughputs this fix there) and FAILS `switchyard-stage` at
`stall_usage_p50 = 1.0000`, which is the live signature the old gate could not see.

**Still only checkable on hardware** -- the replay does not model: the spill/cold-restore cost
of a reclaimed lease (it treats release as free, where the server writes a checkpoint and may
fail the restore with "Eviction did not free enough space"); non-reclaimable/explicit leases,
which age out only on TTL; GDN state slots and the `mamba-slot 96/96` regime; and the real
per-pass CPU cost, which is reported as `match_calls`/`match_tokens` rather than charged to
the clock. A live 16-way soak on both routes remains the acceptance test.
- [x] Server bug (found by replay work, b030c7f): a client that disconnects while its request is
      still in prefill is never aborted (stream_with_cancellation checks is_disconnected only
      after the first yielded chunk; non-streaming path only on CancelledError) → its table
      slot + forwarded KV leak for the server's life. **DONE 2026-09-05**: new
      `python/freetoken/server/disconnect.py` holds two primitives that race the awaited work
      against a periodic `is_disconnected()` poll (`POLL_INTERVAL_S = 0.25`) and raise
      `CancelledError` the moment the client is gone -- `aiter_or_disconnect` for the streaming
      generators (the poll now covers the wait for the FIRST chunk, i.e. the whole prefill
      window) and `await_or_disconnect` for the non-streaming `generate_full`. Every surface
      that shares the path is covered: `/generate`, openai chat + completions (stream and
      non-stream, including the session-busy resubmit), anthropic `/v1/messages`, responses
      `/v1/responses`. The abort was NOT duplicated -- each caller's existing
      `except asyncio.CancelledError` still sends the AbortMsg, so it stays exactly one per
      request; `stream_with_cancellation` now keeps a strong reference to its fire-and-forget
      abort task (`FrontendManager.spawn_abort`), which a bare `create_task` did not.
      Tests: `tests/server/test_disconnect_abort.py` (12; fake client + fake engine, no GPU) --
      prefill-time drop for streaming and for chat/completions/anthropic/responses non-stream,
      exactly-one-AbortMsg, ack_map/event_map released, post-first-chunk drop still aborts,
      normal completion unchanged, probe failure treated as a disconnect. tests/server 761
      passed, tests/scheduler 194 passed, ruff clean, diff whitespace clean.
      Scheduler side checked, no ticket needed: `PrefillManager.abort_req` pops the queued
      `PendingReq` and returns `req.chunked_req`, which is `None` for a request that was never
      admitted -- so aborting a queued request drops its reservation, frees nothing else (there
      is nothing else), and `_pending_abort_acks` still emits the terminal ack that closes the
      frontend's accounting. No scheduler change required.
      Known pre-existing gap left alone: in the streaming chat path a disconnect AFTER an
      auto-session-busy resubmit aborts the pre-fallback uid (the wrapper was bound to it);
      the fallback request finishes on its own, so it does not leak -- see the comment in
      `openai_api.py::stream_chat_completion_chunks`.
- [x] Live soak against b030c7f with FREETOKEN_SCHEDULER_INVARIANT=warn (after prefill profile).
      **DONE 2026-09-05, PASS on both routes against `4a99e34`** (clean tree: d685e99 gate +
      standing reservation + `max_chunked_prefills=8` + finishability invariant, `ff470e7`
      disconnect-abort, `acc91e9` decode, `4a99e34` prefill). Stage 470 req / **0 err / 0
      STALLED**, p50/p95/p99 24,283/145,840/230,183 ms; passthrough 1,600 req / **0 err** /
      1 STALLED, 7,527/32,906/83,354 ms. **0 invariant warnings** across ~3,141 passes
      (`ea7ed7c` violated it on 566). 0 `committed_pages_required`, 0 `LinearStatePool
      exhausted`, 0 tracebacks, 0 oversize warnings, 0 `Eviction did not free enough space`,
      0 ERROR/CRITICAL. Trailing silence 1 s on both phases (the §T deadlock signature;
      `ea7ed7c` had 2,616 s); scheduling wall clock 97.2 % / 95.7 % of the phase. Throughput
      up on every axis vs §R4/§R6: decode @16 lanes 81.6 -> 96.8 tok/s stage and 161.4 ->
      177.5 passthrough, per-stream 5.10 -> 6.05 / 10.09 -> 11.09, prefill instant median
      1,637 -> 1,851 / 1,496 -> 1,838; stage p95 -27 %. Mean lanes per prefill batch 1.83
      (§R6 2.37) -- the standing reservation seats fewer and that is the trade that bought
      0 STALLED. Graceful shutdown 3 s, GPU 0 MiB. Write-up: soak results §U ("Run against
      4a99e34"); drivers `scratchpad/soak7/`.
- [x] **Scheduler/server observability the soak could not get (§U5/§U6) -- DONE.**
      `/v1/stats` gains a `scheduler` block (`null` until the engine publishes one, which it
      never does offline or on a non-primary TP rank -- deliberately distinct from an
      all-zero document) and `requests.aborts`.
      - New `python/freetoken/scheduler/counters.py` owns the document's shape:
        `PrefillCounters` (`passes`, `fresh_admits_blocked_by_cap`, `deferred_chunks`,
        `refusals`, `chunked_inflight` + `_max`, `seatable_lanes_last` + a per-lane/geometric
        histogram, and `invariant.{checks,violations,worst_shortfall}`), `SpillCounters`
        (spills/restores/prefetches, each with its failure channel, plus
        `restores_diverged`), and `build_scheduler_counters()`, which renders them next to
        `SpecStats.as_dict()` (declines by reason + a new accepted-token histogram).
      - **The finishability invariant is now evaluated and counted on EVERY pass**, not only
        under `FREETOKEN_SCHEDULER_INVARIANT`: the comparison is three attribute reads next
        to the radix walk the same pass runs, and the env var now only decides whether a
        violation is additionally logged (`warn`) or raised. The soak that needed the number
        was not running with the var set, which is why it had none.
      - Transport: `SchedulerCountersMsg` -> `SchedulerCountersReply`, published at most
        every 2 s and only when the document moved, plus one forced flush before the
        scheduler parks on a blocking receive (so an idle poll is never 2 s stale).
      - Aborts are counted at the frontend call site (`abort_user(..., reason=)`), the only
        place the reason is knowable -- the wire carries one untagged `AbortMsg`:
        `client_disconnect` / `explicit` (prepare-stop drain) / `error`. The `error` count
        lives inside `observe`'s not-aborting branch: an abort's own terminal ack is an
        `ErrorReplyMsg("request aborted")`, so counting outside it scores every disconnect
        twice.
      - Batch log line gains `#seatable-lane` and `#chunked-inflight` (optional in
        `analyze.py`'s regex, so old soak logs still parse).
      - `benchmarks/switchyard_soak/analyze.py` now also reads `stats_*.json` snapshots and
        prints the per-phase delta between consecutive ones (the counters are cumulative for
        the server process).
      Tests: `tests/scheduler/test_scheduler_counters.py` (13, real `PrefillManager`),
      `tests/server/test_stats_counters.py` (10), plus two log-line tests in
      `test_scheduler_status.py`.
- [x] **§R7 ticket 1 CLOSED live (2026-09-05, `13af13d`).** `812bc57` divides the interleave
      share by the lanes the pass will actually SEAT instead of by queue depth. The starvation
      signature (`#new-seq: 1`, `#new-token <= 512`, `#queue-req >= 8`) goes 1,278/2,091 (61 %)
      stage and 200/1,050 (19 %) passthrough -> **0 of 1,202 prefill passes (0.0 %) on both
      routes**. Stage median `#new-token` 5,689. Full re-run: **stage 492 req / 0 err / 0
      STALLED**, p50/p95/p99 29,820/109,395/149,081 ms; **passthrough 1,904 / 0 err / 0
      STALLED**, 6,888/24,580/46,695. p95 -25 % on both routes, p99 -35 % / -44 %, requests
      +4.7 % / +19 %, effective new-token prefill rate 1,830 -> 2,310 tok/s. 0 invariant
      warnings, 0 committed_pages_required, 0 LinearStatePool exhausted, 0 eviction failures,
      0 tracebacks, 0 ERROR/CRITICAL; trailing silence 1 s / 2 s and **scheduling wall clock
      99.8 % of both phases**. Disconnect-abort re-verified (active 1 -> 0 in 5 s). Graceful
      shutdown 4 s, GPU 0 MiB. Write-up: soak results §V ("Run against 13af13d").
- [ ] **Watch mean lanes per prefill batch now that the divisor no longer caps it.** 1.83 ->
      3.43 (stage) and 3.53 -> 4.92 (passthrough) at `13af13d`. Passthrough sits inside the
      4.7-6.6 band the failing trees hit, but those were *stage-route* numbers with 15/32
      errors; stage here is 3.43 with 0 errors and the best p95 on record. Record lanes every
      soak; stage >~5 **together with** rising errors or p95 is the §R6/§R7 mode returning.
- [x] batch_memcpy probe stream ordering (`13af13d`): the probe zeroed `dst` on the ambient
      stream and copied on a private one with no `wait_stream`, so a busy caller could make it
      read its own memset late and latch `OffloadMoeCache._batch_memcpy = False` process-wide,
      silently disabling prefill hit-D2D. Fixed with `stream.wait_stream(current_stream())` +
      `dst.record_stream(stream)` (same pattern in `tests/moe/test_prefill_hit_d2d.py::
      test_batch_memcpy_roundtrip`), plus a new
      `test_batch_memcpy_probe_survives_busy_ambient_stream` that queues ~0.5 s of ambient
      work before calling `load_batch_memcpy()`. `uv run pytest tests/moe -q` whole-dir:
      **161 passed, 5 skipped**. NOTE: `--moe-prefill-hit-d2d` is OFF in the P2 serve profile
      (`moe_prefill_hit_d2d=False`), so a soak against that line exercises no
      `cudaMemcpyBatchAsync` at all and cannot confirm the probe latched True -- the test is
      the evidence. A soak that grades hit-D2D must pass the flag.
- [x] **Soak drivers moved into the repo at `benchmarks/switchyard_soak/`** (`run.sh`,
      `serve.sh`, `sample.sh`, `split.py`, `analyze.py`, `gaps.py`; `runs/` gitignored). The
      08:59 WSL OOM restart destroyed `scratchpad/soak7/` and an in-flight soak with it.
      `run.sh` refuses to start below 26 GiB `MemAvailable`; `sample.sh` records it per 5 s;
      `analyze.py` now reports lanes/batch and the starvation-signature fraction directly.
- [x] Merged fork/main (14 Ada/Ornith commits) into nemotron35 at 32cc504. Before deploying on Ada:
      rebuild the _gguf extension (multiwarp bool→warps int64; stale .so silently picks 4-warp path).
- [x] Scheduler.__init__'s unconditional torch.cuda.get_device_capability (from fork/main) is now
      routed through `_device_compute_capability`, which answers None off-GPU; the batch profile
      treats None as "no crossover" (grouping off) and keeps Ada's 1536 / Blackwell's 1280 exactly.
      Covered by tests/scheduler/test_batch_profile_capability.py. Full Scheduler construction on
      CPU is still out of reach for a small fixture (`__init__` builds a real Engine and a
      torch.cuda.Stream), so the tests exercise the helpers, not a constructed Scheduler.
- [x] Fixed the tests/scheduler/test_session_spill.py prefetch flake at the source: `start_prefetch`
      reaps a finished predecessor, and when the reader thread happened to land first that reap
      installed (and logged) the promotion, then dropped its id — so the caller's own
      `collect_prefetch(sid, wait=True)` found `_prefetch is None` and answered None. The store now
      parks such an unasked-for promotion in `_promoted` and hands it to the first caller that asks.
- [ ] fork/main fast-forward to the merge is the user's call (nemotron35 branch carries it).
- [x] **Extend-path MoE ticket 1** (from the n-gram NO-GO §6): the 11.6 ms per MoE layer per
      forward is not host-side planning, it is `_prefill_routed` streaming **every** expert of the
      layer into its double buffer on **every** forward -- nothing in that movement path reads
      `topk_ids`, which is why it is flat from 1 to 32 tokens. 128 x 5.612 MB = 718 MB per layer,
      16.5 GB per forward (the whole 15.4 GiB bank set), and 16.5 GB / 267 ms = **61.9 GB/s**: a
      saturated PCIe 5.0 x16 link. A 1-token extend routes 6 experts per layer and moves 128 --
      **21.3x, in bytes**. Fixed by `--moe-extend-cache-tokens` (default 64, 0 disables): below the
      threshold an extend takes the *decode* movement (`ensure_experts` + `copy_missing`) and the
      *prefill* grouped GEMM; movement and kernel were already independent arguments of
      `_expert_gemm`, the extend path just had them paired wrong. 64 is where the measured
      `D(m) ~ 6.2*m^0.75` distinct-expert curve reaches num_experts. **Measured 2026-09-05:
      forward 282.7 -> 27.7 ms (m=1), 282.7 -> 30.2 (m=8), 282.5 -> 30.9 (m=32); MoE 11.4 ->
      0.42-0.48 ms/layer, i.e. 9.2-10.2x on the forward and 23.6-27.3x on the MoE. 131K prefill
      5,059 -> 5,105 tok/s (+0.9 %, noise), needle recalled both arms, a long prompt with a short
      last chunk is greedy token-identical. A verify step is now 4.4x a graphed decode step
      instead of 42x, projecting 1.63x on the copy class.** The first attempt pointed the grouped
      prefill GEMM at the slot cache and faulted (sgl `moe_align_block_size` over ~1,800 experts);
      the GEMV is what ships, and the grouped variant is ticket 1 of the write-up. There is **no
      per-forward re-planning to hoist** above the threshold on the default configuration (six
      `Tensor.copy_` per layer, no host sync, plan precomputed in `_build_copy_plan`; the only
      per-layer host work belongs to the opt-in `--moe-prefill-hit-d2d` split), so the large-M path
      is deliberately untouched. Write-up:
      `benchmarks/results/nemotron35_lightning_5080_extend_moe_2026-09-05.md`; tests
      `tests/moe/test_extend_cache.py`.
- [ ] Extend-cache follow-ups (from the same run): (a) whether to raise the threshold to cover the
      scheduler's 512-token interleave chunks -- decidable in one run by per-chunk time and the
      following decode's miss rate at `--moe-extend-cache-tokens` 64/512/2048 on a 131K prompt;
      (b) `--moe-collect-stats`' "MoE decode miss stats" and the pageable-layer profile now also
      count extend routings below the threshold (`ensure_experts` bumps `lru_stats`/`decode_freq`);
      (c) `_ensure_experts_sized_kernel` evicts serially in a `(1,)` grid -- one argmin over
      `next_pow2(cache_size)` lanes per miss, fine at a decode step's 6 and the first suspect if
      the M = 32 number lands above prediction.
- [x] **`--speculative ngram`** (ticket 3 of the n-gram NO-GO §6, unblocked by the extend-MoE fix).
      Prompt-lookup speculation, greedy-only, single-stream v1. Plan and what shipped:
      - [x] `NgramDrafter` (incremental most-recent-occurrence index over prompt + output),
            `accepted_count`, adaptive draft length -- `python/freetoken/scheduler/spec_ngram.py`.
      - [x] Verify step: one extend forward over `m = k + 1` positions with **every** logits row
            kept (`Batch.logits_indices` + `Batch.last_indices`, replacing the direct
            `attn_metadata.get_last_indices` call in all six LM-head sites), greedy argmax in a
            dedicated `Engine.spec_verify_forward` (no `complete_one`, no sampler, always eager).
      - [x] Mamba-2 state: verify into a **private scratch slot** (`Req.spec_scratch_slot`, drawn
            from the LinearStatePool free-list, freed by `_free_req_slots`), never the live slot;
            each mixer records its scan inputs into `SpecScanCapture`
            (`models/nemotron_h/spec_scan.py`) and the accepted prefix is committed with one varlen
            SSD scan per layer plus a conv-window slide. A full acceptance skips the replay and
            copies the scratch slot back. NOT the already-reserved ping-pong pair the design
            sketched: those hold the tool-call anchor freeze and the radix chunk snapshot.
      - [x] KV rollback: `CacheManager.free_spec_tail` returns the pages the rejected positions
            allocated, restoring the allocator's invariant (pages exist exactly to
            `page_ceil(cached_len)`). Without it every partial rejection leaked `k - j` pages.
      - [x] Prefix cache never sees a rejected token: a verify batch is drained by this module,
            not by `_process_last_data`, so no `cache_req` runs on it, and the finish-time insert
            boundary is `cached_len`, which only ever advances by the accepted count.
      - [x] EOS / stop-string / length truncation inside an accepted run; one `DetokenizeMsg` per
            accepted token with `finished` on the last only.
      - [x] Engagement: a pre-drain `peek()` on the one-token-stale token list decides whether to
            drain and run a (necessarily synchronous) verify step. Overlap is what makes a plain
            decode step 6.9 ms rather than ~9, so running the whole loop drained would cost ~30 %
            on the ~99.6 % of code/prose steps that never draft.
      - [x] Flags `--speculative ngram`, `--spec-ngram-n 8`, `--spec-draft-len 8`,
            `--no-spec-adaptive`; refused (with a warning, falling back to ordinary decode) on SWA,
            on non-radix recurrent caching, and per-step on sampling / multi-request / multimodal /
            hidden-state-probe requests.
      - [x] Tests: `tests/scheduler/test_spec_ngram.py` (26 CPU tests: drafter, acceptance,
            adaptive k, and the full verify step against a faked scheduler -- partial/total/full
            acceptance, EOS truncation, budget and tool-call-anchor clamps, scratch-slot lifecycle,
            pool exhaustion); `tests/e2e/test_spec_ngram_equivalence.py` (CUDA, `needs_weights`).
      - [x] Benchmarks: `benchmarks/probe_spec_ngram_impl.py` (one model load, both arms toggled on
            the live scheduler, warm prefix tree so only decode is compared).
      - [x] Break-even gate: the verify/decode cost ratio is ~7x at short context and ~10x at
            131K, so the decoder measures both terms online (the gap between consecutive peeks IS
            a decode step) and stops drafting when `accepted + 1` can no longer pay. Verify uses a
            running MINIMUM (its first sample pays Triton autotune) and decode an EWMA (its
            minimum is far below a real step) -- getting either wrong cost a GPU run each.
      - [x] **Measured** (write-up §5): code 1.03x, prose 1.02x, copy 1.01x (1.11x ungated, same
            drafter stats -- read it as ~1.05x), 131K needle 0.89x. Commit self-check bit-exact
            (0.000e+00). 16-way soak PASS both arms, 0 errors, p50 and request count flat.
- [ ] Speculation follow-ups, by measured upside: (a) **~40 % of a verify step is not the forward**
      (~52 ms end-to-end vs a ~30 ms extend forward) -- the commit issues 46 eager kernel launches
      and `_prepare_batch` rebuilds pinned staging for a one-request batch; taking the step from 7x
      to 4x a decode step moves the copy class from 1.03x to ~1.12x; (b) the **burst-entry
      hysteresis** costs a factor of ~4 in draft rate (0.079 measured vs 0.353 offline) -- a
      stickiness latch, decidable in one run at latch 0/2/4; (c) `--spec-draft-len` 4/8/16/24 on
      the 131K needle, since break-even there needs `accepted + 1 > ~10` against a ceiling of 9;
      (d) the soak tail (p95 +22 %, p99 +131 % on ONE 10-minute pair with different session mixes;
      re-run at the reference 20-minute phase length); (e) batched (bs > 1) verify --
      `_make_write_tuple` and the drain loop are one-token-per-request; (f) a graph-captured
      fixed-width verify forward; (g) sampling (non-greedy) speculation needs `Sampler.prepare` to
      repeat-interleave its per-request parameter rows by k; (h) the drafter indexes the whole
      prompt on first engagement (~0.1-0.2 s at 131K).

## MoE prefill GEMM (ticket §9.2 of the prefill profile) — 2026-09-05

Ticket: *"MoE prefill is now the flat term and it is 33 TFLOP/s"* — 29.5 ms per layer at
M=8192, 23 layers, 79 % of the position-independent per-chunk cost.

- [x] Read the prefill profile / q8 ceiling / extend-MoE / cache-study write-ups first, and
      re-derive the denominators: the honest ceilings are **123 TFLOP/s** (cuBLAS bf16) and
      **118 TFLOP/s** (Triton `tl.dot`), not the 225 spec sheet.
- [x] **The "b12x returned nan at M=8192" ticket is a reporting artifact, not a bug.**
      `bench_nvfp4_moe_kernels.py`'s summary prints `nan` for every column when a backend
      produced no rows; re-run on this host, b12x runs clean and is 12.60 ms at M=8192
      (78 TFLOP/s) against Triton's 29.47. Nothing to fix; the number is the target.
- [x] `benchmarks/bench_moe_prefill_gemm.py` (new): the prefill grouped GEMM alone, TFLOP/s
      against the measured ceiling (not the HBM roofline, which is the wrong figure of merit
      at M=8192), with a tile-grid sweep, a routing-skew knob and a real-`topk_ids` replay.
- [x] Tile sweep at M=8192: the tuner's grid is exhausted — best in it is 26.4 ms (1.12x).
      **The kernel structure, not the tile, was the limit.**
- [x] Root cause, by ablation: the K-loop loaded ONE e4m3 scale **per packed byte**, i.e.
      every value 8 times (one scale covers 16 k-values = 8 bytes). Loading the distinct
      `[BLOCK_KB/8, BLOCK_N]` rows and broadcasting is **1.73x** on its own and drops the
      kernel's shared memory 28 KB -> 12 KB, which is what unlocks the wider tile.
- [x] Second term: `cvt.rn.f16x2.e2m1x2`, the Blackwell hardware FP4 decode (the same
      instruction flashinfer's b12x kernel uses), replaces `_e2m1_decode`'s ~14-op bit
      construction — bit-identical for all 256 packed bytes, worth a further 1.04x.
- [x] Result: **29.47 -> 16.67 ms per layer at M=8192, 1.77x, 58.8 TFLOP/s = 50 % of the
      Triton `tl.dot` ceiling** (was 28 %), on the tile `64/256/16/1/4/3`. Bit-identical
      (`0.000e+00`) to the old kernel at every tile swept.
- [x] `FREETOKEN_NVFP4_PREFILL_{BLOCK_M,BLOCK_N,BLOCK_KB,GROUP_M,NUM_WARPS,NUM_STAGES}` —
      the twins of `FREETOKEN_EXTEND_*` / `FREETOKEN_DECODE_*`, so a tile A/B is two
      invocations of one binary.
- [x] `benchmarks/tune_nvfp4_moe.py`: `BLOCK_N=256` and `BLOCK_KB=16` added to the grid
      (unreachable before the scale fix), plus `--merge` so a table can be rebuilt in passes.
- [x] VMM int32/int64 dtypes (`csrc/vmm_tensor.cpp` + `kernel/vmm.py`): growable KV plus
      `--nvfp4-backend flashinfer` died at startup on the b12x int32 bank.
- [x] Re-tuned per-M-bucket tables (the tuner picks `64/256/16/8/4/3` at M=4096/8192
      independently of the hand sweep): 256 1.82 -> 1.275 ms, 2048 8.47 -> 5.107,
      **8192 29.47 -> 16.95 ms (57.9 TFLOP/s)**. Routing skew is worth 0.7 % at this M
      (Dirichlet alpha 0.25/1.0/4.0), so the real-`topk_ids` capture the ticket asked for is
      not the discriminator here — the harness keeps `--routing-file` for the small buckets.
- [x] e2e A/B, two servers (before arm = worktree at `e4070da`), same prompts:
      **131,088 tokens 26.69 -> 22.29 s TTFT (1.198x), 262,160 75.31 -> 66.60 s (1.131x)**,
      needle recalled in all four runs, answers byte-identical, greedy short prompt
      byte-identical. Δ per chunk 275/272 ms vs the microbench's predicted 288.
- [x] `--nvfp4-backend flashinfer --kv-grow-step-tokens` starts and serves (was a startup
      `ValueError`); `tests/kernels/test_vmm_tensor.py` pins int16/int32/int64.
- [x] 193 tests pass (`tests/moe` + `tests/kernels/test_vmm_tensor.py`), 5 skipped.
- [ ] Follow-ups, in measured order (§10 of the write-up): (a) b12x is still 1.34x on the
      M=8192 grouped GEMM and the gap is the **operand path**, not the dequant — the
      no-scale-at-all ablation is *slower* than the shipped kernel — so closing it means
      adopting a swizzled bank layout, which is a load-time global decision that costs
      decode 1.6-1.9x; (b) `a_ptrs_lo`/`a_ptrs_hi` read the same span at a 2-element stride
      so neither activation load vectorizes — a one-off deinterleave of `A` per GEMM is
      ~0.5 ms of HBM and is the most likely remaining Triton-side win; (c) the decode GEMVs
      load their block scale the same redundant way and were not touched (decode is
      HBM-bound, so it may be worth nothing — but it is the same three lines);
      (d) `bench_nvfp4_moe_kernels.py --gate` still asserts the 2B1 targets ("b12x >= 2x
      triton at M=8/16"), which are now inverted and fail on a healthy tree; (e) the M=256
      bucket runs at 20 % of the ceiling with +53 % padding waste at `BLOCK_M=16`, and the
      scheduler's interleave share really does produce 512-token chunks.

## 2026-09-05 — 16-way batched decode: profile + fix (Switchyard's real regime)

Question: at 16 decode lanes the aggregate is 97 tok/s (stage) / 178-190 (passthrough),
6-11 tok/s per stream, against 145 tok/s single-stream at 131K. Where does the step go?

- [x] Phase A (weightless kernel sweep, one lock, 5 min) — attribution for (b) MoE GEMV
      at m=16, (c) decode attention at 16 lanes x long ctx, (d) Mamba-2 at batch 16,
      (a) expert-gather bandwidth. `benchmarks/decode16/phaseA.sh`.
- [x] (a) from the existing cache study + a nemotron profile added to
      `bench_offload_cache_copy`: the gather runs at 51-52 GB/s at every batch and miss
      count, i.e. at the measured PCIe ceiling (52.9). Bytes, not the kernel.
- [x] (c) ruled out: 64 splits is still the best config at bs=16 x 131K (1.476 ms/layer,
      773 GB/s = 80 % roofline), 1.51x the old 8-split default. Not over-split.
- [x] (d) ruled out: 38.4 us/layer graphed at bs=16 -> 0.9 ms/step over 23 mamba layers.
- [x] (e) FOUND — `_elastic_graph_batch_sizes` returned `[1,2,3,4,8]`, and
      `can_use_cuda_graph` gates on `max(list)`, so every decode batch of 9-16 lanes on
      the P2 profile ran EAGER. 314/427 decode batches (73.5 %) of the 13af13d soak.
- [x] Fix: always capture the tier's own capacity; keep the sparse power-of-two set.
      `FREETOKEN_ELASTIC_GRAPH_MAX_BS` caps it so before/after is one binary.
- [x] Tests: `tests/engine/test_elastic_graph_sizes.py` (+5 cases, incl. the invariant
      `max(sizes) == capacity` for every tier 1..64).
- [x] `bench_decode_moe --pad-lanes/--pad-tokens`: mixed-context decode batches.
- [x] Phase B/C: before/after at 16 lanes (mixed and uniform contexts), single-stream,
      131K needle. Engine-side +3.9 % at bs=16; the client aggregate cannot resolve it.
- [x] **Phase E caught the first fix being a NET LOSS.** A sparse `[1,2,3,4,8,16]` set makes
      a 12-lane batch pad up to bs-16 with dummy rows that route their own experts: 88.0 ms
      vs 82.2 ms eager (−6.7 %). And 421 of the soak's 427 decode batches run at capacity 16
      with batch sizes spread across the band (164 at 9-15). Sparse would have helped 149
      batches by 3 % and hurt 164 by 7 %.
- [x] Fix v2: the set is **dense to `_DENSE_GRAPH_BS = 16`**, then a 1.33-1.5x ladder, with
      the tier's capacity always appended. Tests rewritten as invariants (no batch pads; the
      capacity is always captured; the ladder stays sparse above 16).
- [x] Phase F (headline): 12 lanes in a 16-request pool, eager -> exact graph,
      **143.21 -> 153.84 tok/s = 1.074x**, step 83.8 -> 78.0 ms, P = 0.85, n = 11/10.
      Dense set costs **80 MiB** and <1 s per resize; MoE cache unchanged at 976 slots.
- [x] Write-up `benchmarks/results/nemotron35_lightning_5080_decode16_2026-09-05.md`,
      docs/nemotron.md, tasks/lessons.md.

### Review

The largest term at 16 lanes is **not** fixable in a bounded way: 74 % of the step is the
PCIe expert gather, measured at 51-52 GB/s = the link roofline, with a working set of
~1,417 expert-layer slots against 976 in the pool. Attention (c), the MoE GEMV (b) and
Mamba-2 (d) were all measured and are all fine at batch 16. What *was* broken is (e): the
elastic decode-graph set stopped at 8, so 73.5 % of the soak's decode batches ran eager.
Fixed, worth **1.074x at 12 lanes / 1.039x at 16**, and ~5 % weighted over the soak's own
batch histogram. Single-stream and the 131K needle are untouched by construction (the helper
is only consulted when `--elastic-initial-requests` is set) and were measured anyway:
135.84 tok/s at bs=1, 132.32 tok/s decoding at a 130,016-token context with the needle exact.

Not committed, per instruction. Open tickets in §7 of the write-up; the first
(`_determine_cuda_graph_bs` has the same padding defect on the non-elastic path) is the one
to pick up next, and its experiment is already written as `phaseE.sh`.

Note (disagreeing with the brief): "8 lanes at 100K+" is not a regime this engine can
reach — `--num-tokens 262144` is the whole KV budget, so 16 lanes average 16K each. The
stage route's decode lines show `#token: 27733` at 15 running requests (1.8K/lane). The
mixed arm therefore uses 8 lanes x 16K + 8 short, which is the honest saturated shape.
