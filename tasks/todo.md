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
- [x] 3A wire/errors  - [x] 3B JSON mode  - [x] 3C sessions+parsers  - [ ] 3D soak run (**FAILS at `ea7ed7c`+`acc91e9`, 2026-09-05, WORSE than 81ab30e**: stage 176 req/32 timeouts/12 STALLED, passthrough 32/32 = 100% error/18 STALLED; permanent deadlock, last batch 5m35s in, 2,616 s of unbroken silence; 14 chunked prefills own the pool at 1.76x its size. Last passing tree is still `befcde6`+reserved_pages fix -- see soak results §"Run against ea7ed7c" and handover item 2)  - [x] 3E residency: spill on demand + capacity/age retention + restart-persistent checkpoints  - [x] 3F prefetch next queued checkpoint to RAM  - [x] 3G partial-prefix restore + stray `</think>` + prefill-time boundary capture

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
- [ ] Q5: profile the prefill curve (3,000 tok/s @131K → 1,064 @524K): attention vs SSD scan vs
      KV grow; fix or file
- [x] Q1: standing cross-engine oracle — tool landed in `5f7c0d6`, **first live sweep run
      2026-09-05 at 262K** (`benchmarks/results/nemotron35_lightning_5080_oracle_2026-09-05.md`):
      FreeToken 19/24 vs llama.cpp 17/24, 2 `freetoken-only-miss` (both composition, not
      retrieval), **0 retention failures on either engine**. Top-k logit comparison still
      blocked on FreeToken having no logprobs in `SamplingParams` (see docs/oracle.md).
      **1M FreeToken leg also run** (no llama.cpp leg): 7/19 turns, 0 `retention`, 5/6 needles
      recovered by leak-free reverse probes — interference, not retention. Remaining: the 131K
      and 524K rungs, and the llama.cpp 1M leg.

## Backlog
- [ ] Prompt-lookup (n-gram) speculative decoding for agent-session decode
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
