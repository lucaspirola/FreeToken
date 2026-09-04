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
  - [x] 2B2 triton fallback tuning  - [x] 2B4 cache sizing study (triton default, LFU for 16-way, hybrid rejected)  - [ ] 1M single-session spill gate (also measures restore NVMe vs RAM for candidate 3F prefetch)

## Phase 3 — Switchyard
- [x] 3A wire/errors  - [x] 3B JSON mode  - [x] 3C sessions+parsers  - [ ] 3D soak run (prep done)  - [x] 3E residency: spill on demand + capacity/age retention + restart-persistent checkpoints  - [x] 3F prefetch next queued checkpoint to RAM  - [x] 3G partial-prefix restore + stray `</think>` + prefill-time boundary capture

## Phase 3H — hidden-state export (Switchyard probe contract)
- [x] 3H implement (1f2de67)  - [ ] 3H GPU parity check

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
- [ ] Follow-up from that bisect (perf, not correctness): `decode_launch_config` has no
  context-length key and every tuned branch needs the Ornith head shape, so Nemotron always takes
  the `(kv_splits=8, block_n=32, warps=4)` fallback -- 16 CTAs on 84 SMs at 262K KV. Likely the
  72.6 -> 51.8 -> 32.0 tok/s decode curve. `num_kv_splits_ptr` is passed to both decode kernels
  and dereferenced in neither.
- [ ] Follow-up from that bisect (1M blocker): Triton KV loaders widen slot ids to int64 on store
  (`kv_quant.py:52,161`) but not on load (`attention.py:621,1119,1271`). Safe at Nemotron's
  `stride_ks=256` (~8.4M slots) but ~1.05M at head_dim 256 / 8 kv heads.
