# FreeToken — Nemotron 3.5 Lightning (Switchyard) on RTX 5080

Handover: `tasks/nemotron35-handover.md`. Plan: `tasks/nemotron35-plan.md`. Rules:
`tasks/lessons.md`. Results files below are in `benchmarks/results/` unless stated;
`N35 =nemotron35_lightning_5080_`.

HEAD `14c1bd8`, tree clean. `fork/nemotron35` == HEAD (pushed); `fork/main` is 77 behind, 0 ahead.

---

## Open

### Perf, ranked by measured upside
- [ ] **n-gram verify overhead.** ~40 % of a verify step is not the forward (~52 ms vs a ~30 ms
      extend forward): 46 eager kernel launches in the commit, `_prepare_batch` rebuilding pinned
      staging for a one-request batch. 7x → 4x a decode step moves the copy class 1.03x → ~1.12x.
      Same ticket list: burst-entry hysteresis costs ~4x in draft rate (0.079 measured vs 0.353
      offline) — decidable in one run at latch 0/2/4; `--spec-draft-len` 4/8/16/24 on the 131K
      needle; the soak tail (p95 +22 %, p99 +131 % on one 10-minute pair — re-run at 20 min);
      batched (bs>1) verify; a graph-captured fixed-width verify forward; non-greedy speculation
      needs `Sampler.prepare` to repeat-interleave parameter rows by k; the drafter indexes the
      whole prompt on first engagement (~0.1–0.2 s at 131K).
      Evidence: `N35ngram_spec_impl_2026-09-05.md` §6.
- [ ] **`graph.py::_determine_cuda_graph_bs` padding defect on the NON-elastic path.**
      `[1,2,4] + range(8, max_bs+1, 8)` pads 3→4, 5-8→8, 9-16→16; the 12→16 case measured
      **−6.7 %** on this checkpoint (a padded row routes its own experts). Blast radius is every
      model and profile, so measure one dense model first — the experiment is already written as
      `benchmarks/decode16/phaseE.sh` (two servers, `--cuda-graph-max-bs 8` vs `16`, c=12).
      Evidence: `N35decode16_2026-09-05.md` §7.1. Expected: ~1.07x at partial lane counts.
- [ ] **MoE A-operand vectorization.** `a_ptrs_lo`/`a_ptrs_hi` read the same span at a 2-element
      stride so neither activation load vectorizes; a one-off deinterleave of `A` per GEMM is
      ~0.5 ms of HBM. Most likely remaining Triton-side win against b12x's residual 1.34x (the
      no-scale ablation is *slower* than the shipped kernel, so the gap is the operand path, not
      the dequant; adopting b12x's swizzled bank layout is a load-time global decision that costs
      decode 1.6–1.9x and is therefore rejected).
      Evidence: `N35moe_prefill_gemm_2026-09-05.md` §10(a)(b).
- [ ] **Extend-cache threshold vs the scheduler's 512-token chunks.** One run decides it:
      per-chunk time and the following decode's miss rate at `--moe-extend-cache-tokens`
      64 / 512 / 2048 on a 131K prompt. Related from the same write-up: `--moe-collect-stats` and
      the pageable-layer profile now also count sub-threshold extend routings;
      `_ensure_experts_sized_kernel` evicts serially in a `(1,)` grid (first suspect if M=32 lands
      above prediction). And the M=256 GEMM bucket runs at 20 % of ceiling with +53 % padding
      waste at `BLOCK_M=16`.
      Evidence: `N35extend_moe_2026-09-05.md`; `N35moe_prefill_gemm_2026-09-05.md` §10(e).
- [ ] **16-way decode is at the hardware ceiling — do not re-litigate.** 74 % of the step is the
      PCIe expert gather at 51–52 GB/s against a measured 52.9 GB/s link; working set ~1,417
      expert-layer slots against 976 in the pool. Attention (64 splits, 80 % roofline), the MoE
      GEMV and Mamba-2 were all measured fine at batch 16. Only two levers remain:
      (a) `--moe-backend hybrid` at 16 lanes, never measured — the auto-threshold compares
      standalone CPU BW vs standalone PCIe BW (1.26 here) when the right criterion is
      `(cpu_ov + pcie_ov) / pcie_alone` = 1.73; (b) quantify what the 64-slot Mamba-2 snapshot
      cache (~2.4 GiB of expert slots) buys in prefix reuse.
      Evidence: `N35decode16_2026-09-05.md` §0/§2/§7.2–7.3.

### Scheduler / server tickets
- [ ] **An over-pool prompt has no client rejection path.** `PrefillManager.schedule_next_batch`
      skips a fresh request whose `input_len + output_len > cache_manager.max_size`, logs one
      warning and `continue`s — never removing it from `pending_list`, never failing the request,
      so the client hangs until its own timeout. It also keeps inflating `waiting`, shrinking every
      other lane's chunk, and `_seatable_lanes` lacks the same skip, so one unadmittable prompt
      pins the seatable-lane estimate. Fix: 400/413 at admission, and mirror the skip in
      `_seatable_lanes`.
- [ ] **`stopped_for_lane_cap` rotation is dead code on this model.** `lane_cap =
      _resolve_max_prefill_seqs(config)` is 0 for Nemotron (confirmed live with `py-spy dump
      --locals`), so the interleaved `pending_list = remaining + chunked_list` branch is
      unreachable exactly on the profile that turns interleaving on. Its comment describes
      `blocked_fresh` / the refusals break. Set the flag there or delete the branch.
- [ ] **A refused prefill pass costs O(queue × prompt) radix walks.** 16 pending 118K-token
      prompts are re-`match_prefix`ed from scratch on every pass that returns `None` (4 of 5
      py-spy samples during the `81ab30e` stall were inside `fast_compare_key`). Cache the match
      per pending request, or skip the walk once the pass has refused a fresh admit.
- [ ] **`_maybe_shrink_growable_kv` calls `evict_all_unlocked_prefixes()` before computing
      whether a shrink is possible**, so above the initial KV step every idle moment wipes the
      whole prefix cache and often shrinks nothing (`server.gen1.log` 09:44–09:47).
- [ ] **`benchmarks/scheduler_replay.py` is not an acceptance gate for scheduler policy.** It
      scored `81ab30e` at 2.49x tokens / 2.14x completions — the commit that then failed the live
      soak on both routes. It still models no spill/cold-restore cost for a reclaimed lease, no
      non-reclaimable leases, no GDN state slots, and charges per-pass CPU to `match_calls` rather
      than the clock. Extend it or demote it.
- [ ] **Session residency leftovers** (from the 1M gate, §6 of `N351m_sessions_2026-09-04.md`): a
      *resident* session is never checkpointed, so a restart loses it (spill-on-shutdown flag);
      `_evict_one_lru` can evict the record the pending admission is about to restore.
- [ ] **Watch mean lanes per prefill batch every soak.** 1.83 → 3.43 (stage) and 3.53 → 4.92
      (passthrough) at `13af13d`, now that the divisor no longer caps it. Stage >~5 **together
      with** rising errors or p95 is the §R6/§R7 failure mode returning. Passthrough sitting in the
      old 4.7–6.6 band is not a regression — that band was a *stage-route* measurement.
- [ ] `num_kv_splits_ptr` is passed to both decode kernels and dereferenced in neither (the split
      count is a constexpr). Delete it or use it.

### Recall / oracle
- [ ] **131K rung on both engines** (~10 min total) — the only unrun length.
- [ ] **524K `direct:harbour` lead.** The one leak-free direct probe llama.cpp holds and FreeToken
      loses (it returns the *orchard* code — `interference-cross`; `reverse:harbour` recovers it
      two turns later). It is also turn 2, the one turn whose TTFT was 50.0 s against 2.4 s for
      turns 3–19, i.e. a partial-prefix re-prefill. Cheap re-probe: re-run 524K with
      `--filler-cursor 65` and see whether the pairing repeats. `N35oracle_2026-09-05.md`.
- [ ] Optional: the llama.cpp leg alongside the 1M FreeToken leg is **impossible on this card**
      (~20 h of prefill against a 4 h lock cap) — recorded as a host limit, not a gap to close.

### Process / repo
- [ ] **fork/main fast-forward to the merge is the user's call.** `fork/main` (`62f5a66`) is a
      strict ancestor of HEAD; `fork/nemotron35` already carries it.
- [ ] **Rebuild the `_gguf` native extension before deploying on Ada.** The fork/main merge
      (`32cc504`) changed multiwarp bool → warps int64; a stale `.so` silently picks the 4-warp path.
- [ ] CI follow-ups: (a) re-run the full unit-test step with no GPU job live and record the wall
      time (the 14.2 s figure predates the current GPU session); (b) fold `tests/moe` +
      `tests/kernels` into the CI list — only `tests/moe/test_offload.py::
      test_adjust_config_converts_moe_cache_rate_to_cache_size` blocks it (asks for the `fi`
      backend and errors instead of skipping without flashinfer); the other 154 pass on CPU;
      (c) the `[tool.ruff.lint] ignore` list in `pyproject.toml` is the pre-existing violation set
      (E741 ×79, E702 ×47, E731 ×18, F401 ×17, F841 ×13, …) — clearing any line lets it be deleted.
- [ ] `bench_nvfp4_moe_kernels.py --gate` still asserts the 2B1 targets ("b12x >= 2x triton at
      M=8/16"), which are now inverted and fail on a healthy tree.
- [ ] Replay real Switchyard traces as the benchmark instead of synthetic soaks/needles.
- [ ] **Pre-existing, not this effort's.** `tests/server/test_muse_glimmer_parsers.py` is reported
      to need the repo root on `PYTHONPATH` to collect. `tests/models/test_laguna_modules.py`
      raised 6 RuntimeErrors only while a sibling agent's server was loading and passes alone — a
      suite-ordering/GPU-contention artifact; re-run a suspicious failure alone before attributing
      it to a diff. Neither re-verified this session (no pytest while a model is loaded).
- [ ] Remove or ignore two stale worktrees that predate this effort:
      `.claude/worktrees/agent-a45f827ae98e76526`, `.claude/worktrees/agent-a4ce5e26d2ccafdb6`.

---

## Done

### Nemotron 3.5 Lightning — phases
- [x] **Phase 1 bring-up** — model package, engine/CLI/AOT/docs/preflight, all gates (parity,
      invariance, prefix, elastic, tool, 64K/128K needle; the 128K miss was a `trim_filler`
      benchmark bug).
- [x] **Phase 2 kernels** — flashinfer SSU probe, Mamba-2 layout/metadata, SSD prefill, decode
      SSU + gated norm, b12x relu2, dense NVFP4 tuning, Triton fallback tuning, cache-sizing study
      (Triton default, LFU for 16-way, hybrid rejected), CUDA-graph use-after-free fix.
      `N35mamba2_2026-09-04.md`, `N35cache_study_2026-09-04.md`.
- [x] **Phase 3 Switchyard** — 3A wire/errors, 3B JSON mode, 3C sessions+parsers, 3D soak,
      3E residency (spill on demand, capacity/age retention, restart-persistent checkpoints),
      3F RAM prefetch, 3G partial-prefix restore + stray `</think>` + prefill-time boundary
      capture. `N35switchyard_2026-09-04.md`.
- [x] **Phase 3H hidden-state export + GPU parity** — all 52 layers cosine ≥ 0.998840 (gate 0.99,
      median 0.999760) against a meta-device streamed transformers reference. `1f2de67`, `befcde6`;
      `N35hidden_states_parity_2026-09-04.md`. The reference needs `--reference-dt-min 0.0`:
      transformers hard-codes the same `dt` floor, independently confirming the 262K root cause.
- [x] **Phase 4 MTP — NO-GO** (verify step 1.63x cost, projected 0.96x gain; flag not built).

### Correctness
- [x] **262K recall root cause — the Mamba-2 `dt` floor.** `dt_limit` was `(time_step_min, inf)`
      = `(1e-3, inf)` in the prefill scan; `time_step_min` is HF's *initializer* range for
      `dt_bias`, not a runtime bound, and the floor caps every head's memory horizon at
      `1/(|A|·1e-3)` tokens. `dt_limit=(0.0, inf)` turns 147,456 and 262,144 @ depth 0.52 from FAIL
      to PASS at identical TTFT. `3ac79ec` (`models/nemotron_h/config.py::_dt_floor`,
      `FREETOKEN_NEMOTRON_DT_MIN` hatch, 3 tests). `N35262k_rootcause_2026-09-04.md`.
      This **retracted** the earlier bisect verdict ("no FreeToken bug, gate mid-depth at 131K
      only") — `N35262k_bisect_2026-09-04.md`; the exonerations it established (growable-vs-static
      KV bit-identical, KV dtype, attention backend, chunk size, dense dequant) still stand.
      Cross-engine confirmation: `N35262k_crossengine_2026-09-04.md`.
- [x] **1M gate — all four criteria PASS.** One session grown to 1,039,989 tokens (needle recalled
      at every length, twice); demand spill 3.53 GiB to NVMe in 2.980 s; a **new** `ft serve`
      adopted the checkpoint and restored 1,040,020 tokens in 2.681 s (1.32 GiB/s) with a
      byte-identical answer; capacity/age eviction verified. `31d606d`;
      `N351m_sessions_2026-09-04.md`.
- [x] **1M multi-needle, one prefill, 8 graded questions (5/8).** The headline is question 8: the
      "missed" depth-0.25 needle was recovered by a composition question, so grade with more than
      one question shape per needle before blaming retention. `benchmarks/bench_multi_needle.py`;
      `N351m_multineedle_2026-09-04.md`.
- [x] **Cross-engine oracle** — `benchmarks/oracle_cross_engine.py` (`5f7c0d6`), `docs/oracle.md`.
      262K: FreeToken 19/24 vs llama.cpp 17/24, 0 retention, 0 selection, 12/12 needles in state.
      524K on both engines and 1M FreeToken-only (`be85ffa`): the `key → code` collapse returns the
      same wrong near-duplicate **byte-for-byte in both engines** — a model property, closed as
      model-limited, no kernel bug. `N35oracle_2026-09-05.md`.
- [x] **`cached_tokens: 0` was a missing `--enable-cache-report`, not a regression.** Fixed as a
      presence rule: `prompt_tokens_details` is emitted whenever reporting is on (explicit 0 for a
      genuine miss) and absent entirely when off; `/v1/responses` always reports, since its schema
      makes the field mandatory. `docs/oracle.md` / `docs/switchyard.md` updated.

### Performance
- [x] **Decode launch config** (`acc91e9`) — `_grid_filling_splits` sizes the split count to the SM
      count for untuned head shapes (64/64/8 here instead of 8/32/4); int64 slot ids on KV load
      behind a compile-time `SLOT_I64`. 82.8→145.3 (131K), 58.7→132.4 (262K), 35.4→113.6 (524K),
      ~20→95.8 (1M) tok/s; prefill unchanged. `N35decode_launch_2026-09-04.md`.
- [x] **Prefill superlinearity** (`4a99e34`) — the whole superlinear term was the extend kernel's
      launch: `_select_extend_tile`'s `head_dim<=128` arm hard-coded `BLOCK_M=128` while sm_120
      takes 4 warps, so the fp32 accumulator spilled (396 slots vs 14). `extend_launch_config` now
      caps `BLOCK_M` by the register budget. 131K 3,230→5,288, 262K 1,965→3,683, 1M 573→1,307
      tok/s; 1M TTFT 1,810→795.8 s. SSD scan, KV grow, page-index build all exonerated.
      `N35prefill_profile_2026-09-05.md`.
- [x] **Native-Q8 extend QK — CLOSED NEGATIVE, kernel unchanged** (`a25e954`). 225 TFLOP/s is the
      spec sheet; the part does 123.0 (cuBLAS bf16) / 118.4 (Triton `tl.dot`), so the kernel is at
      57–60 % of achievable, not 31 %. int8 is 1.04x bf16; the whole q8_0 dequant is worth 1.206x;
      the best accuracy-preserving combination is 1.05x. `N35prefill_q8_2026-09-05.md`.
- [x] **MoE prefill GEMM 1.74x** (`2a139ad`) — the K-loop loaded one e4m3 scale *per packed byte*
      (every value 8×); loading the distinct rows and broadcasting is 1.73x alone and drops shared
      memory 28→12 KB, and `cvt.rn.f16x2.e2m1x2` adds 1.04x. 29.47→16.95 ms/layer at M=8192
      (57.9 TFLOP/s, 49 % of ceiling), **bit-identical**. e2e 131K TTFT 26.69→22.29 s (1.198x),
      262K 75.31→66.60 s. Plus `FREETOKEN_NVFP4_PREFILL_*` knobs, a retuned per-M table, and VMM
      int32/int64 dtypes (which also fixed `--kv-grow-step-tokens` + `--nvfp4-backend flashinfer`
      dying at startup). `N35moe_prefill_gemm_2026-09-05.md`.
- [x] **Extend-path MoE 9–10x** (`89b632b`) — `_prefill_routed` streamed **every** expert of a
      layer into its double buffer on every forward (128 × 5.612 MB = 718 MB/layer, 16.5 GB per
      forward = 61.9 GB/s, a saturated PCIe 5.0 x16 link) because nothing in the movement path
      reads `topk_ids`. `--moe-extend-cache-tokens` (default 64, 0 disables) routes small extends
      through the *decode* movement with the *prefill* GEMM. Forward 282.7→27.7 ms (m=1),
      →30.9 (m=32); MoE 11.4→0.42–0.48 ms/layer. `N35extend_moe_2026-09-05.md`,
      `tests/moe/test_extend_cache.py`.
- [x] **Elastic CUDA graphs** (`14c1bd8`) — `_elastic_graph_batch_sizes` returned `[1,2,3,4,8]` and
      `can_use_cuda_graph` gates on `max(list)`, so 73.5 % of the soak's decode batches (9–16
      lanes) ran **eager**. The first fix (sparse to 16) was a **net loss** (a 12-lane batch pads to
      16 and the dummy rows route their own experts: −6.7 %); v2 is dense to 16 then a 1.33–1.5x
      ladder, capacity always appended. 12 lanes 143.21→153.84 tok/s (1.074x), 16 lanes 1.039x,
      ~5 % weighted over the soak's batch histogram; costs 80 MiB. `N35decode16_2026-09-05.md`.
- [x] **Speculative decoding shipped but not on by default** (`e4070da`, `--speculative ngram`).
      Full design in `N35ngram_spec_impl_2026-09-05.md`: drafter, all-rows verify forward, private
      Mamba-2 scratch slot + varlen SSD commit, `free_spec_tail` KV rollback, prefix cache never
      sees a rejected token, online break-even gate. Measured 1.03x code / 1.02x prose / 1.01x copy
      / 0.89x at 131K; commit self-check bit-exact; 16-way soak PASS both arms. The earlier NO-GO
      (`193da80`, `N35ngram_spec_2026-09-05.md`) was correct at the time and was unblocked by the
      extend-MoE fix.
- [x] **Ornith/Ada line merged** (`32cc504`, 14 commits `cefa4bd..62f5a66`): sm_120 GGUF dispatch
      thresholds, the upstream int8-MMA MMQ port (Q4_K/Q6_K, ~1.75x prefill on Ornith), the Ada
      sm_89 port, asymmetric KV, elastic multi-agent, counter-guided expert cache. `Scheduler.
      __init__`'s unconditional `torch.cuda.get_device_capability` routed through
      `_device_compute_capability` so CPU construction works (`52a6503`).

### Scheduler / server
- [x] **Slot-reclaim crash fix** (`c4486b6`, `fad1fc4`) — `_cache_req_hybrid` reserves the
      replacement ping-pong slot before donating the frozen Mamba snapshot; one escalating reclaim
      path (free-list → LRU snapshot eviction → on-demand spill of the LRU idle lease); `/health`
      503 with a reason when a worker is gone; bounded shutdown on every stop path.
      `tests/scheduler/test_hybrid_pool_exhaustion.py`, `tests/server/test_health_liveness_and_shutdown.py`.
- [x] **Admission gate, third attempt lands** (`d685e99` + `b030c7f` standing reservation +
      finishability invariant). Two failures first: `81ab30e` (charged against the whole pool —
      stalled stage 52 % of the wall clock, reverted in `5bf0bcc`) and `ea7ed7c` (charged against
      `admissible_size` at admission only — permanent deadlock, 14 chunked prefills owning 1.76x
      the pool). The bug family: *a budget checked only at admission is not a budget; the invariant
      that fails is about the set already admitted.* Soak §U PASS (`797d23e`).
- [x] **Seatable-lanes chunk divisor** (`812bc57`) — divide the prefill budget by the lanes the
      pass will actually seat, not by queue depth. The starvation signature (`#new-seq: 1`,
      `#new-token ≤ 512`, `#queue-req ≥ 8`) went 61 % stage / 19 % passthrough → **0 of 1,202
      passes**. Stage 492 req / 0 err / 0 STALLED, passthrough 1,904 / 0 / 0; p95 −25 % both routes,
      p99 −35 %/−44 %; effective prefill rate 1,830→2,310 tok/s; scheduling wall clock 99.8 %.
      `f6ed0b5`; soak §V.
- [x] **Client-disconnect abort during prefill** (`ff470e7`) — `server/disconnect.py`
      (`aiter_or_disconnect`, `await_or_disconnect`, 0.25 s poll) covers `/generate`, openai chat +
      completions, `/v1/messages`, `/v1/responses`, streaming and not. Exactly one AbortMsg per
      request; `FrontendManager.spawn_abort` keeps a strong reference.
      `tests/server/test_disconnect_abort.py` (12). No scheduler change needed.
- [x] **Observability** (`78f29d3`) — `/v1/stats.scheduler` (`null` until the engine publishes,
      deliberately distinct from all-zero) with prefill/spill/spec counters and the finishability
      invariant now **evaluated and counted on every pass**; `requests.aborts` tagged by reason at
      the frontend call site; `#seatable-lane` / `#chunked-inflight` on the batch log line;
      `analyze.py` reads `stats_*.json` deltas. `scheduler/counters.py`, 25 tests.
- [x] `session_spill`: `start_prefetch` reaping a finished predecessor now parks the unasked-for
      promotion in `_promoted` instead of dropping its id (`52a6503`).
- [x] `batch_memcpy` probe stream ordering (`13af13d`) — the probe zeroed `dst` on the ambient
      stream and copied on a private one with no `wait_stream`, so a busy caller could latch
      `OffloadMoeCache._batch_memcpy = False` process-wide and silently disable prefill hit-D2D.
      Fixed with `stream.wait_stream(current_stream())` + `dst.record_stream(stream)`, plus
      `test_batch_memcpy_probe_survives_busy_ambient_stream`. `tests/moe`: 161 passed, 5 skipped.

### Infrastructure
- [x] **CI for CPU-only checks** (`508ea32`) — `.github/workflows/cpu-checks.yml`: `ruff check`,
      the scheduler/server/kvcache/dsv4/Nemotron-H unit tests (1,239 passed / 51 skipped in 14.2 s),
      and `benchmarks/scheduler_replay.py --gate`. `docs/cpu-checks.md`.
- [x] **Soak drivers in the repo** at `benchmarks/switchyard_soak/` (`run.sh`, `serve.sh`,
      `sample.sh`, `split.py`, `analyze.py`, `gaps.py`; `runs/` gitignored). A WSL OOM restart at
      08:59 on 2026-09-05 destroyed `scratchpad/soak7/` and an in-flight soak with it. `run.sh`
      refuses to start below 26 GiB `MemAvailable`; `sample.sh` records it every 5 s.
- [x] Commits pushed as `fork/nemotron35`.

## 2026-09-05 — make `--speculative ngram` pay (verify-step cost, engagement, draft length)

Follow-up to `benchmarks/results/nemotron35_lightning_5080_ngram_spec_impl_2026-09-05.md`
tickets 1, 2, 3 and 6. Write-up:
`benchmarks/results/nemotron35_lightning_5080_ngram_spec_fast_2026-09-05.md`.

- [x] **Engagement decided post-drain.** `NgramDrafter.could_match` + an (n−1)-prefix hash set;
      `peek(stale=...)` in both scheduler loops. Strict superset of the exact test, so no burst
      entry can be missed; the exact test runs post-drain in `run_step`.
- [x] **Verify batch built from its own fixed shape.** `SpecNgramDecoder._prepare_verify`,
      persistent device buffers, metadata cached by (extend width, state slot), no
      `Sampler.prepare`. 0.80 → 0.34 ms/step. Falls back to `_prepare_batch` under
      `--kv-grow-step-tokens`.
- [x] **The 280-launch commit → one fused scan.** `SpecScanCapture._commit_fused` folds the layer
      axis onto the head axis (23 × 64 heads = one 1 472-head sequence). 7.12 → 0.45 ms host,
      **bit-exact** at eight (m, n) shapes (`benchmarks/check_spec_fused_commit.py`, weightless).
- [x] **Graph-captured verify forward — measured NO-GO.** m = 9: 30.6 ms host launch vs 36.4 ms
      GPU; 131K: 31.0 vs 91.8. The launch path already hides under the GPU.
- [x] **Draft-length / n sweep.** k ∈ {4,8,12,16} × n ∈ {6,8,10}, four classes. Plus a
      stream-independent fixed-transcript replay (`benchmarks/spec_engage_replay.py`).
- [x] **Instrumentation.** `SpecStats.cost_ms` (per-phase wall clock + CUDA-event GPU time),
      surfaced on `/v1/stats`.
- [x] Gates: greedy agreement (`off == off2` on 4/4 in every session), 131K needle answered
      identically in both arms, 30 CPU tests, full CPU suite green.

### Review

**Result.** A verify step is 54.0 → 35.6 ms (−34 %). On a fixed transcript with the measured
per-step costs, the copy class goes **1.11× → 1.61×** at the shipped `k = 8` and **1.88×** at
`k = 16`; code and prose are 0.99× at every setting; 131K still regresses (ratio ~12×).

**Two corrections to the previous write-up, both from measurement.**
1. Ticket 2's "burst entry costs a factor of ~4 in draft rate" is wrong — the real gap is 2 %.
   Its 0.079-vs-0.353 evidence was **stream variance**: speculation perturbs its own output and
   the copy prompt's reasoning preamble decides how much of the 1 024-token window is copy. Arms
   of the same binary span 1.04×–1.67×.
2. A single copy-class throughput arm cannot measure this feature. Three byte-identical repeats
   give identical drafter statistics and 1.8 % tok/s spread, so the engine is deterministic —
   the variance is in the comparison, not the measurement. Greedy acceptance is a deterministic
   function of the baseline transcript, so replay it.

**Not done, deliberately.** The default `--spec-draft-len` is left at 8: k = 16 is worth ~1.17×
on copy and neutral elsewhere, but it doubles the price of the break-even gate's two probe steps
at long context, and that trade wants its own confirming session.
