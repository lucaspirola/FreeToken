# FreeToken — Nemotron 3.5 Lightning (Switchyard) on RTX 5080

Handover: `tasks/nemotron35-handover.md`. Plan: `tasks/nemotron35-plan.md`. Rules:
`tasks/lessons.md`. Results files below are in `benchmarks/results/` unless stated;
`N35 =nemotron35_lightning_5080_`.

HEAD `e3a2019` (two uncommitted soak-harness files: `benchmarks/switchyard_soak/run.sh`,
`benchmarks/probe_disconnect_middleware.py`). The four 2026-09-05 tickets are committed in
`ca7e74b` and the three §W tickets in `e3a2019`; the end state is **soak-validated**
(`N35switchyard_soak_2026-09-04.md` §X: traffic PASS on both routes and **0 invariant warnings**
over 1,541 checks — §W's blocker closed). `fork/main` (`62f5a66`) is behind, 0 ahead.

---

## Open

### Perf, ranked by measured upside
- [ ] **n-gram verify overhead.** ~40 % of a verify step is not the forward (~52 ms vs a ~30 ms
      extend forward): 46 eager kernel launches in the commit, `_prepare_batch` rebuilding pinned
      staging for a one-request batch. 7x → 4x a decode step moves the copy class 1.03x → ~1.12x.
      Same ticket list: burst-entry hysteresis costs ~4x in draft rate (0.079 measured vs 0.353
      offline) — decidable in one run at latch 0/2/4; the soak tail (p95 +22 %, p99 +131 % on one
      10-minute pair — re-run at 20 min); batched (bs>1) verify; a graph-captured fixed-width
      verify forward; non-greedy speculation needs `Sampler.prepare` to repeat-interleave
      parameter rows by k; the drafter indexes the whole prompt on first engagement (~0.1–0.2 s at
      131K). **The 131K regression is this same ticket**: at the shipped k=8 the needle case
      measures **0.898x** of spec-off (k=16 0.870x), and the gate cannot refund its own two probe
      steps (163 ms at k=8, 279 ms at k=16, against a 10.4 ms decode step) — a cheaper long-context
      verify step is the only fix, not a draft-length or threshold setting.
      Evidence: `N35ngram_spec_impl_2026-09-05.md` §6; `N35misc_tickets_2026-09-05.md` §4.
- [ ] **Fold gemm2's deinterleave prepass into gemm1's store.** gemm2's A *is* gemm1's output, so
      its 182 MB prepass is removable by having gemm1's store emit the two k-planes directly —
      **~0.3 ms of the 0.551 ms** the prepass costs at M=8192.
      Evidence: `N35misc_tickets_2026-09-05.md` §2.
- [ ] **The extend-cache guard is conservative under LFU, and its JIT is a production hazard.**
      Two leftovers from the threshold study: (a) `use_cached_extend` excludes
      `_size_class_enabled` but not `cache_policy_id == 1`, and LFU takes the **in-repo** sized
      kernel (`offload_kernels.py:50`), which has no `BLOCK_K` width limit — so the new
      1,024-routed-id refusal costs the LFU profile widths it could actually serve (it costs
      nothing today, because the stream wins above the crossover anyway). (b) The m=128 cell cost
      **22 minutes of one-off Triton JIT** at `BLOCK_K = 1024`; if the threshold is ever raised,
      that compile has to happen at warmup, never on a live request.
      Evidence: `N35misc_tickets_2026-09-05.md` §3.
- [x] **DONE (uncommitted) — `--moe-collect-stats` publishes only at an idle boundary, so no
      soak can report an expert-cache hit rate.**
      Fix: `OffloadMoeCache.decode_stat_totals()` returns the same accumulators as raw
      cumulative INTS (a lifetime ratio cannot be differenced back into a window's hit
      rate), and `note_extend_gate()` counts every `use_cached_extend` decision at the
      `layers/moe.py:_prefill_routed` call site — two host ints, no device work, so they
      publish with or without the flag. `counters.build_moe_counters` renders both under
      `/v1/stats.scheduler.moe` (`extend_cache` always, `decode` only under
      `--moe-collect-stats`, since reading it costs a few `.item()` syncs), the scheduler
      passes `engine.moe_offload_cache` into `build_scheduler_counters`, and `analyze.py`
      prints both blocks. `run_when_idle`'s log lines are untouched. Tests:
      `tests/scheduler/test_scheduler_counters.py` (5 new),
      `tests/moe/test_extend_cache.py` (the gate counter survives `reset_stats`).
      Original ticket: Every `MoE decode miss stats` / `GPU batch profile` line is emitted
      from `Scheduler.run_when_idle` (`scheduler.py:346-408`), and `Scheduler is idle` appeared
      **0 times in 41 minutes** at c=16 — the flag was on for the whole `ca7e74b` soak and returned
      nothing. `decode_miss_stats()` is already a dict of ints; hang it off `/v1/stats` next to
      `scheduler.prefill`, the way `78f29d3` did for the admission counters. Add a counter to the
      **extend-cache gate** (`use_cached_extend`) in the same change — today it can only be
      inferred from `#new-token <= --moe-extend-cache-tokens` (76 of 1,522 passes, 5.0 %, vs 70 of
      1,210 at `13af13d`). soak §W7.
- [ ] **The M=256 GEMM bucket** runs at 20 % of ceiling with +53 % padding waste at `BLOCK_M=16`;
      `_ensure_experts_sized_kernel` evicts serially in a `(1,)` grid. `--moe-collect-stats` and
      the pageable-layer profile now also count sub-threshold extend routings.
      Evidence: `N35moe_prefill_gemm_2026-09-05.md` §10(e); `N35extend_moe_2026-09-05.md`.
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
- [x] **DONE (uncommitted) — 9 finishability-invariant warnings in the `ca7e74b` soak.**
      Root cause confirmed, and it is the hypothesis below: `Scheduler._restore_cold_session`
      is the one thing that spends pool pages BETWEEN two prefill passes — it runs from
      `_process_one_msg` (before `add_one_req`) and from `_reclaim_for_blocked_prefill`,
      neither an admission gate — and it ends with the restored prefix LOCKED. That takes
      those tokens out of `available_size` whether it allocated pages for them or merely
      re-protected a prefix the tree still held as evictable, while `owed` does not move.
      Reproduced in `benchmarks/scheduler_replay.py`: the new `switchyard-restore` profile
      models the missing half of the session cycle (a reclaim CHECKPOINTS before it
      unlocks; the session's next turn restores). Pre-fix it scores **43 violations at
      seed 7, short by 84,234 tokens** — one restore of a 127,204-token prefix of which
      121,865 tokens were still counted in `available_size`.
      Fix: `CacheManager.session_restore_footprint()` (what the restore takes out of
      `available_size`, verified equal to the measured drop) charged against
      `available_size - PrefillManager.finishability_reservation()` (the exact left-hand
      side of `_check_finishability`), and the restore is DEFERRED with its checkpoint
      intact when it does not fit — counted as `session_spill.restores_deferred`. Cannot
      deadlock: reuse is an optimization, so a deferred session re-prefills through the
      normal gated path, `_reclaim_for_blocked_prefill` retries after the next release, and
      the reservation it is charged against drains by a chunk per pass.
      Post-fix: 0 violations and `deadlock` False on all of seeds {1,3,5,7,11,13,17,23},
      seed 7 prefilled tokens +10.8% and error rate 0.3510 → 0.3391, with 12 restores and 2
      deferrals. `switchyard-restore` is now the 5th `--gate` case (floors ~5% under the
      measurement, plus `session_restores >= 8` so the profile cannot pass by doing
      nothing); the four pre-existing profiles are bit-identical. Tests:
      `tests/scheduler/test_prefill_finishability.py` (4 new). Docs: `docs/cpu-checks.md`.
      `FREETOKEN_SCHEDULER_INVARIANT=raise` is now safe to soak — but soak it before
      trusting that.
      Original ticket: 18:38:30–18:38:49 of the passthrough phase, 9 of 702 checks: two in-flight chunked
      prefills over-promise the pool by a **constant 1,401 tokens** (0.5 % of 262,144) while both
      `owed` and `available_size` fall by one 8,192-token chunk per pass. It resolved itself —
      queue drained 14 → 0, no error, no stall, no fatal, graceful shutdown in 2 s. Leading
      hypothesis: a **cold session restore** materialises committed pages *after* admission
      (four restores, one of 79,104 tokens, in the 2 s before the first warning), shrinking
      `cache_manager.available_size` without shrinking the standing reservation that
      `_check_finishability` compares it against (`prefill.py:503-546`). Not proven — §V had 441
      restores and 0 warnings. **Next step is CPU-only:** extend `benchmarks/scheduler_replay.py`
      with a restore landing between a chunked prefill's admission and its next chunk and assert
      the invariant; if it reproduces, charge the restore against the standing reservation (or
      re-check finishability after a restore) rather than loosening the invariant.
      **Do not run `FREETOKEN_SCHEDULER_INVARIANT=raise` in a soak until this is understood** —
      it would have killed an otherwise clean run. Evidence: `N35switchyard_soak_2026-09-04.md` §W6.
- [ ] **`max_chunked_prefills = 8` binds: `fresh_admits_blocked_by_cap` = 435** (27 stage / 408
      passthrough) over the `ca7e74b` soak, plus 500 `refusals`. §U5 could not prove the cap ever
      bound; `78f29d3` now proves it does. Goodput went *up* in the same run, so this is evidence
      for the §U8-ticket-9 reservation arithmetic, not a demonstrated cost. soak §W3/§W9.
- [x] **DONE (uncommitted) — the `client_disconnect` abort counter stays 0 through a probe
      that demonstrably aborted.**
      Root cause: `FrontendManager.abort_user` opens with a 0.1 s settling sleep and the
      NON-streaming endpoints (`openai_api.py:456/469/871`, `anthropic_api.py:165`,
      `responses_api.py:208` — and the §W5 probe was `"stream": false`) *await* it from
      inside their own `except asyncio.CancelledError` handler. Any cancellation of the
      request task during that window discards the coroutine before `stats.on_abort` and
      before the `AbortMsg` is sent, so the disconnect is invisible on `/v1/stats` **and**
      the request keeps its pending entry, table slot and forwarded KV — the leak the path
      exists to close. The streaming path was never exposed: `spawn_abort` runs it as its
      own task. Reproduced with the fake-client fixture (0 aborts, 0 AbortMsgs).
      Fix: `abort_user` now dispatches `_dispatch_abort` as a tracked task and awaits it
      through `asyncio.shield`, so the delivery completes even when the caller is
      cancelled. No call-site or ordering change. Tests: 5 new in
      `tests/server/test_disconnect_abort.py`, including the cancellation case and that
      `explicit` (prepare-stop drain) stays distinguishable from `client_disconnect`.
      Caveat worth one soak line: this is the only mechanism in the tree that produces a
      0 counter after an abort, but §W5 also saw `active` return to 0, which needs the
      AbortMsg to have been delivered — so re-run the probe and read
      `stats_after_probe.json` directly rather than a phase snapshot.
      Original ticket:
      `78f29d3` publishes `requests.aborts`, and the §W5 disconnect probe took `active` 0 → 1 → 0
      in 2 s while `client_disconnect` never left 0 — the counter exists but the disconnect path
      does not increment it. Half of §U8 ticket 12. soak §W5.
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
      (passthrough) at `13af13d`; **3.18 / 4.96 at `ca7e74b`**, stage moving *down* while requests
      rose 30 %, now that the divisor no longer caps it. Stage >~5 **together
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
- [x] **Non-elastic CUDA-graph ladder, dense to 16** — `_determine_cuda_graph_bs` built
      `[1,2,4] + range(8, max_bs+1, 8)`, so a 12-lane batch replayed the bs-16 graph with four
      dummy rows that route their own top-6 experts. It now unions `range(1, min(max_bs,16)+1)`
      **for offload-MoE models only** (`GraphRunner` passes `offload_moe=moe_offload_cache is not
      None`); dense models keep the historical list byte-for-byte, pinned by a test. Three
      alternating repeats per arm out of one binary at 12 lanes: **140.43 → 150.90 tok/s
      (1.074x)**, perfect separation, event-gap p50 83.0–87.0 → 77.2–79.3 ms; 11 extra graphs,
      ~80 MiB, ~0.8 s of startup. Hatch `FREETOKEN_GRAPH_DENSE_BS=0|1`; 8 new tests in
      `tests/engine/test_elastic_graph_sizes.py`. `N35misc_tickets_2026-09-05.md` §1.
- [x] **NVFP4 MoE prefill A-operand deinterleave — shipped ON by default.** Both `a_ptrs_lo/hi`
      were stride-2 on the contiguous axis; a `DEINTERLEAVED_A` constexpr arm plus a host prepass
      (`a.view(M, K//2, 2).permute(0,2,1)`) makes them unit-stride at an unchanged reduction order.
      **16.960 → 13.961 ms at M=8192 (1.215x), bit-exact (0.000e+00) at every M**, 70.3 TFLOP/s =
      59 % of `tl.dot`; residual gap to b12x **1.34x → 1.10x**. End to end at 131K:
      **6,124.7 → 6,577.8 tok/s (1.074x)**, engine average 5,728.6 → 6,177.6, **TTFT 21.6 → 19.8 s**,
      decode unchanged, needle PASS ×4. Hatch `FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A=0`; test
      `tests/moe/test_nvfp4_backends.py::test_deinterleaved_a_is_bit_identical_to_the_interleaved_kernel`.
      `N35misc_tickets_2026-09-05.md` §2.
- [x] **`--moe-extend-cache-tokens` stays 64, plus a crash guard.** New harness
      `benchmarks/bench_extend_moe_threshold.py` + `benchmarks/extend_moe/run_threshold.sh`
      (one model load, 7 timed extends per cell, fresh tail per call, arm proven per row).
      Wall ms stream/cached: 64 → 281.1/**249.4**, 80 → **285.3**/294.8, 96 → **284.0**/330.5,
      128 → **274.1**/370.3 — **crossover between 64 and 80**, i.e. the shipped default. At m=256
      the cached path does not merely lose, it **cannot execute**: flashlib's `lru_ensure` builds a
      `[BLOCK_K, BLOCK_K]` dedup block at `BLOCK_K = next_pow2(num_tokens*top_k)` and Triton caps a
      tensor at 1,048,576 elements, so `m ≤ 170` at top-6 and `--moe-extend-cache-tokens 256`
      killed the engine mid-forward. `use_cached_extend` now refuses above 1,024 routed ids and
      falls back to the stream; 3 new tests plus `test_every_copy_of_the_default_agrees` pinning
      the four hardcoded copies of the default. `N35misc_tickets_2026-09-05.md` §3.
- [x] **`--spec-draft-len 16` as the default — NO-GO, stays 8.** 131K non-copy measures **0.870x**
      of spec-off at k=16 (k=8 0.898x) against a ±2 % criterion and a 1 % control spread, and at
      k=16 the break-even gate **never closed** (`declined_uneconomic` 0 of 55 peeks, vs 16 at
      k=8): a longer draft raises `emit` about as fast as `verify_ms`. Short-context step cost
      35.8 ms at k=8 vs 49.8 at k=16. Pinned by `test_spec_draft_len_default_stays_8` and
      `test_the_gate_does_not_close_at_the_k16_operating_point`; copy-heavy traffic still passes
      16 explicitly. `N35misc_tickets_2026-09-05.md` §4.
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
at long context, and that trade wants its own confirming session. **That session ran on
2026-09-05 and 8 is now the pinned answer** — k = 16 is 0.870x of off at 131K and the gate stops
closing entirely (`N35misc_tickets_2026-09-05.md` §4).

---

## 2026-09-05 — final validation soak of the end state (`ca7e74b`)

- [x] **Run** — `SOAK_EXTRA_ARGS="--moe-collect-stats" benchmarks/switchyard_soak/run.sh ca7e74b 20m`;
      stage 20 m then passthrough 20 m at c=16, `FREETOKEN_SCHEDULER_INVARIANT=warn`,
      `--enable-cache-report`, server under `scripts/gpu_lock.sh`. 17:55:01 → 18:39:17,
      READY in 33 s, both phases `exit=0`, GPU back to **0 MiB**, no leftover venv processes.
- [x] **Grade** — `split.py`, `analyze.py` (logs **and** the four `/v1/stats` snapshots),
      `gaps.py`. Full write-up: `N35switchyard_soak_2026-09-04.md` **§W**.
- [x] **Disconnect probe** on the same server — `active` 0 → 1 → 0, back to 0 **2 s** after the
      socket close (§V measured 5 s, §U 7 s).

### Review

**Traffic: PASS on both routes, and every headline beats the §V (`13af13d`) baseline.**
Stage 492 → **639** requests (+29.9 %) at p95 109,395 → **72,094 ms** (−34.1 %) and p99 −22.8 %.
Passthrough 1,904 → **2,155** (+13.2 %) at p50 −13.3 % but p95 **+9.7 %** and p99 **+15.3 %** —
goodput bought with a slightly worse tail, not a slower engine: per-stream decode at 16 lanes is
11.91 → **13.58 tok/s** (+14 %) and the effective new-token prefill rate 2,008 → **2,410** (+20 %).
0 errors, 0 STALLED, 0 fatals, 0 tracebacks, 0 ERROR/CRITICAL lines, trailing silence 1 s / 3 s,
scheduling wall clock 99.8 % / 99.5 %, 0 spill or restore failures in 1,734 spills and 642 restores.

**The dense graph ladder is confirmed live and is the cleanest result of the run.** `13af13d`
captured `(1, 2, 3, 4, 8)` at every elastic tier and ran **314 of 427 decode batches (73.5 %)
eager**; `ca7e74b` captures `1..16` at the 16-request tier and ran **0 of 485 eager**.

**Verdict is a qualified PASS: 9 finishability-invariant warnings** in the last 20 s of the
passthrough phase break the stated "0 invariant violations" criterion. Nothing downstream went
wrong — the episode is a constant 1,401-token over-promise that resolved itself — but it is the
one open blocker and is ticketed above with a CPU-only repro to try.

**Two measurement lessons this run bought.**
1. **A counter that only publishes at an idle boundary does not exist on a busy server.**
   `--moe-collect-stats` was on for 41 minutes at c=16 and emitted nothing, because every one of
   its log lines comes out of `run_when_idle` and `Scheduler is idle` never fired. The soak
   therefore has **no expert-cache hit rate**, and no future soak will until the counters move to
   `/v1/stats`. Before asking a run to report a metric, check that the metric's publication path
   is reachable in that run's regime.
2. **A 14-sample bucket is not a measurement.** Stage `#running-req == 16` aggregate reads
   99.9 → 85.7 tok/s and would look like a 14 % regression; both sides are n=14 out of ~170
   decode batches. The `>= 12` bucket (n=79 → 114) has identical medians (86.2) and a *rising*
   mean. Report the sample size next to any bucketed soak number, or do not report the bucket.

---

## 2026-09-05 — validation soak of the three §W fixes (`e3a2019`)

- [x] **Run** — `SOAK_EXTRA_ARGS="--moe-collect-stats" benchmarks/switchyard_soak/run.sh e3a2019 20m`;
      stage 20 m then passthrough 20 m at c=16, `FREETOKEN_SCHEDULER_INVARIANT=warn`,
      `--enable-cache-report`, server under `scripts/gpu_lock.sh`. 19:22:52 → 20:07:41, READY in
      24 s, both phases `exit=0`, graceful shutdown in 4 s, GPU back to **0 MiB**, no leftovers.
- [x] **Grade** — `split.py`, `analyze.py` (logs **and** the three `/v1/stats` snapshots),
      `gaps.py`. Full write-up: `N35switchyard_soak_2026-09-04.md` **§X**.
- [x] **New check: `session_spill.restores_deferred`** — **0** over 568 restores (0 failed), with
      0 invariant violations. The charging alone held; the deferral arm stayed unexercised.
- [x] **New check: MoE counters from `/v1/stats.scheduler.moe`** — extend-cache gate **3.1 %**
      (1,104 hits / 34,224 misses of 35,328 routed extend layer-forwards). Decode expert-cache hit
      rate **still not soak-measurable** — see the new ticket.
- [x] **New check: disconnect probe on BOTH shapes**, asserted against `stats_after_probe.json`.
      **FAIL**: `client_disconnect` = **1**, required ≥ 2. `active` did return to 0. Diagnosed to
      root cause with a CPU-only repro; no fix applied (not a one-liner).

### Review

**PASS on the stated acceptance criteria.** 571 stage / 2,149 passthrough requests, 0 errors,
0 STALLED, 0 fatals, 0 tracebacks, 0 ERROR/CRITICAL lines, no `health_bad.log`, trailing silence
**1 s / 1 s**, scheduling wall clock 99.7 % / 99.9 %, 0 of 503 decode batches eager, 0 spill or
restore failures in 1,715 spills / 568 restores.

**§W6 is closed: 1,541 invariant checks, 0 violations, worst shortfall 0 tokens** — on the same
profile and the same route (passthrough) that produced §W's nine warnings, at unchanged
throughput: requests −0.3 %, p50 +0.4 %, p95 +2.0 %, p99 −0.1 %, decode aggregate median +0.5 %.

**Stage's −10.6 % requests / +13.9 % p95 is workload, not engine.** Prefix reuse 83.7 % → 79.8 %,
so the same 20 minutes carried 13 % more new prefill tokens (2,894 vs 2,566 tok/s effective) over
11 % fewer requests — **+26 % new tokens per request** — at 17 % higher instant prefill
throughput. Stage is the low-count high-variance route (the §V→§W swing on it was +30 % requests /
−34 % p95), and no marker, gap or pressure counter shows the §R6/§R7 mode.

**The one FAIL is a pre-existing defect the new check was the first to look for.**
`client_disconnect` counted the streaming probe and not the non-streaming one — and the reason is
not the counter. On a drained server a 60 K-token prefill finishes in ~6.5 s, so the old probe's
fixed 6 s sleep had been timing a *completion*, in §W too (§W5's "0 → 1 → 0 in 2 s" is that
artefact). With a probe that closes on `requests.active >= 1`, the non-streaming request still ran
to completion and answered 200 OK into a dead socket five seconds after the close. Root cause,
proven CPU-only in the new `benchmarks/probe_disconnect_middleware.py` (middleware off: seen in
2.01 s; on: never seen): `api_server.py`'s `@app.middleware("http")` request-ring recorder is a
Starlette `BaseHTTPMiddleware`, which owns the ASGI receive channel and never forwards
`http.disconnect`, so `disconnect.py`'s 0.25 s poll of `Request.is_disconnected()` reads False
forever and the handler that sends the AbortMsg is never entered. Streaming is immune because its
abort comes from the send side. `e3a2019`'s `asyncio.shield` is correct but unreachable there.
Ticketed as open item 0; no fix in this session (a pure-ASGI middleware plus a uvicorn-level test).

**A second measurement lesson.** `/v1/stats.scheduler.moe` publishes the decode expert-cache
counters now, but `OffloadCache`'s bank rebuild calls `lru_stats.zero_()` and this run rebuilt 30
times, so a snapshot only carries the traffic since the last rebuild: `layer_calls` read **115**
after a 20-minute phase and **2,576** after a 26-second probe. Moving a counter off the idle path
was necessary but not sufficient — **a cumulative counter that something else resets is still not
readable as a delta**. Ticketed as open item 0b.

**Host:** `MemAvailable` bottomed at **2.1 GiB** (§W 3.1, §V 5.1) over 495 samples, 0.1 GiB above
`run.sh`'s own abort watchdog. GPU 13.85 GiB median / 14.78 peak; top-process RSS peak 23.4 GiB;
30 elastic capacity changes (§W 50).
