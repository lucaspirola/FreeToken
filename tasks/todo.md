# FreeToken — Nemotron 3.5 Lightning (Switchyard) on RTX 5080

Handover: `tasks/nemotron35-handover.md`. Plan: `tasks/nemotron35-plan.md`. Rules:
`tasks/lessons.md`. Results files below are in `benchmarks/results/` unless stated;
`N35 =nemotron35_lightning_5080_`.

HEAD `8e064d4`, **working tree clean** — everything below is committed, and `fork/main` is
fast-forwarded (at `9afd48c`, 0 ahead). Baseline: `N35switchyard_soak_2026-09-04.md` **§AB**
(`670969f`) — PASS on every criterion, both routes, **0 violations of 1,452 invariant checks**,
**0 `Exception in ASGI application` for either request shape**, 0 of 459 decode batches eager.
**Every item below is closed with a result and a commit, or carries a written `Deferred:` reason
why it does not affect production.**


---

## Open

### Perf, ranked by measured upside
- [ ] **n-gram verify at long context — the residual.** The launch-overhead half is **closed by
      `b84ecb7`** (one fused SSD scan for 23 layers, verify batch built from its own fixed shape):
      step 54.0 → 35.6 ms, non-forward now ~0.5 ms of 35.6; "burst-entry hysteresis costs ~4x in
      draft rate" was **stream variance**, real gap 2 %. What is left at 131K (**0.898x** at k=8)
      is two things, neither of them the launch path: (a) the extend attention reads the whole KV
      history once per query token, so verify/decode is ~9–12x against a `k+1 = 9` ceiling — a
      fused multi-query extend kernel is the shape of that fix; (b) the gate cannot refund its own
      probe, and **seeding only the cost side is measured and NEGATIVE**. `FREETOKEN_SPEC_GATE_SEED`
      fits `verify_ms(m) = a + b·m` from two narrow probes to 0.1 % of both measured operating
      points, primes the gate — and still runs a full-width verify step, because `emit` is
      untouched at its `max_k + 1 = 9` prior: **226 ms spent pricing the gate against the shipped
      arm's 182 ms**. Ships default off, kept in the tree, 8 CPU tests.
      Next step is **both sides or neither, on the replay, not the GPU**: prime `emit` from the
      probes' own acceptance (widen `_SEED_WIDTHS` to e.g. `(3, 6)` so at least one probe usually
      *rejects* — under greedy decoding a rejecting probe is a true full-width sample), or compare
      against the `max_k + 1` ceiling while `emit` is still a prior (a real policy change). Extend
      `benchmarks/spec_engage_replay.py` with a **gate-policy axis** first: 2–3 verify steps on a
      79-token generation cannot resolve a 4 % effect against an 18 % baseline spread, and the
      tok/s column of that run is not attributable (the arms emitted 96 vs 70 tokens and plain
      decode itself moved 18 % between arms).
      *Deferred:* speculation is **off by default** (`--speculative ngram` is opt-in, 1.01–1.03x),
      so no production path executes the gate at all.
      Evidence: `ngram_spec_gate_seed_2026-09-05.md` §2/§3/§4;
      `N35ngram_spec_fast_2026-09-05.md` §2/§4/§5/§8; `N35misc_tickets_2026-09-05.md` §4.
- [x] **DONE (`9dc283e`) — fold gemm2's deinterleave prepass into gemm1's store.** gemm1 now emits
      gemm2's two k-planes itself via a permuted **B/scale gather** (`PLANAR_OUT`: tile column `d`
      takes weight row `(d % (N//2))*2 + d//(N//2)`), so the C store stays contiguous and the
      permutation lands on the axis that was strided anyway. **1.018x at M=8192 (1.237x over the
      pre-`ca7e74b` kernel), bit-exact (0.000e+00) at every M**, `%tl.dot` 59.3 → 60.4. The removed
      prepass is 0.445 ms and the arm recovers 0.250 of it (56 %) — the permuted row set costs
      ~0.2 ms in L2/TLB locality, and is still a strict win at every M. On by default;
      `FREETOKEN_NVFP4_PREFILL_FUSED_PLANES=0` is the hatch, and the flag is inert unless the
      deinterleave is on **and** the activation is epilogue-fused (a gated activation reads gemm1's
      output row-wise and would be silently corrupted). Test:
      `test_fused_k_planes_are_bit_identical_to_the_gemm2_prepass`. Neutral-to-positive live in
      soak §AA3. Evidence: `N35moe_prefill_leftovers_2026-09-05.md` §1.
      **Not done:** the end-to-end A/B (at 1.018x on a term that is ~14 % of a long prefill the
      expected move is ~0.3 %, below `bench_long_context.py`'s spread — a paired A/B or nothing).
- [ ] **The extend-cache guard is conservative under LFU, and its JIT is a production hazard.**
      `use_cached_extend` refuses above 1,024 routed ids because flashlib's `lru_ensure` builds a
      `[BLOCK_K, BLOCK_K]` dedup block and Triton caps a tensor at 1,048,576 elements — but the
      serving profile runs **LFU**, and `offload_kernels.py:50` routes `cache_policy_id == 1` to
      the **in-repo** sized kernel, whose compile-time widths are fixed by the model and the cache
      size, not by the query. So the guard is a flashlib-LRU constraint applied to a policy that
      does not have it. The correct shape, if ever wanted, is `_MAX_ENSURE_QUERY` as a property of
      the *policy* — and only after a measurement says the cached path wins somewhere above 170
      tokens, which today it does not (1.25x GPU-time **loss** at m=128). If the threshold is ever
      raised, the last width that compiles under LRU costs **22 minutes of one-off Triton JIT**,
      which must be a warmup pass before the server takes traffic, never a live request (a
      22-minute stall inside a forward is indistinguishable from a hang to the finishability
      invariant). **Documented, no code changed.**
      *Deferred:* raising it buys access to a path measured **slower** at every width above the
      m=64 crossover, so nothing in production is refused work it could serve faster.
      Evidence: `N35moe_prefill_leftovers_2026-09-05.md` §3; `N35misc_tickets_2026-09-05.md` §3.
- [x] **DONE (`e3a2019`) — `--moe-collect-stats` publishes only at an idle boundary, so no
      soak can report an expert-cache hit rate.** Half of this was still broken and is closed by
      `38617a7`: `OffloadCache`'s bank rebuild called `lru_stats.zero_()`, so a snapshot only
      carried the traffic since the last rebuild (`layer_calls` read **115** after a 20-minute
      phase and **2,576** after a 26-second probe). `decode_stat_totals` now folds the windowed
      counters into a host base before every `reset_stats` — a lifetime accumulator, not a
      `rebuild_epoch`. Validated in soaks §Y6/§Z6/§AA6: monotone across 20–26 elastic capacity
      changes, and the decode expert-cache hit rate is finally a soak measurement —
      **53.0 / 52.9 / 53.3 %** in three consecutive runs (lifetime 50.8 %), extend-cache gate
      5.6 / 6.5 / 5.7 %.
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
- [x] **DONE (`670969f`) — M=512 was served by the wrong MoE-prefill bucket.** `PREFILL_M_BUCKETS`
      gains **512 at `BLOCK_M=32`** — the "256" entry with *nothing else changed*, which is exactly
      the pair the microbench measured (1.10x / 1.106x on two independent routings, 1.429 → 1.296
      ms; the 1024 bucket's wider tile was never measured at this M, and borrowing a launch
      constant swept on another geometry is the 2026-09-05 lesson). Nearest bucket with **ties to
      the smaller**, so `384 < M ≤ 768` uses it (`640 < M ≤ 768` moves *down* from 1024).
      `FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS=512` restores the pre-2026-09-06 table exactly — the
      bucket-boundary hatch, which `FREETOKEN_NVFP4_PREFILL_BLOCK_M` cannot express (it would drag
      4096/8192 off their tuned tile too). `tests/moe/test_nvfp4_triton_tuning.py`.
      Live in soak §AB: **stage clearly positive** — instant prefill median 3,194 tok/s (+17.5 %,
      best since §X), effective new-token rate 2,581 (+7.1 %), p95 −6.8 %, p99 −14.0 %, at *lower*
      prefix reuse over more requests. Passthrough confounded — see the new §AB9.1 note below.
      The retired half of the original ticket: "M=256 runs at 20 % of ceiling" measured against a
      `tl.dot` ceiling that does not bind. At ~12 routed rows per expert both GEMMs must read every
      expert bank once = 718.5 MB = **0.748 ms at 960 GB/s** against 1.078 measured — **69.4 % of
      the HBM roofline**, headroom ≤1.44x, and `BLOCK_M=32` there is a **12 % loss**. The 256
      bucket is correct as tuned. **Still unswept:** `BLOCK_KB` / `num_stages` at the small-M
      bucket (`--grid smallm`, 216 tiles). *Deferred:* microbench-only, and the 512 result shows
      the tile axis was the one that mattered. Evidence: `N35moe_prefill_leftovers_2026-09-05.md`
      §2; soak §AB3/§AB9.1.
- [ ] **The 512 bucket's aggregate prefill rate (new, soak §AB9.1).** §AB pushed 6.17 M new tokens
      in 2,694 s = **2,290 tok/s** against §AA's 2,431 (**−5.8 %**) and §Z's 2,357 (−2.8 %) — while
      the route that pushed the most new tokens per second got *faster*. The passthrough phase
      carried 19.8 % fewer new tokens at higher reuse (89.8 % vs 87.9) with a median chunk
      `#new-token` of 2,454 against §AA's 4,486, which depresses a per-pass instant rate
      mechanically; the other candidate fingerprint is the stage extend-cache gate at 1.5 % vs
      §AA's 4.7 % (a routing-mix observation, not a fault).
      *Deferred:* every acceptance criterion passed on both routes, nothing shows the signature of
      a slower kernel, and the revert is one env var with no rebuild. Settle it with a repeat
      **passthrough** phase under `FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS=512` when a GPU slot is
      free.
- [ ] **16-way decode is at the hardware ceiling — do not re-litigate.** 74 % of the step is the
      PCIe expert gather at 51–52 GB/s against a measured 52.9 GB/s link; working set ~1,417
      expert-layer slots against 976 in the pool. Attention (64 splits, 80 % roofline), the MoE
      GEMV and Mamba-2 were all measured fine at batch 16. Only two levers remain:
      (a) `--moe-backend hybrid` at 16 lanes, never measured — the auto-threshold compares
      standalone CPU BW vs standalone PCIe BW (1.26 here) when the right criterion is
      `(cpu_ov + pcie_ov) / pcie_alone` = 1.73; (b) quantify what the 64-slot Mamba-2 snapshot
      cache (~2.4 GiB of expert slots) buys in prefix reuse.
      *Deferred:* the ceiling is the hardware; neither remaining lever is a defect.
      Evidence: `N35decode16_2026-09-05.md` §0/§2/§7.2–7.3.

### Scheduler / server tickets
- [x] **DONE (`e3a2019`) — 9 finishability-invariant warnings in the `ca7e74b` soak.** Soak-clean
      ever since: 0 violations over 1,541 (§X), 1,868,852 (§Y), 1,486 (§Z) and 1,551 (§AA) checks.
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
- [ ] **`max_chunked_prefills = 8` still binds.** `fresh_admits_blocked_by_cap` per run: 388 (§X)
      → 204 (§Y) → 301 (§Z) → 376 (§AA) → **185** (§AB), the lowest of the four post-livelock
      soaks. *Deferred:* goodput rose in every one of those runs, and §AB is both the lowest count
      and the best stage prefill rate — evidence for the §U8-ticket-9 reservation arithmetic, not a
      demonstrated cost. soak §W3/§W9, §AA9.3, §AB9.4.
- [x] **DONE (`e3a2019` + `38617a7`) — the `client_disconnect` abort counter stays 0 through a
      probe that demonstrably aborted.** The shield below was correct but **unreachable**: the
      request-ring recorder was a Starlette `BaseHTTPMiddleware`, which proxies the ASGI receive
      channel through its own task and never forwards `http.disconnect`, so `disconnect.py`'s
      0.25 s poll of `Request.is_disconnected()` read False forever and a non-streaming request
      whose socket closed ran to completion and answered 200 OK into a dead socket. `38617a7`
      rewrites it as a **pure-ASGI middleware** that passes `receive` through untouched
      (`server/request_ring.py`), with `tests/server/test_disconnect_middleware_asgi.py` driving a
      real uvicorn and the CPU A/B in `benchmarks/probe_disconnect_middleware.py` (middleware off:
      seen in 2.01 s; on: never). **Validated live in soak §Y5**: the probe's non-streaming arm
      moved the counter on its own, plus **11 real `client_disconnect` aborts in stage traffic**
      with 11 matching abort log lines, against 0 in §W/§X.
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
- [x] **DONE (`125da19`) — the 576 s admission livelock (soak §Y5b), formerly filed as "an
      over-pool prompt has no client rejection path".** Validated by soak §Z: **refusals
      1,867,771 → 190** on stage (0.6 passes/s against §Y's 3,240/s), seatable-lane histogram
      unpinned, **0 gaps ≥ 30 s** (first soak with none, largest 13 s / 18 s), 704 stage requests
      at p95 70.9 s — the best stage phase of the effort. `fresh_admits_deferred` 317 / 107.
      Root cause: the admission loop was FIFO with a single stopping rule. A fresh prompt the
      pools refused made the pass `break`, so every request behind it went unexamined — with
      1 running request draining to 0, 15 queued, `token usage 0.59` and 108 K tokens of the
      pool free. Nothing was running, so no page came back, so every subsequent pass took the
      identical decision; `normal_loop`'s `blocking`/`_only_idle_sessions` test requires an
      EMPTY prefill queue, which a queue nobody can admit never is, so the loop never took its
      10 ms nap: 1,867,771 refusals in 576 s (~3,240/s) on one core at 102 %. It ended only
      when the clients' 600 s timeouts fired. The seatable-lane histogram pinned at bucket 2
      is `would_seat` (which does not `lock()` the matched prefix) disagreeing with
      `_try_allocate_one` (which locks, and re-checks the gate against the smaller budget).
      Fix (three lines of policy):
      (a) a refused FRESH admit is SKIPPED, not stopped on, while `PrefillAdder.headroom` > 0
          — the same shape as the existing `max_chunked_prefills` skip, and safe for the same
          reason: `reserved_size` carries every claim already made, so a lane admitted after
          the skip cannot re-sell the refused prompt's pages (this is the term `ea7ed7c`'s
          continue-past-refusals lacked). Counted as `fresh_admits_deferred` on `/v1/stats`.
      (b) `headroom` (token budget, table slots, pool minus reservations) stops the walk when
          nothing of any size can be seated, so a genuinely full pool does not radix-match the
          whole queue.
      (c) `Scheduler._only_idle_sessions` also returns True when the last pass scheduled
          nothing (`_admission_stalled`), so a refused pass takes the 10 ms nap instead of
          spinning. Cleared by `_process_last_data` (a drain changes the state).
      **NOT done, and it turns out not to be reachable: the client-rejection path.** The
      original ticket's `max_size` skip no longer exists in the tree, and an over-pool prompt
      cannot occur: `engine.py:546` sets `self.max_seq_len = min(config.max_seq_len,
      num_tokens)`, and `_process_one_msg` already rejects `input_len >= max_seq_len` with
      `ErrorReplyMsg(code="context_length_exceeded")` and clamps `max_tokens` to
      `max_seq_len - input_len` — so `input_len + output_len <= max_seq_len <= pool tokens`
      by construction. §Y5b's head was refused by the pool's *current* protected/live
      commitment, not by the pool's size, so rejecting it would have been wrong; deferral is
      the correct treatment and is what shipped. `_seatable_lanes` was deliberately NOT given
      the same skip: `would_seat` is optimistic and its optimism is only bounded by stopping
      at the first refusal, so mirroring inflated the chunk-share divisor and cost 20 % of
      prefill throughput on all five replay profiles (util 0.96 -> 0.79). Under-counting the
      divisor is the safe direction. Tests: `tests/scheduler/test_admission_livelock.py`
      (5 new, 4 of them fail on `efa37da`) + 2 in `test_scheduler_counters.py`.
- [ ] **`stopped_for_lane_cap` rotation is dead code on this model.** *(Deferred: unreachable on
      this profile, so it cannot misbehave — delete or wire it when a model makes `lane_cap` > 0.)*
      `lane_cap =
      _resolve_max_prefill_seqs(config)` is 0 for Nemotron (confirmed live with `py-spy dump
      --locals`), so the interleaved `pending_list = remaining + chunked_list` branch is
      unreachable exactly on the profile that turns interleaving on. Its comment describes
      `blocked_fresh` / the refusals break. Set the flag there or delete the branch.
- [~] **PARTLY DONE (`8429411`) — a refused prefill pass costs O(queue × prompt) radix walks.**
      Live in soak §AA3: **~101 K matched tokens per pass over 8,360 walks, 40.8 % memo hit rate**
      run-wide (43.1 % stage / 38.8 % passthrough) at no measurable scheduler CPU (busiest process
      107.9 % median, no sustained-100 % window).
      Shipped: a **strict per-pass memo** of `CacheManager.match_req`, owned by
      `PrefillManager.match_memo`, shared by the seat scan, the admission loop and the
      post-refusal reclaim, so each queued prompt is walked **at most once per scheduler
      iteration** instead of up to three times. Validity is enumerated rather than assumed
      (`PrefillAdder._match`): the only thing that can falsify an entry is a node LEAVING the
      tree, `lock`/`unlock`/slot-allocation/page-table writes cannot, so the sole in-pass
      invalidation is `reserve_mamba_slots` — and only when it is about to escalate past the
      free list into `evict_mamba`/the lease spill. Outside a pass the memo is dropped by a
      released lease and cleared at the top of every pass. This is NOT `ea7ed7c`'s memo: that
      one lived ACROSS passes.
      New counters `scheduler.prefill.match.{calls,tokens,memo_hits,tokens_per_pass}` on
      `/v1/stats` and in `analyze.py`, so a soak can be read against the replay's
      `match_tokens_per_prefill_pass`. **Measured** (replay, seed 7, 20,000 ticks, 9dc283e ->
      this tree):

      | profile | tokens/pass before | after | delta | match_calls |
      |---|---|---|---|---|
      | stage | 213,006 | 174,144 | **-18.2 %** | 2,545 -> 2,124 |
      | pressure | 444,522 | 403,736 | **-9.2 %** | 3,758 -> 3,419 |
      | switchyard-stage | 518,410 | 437,907 | **-15.5 %** | 5,265 -> 4,725 |
      | switchyard-deadlock | 453,395 | 412,720 | **-9.0 %** | 7,339 -> 6,697 |
      | switchyard-restore | 563,156 | 477,331 | **-15.2 %** | 5,455 -> 4,897 |

      Every outcome metric is byte-identical on all five profiles — `prefilled_tokens`,
      `completed`, `error_rate`, `stall_frac`, `prefill_batches`, `empty_prefill_passes`,
      `session_restores`, `invariant_violations` — which is the evidence that the memo moved
      no scheduling decision, only the cost of reaching it.
      **Residual, deliberately not taken:** the walk still scales with queue × prompt WITHIN a
      pass, because a pass must ask each queued request whether it fits. The remaining win is
      a memo that survives a *run of identical refused passes*, and the scheduler already has
      the exact predicate for "nothing changed since the last pass" — `_admission_stalled`
      (125da19), which is true precisely when no batch was scheduled, none drained and no
      message arrived. Keeping the memo across passes gated on it would be sound, but it means
      enumerating every mutation that can reach the manager from outside a pass, and a missed
      one is exactly `ea7ed7c`'s stale `cached_len`. Not worth it before a soak measures what
      the per-iteration memo already saved.
- [x] **DONE (`8429411`) — `_reclaim_soft_sessions_for_pending` measured a budget the admission
      gate does not feel, so it froze nothing at exactly the moment it should.** Validated by soak
      §AA: admission-pressure releases **507 stage (+15 %) / 667 passthrough (+18 %)** against §Z,
      and the 4-deep scan took `fresh_admits_deferred` on stage **317 → 1,151** while `refusals`
      stayed flat (190 → 163). It over-spills slightly — see the new watch ticket below.
      It called it pressure only when `needed > cm.available_size` — the *pre-lock* budget —
      while `_try_allocate_one` locks the matched prefix first and re-checks the gate against
      what is left. A turn reusing a large evictable prefix therefore looked comfortable here
      and was refused there. Soak §Y5b: **zero** `Released soft session ... KV protection
      (admission pressure)` lines across the whole 576 s window.
      Fix: new `CacheManager.lock_delta(handle)` — a pure `node..root` walk summing `length`
      over `ref_count == 0` nodes, which is exactly what `lock()` moves out of `evictable` in
      all three tree flavours, computed without touching a ref count. The pressure test is now
      `needed > available_size - lock_delta`. A continuation passes `cached_len` and is charged
      no delta (it locked its prefix in the pass that admitted it) and costs no match.
      `_reclaim_for_blocked_prefill` also stops after `pending_list[0]`: it now scans up to
      `_RECLAIM_SCAN_DEPTH = 4` and returns on the first release, because since 125da19 a pass
      admits PAST a prompt it cannot seat, so the blocked request is often behind the head.
      Bounded because each fresh entry costs a radix walk — one the per-pass memo has usually
      already paid for.
      **Deliberately NOT added:** the gate also charges
      `PrefillManager.finishability_reservation()`. Including it would make the test exact and
      would also make the message-path caller (`_reclaim_soft_sessions_for_admission`, which
      runs on every arriving request, refused or not) spill idle conversations more eagerly
      than any measurement asks for. The lock delta is the term §Y5b's evidence names; the
      reservation term waits for a run that shows it costing something.
      Behavioural A/B on the §Y shape (same pool, same request, `needed` 255, pre-lock budget
      824, `cached_len` 700): before -> `released=False`, nothing spilled; after ->
      `824 - 700 = 124 < 255`, the lease is released and the pool goes back to 1,024. Note the
      replay does NOT cover this: `scheduler_replay.py` re-implements the loop's reclaim
      inline rather than calling `Scheduler._reclaim_for_blocked_prefill`, so the gate is
      blind to it and `tests/scheduler/test_reclaim_and_match_memo.py` is its only coverage.

- [x] **DONE (`670969f`) — the streaming disconnect path raised `CancelledError` out of the ASGI
      app.** `aiter_or_disconnect` now raises `disconnect.ClientGone` (not a bare
      `CancelledError`) when the poll finds the client gone, and
      `FrontendManager.stream_with_cancellation` **returns** instead of re-raising once
      `spawn_abort` is in flight: a StreamingResponse has no response object left to hand back, so
      ending the generator is the streaming twin of the non-stream endpoints'
      `return client_gone_response()` (`125da19`). A genuine outer cancellation (shutdown, uvicorn
      tearing the cycle down) is still a plain `CancelledError` and still propagates.
      Validated **by absence** in soak §AB: **0 `Exception in ASGI application` in the whole run**
      with the criterion now covering the streaming shape, and the probe's two arms both taking
      `active` 0 → 1 → 0 in 2 s and each incrementing `client_disconnect` on its own.
      `tests/server/test_disconnect_abort.py`.

- [x] **DONE (`8429411`) — `_maybe_shrink_growable_kv` evicted the whole prefix cache
      before computing whether a shrink was possible**, so above the initial KV step every
      idle moment wiped it and often shrank nothing (`server.gen1.log` 09:44–09:47; soak
      §Y5b's timeline opens with two such lines 4 s apart, immediately before 576 s spent
      re-prefilling).
      The small targeted fix exists and is exact. `page_usage()` already returns
      `committed - free - evictable`, which is precisely what `evict_all_unlocked_prefixes()`
      would leave occupied, so the best target the rest of the function could reach is
      computable *before* the eviction — and `compact_active_pages` returns
      `max(target_pages, required)`, so compaction can only raise that target, never beat it.
      When `max(initial, ceil(used_pages / step) * step) >= committed_pages` the eviction was
      always going to be pure loss, and the function now returns (at debug level) with the
      prefix cache intact. Provably never skips a shrink that would have happened.
- [ ] **`benchmarks/scheduler_replay.py` is not an acceptance gate for scheduler policy.** It
      scored `81ab30e` at 2.49x tokens / 2.14x completions — the commit that then failed the live
      soak on both routes. It still models no spill/cold-restore cost for a reclaimed lease, no
      non-reclaimable leases, no GDN state slots, and charges per-pass CPU to `match_calls` rather
      than the clock. **And it has a blind spot with a name now:** it re-implements the loop's
      reclaim *inline* instead of calling `Scheduler._reclaim_for_blocked_prefill`, so the whole
      `8429411` reclaim change is invisible to the gate and
      `tests/scheduler/test_reclaim_and_match_memo.py` is its only coverage. It did earn one thing:
      the `stall_frac` column added in `125da19` reproduces §Y5b (0.68 → 0 on `switchyard-stage`,
      0.72 → 0 on `switchyard-restore`) — the livelock had been sitting in `switchyard-stage`'s
      metrics dict all along with no column printing it. Extend it or demote it.
      *Deferred:* `tests/scheduler/test_reclaim_and_match_memo.py` plus the live soak cover what
      the gate cannot see, so no change ships on the gate's word alone.
- [ ] **Session residency leftovers** *(Deferred: spill-on-demand is by design and documented in
      `docs/nemotron.md`; a resident session losing its state across a restart is the stated
      contract, not a defect.)* (from the 1M gate, §6 of `N351m_sessions_2026-09-04.md`): a
      *resident* session is never checkpointed, so a restart loses it (spill-on-shutdown flag);
      `_evict_one_lru` can evict the record the pending admission is about to restore.
- [ ] **Watch mean lanes per prefill batch every soak.** Stage 3.13 (§X) / 3.55 (§Y) / 3.79 (§Z)
      / 3.04 (§AA) / **3.43** (§AB); passthrough 4.88 / 4.53 / 4.43 / 4.62 / **4.09**. No trend,
      all far under ~5, 0 errors and 0 STALLED on every one, starvation signature 0.0 % on both
      routes in §AB. *Deferred:* a standing watch, not a ticket. Stage >~5 **together with** rising
      errors or p95 is the §R6/§R7 failure mode returning. Passthrough sitting in the old 4.7–6.6
      band is not a regression — that band was a *stage-route* measurement.
- [ ] **Reclaim over-spill watch (soak §AA9.1) — stable in §AB, not worsening.** The `8429411`
      reclaim arm spends some prefix cache to buy admission throughput. §AA → §AB: pressure
      releases 1,174 → **1,189** (+1.3 %), spills 1,542 → **1,475** (−4.3 %), restores 479 →
      **421** (−12.1 %), `restores_deferred` 3 → 3, 0 failures; passthrough prefix reuse
      **recovered to 89.8 %** (above §Z's 88.9), stage drifted 1.1 pp further down (83.6 → 82.5) on
      a phase deliberately carrying more new tokens.
      *Deferred:* watch-only — two soaks show the cost flat and the arm paying for itself, with 0
      errors, 0 STALLED and 0 restore failures on both. `_RECLAIM_SCAN_DEPTH = 4` is the knob if a
      later soak ever shows reuse falling **with restores climbing**.
- [ ] **The reclaim pressure test still does not charge `finishability_reservation` (new).**
      Deliberately left out of `8429411`: including it would make the test exact, but the
      message-path caller `_reclaim_soft_sessions_for_admission` runs on **every** arriving request
      and would spill idle conversations more eagerly than any measurement asks for. The lock delta
      is the term §Y5b's evidence names; this term waits for a run that shows it costing something,
      and §AA9.1's over-spill is the counter-evidence to weigh against it.
      *Deferred:* no observed cost — two soaks with the lock-delta term alone show 0 invariant
      violations, 0 restore failures and the over-spill flat.
- [ ] **A cross-pass match memo (new).** The per-pass memo is measured; the walk still scales with
      queue × prompt *within* a pass. The remaining win is a memo that survives a run of identical
      refused passes, and `_admission_stalled` (`125da19`) is exactly the predicate for "nothing
      changed since the last pass". Sound, but it means enumerating every mutation that can reach
      the manager from outside a pass — a missed one is exactly `ea7ed7c`'s stale `cached_len`.
      *Deferred:* the cost it would remove is not observable — busiest-process CPU is flat at
      107.9 / 108.9 / 107.9 / **108.8 %** median across four soaks with no sustained-100 % window,
      and `PrefillAdder.headroom` already bounds the walk.
- [ ] **Serving-phase RAM floor 2.3 GiB (new, soak §AB9.2).** The worst since §Y, 0.3 GiB above
      the `SOAK_RAM_ABORT_GIB=2` watchdog, which did not fire (median 6.6 GiB over the run).
      *Deferred:* a soak-harness margin, not a server defect — the watchdog exists for exactly this
      and the run finished with a graceful shutdown. Set `SOAK_HOST_RAM_RESERVE_GB=7` on the next
      soak and record it as a profile deviation.
- [ ] **The load-phase RAM watchdog is unproven under a cold load (new, soak §AB9.3).** §AB's load
      was warm (`READY after 29 s`, load-phase minimum 8.0 GiB over 6 samples, cleared to 11.4),
      so the new floor was armed and reported but never approached. §AA's cold start (85 s,
      1.1 GiB minimum) is the reproduction.
      *Deferred:* it can only fail **safe** — the worst case is an abort that would otherwise have
      been a WSL OOM restart. The next cold-cache start is its first real test.
- [ ] `num_kv_splits_ptr` is passed to both decode kernels and dereferenced in neither (the split
      count is a constexpr). Delete it or use it. *(Deferred: dead argument, no behaviour.)*

### Recall / oracle
- [x] **DONE (`5d59c05`) — the 131K rung on both engines.** FreeToken **22/24** vs llama.cpp
      **23/24** on a byte-identical 131,042-token prompt (sha256 `8d4f380ac9179e60`): **direct 6/6
      each, reverse 6/6 each, control pass, 0 both-miss, 0 `retention`, 0 `selection`,
      0 `interference-*`**, all six needles in state. Every miss on either engine is *arithmetic on
      a combined question* — FreeToken's two are off by one (`5663623 + 4190877 = 9854499` for an
      expected 9854500), llama.cpp's one names the wrong larger code. 131K therefore bounds the
      262K → 524K `key → code` collapse from below. Cost: turn-1 TTFT **20.0 s at 6,559 tok/s**
      (3.7x llama.cpp's 1,763), whole rung 5 min GPU against a ~15 min budget. `--n-cpu-moe 14` is
      correct at this length; no doc change. `N35oracle_2026-09-05.md` §14.
- [x] **DONE (`5d59c05`) — the 524K `direct:harbour` lead: the re-prefill explanation is REFUTED.**
      Re-run on a *rotated* haystack (`--filler-cursor 65`, sha256 `9e82fd972d04de7a`, cursor not a
      multiple of 64 so no checkpoint could match): turn-2 TTFT **2.62 s** against the cursor-0
      run's 50.0 s, and it **still misses** — to a *different* wrong code (`interference-cross` →
      `interference-near`). "The 50 s TTFT was not the cause." The whole matrix reproduces exactly
      (agree 8/8, both-miss 9/9, freetoken-only-miss 2/2), FreeToken 8/19 vs llama.cpp 10/19.
      Handover item 6 is **closed**. `N35oracle_2026-09-05.md` §15/§16.
- [ ] **Re-word, do not re-open:** what remains at 524K is a single interference-class probe where
      FreeToken is weaker than llama.cpp under a quantization confound (NVFP4 vs Q4_0) that cannot
      be lifted on this card. Not worth GPU time on its own; worth re-checking the day a GGUF
      loader or an NVFP4 llama.cpp path removes the confound. *Deferred:* a model/quantization
      property, not an engine defect; 131K and 524K both reproduce it on two haystacks.
- [ ] Unexplained, does not touch any verdict: llama.cpp's decode median was **6.1 tok/s** at
      cursor 65 against **23.6 tok/s** in the cursor-0 run at the same `--n-cpu-moe 23`; prefill
      chunk cost, prefix-cache hits and recall all matched. `N35oracle_2026-09-05.md` §15a.
      *Deferred:* a llama.cpp-side observation on the reference engine, not a FreeToken path.
- [ ] Optional: the llama.cpp leg alongside the 1M FreeToken leg is **impossible on this card**
      (~20 h of prefill against a 4 h lock cap) — recorded as a host limit, not a gap to close.
      *Deferred:* a host limit; the 1M FreeToken leg already ran.
- [ ] `docs/oracle.md` correction: the Phase-A line's `ft serve` needs an absolute `.venv/bin/ft`
      inside a wrapper script — `gpu_lock.sh` does not run through the venv shim. *(Deferred: a
      benchmark-runbook nit; the workaround is in the handover's "How to run things".)*

### Process / repo
- [x] **DONE — `fork/main` fast-forward (user decision).** `fork/main` is at `9afd48c`, 0 ahead of
      HEAD's line; `fork/nemotron35` already carried the merge.
- [x] **DONE (`9afd48c`) — a stale `_gguf` build is now loud instead of silent.** The fork/main
      merge (`32cc504`) changed the shared MMVQ binding's `multiwarp` bool to an int64 `warps`,
      and pybind11 converts int → bool silently, so a pre-merge cached `.so` would run the wrong
      kernel width **with no error**. The loader reads the bound signature, refuses a stale build,
      and names the cache directory to delete.
      *Deferred (the rebuild itself):* not applicable to this card — the Nemotron NVFP4 path does
      not touch the GGUF kernels, and the silent failure mode is now impossible. It is a deployment
      step for an Ada box, not an open defect.
- [x] **DONE — CI follow-up (a).** The full CPU test step (`tests/scheduler tests/server
      tests/kvcache tests/dsv4 tests/models`) with **no GPU job live: 118 s wall** (`ee2e7bf`); the
      old 14.2 s figure was measured under different conditions.
- [~] **CI follow-up (b) — DEFERRED with a reason, not a blocker.** Folding `tests/moe` +
      `tests/kernels` into CI is **infeasible**: the GitHub runner is CPU-only ubuntu and those
      suites want a GPU. They are run locally instead and pass on the tree (`22478ef`).
      *Deferred:* no production path is uncovered — the suites run, just not in CI.
- [~] **CI follow-up (c) — PARTLY DONE (`22478ef`).** F401, F841, E402, E712, E714, E742, E701 and
      F541 are now enforced (45 violations fixed); the `[tool.ruff.lint] ignore` list is down to
      **E702 / E731 / E741**. E741 (×79) is the big remaining one and is mostly `l` as a loop
      variable in kernel code. *Deferred:* style only.
- [ ] `bench_nvfp4_moe_kernels.py --gate` still asserts the 2B1 targets ("b12x >= 2x triton at
      M=8/16"), which are now inverted and fail on a healthy tree. *Deferred:* a benchmark gate,
      not a shipped path.
- [x] **DONE (`8878659`, `7261fa8`) — replay real Switchyard traces instead of synthetic soaks.**
      `--trace-dir` captures one JSON line per completed request (no prompt text — a prefix hash
      chain), `benchmarks/trace_replay.py` replays it and `trace_to_profile.py` turns it into a
      `scheduler_replay.py` profile; `benchmarks/trace_load_report.py` prints the real load's
      arrival rate, concurrency, token distributions, prefix reuse and latency quantiles beside the
      soak's with a verdict naming which soak assumptions hold. `docs/switchyard.md` §9/§11.
- [x] **DONE — pre-existing test issues (`ee2e7bf`, `0f6ff4b`).** `pythonpath = ["."]` in
      `pyproject.toml` so `tests/server/test_muse_glimmer_parsers.py` collects its sibling module.
      `tests/models/test_laguna_modules.py`'s 6 errors in a full run were **not** GPU contention:
      the TP fixture asserted TP info was unset, which a suite-ordering predecessor had already
      set; the fixture now tolerates it. Both verified with no GPU job live.
- [ ] Remove or ignore two stale worktrees that predate this effort:
      `.claude/worktrees/agent-a45f827ae98e76526`, `.claude/worktrees/agent-a4ce5e26d2ccafdb6`.
      *(Deferred: housekeeping outside the repo's tracked tree.)*

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

---

## 2026-09-06 — three soaks, the livelock, and the ticket-7 leftovers

Runs, in order. Full write-ups in `N35switchyard_soak_2026-09-04.md` §Y/§Z/§AA,
`ngram_spec_gate_seed_2026-09-05.md`, `N35moe_prefill_leftovers_2026-09-05.md`,
`N35oracle_2026-09-05.md` §14–16.

- [x] **§Y — soak vs `38617a7`: both fixes VALIDATED, run is a FAIL.** Non-streaming disconnect
      abort real for the first time (11 in stage traffic + the probe's own arm); MoE decode
      counters monotone across 16 capacity changes, 53.0 % hit rate. Stage **FAIL**: 447 req,
      2.24 % errors, 5 STALLED, p99 600 s — a **576 s admission livelock** (handover ticket 7),
      1,867,771 refusals at ~3,240/s on one core at 102 %, seatable-lane histogram pinned at
      bucket 2 for 1.70 M of 1.87 M passes, ended only by the clients' 600 s timeouts. 0 invariant
      violations across all 1.87 M checks. Also: 10 ASGI tracebacks (`CancelledError` out of the
      non-streaming endpoint), structurally impossible before `38617a7`.
- [x] **`125da19` — livelock fix.** Defer refused fresh admits past the head while
      `PrefillAdder.headroom > 0`; `headroom` stops the walk when nothing of any size can be
      seated; nap on a pass that admits nothing; quiet 499 (`ClientGone`) on non-stream disconnect.
      Replay gains a `stall_frac` column that reproduces §Y5b. `_seatable_lanes` deliberately NOT
      mirrored (it cost 20 % of prefill throughput on all five profiles).
      `tests/scheduler/test_admission_livelock.py`.
- [x] **`785a278` — spec gate seed, measured NEGATIVE, default off.** Handover item 1 rewritten to
      the live residual; the launch-overhead half was already closed by `b84ecb7`.
- [x] **§Z — soak vs `785a278`: PASS both routes.** Refusals 1.87 M → 190 stage, 0 gaps ≥ 30 s
      (first soak with none), 704 stage req at p95 70.9 s (best stage phase of the effort), 2,016
      passthrough, 0 errors / 0 STALLED / 0 ASGI tracebacks, 0 violations of 1,486 checks.
      MemAvailable floor 4.0 GiB.
- [x] **`9dc283e` — MoE fused k-planes, default on**, bit-exact 1.018x at M=8192 (item 2a). Item 2b
      measured: `BLOCK_M=32` loses 12 % at M=256 (the bucket is at 69 % of the **HBM** roofline,
      not 20 % of a dot ceiling) and wins 1.10x at M=512, which the nearest-bucket rule serves from
      the 256 table — new ticket. Item 2c documented, no code.
- [x] **`5d59c05` — oracle §14–16.** 131K rung clean on both engines (22/24 vs 23/24, direct 6/6
      each); the 524K harbour re-prefill hypothesis refuted at `--filler-cursor 65` (2.62 s TTFT,
      still misses, different wrong code). Handover item 6 closed.
- [x] **`ee2e7bf` + `0f6ff4b` — handover item 10 closed**, and the full CPU step timed at 118 s
      with no GPU job live.
- [x] **`8429411` — ticket-7 leftovers.** Reclaim pressure test charges `CacheManager.lock_delta`
      and scans `_RECLAIM_SCAN_DEPTH = 4` past the head; strict per-pass radix match memo with
      `scheduler.prefill.match.*` on `/v1/stats`; idle growable-KV shrink skipped when it cannot
      shrink. Replay: `match_tokens_per_prefill_pass` −9…−18 % on all five profiles with
      byte-identical outcomes. `tests/scheduler/test_reclaim_and_match_memo.py`.
- [x] **§AA — soak vs `8429411`: PASS both routes, the current baseline.** 608 stage / 2,138
      passthrough, 0 errors / 0 STALLED / 0 fatals / 0 tracebacks, 0 violations of 1,551 checks,
      0 of 489 decode batches eager, shutdown 3 s. Best passthrough tail of the effort (p95 26.0 s,
      p99 38.8 s) at +12.2 % effective prefill rate. Reclaim arm fires (+15 %/+18 % admission-
      pressure releases, `fresh_admits_deferred` 317 → 1,151 on stage); memo 43.1 %/38.8 % hits at
      ~101 K matched tokens per pass; prefix reuse −2.4 pp / −1.0 pp. Two new tickets: §AA9.1
      reclaim over-spill watch, §AA9.2 the model-load RAM window is unguarded (host hit **1.1 GiB**
      during the load, 0.9 GiB below the watchdog floor that only arms after `READY`).
- [x] **`22478ef` — ruff ignore list down to E702/E731/E741** (45 violations fixed, item 9 partly).
- [x] **`9afd48c` — `_gguf` stale-build guard.** pybind11 converts int → bool silently, so a
      pre-`32cc504` cached `.so` ran the wrong MMVQ kernel width with no error; the loader now
      reads the bound signature, refuses the stale build and names the cache directory to delete.
      Closes the Ada-rebuild item as a *silent* failure.
- [x] **`7261fa8` — production observability.** `benchmarks/ops/stats_sampler.py`
      (`sample` / `summarize`, stdlib-only, window-not-lifetime rates, restart detection) + a
      systemd user unit that runs on `/usr/bin/python3` and records an outage rather than exiting;
      `benchmarks/trace_load_report.py` comparing a real `--trace-dir` capture against a soak run
      and printing the `trace_to_profile.py` command that closes the loop back to the replay;
      `docs/switchyard.md` §11. `tests/benchmarks/test_ops_observability.py`.
- [x] **`670969f` — handover items 2, 3 and 7 in one commit.** (2) load-phase RAM watchdog
      `SOAK_RAM_LOAD_ABORT_GIB` (default 0.8) armed on the READY poll + `SOAK_HOST_RAM_RESERVE_GB`
      on `serve.sh`; (3) the MoE prefill **512 bucket** at `BLOCK_M=32` with
      `FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS=512` as the revert; (7) **streaming** client disconnect
      ends the stream quietly via `ClientGone`.
- [x] **§AB — soak vs `670969f`: PASS on every criterion, both routes.** 635 stage / 1,977
      passthrough, 0 errors / 0 STALLED / 0 fatals, 0 violations of 1,452 checks, **0 `Exception in
      ASGI application` for either request shape**, 0 of 459 decode batches eager, largest gap
      23 s / 32 s, shutdown 4 s. Stage prefill +7.1 % effective / +17.5 % instant, p95 −6.8 %,
      p99 −14.0 %. Load watchdog armed, reported and cleared at an 8.0 GiB minimum (warm load).
      §AA9.1 stable, not worsening. Three new notes: §AB9.1 (aggregate prefill −5.8 % on a lighter
      passthrough workload), §AB9.2 (serving RAM floor 2.3 GiB), §AB9.3 (load watchdog unproven on
      a cold load).

### Review

**Five soaks in one session, and each one earned the next.** §Y validated two fixes and failed on
a defect neither of them caused; §Z closed that defect and produced the best stage phase of the
effort; §AA validated the scheduler leftovers and produced the best passthrough tail; §AB validated
the last three tickets and passed every criterion with the ASGI check now covering both request
shapes. Counters that did not exist before `38617a7` are now reproducible across runs: decode
expert-cache hit rate 53.0 / 52.9 / 53.3 / **52.6 %** (0.7 pp), radix memo hit rate 40.8 / **40.9 %**
(0.1 pp), busiest-process CPU 107.9 / 108.9 / 107.9 / **108.8 %** median.

**Nothing is open as a blocker.** Every remaining line in this file is either closed with a result
and a commit, or carries a written `Deferred:` reason why it does not affect production — the
reclaim over-spill watch, the 512-bucket aggregate (§AB9.1, one env var reverts it), the soak's own
RAM floor (§AB9.2) and cold-load watchdog (§AB9.3), both-sides gate seeding (speculation is off by
default), the cross-pass memo (CPU flat across four soaks), the `finishability_reservation` term
(no observed cost), `blocked_by_cap` (goodput rising, lowest in §AB), the hardware ceiling, and the
Ada rebuild (now a loud refusal, and not on this card's path).

---

## Switchyard "hidden" session requests (2026-09-06, owner GO)
Requested by the Switchyard hidden-state routing session via cross-session message; owner
gave standing GO for agent-requested code adaptation. Server on :1919 restarted 07:15 with
`--hidden-states-dir /home/lucas/.cache/freetoken/hidden-states --hidden-states-max-tokens 4096`.
- [x] (a) Inline pooled hidden states (6b18742 + cf377e4; live parity cos 1.000000 all layers, 13k-token chunked prefill OK): `kv_transfer_params.pooling` = mean|last|both, arbitrary
      sorted layer_ids subset, no file, no max-tokens cap, O(layers x hidden) accumulation across
      prefill chunks; response `kv_transfer_params.pooled` {layer_ids, hidden, prompt_tokens,
      dtype float32, mean/last base64 row-major}. Unit tests + parity vs file artifact + chunked.
- [x] (b) First-decode-step top-k logprobs (3928b4f; live-verified): `logprobs: true, top_logprobs: k (1..20)` →
      `choices[0].logprobs.content` with exactly one entry. Docs: later steps not populated;
      with thinking on the first token is the first reasoning token.
- [x] Restarted :1919 at cf377e4 2026-09-06 12:09 (after a host OOM at 11:51 took the previous instance; see lessons); live parity done.
- [ ] Later, gated on results: (c) pooled as first stream event after prefill; (d) launch
      checklist + serve.sh gain the two hidden-states flags.
