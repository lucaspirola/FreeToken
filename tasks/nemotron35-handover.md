# Nemotron 3.5 Lightning on FreeToken — handover

State as of 2026-09-05 ~10:00, HEAD 13af13d on `main` (not pushed). Read this, then
`tasks/nemotron35-plan.md` (spec + decisions), `tasks/todo.md` (checklist), `tasks/lessons.md`
(rules), `docs/nemotron.md` (profiles), `docs/switchyard.md` (router contract + e2e harness).
Memory notes: host embedder service, host-OOM rules, plan pointer.

## One-line status
Model serves correctly on the RTX 5080 with the Triton Mamba-2 SSD kernels and NVFP4 experts;
Switchyard's text-based escalation contract is met (contract 12/12); session residency
(spill on demand, capacity/age retention, restart persistence, RAM prefetch, partial-prefix
restore) is implemented and unit-tested. **Items 1-4 are all closed as of 2026-09-05**, the
last of them being the 16-way Switchyard soak (item 2, PASS on both routes against `4a99e34`).
What is left is the ticket list (5-12) and the 1M llama.cpp oracle leg (item 12).

## Done (commits 80f2838..e50bc22, ~26 commits)
- Phase 1 bring-up, Phase 2 kernels (SSD prefill/decode, b12x relu2, Triton MoE + dense tuning,
  cache study → Triton default, LFU for 16-way), Phase 3A–3G (wire contract, JSON mode,
  sessions/parsers, soak harness, residency policy, prefetch, partial restore + prefill-time
  state capture), slot-reclaim crash fix (+ /health 503, bounded shutdown), MTP NO-GO.
- Results: benchmarks/results/nemotron35_lightning_5080_{,mamba2_,cache_study_,switchyard_}2026-09-04.md
- Numbers: 131K prefill ~3,000 tok/s, decode 63–73 tok/s single stream (**superseded
  2026-09-05: 145.3 tok/s after the decode launch-config fix**), 16-way aggregate
  ~168 tok/s with LFU; 131K needle passes (chat endpoint); spill 2.7–3 GiB/s, RAM restore
  5–8 GiB/s, NVMe restore ~1.3 GiB/s.

## Items 1-4 (all closed) — history and evidence; ONE GPU job at a time under scripts/gpu_lock.sh
1. **262K recall — CLOSED 2026-09-04, root cause fixed.** The Mamba-2 prefill scan floored
   the discretized timestep at `dt >= config.time_step_min` (1e-3). `time_step_min` is HF's
   *initializer* range for `dt_bias`, not a runtime bound; the floor caps every head's memory
   horizon at `1/(|A|*1e-3)` tokens. `dt_limit=(0.0, inf)` — vLLM's value, and what llama.cpp
   and FreeToken's own decode kernel always did — turns 147,456 and 262,144 @ depth 0.52 from
   FAIL to PASS at identical TTFT. Fix: `models/nemotron_h/config.py::_dt_floor`
   (`FREETOKEN_NEMOTRON_DT_MIN=<float>` restores a floor for A/B) + 3 tests in
   `tests/models/test_nemotron_h.py`. Write-ups:
   `benchmarks/results/nemotron35_lightning_5080_262k_{rootcause,crossengine}_2026-09-04.md`.
   The bisect's "model/quant limit" verdict and its "gate mid-depth needles at depth <=0.1"
   acceptance bar are **retracted**; retest the 262K/524K rows in the cache study and the 1M
   gate against the fix. The two perf tickets from the bisect are **CLOSED 2026-09-05**:
   `decode_launch_config` now sizes the decode grid to the GPU for untuned head shapes
   (`_grid_filling_splits`: 64 splits / BLOCK_N 64 / 8 warps here instead of 8/32/4), and the
   KV *load* path widens slot ids to int64 behind a compile-time `SLOT_I64` predicate. Result:
   single-stream decode 82.8 -> 145.3 tok/s at 131K, 58.7 -> 132.4 at 262K, 35.4 -> 113.6 at
   524K (paired arms of one A/B), and 95.8 at 1,040,016 against the ~20 recorded on
   2026-09-04; prefill unchanged. Write-up:
   `benchmarks/results/nemotron35_lightning_5080_decode_launch_2026-09-04.md`.
   Exonerated with evidence along the way: the FP8 W8A8 Mamba in/out projections (11 of 46
   saturate their calibrated `input_scale`, but by the same 1.8e-5 clipped fraction at the
   passing 131K and the failing 147K — see `FREETOKEN_DEBUG_FP8_ACT_STATS`) and the whole
   NVFP4 path (W4A16 end to end, no activation quantization anywhere).
2. **16-way Switchyard soak — RE-RUN 2026-09-05 against `13af13d`, PASS on both routes, and
   §R7 ticket 1 (the starvation signature) is CLOSED.** Write-up: soak results file
   §"Run against 13af13d" (§V). Tree: `812bc57` seatable-lanes chunk divisor + `32cc504`
   fork/main Ada merge + `52a6503` + `193da80` + `13af13d`.
   - **Stage 492 req / 0 err / 0 STALLED**, p50/p95/p99 **29,820 / 109,395 / 149,081 ms**;
     **passthrough 1,904 req / 0 err / 0 STALLED**, 6,888 / 24,580 / 46,695 ms. 2,396
     requests, zero failures, zero error records, 5/5 scenarios clean on both routes.
   - **The starvation signature is 0 / 1,202 prefill passes (0.0 %)** on both routes, against
     61 % (stage) and 19 % (passthrough) at `4a99e34`. Stage p95 **−25 %**, p99 **−35 %**;
     passthrough p95 −25 %, p99 −44 %; requests +4.7 % / +19 %. Stage median `#new-token`
     5,689 instead of §U's 512-token crawls; effective new-token prefill rate 1,830 → 2,310.
   - 0 invariant warnings, 0 `committed_pages_required`, 0 `LinearStatePool exhausted`,
     0 `Eviction did not free enough space`, 0 oversize skips, 0 tracebacks, 0 ERROR/CRITICAL,
     40/40 `/health` ok. Trailing silence 1 s / 2 s; **scheduling wall clock 99.8 % of both
     phases** (§U: 97.2 / 95.7). Stage 0 gaps ≥ 30 s; passthrough 1 (31 s) and it is a
     session-spill burst — 291 non-batch lines, `#queue-req 11` with `#running-req 3` — not a
     stall. `#mamba-slot: 96/96` reached on 60/761 and 32/866 batch lines, longest run
     **7 batches**, requests still completing inside, pool exhaustion never reached.
   - Mean lanes per prefill batch **1.83 → 3.43 (stage)** and **3.53 → 4.92 (passthrough)**.
     Passthrough is inside the 4.7–6.6 band the failing trees occupied — but those were
     **stage-route** numbers with 15/32 errors and p95 200.7 s; stage here is 3.43, below
     §R6's 4.71, with 0 errors. Lanes are now a free variable: watch mean stage lanes every
     soak and treat >~5 *with* moving errors/p95 as the §R6/§R7 mode returning.
   - Disconnect-abort re-verified: `/v1/stats.requests.active` 0 → 1 → **0 five seconds** after
     the socket close; 0 spurious aborts on 2,396 requests.
   - Host: busiest process median 101.0 % CPU, GPU 13.9 GiB median / 15.8 GiB peak, RSS peak
     23.9 GB, `MemAvailable` floor 5.1 GiB, 441 cold restores 0 failures, 309 idle expiries,
     **graceful shutdown 4 s, GPU 0 MiB**, no leftovers.
   - **Drivers are now tracked at `benchmarks/switchyard_soak/`** (`run.sh`, `serve.sh`,
     `sample.sh`, `split.py`, `analyze.py`, `gaps.py`); outputs go to `runs/<tag>/`
     (gitignored). They moved out of the session scratchpad because the 08:59 WSL OOM restart
     destroyed `scratchpad/soak7/` *and* an in-flight soak. `run.sh` now refuses to start
     below 26 GiB `MemAvailable` and `sample.sh` records it every 5 s.
   - **Caveat:** `--moe-prefill-hit-d2d` is OFF in the P2 serve profile, so this soak
     exercised no `cudaMemcpyBatchAsync` path; the `13af13d` probe fix is covered by
     `tests/moe/test_prefill_hit_d2d.py::test_batch_memcpy_probe_survives_busy_ambient_stream`,
     not by the soak.

2b. **Previous soak — PASS on both routes against `4a99e34`** (superseded by the above; kept
   for the history of what each tree bought).**
   Write-up: `benchmarks/results/nemotron35_lightning_5080_switchyard_soak_2026-09-04.md`
   §"Run against 4a99e34" (§U). Tree: `d685e99`'s gate restored **plus** `b030c7f`'s standing
   reservation (`PrefillManager._standing_reservation` seeded into `PrefillAdder.reserved_size`,
   charged to fresh admits only, kept out of `reserved_pages`), `max_chunked_prefills = 8`,
   and the `_check_finishability` invariant; `ff470e7` disconnect-abort; `acc91e9` decode
   launch config; `4a99e34` prefill `BLOCK_M` register cap. Server run under
   `scripts/gpu_lock.sh` with `FREETOKEN_SCHEDULER_INVARIANT=warn`.
   - **Stage 470 req / 0 err / 0 STALLED**, p50/p95/p99 **24,283 / 145,840 / 230,183 ms**;
     **passthrough 1,600 req / 0 err / 1 STALLED**, 7,527 / 32,906 / 83,354 ms. 2,070
     requests, zero failures, zero error records, 5/5 scenarios clean on both routes.
   - **0 finishability-invariant warnings** across ~3,141 prefill passes (`ea7ed7c` violated
     it on 566 stage passes). 0 `committed_pages_required`, 0 `LinearStatePool exhausted`,
     0 tracebacks, 0 oversize `can never be admitted`, **0 `Eviction did not free enough
     space`** (the `ea7ed7c` run had 16), 0 ERROR/CRITICAL. `/health` ok on all 40 checks.
   - **Trailing silence 1 s (0.1 %) on both phases** — the §T deadlock signature, now reported
     by `gaps.py` (leading + trailing silence against the driver's phase window, warn at
     ≥ 120 s). `ea7ed7c` had 2,616 s of it. Scheduling wall clock 97.2 % (stage) / 95.7 %
     (passthrough) of the phase; stage 0 gaps ≥ 30 s, passthrough 1 (54 s, and it is a
     session spill/restore burst — the batch that ends it restores 589,680 cached tokens
     across 6 lanes — not a scheduler stall).
   - **Throughput up on every axis** vs §R4/§R6: decode aggregate @ `#running-req == 16`
     **81.6 → 96.8 tok/s** (stage) and **161.4 → 177.5** (passthrough), per-stream 5.10 →
     6.05 and 10.09 → 11.09; prefill instant median **1,637 → 1,851** and **1,496 → 1,838**;
     effective new-token prefill rate 1,830 / 1,879 tok/s; prefix reuse 85.0 % / 88.8 %;
     stage p95 **−27 %** vs §R6. `acc91e9` and `4a99e34` are measurable here (the `ea7ed7c`
     run was starved 92 % of the wall clock and could not read them).
   - **Mean lanes per prefill batch 1.83** against §R6's 2.37 — the standing reservation makes
     an in-flight prefill keep costing admission until it finishes, so fewer fresh prompts are
     seated. The series is now 2.37 → 4.71 → 6.57 → **1.83** lanes with 0 → 15 → 32 → **0**
     errors: fewer lanes, zero errors, best latency of any run. More lanes is not the metric.
   - **`max_chunked_prefills = 8` shows no sign of binding, and that cannot be proven from a
     log**: the cap is a silent `continue` and there is no counter. Inference (§U5): a pass
     with `#cached-token > 0` necessarily admitted a *fresh* request, so `chunked_inflight < 8`
     then; 282/2,091 (stage) and 279/1,050 (passthrough) such passes, median **2 s** apart,
     worst window 73 s / 102 s, with no symptom in those windows. Ticket 12 below.
   - **Disconnect-abort (`ff470e7`) verified live**: a ~60 K-token non-streaming request was
     dropped on a raw socket mid-prefill; `/v1/stats.requests.active` went 0 → 1 → **0 seven
     seconds after the close**. No spurious aborts: 0 client failures on 2,070 requests, and
     3/3 invalid-request canaries still correct. The abort *count* during the soak is not
     measurable (`"Aborting request %d"` is debug-level, `StatsTracker` has no cumulative
     counter) — ticket 12.
   - Host: busiest process median 109.9 % / 107.0 % CPU (no spin), GPU 14.1 GiB median /
     15.8 GiB peak, RSS peak 24.2 GB, 423 cold restores with 0 failures, 396 session idle
     expiries, **graceful shutdown in 3 s, GPU 0 MiB**, no leftover workers.
   - **Still open after the pass** (§U8): §R7 ticket 1 is now the stage route's binding limit —
     `chunk_limit = token_budget // waiting` produced the starvation signature (`#new-seq: 1`,
     `#new-token ≤ 512`, `#queue-req ≥ 8`) on **1,278 of 2,091 stage passes (61 %)** and
     200/1,050 passthrough (19 %); that, not the scheduler gate, is why stage p95 is 146 s.
     Also 761 benign INFO `Discarded cold session …: client tokens diverge at 3` (soak6: 179),
     the message the item-3 `.valid` ticket would disambiguate.
   - History (kept): `fad1fc4` closed the `committed_pages_required` fatal but starved stage by
     charging the chunk cap in `reserved_size`; the §R6 `PrefillAdder.reserved_pages` fix
     restored it (471/0/1, 2.37 lanes). `81ab30e` (gate against the whole pool) regressed to
     268/15/7 and 720/16/10. `ea7ed7c` (gate against `admissible_size`) **deadlocked** — 14
     chunked prefills owning 1.76x the pool, last batch 5 m 35 s in, 2,616 s of silence.
     The bug family in all three: a budget checked only at admission is not a budget; the
     invariant that fails is about the *set* already admitted.
   - Drivers: `scratchpad/soak7/` (this run; `gaps.py` there and in `soak6/` now report
     trailing silence), `scratchpad/soak6/`, `soak5/`, `soak3/`, `soak2/` under
     /tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-.../scratchpad/.
   - Open tickets from these runs are 8-12 below.
3. **1M gate — CLOSED 2026-09-04, all four criteria PASS.** Write-up:
   `benchmarks/results/nemotron35_lightning_5080_1m_sessions_2026-09-04.md`. One session grown
   to **1,039,989 tokens** (8 turns × 130K, needle recalled at every length, twice); demand
   spill of the resident 1M session **3.53 GiB to NVMe in 2.980 s (1.18 GiB/s)**; a **new**
   `ft serve` process adopted the checkpoint (`adopted 1 checkpoint(s)`) and its next turn
   **restored 1,040,020/1,040,020 tokens from disk in 2.681 s (1.32 GiB/s)** — 9.8 s wall
   against the 1,861 s of prefill that built the prefix, with a byte-identical (correct)
   answer across the restart. Capacity/age eviction verified at a 1.6 GiB cap: the third spill
   evicted the older of two candidates by `last_used_at`, survivors still restored (0.255 s),
   and a record larger than the whole cap is refused rather than evicting the world.
   262K/524K needles re-run through `/v1/chat/completions` at depth 0.50 after the `dt` fix:
   **262,160 PASS** (1,925 tok/s prefill, 56.3 decode) and **524,304 PASS** (1,064, 34.5) —
   the cache study's "~131K–256K coherent ceiling" caveat is retracted.
   Notes/tickets from the run (§6 of the write-up): `_restore_cold_session` uses
   `session.spill` without checking `.valid`, so a capacity eviction is reported as
   "client tokens diverge" (one-line fix, not applied — scheduler.py is another agent's file);
   a *resident* session is never checkpointed, so a restart loses it (spill-on-shutdown flag
   would fix); `_evict_one_lru` can evict the record the pending admission is about to restore;
   `--session-spill-ram-gb 0` is what forces the NVMe tier for this test (the 4 GiB default
   keeps a 3.5 GiB checkpoint in RAM, where a restart destroys it). Drivers:
   scratchpad/1m2/{serve.sh,drive.py,trigger.py,hold1b.sh,hold4_lru.sh,hold3_evict.sh,
   hold2_needles.sh} under /tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-.../scratchpad/.
4. **Phase 3H hidden-state export — CLOSED 2026-09-04, parity PASS.** All 52 exported
   layers match transformers' own `NemotronHBlock` stack at cosine **>= 0.998840** on the
   mean-pooled residual (gate 0.99); median 0.999760. Write-up:
   `benchmarks/results/nemotron35_lightning_5080_hidden_states_parity_2026-09-04.md`.
   Two things had to be fixed in `benchmarks/probe_hidden_states_parity.py` first (only
   file changed, uncommitted): (a) its `AutoModelForCausalLM.from_pretrained` reference is
   impossible on this release — modelopt MIXED_PRECISION, which transformers 5.15 has no
   quantizer for, `backbone.*` vs `model.*` names, per-expert NVFP4 tensors vs a fused 3-D
   parameter (400 missing / 18 486 unexpected keys), and 58.8 GiB dense bf16 against a
   34 GiB host. It now builds the model on `meta` and streams one block at a time
   (dequant on the sibling scales, ~3.5 GiB VRAM, ~10-22 s for the whole forward), with
   the per-block forward hook recording `residual + mixer` directly. (b) `--capture-only`
   / `--artifact <path>` split the run into two phases (server up, then server stopped),
   since the served model and the reference cannot be resident together.
   **The reference needs `--reference-dt-min 0.0` (the default).** transformers hard-codes
   the same 1e-3 `dt` floor that item 1 identified as a bug; leaving it in fails 12
   shallow layers (worst 0.9406 at layer 3) — an independent confirmation of item 1.
4b. **1M multi-needle recall — 2026-09-04, 5/8.** One 1,039,994-token prefill (TTFT
   **1,815 s**, 573 tok/s whole-prompt), then eight questions on the same chat prefix, each
   hitting the prefix cache for 99.9954 %+ of its prompt (TTFT 4.7–7.0 s, decode 19.2–20.3
   tok/s). Write-up:
   `benchmarks/results/nemotron35_lightning_5080_1m_multineedle_2026-09-04.md`; harness:
   **`benchmarks/bench_multi_needle.py`** (new, untracked). Depths 0.05 / 0.75 / 0.95 recall,
   0.25 / 0.50 / 0.60 do not; the control (a key absent from the text) is correctly denied
   with no fabrication. **The headline is question 8**: asked which of the depth-0.05 and
   depth-0.25 codes is larger and for their sum, the model returned
   9,854,500 = 5,663,623 + 4,190,877 — so the depth-0.25 needle it had just "missed" when
   asked directly *is* in the state. Grade long-context recall with more than one question
   shape per needle before blaming retention.

5. Ticket: `--kv-grow-step-tokens` + `--nvfp4-backend flashinfer` crashes (VMM int32 bank).
6. Ticket: `_maybe_shrink_growable_kv` evicts all unlocked prefixes before checking whether a
   shrink is possible (wipes the prefix cache at idle above the initial KV step).
7. Ticket: tests/moe/test_prefill_hit_d2d.py order-dependent flake.
8. Ticket: **an over-pool prompt has no client rejection path.** `PrefillManager.
   schedule_next_batch` skips a fresh request whose `input_len + output_len >
   cache_manager.max_size`, logs one `... can never be admitted and is being skipped` warning
   and `continue`s — but never removes it from `pending_list` and never fails the request, so
   the client hangs until its own timeout with no error. It also keeps inflating `waiting =
   len(pending_list) - index`, shrinking every other lane's interleave chunk. Worse,
   `_seatable_lanes` does **not** have the same skip: it sets `blocked_fresh = True` on the
   first request whose cost exceeds the budget, so one permanently unadmittable prompt pins
   the seatable-lane estimate at the number of continuations for as long as it sits in the
   queue. Fix: fail it with a 400/413 at admission (or at `add_one_req`), and mirror the skip
   in `_seatable_lanes`.
9. Ticket: **`stopped_for_lane_cap` rotation is dead code on this model.**
   `stopped_for_lane_cap` is only assigned inside `if lane_cap and len(reqs) >= lane_cap`, and
   `lane_cap = max_batch_seqs = _resolve_max_prefill_seqs(config)` is **0** for Nemotron
   (it returns 1 only when the checkpoint has `gguf_expert_types`) — confirmed live by
   `py-spy dump --locals` during the `81ab30e` soak (`lane_cap: 0`). So the interleaved branch
   `self.pending_list = remaining + chunked_list` is unreachable exactly on the profile that
   turns interleaving on. Its own comment describes a different trigger ("admission stopped on
   a resource-constrained request"), which is `blocked_fresh` / the `refusals` break, not the
   lane cap. Decide which one it means and set the flag there, or delete the branch.
10. Ticket: a refused prefill pass costs `O(queue x prompt)` radix walks — 16 pending
   118 K-token prompts are re-`match_prefix`ed from scratch on every pass that returns `None`
   (4 of 5 py-spy samples during the stall were inside `fast_compare_key`). Cache the match
   per pending request until its prompt or the tree changes, or skip the walk when the pass
   has already refused a fresh admit.
12. Ticket (from the passing soak, §U5/§U6): **the two admission bounds and the abort path
    are unobservable.** Neither the standing reservation's refusals nor `max_chunked_prefills`
    leaves a trace, so "the cap never bound" can only be inferred; and
    `Scheduler._process_one_msg` logs `"Aborting request %d"` at *debug* while `StatsTracker`
    keeps only a live `_aborting` set, so disconnect-aborts cannot be counted from a soak.
    Add `chunked_prefills_inflight`, `fresh_admits_blocked_by_cap`, `deferred_prefill_chunks`
    and a cumulative `aborted` to `/v1/stats`.
11. Ticket: `benchmarks/scheduler_replay.py` (the CPU replay gate added in `508ea32`) scored
   `81ab30e` at "2.49x tokens / 2.14x completions" — the commit that then failed the live soak
   on both routes. It models neither retained session leases, nor decode residency, nor the
   idle timeout, which is where the stall comes from. Either extend it or stop treating it as
   an acceptance gate for scheduler policy.

12. **Cross-engine oracle — first live sweep done 2026-09-05, 262K: NO ENGINE BUG.**
    Write-up: `benchmarks/results/nemotron35_lightning_5080_oracle_2026-09-05.md`; generated
    report and recordings in `~/ai/bench/oracle/2026-09-05/`. FreeToken **19/24** graded turns
    vs llama.cpp **17/24** on a byte-identical 262,076-token prompt: 15 `agree`, 3
    `both-miss`, 4 `llamacpp-only-miss`, **2 `freetoken-only-miss`** — and both of those are
    composition failures on turns where retrieval demonstrably succeeded (one adds two
    correctly-retrieved codes **off by one**; the other gets the sum exactly right and names
    the wrong key as larger). **Zero `retention`, zero `selection`, 12/12 needles `in state`
    on both engines.** The one mid-depth direct miss (quarry, depth 0.500) is made by BOTH
    engines, both returning the near-duplicate `quarry register` twin — a model interference
    limit, not an engine or NVFP4 effect. llama.cpp shows three `interference-near` classes to
    FreeToken's one. Cost: 7 min of GPU for both engines at 262K (one prefill, 19 cached
    turns at 99.978 % cached).
    **`acc91e9` confirmed in a real workload**: 105.5 tok/s median decode at a 262 K context
    (56.3 on 2026-09-04) and **2.7x llama.cpp** on the same prompt/card; prefill 1,949 vs
    1,740 tok/s, TTFT 134.5 vs 150.6 s.
    **1M FreeToken leg (no llama.cpp leg — optional and not attempted): 7/19 turns, and
    ZERO `retention`.** 1,044,416-token haystack, TTFT 1,813.8 s at 576 tok/s (prefill
    unchanged), turns 2-19 at 4.61 s TTFT / **80.7 tok/s decode** (~20 on 2026-09-04 —
    `acc91e9` confirmed at 1M). Direct probes collapse (1/6; the depth-0.03 `quarry register`
    code 1,607,392 acts as an attractor) but **leak-free reverse `code -> key` probes recover
    5 of 6 needles exactly**, so the state holds them and *addressing* is what fails:
    5 `interference-near` + 1 `recall-partial`, in-state 5/6. A single-question-shape gate
    would have filed six retention bugs; the correct count is zero. Composition is the weak
    axis (every combined sum wrong at 1M vs 4/6 right at 262K). The control is still denied.
    **Procedure bug found and fixed in the doc**: `--target-prompt-tokens 1048576` against
    `--num-tokens 1048576` fails at *turn 2* with `context_length_exceeded` (1,048,623 >
    1,048,576) after paying the full 1,818 s prefill — the suite is a conversation and grows.
    Use `--target-prompt-tokens 1044480` at the top rung; the failed recording is kept as
    `ft_1048576_ctxoverflow.json`. Still open at 1M: run the llama.cpp leg alongside it — the
    prediction to test is that it shows the same interference collapse (it already showed more
    of it than FreeToken at 262K).
    Doc bug: `docs/oracle.md`'s Phase-A serve line names `--session-spill-dir
    /mnt/nvme/ft-spill`, which does not exist on this host — the drivers use
    `~/.cache/freetoken/oracle-spill`. Drivers: `scratchpad/oracle/{serve_ft.sh,phaseA.sh,
    phaseB.sh}`.

Scratchpad root (survives Claude restarts, not WSL restarts):
/tmp/claude-1000/-home-lucas-ai-FreeToken/af23ede4-e8ad-4c8d-8b38-c8be515d8870/scratchpad/

## Host rules (all learned the hard way)
- `systemctl --user stop piro-board-embedder.service` before GPU work (Restart=always; holds 4–10 GB).
- Host RAM is the constraint (34 GB WSL): ANY job that loads the checkpoint runs under
  scripts/gpu_lock.sh (refuses < 22 GiB available, 4 h cap, oom_score_adj 1000, reaps workers).
  Claude/tmux are protected by the root timer protect-terminal-oom.timer (oom_score_adj −900).
- Max two subagents at once; never one that ends its turn while its GPU job continues.
- Never `git stash`/`commit -a`; implementers use worktrees; kill workers by venv path.
- FREETOKEN_PIN_BUDGET_GB=17 pins all expert banks (no --moe-pageable-gpu); ratio 0.85;
  8K chunks; q8_0 KV + Triton attention; `--nvfp4-backend` auto→triton; LFU for 16-way.
- Needle gate goes through /v1/chat/completions, digit-free filler; never grade raw SSE.

## Two stale worktrees to ignore/remove
.claude/worktrees/agent-a45f827ae98e76526 and agent-a4ce5e26d2ccafdb6 predate this effort.
