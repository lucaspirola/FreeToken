# Nemotron 3.5 Lightning on FreeToken — handover

State as of 2026-09-06, HEAD `22478ef` on `main`, **working tree clean**. Everything below is
committed. The two 2026-09-05 blockers (non-streaming disconnect never detected, MoE decode
counters zeroed by every bank rebuild) are fixed in `38617a7` and **validated live** by soak §Y;
that same run exposed a pre-existing **576 s stage admission livelock**, fixed in `125da19` and
validated by soak §Z (refusals 1.87 M → 190, 0 gaps ≥ 30 s, the best stage phase of the effort).
The scheduler leftovers of ticket 7 landed in `8429411` and are validated by soak §AA, which is
**the current baseline**. `fork/main` (`62f5a66`) is behind and 0 ahead — a fast-forward is
available and is the user's call. Read this, then `tasks/nemotron35-plan.md` (spec),
`tasks/todo.md` (open checklist), `tasks/lessons.md` (rules — read before touching the GPU),
`docs/nemotron.md` (profiles + numbers), `docs/switchyard.md`, `docs/oracle.md`,
`docs/cpu-checks.md`.

## Status
Serving is correct and fast on the RTX 5080; the 262K/1M recall, scheduler-stall,
finishability-invariant, disconnect-detection and admission-livelock blockers are all closed, and
the oracle ladder is complete at 131K/262K/524K/1M. The current baseline is the **16-way soak
against `8429411` (soak §AA, 2026-09-06): PASS on both routes** — 608 stage / 2,138
passthrough requests, 0 errors, 0 STALLED, 0 fatals, 0 ASGI tracebacks, **0 violations over 1,551
invariant checks** (worst shortfall 0 tokens), 0 of 489 decode batches eager, graceful shutdown in
3 s with GPU back to 0 MiB. Passthrough is the **best passthrough phase of the effort on the tail**
(p95 26.0 s, p99 38.8 s) at +12.2 % effective prefill rate; the best *stage* phase remains §Z (704
requests, p95 70.9 s). Three soaks in a row now agree on the decode expert-cache hit rate to within
0.4 pp (53.0 / 52.9 / **53.3 %**), which only became measurable in `38617a7`.

What is left is a short watch list, not a blocker list: the reclaim arm's over-spill (§AA9.1), an
unguarded host-RAM window during model load (§AA9.2), one measured MoE-prefill bucket-boundary win
(M=512), and the usual two user decisions.

## Performance (start of effort → now)

| metric | start (2026-09-04, ~`508ea32`) | now (`22478ef`) | evidence |
|---|---|---|---|
| decode 131K | 82.8 tok/s | **145.3** (1.75x) | decode_launch_2026-09-04 |
| decode 262K | 58.7 | **132.4** (2.26x) | decode_launch_2026-09-04 |
| decode 524K | 35.4 | **113.6** (3.21x) | decode_launch_2026-09-04 |
| decode 1M | ~20 | **95.8** (single sample; 80.7 on the oracle's 1M leg) | decode_launch, oracle |
| prefill 131K | 3,230 tok/s | **6,559** live (6,577.8 on the bench; TTFT 20.0 s) | oracle §14a |
| prefill 262K | 1,965 | **3,936** (2.00x) | prefill_profile + moe_prefill_gemm |
| prefill 524K | 1,064 | **2,467** (2.32x; +7.4 % over `2a139ad`'s 2,297, TTFT 212.5 s) | oracle §15a |
| prefill 1M | 573–576 | **1,307** (2.28x; MoE-GEMM gain not re-measured at 1M) | prefill_profile |
| 1M TTFT | 1,810–1,824 s | **795.8 s** | prefill_profile |
| MoE prefill GEMM @ M=8192 | 29.47 ms/layer | **13.708** (2.15x, bit-exact) | moe_prefill_leftovers §1 |
| 16-way decode (engine, 12 lanes) | 143.21 eager | **153.84** (1.074x; 1.039x at 16) | decode16 |
| soak decode aggregate @16 lanes | 81.6 stage / 161.4 passthrough tok/s | 87.2 (n=9) / **200.5** (n=122) | soak §AA |
| soak stage | FAIL (crash, then stalls/deadlock at `81ab30e`/`ea7ed7c`) | **PASS** 608 req / 0 err / 0 STALLED, p95 77.8 s (§Z best: 704 req, p95 70.9 s) | soak §AA/§Z |
| soak passthrough | FAIL | **PASS** 2,138 req / 0 err / 0 STALLED, p95 26.0 s | soak §AA |
| soak finishability-invariant warnings | 9 (§W passthrough tail) | **0 of 1,551 checks** | soak §AA4 |
| prefill starvation signature | 61 % stage / 19 % passthrough of passes | **0.1 % / 0.0 %** | soak §AA |
| soak decode batches run eager | — | 73.5 % at `13af13d` → **0 of 489** | soak §AA |
| soak effective prefill rate | 1,830 stage / 1,879 passthrough tok/s | **2,411 / 2,448** | soak §AA3 |
| soak admission refusals (stage) | — | 1,867,771 at `38617a7` → **163** | soak §Y5b/§AA3 |
| decode expert-cache hit rate | not measurable | **53.3 %** (passthrough phase; 50.8 % lifetime) | soak §AA6 |

Oracle recall by question shape (needles all present in state; **0 `retention`, 0 `selection`
at every length on both engines**). 131K and 262K keep the generic turns (24 rows); 524K and 1M
ran `--no-generic` (19 rows):

| length | direct (key→code) | combined | reverse (code→key) | control | FreeToken total | llama.cpp |
|---|---|---|---|---|---|---|
| 131K | 6/6 (llama.cpp 6/6) | 4/6 (5/6) | 6/6 (6/6) | pass | **22/24** | 23/24 |
| 262K | 5/6 (3/6) | 2/6 (2/6) | 6/6 (6/6) | pass | **19/24** | 17/24 |
| 524K | 1/6 (2/6) | 0/6 (1/6) | 6/6 (6/6) | pass | 8/19 | 10/19 |
| 1M | 1/6 | 0/6 | 5/6 | pass | 7/19 | not runnable on this card |

131K bounds the collapse from below: **0 both-miss, 0 retention, 0 selection, direct 6/6 on both
engines**, and every 131K miss on either side is *arithmetic* on a combined question (FreeToken's
two are off by one). The `key → code` collapse between 262K and 524K returns the **same wrong
near-duplicate code byte-for-byte in both engines** — a model property, not an engine defect
(oracle §14/§16).

## Closed
- **262K recall** — Mamba-2 prefill `dt` floor (`time_step_min` is an *initializer* range, not
  a runtime bound). `3ac79ec`; `262k_rootcause_2026-09-04.md`, `262k_crossengine_2026-09-04.md`.
- **Decode launch config** — `_grid_filling_splits` sizes the split count to the SM count for
  untuned head shapes; int64 slot ids on KV load. `acc91e9`; `decode_launch_2026-09-04.md`.
- **Prefill superlinearity** — extend-attention `BLOCK_M` capped by the fp32 accumulator's
  register budget (396 spill slots → 14). `4a99e34`; `prefill_profile_2026-09-05.md`.
- **Native-Q8 extend QK** — closed NEGATIVE, kernel unchanged (the 225 TFLOP/s premise is the
  spec sheet; the kernel is at 57–60 % of the achievable 123). `prefill_q8_2026-09-05.md`.
- **MoE prefill GEMM 1.74x** — one e4m3 block scale per 8 bytes + hardware `cvt.rn.f16x2.e2m1x2`;
  29.47 → 16.95 ms/layer at M=8192, bit-identical. `2a139ad`; `moe_prefill_gemm_2026-09-05.md`.
- **MoE prefill A-operand deinterleave** — both A gathers were stride-2; an even-k/odd-k plane
  rewrite makes them unit-stride at an unchanged reduction order. **1.215x at M=8192, bit-exact**;
  131K prefill 6,124.7 → 6,577.8 tok/s, TTFT 21.6 → 19.8 s. On by default,
  `FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A=0` disables. `ca7e74b`; `misc_tickets_2026-09-05.md` §2.
- **MoE prefill fused k-planes (item 2a)** — gemm2's A *is* gemm1's output, so gemm1 now emits the
  two k-planes itself via a permuted **B/scale gather** (`PLANAR_OUT`), and the gemm2 deinterleave
  prepass disappears. **1.018x at M=8192 (1.237x over the pre-`ca7e74b` kernel), bit-exact at every
  M**; `%tl.dot` 59.3 → 60.4. On by default, `FREETOKEN_NVFP4_PREFILL_FUSED_PLANES=0` disables; the
  flag is inert unless the deinterleave is on and the activation is epilogue-fused (a gated
  activation reads gemm1's output row-wise and would be corrupted). `9dc283e`;
  `moe_prefill_leftovers_2026-09-05.md` §1. Neutral-to-positive live in soak §AA3.
- **Extend-path MoE 9–10x** — `--moe-extend-cache-tokens` (default 64): small extends take the
  decode movement path instead of streaming all 128 experts (16.5 GB/forward at the PCIe
  roofline). `89b632b`; `extend_moe_2026-09-05.md`.
- **Extend-cache threshold** — stays **64** (crossover between 64 and 80 on wall time), plus a live
  crash guard: `use_cached_extend` refuses above 1,024 routed ids because flashlib's `lru_ensure`
  cannot compile past `BLOCK_K = 1024`. `misc_tickets_2026-09-05.md` §3.
- **Elastic CUDA graphs + non-elastic graph ladder** — dense batch sizes to 16 (offload-MoE models
  only; dense models keep the historical sparse list, pinned by a test). 73.5 % of soak decode
  batches ran eager at `13af13d`, **0 %** since; 140.43 → 150.90 tok/s at 12 lanes (1.074x).
  `14c1bd8`, `ca7e74b`; `decode16_2026-09-05.md`, `misc_tickets_2026-09-05.md` §1.
- **`--spec-draft-len` default** — stays **8**: k=16 is 0.870x of spec-off at 131K (k=8 0.898x)
  against a ±2 % criterion, and at k=16 the break-even gate never closes.
  `misc_tickets_2026-09-05.md` §4.
- **Spec-gate seeding — closed NEGATIVE, shipped default off.** `FREETOKEN_SPEC_GATE_SEED=1` fits
  `verify_ms(m) = a + b·m` from two narrow probes to 0.1 % of both measured operating points and
  primes the gate — and then runs a full-width verify step anyway, because `emit` is still on its
  optimistic `max_k + 1 = 9` prior: **226 ms spent pricing the gate against the shipped arm's
  182 ms**. Correct, cheap, unit-tested, kept in the tree, off. The tok/s column is *not* the
  verdict (the arms emitted 96 vs 70 tokens and plain decode itself moved 18 % between arms).
  `785a278`; `ngram_spec_gate_seed_2026-09-05.md`.
- **Scheduler admission** — standing reservation + finishability invariant (`b030c7f`), then
  the seatable-lanes chunk divisor (`812bc57`). Two failed attempts reverted first
  (`81ab30e`→`5bf0bcc`, `ea7ed7c` deadlock). `f6ed0b5` soak PASS; soak §U/§V.
- **Finishability invariant vs cold-session restores (§W6)** — `_restore_cold_session` spends the
  same pool the admission gate proved against, from a path that is not a gate. Charged against an
  exported `PrefillManager.finishability_reservation`, with a deferral rather than a loosened
  invariant. `e3a2019`; validated by soak §X (0 of 1,541 checks) and every soak since.
- **Non-streaming client disconnect is now detected (open item 0).** The request-ring recorder was
  a Starlette `BaseHTTPMiddleware`, which proxies the ASGI receive channel through its own task and
  never forwards `http.disconnect`, so `disconnect.py`'s poll of `Request.is_disconnected()` read
  False forever. Rewritten as a **pure-ASGI middleware** that passes `receive` through untouched.
  `38617a7`; `tests/server/test_disconnect_middleware_asgi.py`, CPU A/B in
  `benchmarks/probe_disconnect_middleware.py`. **Validated live in soak §Y5**: the probe's
  non-streaming arm moved `client_disconnect` on its own, plus **11 real disconnect aborts in stage
  traffic** with 11 matching abort log lines, against 0 in §W/§X.
- **Cumulative MoE decode counters (open item 0b).** `OffloadCache`'s bank rebuild calls
  `lru_stats.zero_()`; `decode_stat_totals` now folds the windowed counters into a host base before
  every `reset_stats`, so the totals are a lifetime accumulator. `38617a7`. **Validated in soak
  §Y6/§Z6/§AA6**: monotone across 20–26 elastic capacity changes and graph captures, and a decode
  expert-cache hit rate is finally a soak measurement — 53.0 / 52.9 / **53.3 %** in three runs.
  Extend-cache gate 5.6 / 6.5 / 5.7 % at `--moe-extend-cache-tokens 64`.
- **The 576 s stage admission livelock (open item 7's head).** A FIFO admission loop with one
  stopping rule: a fresh prompt the pools refused made the pass `break`, so 15 seatable requests
  behind it went unexamined at `usage 0.59` with 108 K tokens free; nothing ran, so no page came
  back, so every pass took the identical decision — **1,867,771 refusals in 576 s (~3,240/s) on one
  core at 102 %**, ended only by the clients' 600 s timeouts. Fix (`125da19`): (a) a refused FRESH
  admit is skipped, not stopped on, while `PrefillAdder.headroom > 0` (safe because
  `reserved_size` already carries every standing claim — the term `ea7ed7c` lacked), counted as
  `fresh_admits_deferred`; (b) `headroom` stops the walk when nothing of any size can be seated;
  (c) `_only_idle_sessions` also returns True when the last pass scheduled nothing, so a refused
  pass takes the 10 ms nap. `_seatable_lanes` was deliberately **not** given the same skip
  (mirroring cost 20 % of prefill throughput on all five replay profiles). The ticket's other half,
  a client-rejection path for an over-pool prompt, is **unreachable by construction**
  (`engine.py:546` clamps `max_seq_len`, and the message path already rejects
  `input_len >= max_seq_len`). Tests: `tests/scheduler/test_admission_livelock.py`. Validated by
  soak §Z: refusals 190 stage, 0 gaps ≥ 30 s, 704 requests at p95 70.9 s.
- **Non-streaming disconnect returns a quiet 499** — `disconnect.ClientGone` (a `CancelledError`
  subclass, so the AbortMsg path is unchanged) + `client_gone_response`, replacing §Y's 10 ASGI
  tracebacks. `125da19`; 0 `Exception in ASGI application` in §Z and §AA.
- **Ticket-7 scheduler leftovers (`8429411`), validated by soak §AA.** Three changes:
  (a) `_reclaim_soft_sessions_for_pending` measured the *pre-lock* budget while `_try_allocate_one`
  locks the matched prefix first, so a turn reusing a large evictable prefix looked comfortable in
  the pressure test and was refused at the gate — §Y5b logged **zero** admission-pressure releases
  across the whole 576 s. New `CacheManager.lock_delta(handle)` (a `node..root` walk over
  `ref_count == 0` nodes) makes the test `needed > available_size - lock_delta`, and
  `_reclaim_for_blocked_prefill` now scans `_RECLAIM_SCAN_DEPTH = 4` deep instead of only the head
  (since `125da19` a pass admits *past* a prompt it cannot seat). Live: admission-pressure releases
  **+15 % stage / +18 % passthrough**, `fresh_admits_deferred` 317 → 1,151 on stage.
  (b) A **strict per-pass memo** of `CacheManager.match_req` shared by the seat scan, the admission
  loop and the post-refusal reclaim, with `scheduler.prefill.match.*` on `/v1/stats`. Replay:
  `match_tokens_per_prefill_pass` −9…−18 % on all five profiles with byte-identical outcomes.
  Live: **~101 K tokens of radix walk per pass, 40.8 % memo hit rate** run-wide, at no measurable
  scheduler CPU.
  (c) `_maybe_shrink_growable_kv` evicted the whole prefix cache *before* computing whether a
  shrink was possible; it now returns early when the best reachable target cannot beat
  `committed_pages`. Provably never skips a shrink that would have happened.
  Tests: `tests/scheduler/test_reclaim_and_match_memo.py`.
- **Oracle 131K rung + the 524K `direct:harbour` lead (open item 6) — CLOSED.** 131K on both
  engines: FreeToken 22/24 vs llama.cpp 23/24, **direct 6/6 each, 0 both-miss, 0 retention, 0
  selection**; every miss on either engine is arithmetic on a combined question. The 524K
  `direct:harbour` re-probe on a *rotated* haystack (`--filler-cursor 65`, sha256
  `9e82fd972d04de7a`) pays a normal **2.62 s cached TTFT** and still misses — and misses to a
  *different* wrong code (`interference-cross` → `interference-near`), so **"the 50 s TTFT was not
  the cause"**; the partial-re-prefill explanation is refuted. What remains is one interference-
  class probe under a quantization confound that cannot be lifted on this card. `5d59c05`;
  `oracle_2026-09-05.md` §14–16.
- **Client-disconnect abort in prefill** — `ff470e7`; `server/disconnect.py`, 12 tests.
  `abort_user` shielded as a tracked task (`e3a2019`).
- **Observability** — `/v1/stats.scheduler` + `requests.aborts`, invariant counted every pass
  (`78f29d3`); MoE counters (`e3a2019`); `fresh_admits_deferred` + replay `stall_frac` (`125da19`);
  `scheduler.prefill.match.*` (`8429411`).
- **1M gate** (restart persistence, eviction, NVMe restore) `31d606d`; **hidden-state parity**
  (52 layers, cosine ≥ 0.998840) `befcde6`; **1M direct-addressing** closed model-limited
  `be85ffa`; **MTP** NO-GO; **n-gram speculation** shipped behind `--speculative ngram` but
  measures 1.01–1.03x (`e4070da`), verify step 54.0 → 35.6 ms (`b84ecb7`).
- **CI** — `.github/workflows/cpu-checks.yml` (ruff + CPU unit tests + scheduler replay gate),
  `508ea32`; `docs/cpu-checks.md`. The full CPU test step, timed with **no GPU job live: 118 s
  wall** (`ee2e7bf`). The `[tool.ruff.lint] ignore` list is down to **E702/E731/E741** — F401,
  F841, E402, E712, E714, E742, E701 and F541 are now enforced, 45 violations fixed (`22478ef`).
- **Pre-existing test issues (open item 10) — CLOSED.** `pythonpath = ["."]` in `pyproject.toml`
  so `tests/server/test_muse_glimmer_parsers.py` collects its sibling module (`ee2e7bf`); the
  laguna TP fixture now tolerates an already-set TP info, which was the real cause of the 6
  ordering errors in a full CPU run (`0f6ff4b`) — not GPU contention.
- **Soak drivers in-repo** — `benchmarks/switchyard_soak/` (`f6ed0b5`), all committed.
- **Soaks §W → §AA.** §W (`ca7e74b`) PASS with 9 invariant warnings; §X (`e3a2019`) PASS, 0 of
  1,541, disconnect defect found; §Y (`38617a7`) both fixes validated but **stage FAIL** on the
  livelock; §Z (`125da19`/`785a278`) PASS both routes, livelock closed; **§AA (`8429411`) PASS both
  routes — the current baseline.**

## Open, ranked by value
1. **Reclaim over-spill watch (soak §AA9.1).** The `8429411` reclaim arm pays for itself but spends
   prefix cache doing it: prefix reuse **86.0 → 83.6 % stage (−2.4 pp)** and 88.9 → 87.9 %
   passthrough, spills 1,377 → 1,542 (+12 %), restores 439 → 479 (+9 %), `restores_deferred` 1 → 3,
   0 failures — bought against a **+12.2 % passthrough prefill rate and the run's best p99**, with
   `refusals` flat (190 → 163). If a later soak shows reuse falling further *with restores still
   climbing*, `_RECLAIM_SCAN_DEPTH = 4` is the knob. Watch, do not revert.
2. **The model-load RAM window is unguarded (soak §AA9.2).** `benchmarks/switchyard_soak/run.sh`
   checks `MemAvailable ≥ 26 GiB` **once, before** the load and arms `SOAK_RAM_ABORT_GIB=2` only
   **after** `READY`. The load itself transiently drove the host to **1.1 GiB** — 0.9 GiB *below*
   the floor that would have TERMed a running server, and the closest this effort has come to
   another WSL OOM restart. Start the watchdog before taking the lock, or gate the load phase on
   its own floor. (The *serving* floor is the best yet: 4.6 GiB, vs §Z 4.0, §Y 2.8, §X 2.1.)
3. **M=512 is served by the wrong MoE-prefill bucket.** `PREFILL_M_BUCKETS` is
   `(16, 64, 256, 1024, 4096, 8192)` and `nvfp4_moe_config` picks the *nearest*, so M=512 lands in
   the "256" bucket at `BLOCK_M=16` — where `BLOCK_M=32` measures **1.10x on two independent
   routings** (1.429 → 1.296 ms; 1.445 → 1.306 at `--seed 7`). Under multi-lane load the
   scheduler's interleave share produces exactly this width. Not changed: a bucket boundary is a
   shipped-table change that wants end-to-end evidence, and the microbench cannot see the real
   chunk-width distribution — grade it with a soak. Note the **original ticket's denominator was
   wrong**: the "M=256 at 20 % of ceiling" figure was against a `tl.dot` ceiling; at ~12 routed
   rows per expert the bucket is weight-streaming bound and already at **69.4 % of the HBM
   roofline** (0.748 ms floor vs 1.078 measured), so `BLOCK_M=32` there is a **12 % loss** and the
   256 bucket is correct as shipped. Still unswept: `BLOCK_KB`/`num_stages` at the small-M bucket
   (`--grid smallm`, 216 tiles). `moe_prefill_leftovers_2026-09-05.md` §2.
4. **Seed both sides of the spec break-even gate, on the replay, not on the GPU.** Seeding the cost
   side alone is measured and negative (closed above). The fix is to prime `emit` too — either from
   the probes' own acceptance (widen `_SEED_WIDTHS` to e.g. `(3, 6)` so at least one probe usually
   *rejects*, which under greedy decoding is a true full-width sample), or by comparing against the
   `max_k + 1` ceiling while `emit` is still a prior (a real policy change). **Do not re-run this
   as end-to-end GPU arms**: 2–3 verify steps on a 79-token generation cannot resolve a 4 % effect
   against an 18 % baseline spread. Extend `benchmarks/spec_engage_replay.py` with a
   **gate-policy axis** — fixed transcript, CPU only, no model load — which is what settled the
   draft-rate question. `ngram_spec_gate_seed_2026-09-05.md` §4.
   The rest of item 1 is unchanged and is not a gate problem: at 131K the extend attention reads
   the whole KV history once per query token, so verify/decode is ~9–12x against a `k+1 = 9`
   ceiling. A fused multi-query extend kernel is the shape of that fix.
5. **A cross-pass match memo.** The per-pass memo (`8429411`) is measured — 40.8 % hits, ~101 K
   tokens/pass — but the walk still scales with queue × prompt *within* a pass. The remaining win
   is a memo that survives a run of identical refused passes, and the scheduler already owns the
   exact predicate: `_admission_stalled` (`125da19`) is true precisely when no batch was scheduled,
   none drained and no message arrived. Sound, but it means enumerating every mutation that can
   reach the manager from outside a pass — a missed one is exactly `ea7ed7c`'s stale `cached_len`.
   soak §AA3, `tasks/todo.md`.
6. **The reclaim pressure test does not charge `finishability_reservation`.** Deliberately left
   out: including it would make the test exact, but the message-path caller
   (`_reclaim_soft_sessions_for_admission`, which runs on every arriving request, refused or not)
   would spill idle conversations more eagerly than any measurement asks for. The lock delta is the
   term §Y5b's evidence names; the reservation term waits for a run that shows it costing
   something. §AA9.1's over-spill is the counter-evidence to watch.
7. **The streaming disconnect path still raises `CancelledError` out of the ASGI app.**
   `FrontendManager.stream_with_cancellation` re-raises after `spawn_abort`, so uvicorn logs
   `Exception in ASGI application` for a StreamingResponse whose client left. The non-streaming
   endpoints now return a quiet 499; the streaming generator needs the equivalent (`return`, not
   `raise`, once the abort is spawned). Left out of the `125da19` fix because §Y's 10 tracebacks
   were all on the non-stream path — so this one is **unobserved, not fixed**. soak §Y8.4.
8. **`fresh_admits_blocked_by_cap` is now the largest remaining admission knob** — 388 (§X) → 204
   (§Y) → 301 (§Z) → **376** (§AA), i.e. `max_chunked_prefills = 8` binds on every run. Goodput has
   gone up in the same runs, so this is evidence for the reservation arithmetic, not a demonstrated
   cost. soak §AA9.3.
9. **16-way decode is at the hardware ceiling — do not re-litigate.** 74 % of the step is the
   PCIe expert gather at 51–52 GB/s against a measured 52.9 GB/s link, working set ~1,417
   expert-layer slots against 976 in the pool. Attention, the MoE GEMV and Mamba-2 were all
   measured fine at batch 16. Only two levers left: `--moe-backend hybrid` at 16 lanes (never
   measured; the auto-threshold asks the wrong question) and the 976-vs-1,417 slot deficit.
   `decode16_2026-09-05.md` §0/§2/§7.
10. **fork/main fast-forward** — user decision. `fork/main` (`62f5a66`) is a strict ancestor of
    HEAD; `fork/nemotron35` already carries the merge.
11. **`_gguf` extension rebuild before deploying on Ada** — user decision. The fork/main merge
    (`32cc504`) changed multiwarp bool → warps int64; a stale `.so` silently picks the 4-warp path.
12. **Watch mean lanes per prefill batch every soak.** Stage 3.13 (§X) / 3.55 (§Y) / 3.79 (§Z) /
    **3.04** (§AA), passthrough 4.88 / 4.53 / 4.43 / 4.62 — no trend, no errors. Stage >~5
    **together with** rising errors or p95 is the §R6/§R7 mode returning.
13. **Smaller, all in `tasks/todo.md` with evidence:** `benchmarks/scheduler_replay.py` is still not
    an acceptance gate for policy (it scored `81ab30e`, which then failed the live soak, and it
    **re-implements the loop's reclaim inline** rather than calling
    `Scheduler._reclaim_for_blocked_prefill`, so the whole `8429411` reclaim change is invisible to
    it); `stopped_for_lane_cap` rotation is dead code on this model; the 1,024-routed-id extend
    guard is a flashlib-LRU constraint applied to an LFU profile that does not have it (and raising
    it costs a 22-minute Triton JIT that must be a warmup job, never a live request); folding
    `tests/moe` + `tests/kernels` into CI is **infeasible** — the CI runner is CPU-only ubuntu and
    those suites need a GPU; the ruff ignore list still carries E702/E731/E741;
    `bench_nvfp4_moe_kernels.py --gate` asserts inverted 2B1 targets; `num_kv_splits_ptr` is passed
    and never dereferenced; session-residency leftovers from the 1M gate; two stale worktrees.

## How to run things
- **Soak**: `benchmarks/switchyard_soak/run.sh [tag] [duration]` — stage 20 m then passthrough
  20 m, c=16, server under `scripts/gpu_lock.sh` with `FREETOKEN_SCHEDULER_INVARIANT=warn`.
  It refuses to start below 26 GiB `MemAvailable` and TERMs the server below 2 GiB while running
  (see open item 2 — the load window itself is *not* covered); everything lands in `runs/<tag>/`
  (gitignored). `SOAK_PHASES=""` (empty) runs the disconnect probe alone — both endpoint shapes,
  ~90 s of GPU. Grade with `analyze.py` (per-route stats, lanes/batch, starvation-signature
  fraction, match counters, `stats_*.json` deltas) and `gaps.py` (leading/trailing silence).
  **The current baseline to diff against is §AA.** Contract/e2e checks:
  `scripts/switchyard_e2e.sh contract|soak|agents` (`docs/switchyard.md` §8).
- **Oracle**: three phases in `docs/oracle.md` §"The three commands" — A FreeToken (you start
  the server; include `--enable-cache-report`), B llama.cpp (starts/stops its own server;
  `--n-cpu-moe 14` at 131K/262K, **23** at 524K), C compare (CPU only). Sweep dimension is
  *length*, never depth. `--target-prompt-tokens 1044480` at the 1M rung (the suite is a
  conversation and grows). Verify prompt identity on CPU with `record --build-only` before taking
  the lock. Phase A's `ft serve` needs an absolute `.venv/bin/ft` inside a wrapper script —
  `gpu_lock.sh` does not run through the venv shim.
- **Ticket harnesses**: `scripts/gpu_lock.sh benchmarks/decode16/phaseE2.sh <outdir>` (graph
  ladder, ~22 min); `benchmarks/bench_moe_prefill_gemm.py --variant tree deint fused prepass
  prepass2 --grid shipped --verify` (~1 min; also prints the weight-byte floor, `%HBM`, padded rows
  and M-block count per `BLOCK_M`); `FREETOKEN_GPU_LOCK_WAIT=7200 scripts/gpu_lock.sh
  benchmarks/extend_moe/run_threshold.sh` (~5 min to m=96 — the first m whose `m*top_k` crosses
  1024 costs **~22 min of Triton JIT**); `benchmarks/probe_spec_ngram_impl.py --sweep-k ...`
  (~2 min). Full commands in each results file's Reproduction section.
- **Replay / CPU gates**: `uv run benchmarks/scheduler_replay.py --gate` (5 profiles, CPU, ~4 s,
  428 MB RSS; the `stall_frac` column reproduces §Y5b); `benchmarks/spec_engage_replay.py` for
  anything about the speculation gate; the full CPU test step is **118 s** with no GPU job live.
  CI runs ruff + the CPU test directories + the replay gate (`docs/cpu-checks.md`).
- **A/B hatches**: `FREETOKEN_DECODE_{KV_SPLITS,BLOCK_N,NUM_WARPS}`,
  `FREETOKEN_EXTEND_{BLOCK_M,BLOCK_N,NUM_WARPS,NUM_STAGES}`,
  `FREETOKEN_NVFP4_PREFILL_{BLOCK_M,BLOCK_N,BLOCK_KB,GROUP_M,NUM_WARPS,NUM_STAGES}`,
  `FREETOKEN_ELASTIC_GRAPH_MAX_BS`, `FREETOKEN_GRAPH_DENSE_BS` (non-elastic ladder, `0|1`),
  `FREETOKEN_NVFP4_PREFILL_DEINTERLEAVE_A` (`=0` restores the interleaved A gathers),
  `FREETOKEN_NVFP4_PREFILL_FUSED_PLANES` (`=0` restores the gemm2 prepass),
  `FREETOKEN_SPEC_GATE_SEED` (`=1` arms the seeded gate probe; default off, measured negative),
  `FREETOKEN_NVFP4_NO_NATIVE_CVT`, `FREETOKEN_NEMOTRON_DT_MIN`, `FREETOKEN_MAMBA2_REF`.
  Both launches are logged once at startup (`Triton decode launch:`, `Triton extend launch:`).

## Host rules (from tasks/lessons.md — all learned the hard way)
- `systemctl --user stop piro-board-embedder.service` before GPU work (Restart=always, 4–10 GB).
- Host RAM (34 GiB WSL, 4 GiB swap) is the constraint, not VRAM. **Any** job that loads the
  checkpoint runs under `scripts/gpu_lock.sh`: refuses below 22 GiB `MemAvailable`, 4 h cap,
  `oom_score_adj=1000`, reaps the worker tree on exit. The **model load** is the peak: soak §AA
  saw `MemAvailable` hit 1.1 GiB during it.
- **Never run pytest, or import torch at all, while a model is loaded.** A CPU test sweep
  overlapping an expert-bank build OOM-restarted WSL twice. Check `pgrep -f "ft serve"` and
  `nvidia-smi` first; if anything is loaded, stay under ~1 GB RSS and do desk work.
- **Never pipe `scripts/gpu_lock.sh`** — its exit trap runs `pkill -9 -g $$` and kills the
  reader. Redirect to a file, then read the file; expect `Killed`/137 after a *successful* run.
  A wrapped script must `exec >` its own log and use `python -u`, or the log is 0 bytes.
- **No detached GPU job**: never end a turn while your GPU job continues. A background `until`
  loop does not make the agent wait; a foreground `until ! kill -0 $PID; do sleep 20; done` does.
  `pgrep -f <script>` matches the `bash -c` wrapper — check `ps -o etime -p <pid>` or an
  artifact's mtime instead.
- The scratchpad does **not** survive a WSL restart. Anything that produced a number in a
  results file belongs under `benchmarks/` or `scripts/`.
- Never `git stash` / `git commit -a` / `git add -A` / `git add -u`; implementers work in
  worktrees and stage explicit paths. Kill leftovers by venv path:
  `pkill -9 -f "FreeToken/.venv/bin/python3"`, then `free -g`. Clear a stale JIT lock on sight:
  `rm ~/.cache/torch_extensions/py312_cu130/*/lock`.
- Serving profile: `FREETOKEN_PIN_BUDGET_GB=17`, `--memory-ratio 0.85`, 8K chunks, q8_0 KV +
  Triton attention, `--nvfp4-backend` auto→triton, LFU for 16-way. Needle gates go through
  `/v1/chat/completions` with digit-free filler; never grade raw SSE.

## Orchestration model
Lead session steers and does not read files it can delegate. Implementation goes to Opus
subagents in `git worktree`s, **at most two at a time**. **One GPU job at a time**, under the
lock, and the agent that started it stays attending it until it exits. CPU-side agents run no
pytest and no torch import while a model is loaded, and stay under ~1 GB RSS. Two stale
worktrees predate this effort and can be ignored or removed:
`.claude/worktrees/agent-a45f827ae98e76526`, `.claude/worktrees/agent-a4ce5e26d2ccafdb6`.

Results files referenced above live in `benchmarks/results/` with the prefix
`nemotron35_lightning_5080_` (except `ngram_spec_gate_seed_2026-09-05.md`).
