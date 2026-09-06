# Nemotron 3.5 Lightning on FreeToken — handover

State as of 2026-09-06, HEAD `8e064d4` on `main`, **working tree clean**. Everything below is
committed, and `fork/main` has been fast-forwarded (it sits at `9afd48c`, 0 ahead). **Every ticket
this effort opened is now either closed with a result and a commit, or explicitly deferred with a
written reason why it does not affect production** — see "Open" below, where every remaining line
carries a `Deferred:` clause. The current baseline is **soak §AB** (`670969f`), the fifth
consecutive 16-way soak and the fourth consecutive PASS on both routes. Read this, then
`tasks/nemotron35-plan.md` (spec), `tasks/todo.md` (open checklist), `tasks/lessons.md` (rules —
read before touching the GPU), `docs/nemotron.md` (profiles + numbers), `docs/switchyard.md`
(§8 soak, §9 traces, **§11 production observability**), `docs/oracle.md`, `docs/cpu-checks.md`.

## Status
Serving is correct and fast on the RTX 5080. Every blocker this effort found is closed: 262K/1M
recall, the scheduler stall, the finishability invariant, non-streaming disconnect detection, the
576 s admission livelock, the ASGI tracebacks on both request shapes, and the unguarded
model-load RAM window. The oracle ladder is complete at 131K/262K/524K/1M.

Baseline: the **16-way soak against `670969f` (soak §AB, 2026-09-06): PASS on every criterion,
both routes** — 635 stage / 1,977 passthrough requests, 0 errors, 0 STALLED, 0 fatals, **0
violations over 1,452 invariant checks** (worst shortfall 0 tokens), **0 `Exception in ASGI
application` for either request shape**, 0 of 459 decode batches eager, no gap ≥ 60 s (largest 23 s
/ 32 s), graceful shutdown in 4 s with GPU back to 0 MiB. Four consecutive soaks now show no
livelock, a decode expert-cache hit rate reproducible to 0.7 pp (53.0 / 52.9 / 53.3 / **52.6 %**),
a radix match-memo hit rate reproducible to 0.1 pp (40.8 / **40.9 %**), and busiest-process CPU
flat at 107.9–108.9 % median.

Production observability shipped with it (`7261fa8`, `docs/switchyard.md` §11): the server log
with the invariant on, a stdlib `/v1/stats` sampler with a systemd user unit, and `--trace-dir`
plus `trace_load_report.py` to check the gate's synthetic traffic against the real thing. See the
**Production launch checklist** below.

## Performance (start of effort → now)

| metric | start (2026-09-04, ~`508ea32`) | now (`8e064d4`) | evidence |
|---|---|---|---|
| decode 131K | 82.8 tok/s | **145.3** (1.75x) | decode_launch_2026-09-04 |
| decode 262K | 58.7 | **132.4** (2.26x) | decode_launch_2026-09-04 |
| decode 524K | 35.4 | **113.6** (3.21x) | decode_launch_2026-09-04 |
| decode 1M | ~20 | **95.8** (single sample; 80.7 on the oracle's 1M leg) | decode_launch, oracle |
| prefill 131K | 3,230 tok/s | **6,559** live (6,577.8 on the bench; TTFT 20.0 s) | oracle §14a |
| prefill 262K | 1,965 | **3,936** (2.00x) | prefill_profile + moe_prefill_gemm |
| prefill 524K | 1,064 | **2,467** (2.32x; +7.4 % over `2a139ad`, TTFT 212.5 s) | oracle §15a |
| prefill 1M | 573–576 | **1,307** (2.28x; MoE-GEMM gain not re-measured at 1M) | prefill_profile |
| 1M TTFT | 1,810–1,824 s | **795.8 s** | prefill_profile |
| MoE prefill GEMM @ M=8192 | 29.47 ms/layer | **13.708** (2.15x, bit-exact) | moe_prefill_leftovers §1 |
| 16-way decode (engine, 12 lanes) | 143.21 eager | **153.84** (1.074x; 1.039x at 16) | decode16 |
| soak decode aggregate @16 lanes | 81.6 stage / 161.4 passthrough tok/s | 93.4 (n=16) / **169.8** (n=84) | soak §AB3 |
| soak stage | FAIL (crash, then stalls/deadlock at `81ab30e`/`ea7ed7c`) | **PASS** 635 req / 0 err / 0 STALLED, p95 72.5 s | soak §AB2 |
| soak passthrough | FAIL | **PASS** 1,977 req / 0 err / 0 STALLED, p95 27.4 s | soak §AB2 |
| soak finishability-invariant warnings | 9 (§W passthrough tail) | **0 of 1,452 checks** | soak §AB4 |
| prefill starvation signature | 61 % stage / 19 % passthrough of passes | **0.0 % / 0.0 %** | soak §AB3 |
| soak decode batches run eager | — | 73.5 % at `13af13d` → **0 of 459** | soak §AB |
| soak effective prefill rate | 1,830 stage / 1,879 passthrough tok/s | **2,581 / 1,986** (§AB9.1) | soak §AB3 |
| soak admission refusals (run) | — | 1,867,771 at `38617a7` → **280** | soak §Y5b/§AB3 |
| soak ASGI tracebacks | 10 at `38617a7` (non-stream) | **0**, both request shapes | soak §AB4 |
| decode expert-cache hit rate | not measurable | **52.6 %** at c=16 (50.3 % lifetime) | soak §AB6 |

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
engines**, and every 131K miss on either side is *arithmetic* on a combined question. The
`key → code` collapse between 262K and 524K returns the **same wrong near-duplicate code
byte-for-byte in both engines** — a model property, not an engine defect (oracle §14/§16).

## Production launch checklist

The reference serving line is `benchmarks/switchyard_soak/serve.sh`, kept verbatim so a result
stays comparable across runs. Deployed, it is:

```bash
FREETOKEN_PIN_BUDGET_GB=17 FREETOKEN_SCHEDULER_INVARIANT=warn \
ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 127.0.0.1 --port 1919 \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 \
  --enable-cache-report --served-model-name nemotron-3.5-lightning \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 16384 \
  --hidden-states-dir /home/lucas/.cache/freetoken/hidden-states --hidden-states-max-tokens 4096 \
  --trace-dir /var/tmp/ft-trace
```

1. **`FREETOKEN_SCHEDULER_INVARIANT=warn`, and redirect the log to a file.** `warn` buys the
   offending pass's context; the *count* is published either way on
   `/v1/stats.scheduler.prefill.invariant`. Never `=raise` — a `warn`-level invariant is worth
   more than a `raise` one, and §W is the run that proved it. `analyze.py <log>` turns the batch
   lines into throughput, occupancy, lanes per prefill batch and the §R7 starvation signature.
2. **`--trace-dir /var/tmp/ft-trace`.** One JSON line per completed request, 0600, **no prompt
   text** — a prefix hash chain instead. This is the only source that sees an individual request.
   Replay with `benchmarks/trace_replay.py`, convert with `trace_to_profile.py`.
3. **The `/v1/stats` sampler as a user service** — the counters are cumulative, so a rate is a
   difference between two snapshots and production needs a time series:
   ```bash
   install -Dm644 benchmarks/ops/freetoken-stats-sampler.service \
       ~/.config/systemd/user/freetoken-stats-sampler.service
   systemctl --user daemon-reload && systemctl --user enable --now freetoken-stats-sampler
   python benchmarks/ops/stats_sampler.py summarize ~/.cache/freetoken/stats/
   ```
   It runs on `/usr/bin/python3`, not the venv (stdlib-only on purpose: a sampler that needed the
   venv would stop working exactly while the venv is being rebuilt), `Restart=always` with no
   dependency on the server unit, and records an outage as `"ok": false` rather than exiting.
   `summarize` gives per-bucket completions, errors, disconnects, refusals,
   `fresh_admits_deferred` / `_blocked_by_cap`, spills / restores / `restores_deferred`, memo hits,
   invariant violations, and the MoE hit rates **computed over the window**, never a lifetime
   average.
4. **Once there is both a trace and a soak run, check the gate against reality:**
   `python benchmarks/trace_load_report.py --trace /var/tmp/ft-trace --soak-run
   benchmarks/switchyard_soak/runs/<tag>` prints the real load's arrival rate, concurrency, token
   distributions, prefix reuse and latency quantiles beside the soak's, with a verdict naming which
   of the soak's assumptions hold. Both ops tools are stdlib-only and need neither torch nor the
   venv.
5. **Host RAM is the constraint, not VRAM.** The reference profile's serving floor was 2.3 GiB in
   §AB (median 6.6). `--host-ram-reserve-gb` raises the *static* pre-load bank preflight only.
6. **The 8 GiB-free variant, if the host must keep 8 GiB free:**
   `--session-spill-ram-gb 0 --host-ram-reserve-gb 9`. This forces the NVMe tier — every cold
   session restore comes off disk (measured 2.681 s / 1.32 GiB/s for a 1M-token session, spill
   1.18 GiB/s) instead of RAM. **This configuration has never been soaked.** At the 4 GiB default
   a 3.5 GiB checkpoint stays in RAM and does not survive a restart; at 0 it always survives.
   Soak it before trusting it under 16-way load.
7. **Hidden-state probe target (docs/switchyard.md §6).** The directory exists before launch
   (`serve.sh` does the `mkdir -p`; `FREETOKEN_HIDDEN_STATES_DIR` overrides the default
   `~/.cache/freetoken/hidden-states`) -- the server refuses a missing `--hidden-states-dir` at
   parse time -- and both `--hidden-states-dir` and `--hidden-states-max-tokens 4096` are on the
   line. FreeToken never cleans the directory: the consumer deletes each artifact once it has
   scored it, so a client that stops deleting fills the disk. Pooled-only requests
   (`kv_transfer_params.pooling`, no `hidden_states_path`) work without the directory and are not
   subject to the token cap; the file path (`hidden_states_path`) needs the directory and is
   capped at 4096 prompt tokens.

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
  two k-planes itself via a permuted **B/scale gather** (`PLANAR_OUT`) and the gemm2 deinterleave
  prepass disappears. **1.018x at M=8192 (1.237x over the pre-`ca7e74b` kernel), bit-exact at every
  M**. On by default; `FREETOKEN_NVFP4_PREFILL_FUSED_PLANES=0` disables, and the flag is inert
  unless the deinterleave is on **and** the activation is epilogue-fused (a gated activation reads
  gemm1's output row-wise and would be corrupted). `9dc283e`;
  `moe_prefill_leftovers_2026-09-05.md` §1. Neutral-to-positive live in soak §AA3.
- **MoE prefill "512" bucket (item 3).** `PREFILL_M_BUCKETS` gains `512` at `BLOCK_M=32` — the
  "256" entry with **nothing else changed**, which is exactly the pair the microbench measured
  (1.10x on two independent routings, 1.429 → 1.296 ms; the 1024 bucket's wider tile was never
  measured at this M, and borrowing a launch constant swept on another geometry is the 2026-09-05
  lesson). Nearest-bucket with **ties going to the smaller** bucket, so `384 < M ≤ 768` uses it.
  `FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS=512` restores the pre-2026-09-06 table exactly — the
  bucket-boundary A/B hatch, which `FREETOKEN_NVFP4_PREFILL_BLOCK_M` cannot express (it would drag
  4096/8192 off their tuned tile too). `670969f`; `tests/moe/test_nvfp4_triton_tuning.py`. Live in
  soak §AB: **stage clearly positive** — instant prefill median 3,194 tok/s (+17.5 %), effective
  new-token rate 2,581 (+7.1 %), p95 −6.8 %, p99 −14.0 % at *lower* prefix reuse — passthrough
  confounded by a lighter workload (see the deferred note §AB9.1).
- **Extend-path MoE 9–10x** — `--moe-extend-cache-tokens` (default 64): small extends take the
  decode movement path instead of streaming all 128 experts (16.5 GB/forward at the PCIe
  roofline). `89b632b`; `extend_moe_2026-09-05.md`. Threshold stays **64** plus a live crash guard
  (`use_cached_extend` refuses above 1,024 routed ids; flashlib's `lru_ensure` cannot compile past
  `BLOCK_K = 1024`). `misc_tickets_2026-09-05.md` §3.
- **Elastic CUDA graphs + non-elastic graph ladder** — dense batch sizes to 16 (offload-MoE models
  only; dense models keep the historical sparse list, pinned by a test). 73.5 % of soak decode
  batches ran eager at `13af13d`, **0 %** in every soak since; 140.43 → 150.90 tok/s at 12 lanes.
  `14c1bd8`, `ca7e74b`; `decode16_2026-09-05.md`, `misc_tickets_2026-09-05.md` §1.
- **`--spec-draft-len` default** — stays **8**: k=16 is 0.870x of spec-off at 131K (k=8 0.898x)
  against a ±2 % criterion, and at k=16 the break-even gate never closes.
  `misc_tickets_2026-09-05.md` §4.
- **Spec-gate seeding — closed NEGATIVE, shipped default off.** `FREETOKEN_SPEC_GATE_SEED=1` fits
  `verify_ms(m) = a + b·m` from two narrow probes to 0.1 % of both measured operating points and
  primes the gate — then runs a full-width verify step anyway, because `emit` is still on its
  optimistic `max_k + 1 = 9` prior: **226 ms spent pricing the gate against the shipped arm's
  182 ms**. Correct, cheap, unit-tested, kept in the tree, off. The tok/s column is *not* the
  verdict (the arms emitted 96 vs 70 tokens and plain decode itself moved 18 % between them).
  `785a278`; `ngram_spec_gate_seed_2026-09-05.md`.
- **Scheduler admission** — standing reservation + finishability invariant (`b030c7f`), then
  the seatable-lanes chunk divisor (`812bc57`). Two failed attempts reverted first
  (`81ab30e`→`5bf0bcc`, `ea7ed7c` deadlock). `f6ed0b5` soak PASS; soak §U/§V.
- **Finishability invariant vs cold-session restores (§W6)** — `_restore_cold_session` spends the
  same pool the admission gate proved against, from a path that is not a gate. Charged against an
  exported `PrefillManager.finishability_reservation`, with a deferral rather than a loosened
  invariant. `e3a2019`; 0 violations in §X/§Y/§Z/§AA/§AB.
- **Non-streaming client disconnect is now detected (item 0).** The request-ring recorder was a
  Starlette `BaseHTTPMiddleware`, which proxies the ASGI receive channel through its own task and
  never forwards `http.disconnect`, so `disconnect.py`'s poll of `Request.is_disconnected()` read
  False forever and an abandoned request answered 200 OK into a dead socket. Rewritten as a
  **pure-ASGI middleware** that passes `receive` through untouched. `38617a7`;
  `tests/server/test_disconnect_middleware_asgi.py`, CPU A/B in
  `benchmarks/probe_disconnect_middleware.py`. Validated in soak §Y5 (11 real disconnect aborts in
  stage traffic against 0 in §W/§X).
- **Both request shapes now end quietly (items 7 and its non-stream twin).** Non-streaming returns
  a quiet 499 (`disconnect.ClientGone`, a `CancelledError` subclass so the AbortMsg path is
  unchanged, + `client_gone_response`) — `125da19`, replacing §Y's 10 ASGI tracebacks. Streaming
  got the twin: `aiter_or_disconnect` raises `ClientGone` and
  `FrontendManager.stream_with_cancellation` **returns** instead of re-raising once the abort is
  spawned, since a StreamingResponse has no response object left to hand back; a genuine outer
  cancellation is still a plain `CancelledError` and still propagates — `670969f`. Validated by
  absence in soak §AB: **0 `Exception in ASGI application` in the whole run**, both probe shapes
  aborting, counting, and logging nothing.
- **Cumulative MoE decode counters (item 0b).** `OffloadCache`'s bank rebuild calls
  `lru_stats.zero_()`; `decode_stat_totals` now folds the windowed counters into a host base before
  every `reset_stats`. `38617a7`. Validated in §Y6/§Z6/§AA6/§AB6 — monotone across 16–26 elastic
  capacity changes, and a decode expert-cache hit rate is finally a soak measurement: **53.0 /
  52.9 / 53.3 / 52.6 %** in four runs.
- **The 576 s stage admission livelock (item 7's head).** A FIFO admission loop with one stopping
  rule: a refused fresh prompt made the pass `break`, so 15 seatable requests behind it went
  unexamined at `usage 0.59` with 108 K tokens free — **1,867,771 refusals in 576 s (~3,240/s) on
  one core at 102 %**, ended only by the clients' 600 s timeouts. Fix (`125da19`): a refused FRESH
  admit is skipped while `PrefillAdder.headroom > 0` (safe because `reserved_size` already carries
  every standing claim — the term `ea7ed7c` lacked), `headroom` stops the walk when nothing of any
  size can be seated, and `_only_idle_sessions` returns True when the last pass scheduled nothing
  so a refused pass takes the 10 ms nap. `_seatable_lanes` deliberately **not** mirrored (it cost
  20 % of prefill throughput on all five replay profiles). The other half — a client-rejection path
  for an over-pool prompt — is **unreachable by construction** (`engine.py:546` clamps
  `max_seq_len`; the message path already rejects `input_len >= max_seq_len`). Validated by soak §Z
  (refusals 190 stage, 0 gaps ≥ 30 s, 704 requests at p95 70.9 s) and three soaks since.
- **Ticket-7 scheduler leftovers (`8429411`), validated by soak §AA.** (a) The reclaim pressure
  test measured the *pre-lock* budget while `_try_allocate_one` locks the matched prefix first, so
  a turn reusing a large evictable prefix looked comfortable there and was refused at the gate —
  §Y5b logged **zero** admission-pressure releases across the whole 576 s. New
  `CacheManager.lock_delta(handle)` makes the test `needed > available_size - lock_delta`, and
  `_reclaim_for_blocked_prefill` scans `_RECLAIM_SCAN_DEPTH = 4` deep (since `125da19` a pass
  admits *past* a prompt it cannot seat). Live: pressure releases +15 %/+18 %,
  `fresh_admits_deferred` 317 → 1,151 on stage. (b) A **strict per-pass memo** of
  `CacheManager.match_req` shared by the seat scan, the admission loop and the post-refusal
  reclaim, with `scheduler.prefill.match.*` on `/v1/stats`; replay −9…−18 % matched tokens per pass
  with byte-identical outcomes, live **40.8 % then 40.9 %** hit rate in two soaks at no measurable
  scheduler CPU. (c) `_maybe_shrink_growable_kv` no longer evicts the whole prefix cache before
  discovering it cannot shrink. `tests/scheduler/test_reclaim_and_match_memo.py`.
- **The model-load RAM window is guarded (item 2).** `run.sh` checked `MemAvailable ≥ 26 GiB` once
  *before* the load and armed its watchdog only *after* `READY`; the load itself transiently drove
  the host to **1.1 GiB** in §AA, below the floor that would have TERMed a running server. Now
  `SOAK_RAM_LOAD_ABORT_GIB` (default 0.8) is armed on the READY poll, and `SOAK_HOST_RAM_RESERVE_GB`
  makes `serve.sh`'s previously hard-coded `--host-ram-reserve-gb` a one-liner. `670969f`. §AB
  reports the gates (`start>=26 load_abort<0.8 warn<4 abort<2 GiB`) and clears the load phase at an
  8.0 GiB minimum — armed and correct, but a warm load did not stress it (see §AB9.3 below).
- **`_gguf` stale-build guard (the Ada rebuild item).** The `fork/main` merge (`32cc504`) changed
  the shared MMVQ binding's `multiwarp` bool to an int64 `warps`, and pybind11 converts int → bool
  silently, so a pre-merge cached `.so` would run the wrong kernel width **with no error**. The
  loader now reads the bound signature, refuses a stale build, and names the cache directory to
  delete. `9afd48c`.
- **Oracle 131K rung + the 524K `direct:harbour` lead (item 6).** 131K on both engines: FreeToken
  22/24 vs llama.cpp 23/24, **direct 6/6 each, 0 both-miss, 0 retention, 0 selection**; every miss
  on either engine is arithmetic on a combined question. The 524K re-probe on a *rotated* haystack
  (`--filler-cursor 65`) pays a normal **2.62 s cached TTFT** and still misses — to a *different*
  wrong code — so **"the 50 s TTFT was not the cause"** and the partial-re-prefill explanation is
  refuted. `5d59c05`; `oracle_2026-09-05.md` §14–16.
- **Production observability (`7261fa8`).** `benchmarks/ops/stats_sampler.py` (`sample` /
  `summarize`, stdlib-only, window-not-lifetime rates, restart detection) + a systemd user unit;
  `benchmarks/trace_load_report.py` comparing a real `--trace-dir` capture against a soak run and
  emitting the `trace_to_profile.py` command that closes the loop back to the replay;
  `docs/switchyard.md` §11. CPU coverage in `tests/benchmarks/test_ops_observability.py`.
- **Client-disconnect abort in prefill** — `ff470e7`; `server/disconnect.py`, 12 tests.
  `abort_user` shielded as a tracked task (`e3a2019`).
- **Observability** — `/v1/stats.scheduler` + `requests.aborts`, invariant counted every pass
  (`78f29d3`); MoE counters (`e3a2019`); `fresh_admits_deferred` + replay `stall_frac` (`125da19`);
  `scheduler.prefill.match.*` (`8429411`); request traces (`8878659`).
- **1M gate** (restart persistence, eviction, NVMe restore) `31d606d`; **hidden-state parity**
  (52 layers, cosine ≥ 0.998840) `befcde6`; **1M direct-addressing** closed model-limited
  `be85ffa`; **MTP** NO-GO; **n-gram speculation** shipped behind `--speculative ngram` but
  measures 1.01–1.03x (`e4070da`), verify step 54.0 → 35.6 ms (`b84ecb7`).
- **CI** — `.github/workflows/cpu-checks.yml` (ruff + CPU unit tests + scheduler replay gate),
  `508ea32`; `docs/cpu-checks.md`. Full CPU test step with no GPU job live: **118 s wall**
  (`ee2e7bf`). Ruff ignore list down to **E702/E731/E741** (`22478ef`).
- **Pre-existing test issues (item 10)** — `pythonpath = ["."]` (`ee2e7bf`); the laguna TP fixture
  tolerates an already-set TP info, which was the real cause of the 6 ordering errors, not GPU
  contention (`0f6ff4b`).
- **`fork/main` fast-forward (user decision, done).** `fork/main` is at `9afd48c`, 0 ahead.
- **Soaks §W → §AB.** §W (`ca7e74b`) PASS with 9 invariant warnings; §X (`e3a2019`) PASS, 0 of
  1,541, disconnect defect found; §Y (`38617a7`) both fixes validated, **stage FAIL** on the
  livelock; §Z (`125da19`/`785a278`) PASS, livelock closed; §AA (`8429411`) PASS; **§AB
  (`670969f`) PASS on every criterion — the current baseline.** Drivers in-repo at
  `benchmarks/switchyard_soak/` (`f6ed0b5`).

## Open, ranked by value
Nothing here is a blocker, and nothing here changes the shipping recommendation for `8e064d4`.
Every line names why it does not affect production.

1. **Reclaim over-spill watch (§AA9.1).** The `8429411` reclaim arm spends some prefix cache to buy
   admission throughput. §AB says it is **stable, not worsening**: pressure releases 1,174 → 1,189
   (+1.3 %), spills 1,542 → 1,475 (−4.3 %), restores 479 → 421 (−12.1 %), 0 failures, and
   passthrough prefix reuse **recovered to 89.8 %**, above §Z's 88.9; stage drifted a further
   1.1 pp down (83.6 → 82.5) on a phase deliberately carrying more new tokens.
   *Deferred:* watch-only — two soaks show the cost flat and the arm paying for itself, with 0
   errors, 0 STALLED and 0 restore failures on both. `_RECLAIM_SCAN_DEPTH = 4` is the knob if a
   later soak ever shows reuse falling **with restores climbing**.
2. **The 512 bucket's aggregate prefill (§AB9.1).** §AB pushed 6.17 M new tokens in 2,694 s =
   2,290 tok/s against §AA's 2,431 (**−5.8 %**) and §Z's 2,357 (−2.8 %) — while the route that
   pushed the most new tokens per second (stage) got **faster** (+7.1 % effective, +17.5 % instant,
   p99 −14.0 %). The passthrough phase carried 19.8 % fewer new tokens at higher reuse and a median
   chunk `#new-token` of 2,454 against §AA's 4,486, which depresses a per-pass instant rate
   mechanically. The other place the bucket could show is the stage extend-cache gate (1.5 % vs
   §AA's 4.7 %) — a routing-mix observation, not a fault.
   *Deferred:* every acceptance criterion passed on both routes, nothing shows the signature of a
   slower kernel, and the revert is one environment variable
   (`FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS=512`) with no rebuild. Settle it with a repeat
   passthrough phase under that variable when a GPU slot is free.
3. **Serving-phase RAM floor 2.3 GiB (§AB9.2).** The worst since §Y, 0.3 GiB above the
   `SOAK_RAM_ABORT_GIB=2` watchdog, which did not fire. `SOAK_HOST_RAM_RESERVE_GB` now makes
   raising the reserve a one-liner.
   *Deferred:* a soak-harness margin, not a server defect — the watchdog exists precisely for this
   and the run completed with a graceful shutdown. Set `SOAK_HOST_RAM_RESERVE_GB=7` on the next
   soak and record it as a profile deviation.
4. **The load-phase watchdog is unproven under a cold load (§AB9.3).** §AB's load was warm
   (`READY after 29 s`, load-phase minimum 8.0 GiB), so the new floor was reported and cleared but
   never approached. §AA's cold start (85 s, 1.1 GiB) is the reproduction.
   *Deferred:* it can only fail *safe* — the worst case is an abort that would otherwise have been
   a WSL OOM restart. The next cold-cache start is its first real test.
5. **Seed both sides of the spec break-even gate.** Seeding the cost side alone is measured and
   negative. The fix is to prime `emit` too — from the probes' own acceptance (widen `_SEED_WIDTHS`
   so a probe usually *rejects*, which under greedy decoding is a true full-width sample) or by
   comparing against the `max_k + 1` ceiling while `emit` is a prior. **Not as GPU arms:** 2–3
   verify steps on a 79-token generation cannot resolve a 4 % effect against an 18 % baseline
   spread — extend `benchmarks/spec_engage_replay.py` with a **gate-policy axis** first.
   *Deferred:* speculation is **off by default** (`--speculative ngram` is opt-in and measures
   1.01–1.03x), so no production path executes the gate at all. The rest of the ticket is a fused
   multi-query extend kernel, which is a project, not a fix.
6. **A cross-pass match memo.** The per-pass memo is measured (40.8 %/40.9 % hits, ~83–101 K matched
   tokens per pass); the walk still scales with queue × prompt *within* a pass. `_admission_stalled`
   (`125da19`) is exactly the predicate for "nothing changed since the last pass".
   *Deferred:* the cost it would remove is not observable — busiest-process CPU is flat at
   107.9 / 108.9 / 107.9 / **108.8 %** median across four soaks with no sustained-100 % window, and
   `PrefillAdder.headroom` already bounds the walk. Doing it means enumerating every mutation that
   can reach the manager from outside a pass, and a missed one is exactly `ea7ed7c`'s stale
   `cached_len`.
7. **The reclaim pressure test does not charge `finishability_reservation`.** Deliberately left out
   of `8429411`: including it would make the test exact, but `_reclaim_soft_sessions_for_admission`
   runs on **every** arriving request and would spill idle conversations more eagerly.
   *Deferred:* no observed cost. Two soaks with the lock-delta term alone show 0 invariant
   violations, 0 restore failures, and the over-spill flat; the reservation term waits for a run
   that shows the omission costing something.
8. **`max_chunked_prefills = 8` still binds.** `fresh_admits_blocked_by_cap` per run: 388 (§X) →
   204 (§Y) → 301 (§Z) → 376 (§AA) → **185** (§AB).
   *Deferred:* goodput rose in every one of those runs, and §AB — the lowest count of the four
   post-livelock soaks — is also the run with the best stage prefill rate. This is evidence for the
   reservation arithmetic, not a demonstrated cost.
9. **16-way decode is at the hardware ceiling.** 74 % of the step is the PCIe expert gather at
   51–52 GB/s against a measured 52.9 GB/s link; working set ~1,417 expert-layer slots against 976
   in the pool. Attention, the MoE GEMV and Mamba-2 were all measured fine at batch 16.
   *Deferred:* the ceiling is the hardware. Only two unmeasured levers remain (`--moe-backend
   hybrid` at 16 lanes, and the 976-vs-1,417 slot deficit) and neither is a defect. **Do not
   re-litigate.** `decode16_2026-09-05.md` §0/§2/§7.
10. **Rebuild `_gguf` before deploying on Ada.** The guard in `9afd48c` turns a stale `.so` from a
    silent wrong-kernel-width into a loud refusal that names the cache directory to delete.
    *Deferred:* not applicable to this card — the Nemotron NVFP4 path does not touch the GGUF
    kernels, and the failure mode is now impossible to hit silently. It is a deployment step for an
    Ada box, not an open defect.
11. **Watch mean lanes per prefill batch every soak.** Stage 3.13 / 3.55 / 3.79 / 3.04 / **3.43**
    (§X→§AB), passthrough 4.88 / 4.53 / 4.43 / 4.62 / **4.09**. No trend, all far under ~5, 0 errors
    and 0 STALLED on every one.
    *Deferred:* a standing watch, not a ticket. Stage >~5 **together with** rising errors or p95 is
    the §R6/§R7 mode returning.
12. **Smaller, all in `tasks/todo.md` with evidence, none production-affecting:**
    `benchmarks/scheduler_replay.py` re-implements the loop's reclaim inline, so the `8429411`
    change is invisible to the gate (*deferred:* `tests/scheduler/test_reclaim_and_match_memo.py`
    plus the live soak cover it, and the gate is not the only evidence);
    `stopped_for_lane_cap` rotation is dead code on this model (*deferred:* unreachable, so it
    cannot misbehave); the 1,024-routed-id extend guard is a flashlib-LRU constraint applied to an
    LFU profile (*deferred:* raising it buys access to a path measured **slower**, and would cost a
    22-minute Triton JIT that must be a warmup pass); folding `tests/moe` + `tests/kernels` into CI
    is infeasible on a CPU-only runner (*deferred:* run locally, green on the tree); the ruff ignore
    list still carries E702/E731/E741 (*deferred:* style only); `bench_nvfp4_moe_kernels.py --gate`
    asserts inverted 2B1 targets (*deferred:* a benchmark gate, not a shipped path);
    `num_kv_splits_ptr` is passed and never dereferenced (*deferred:* dead argument, no behaviour);
    session-residency leftovers from the 1M gate (*deferred:* by-design spill-on-demand, documented
    in `docs/nemotron.md`); two stale worktrees predating this effort.

## How to run things
- **Soak**: `benchmarks/switchyard_soak/run.sh [tag] [duration]` — stage 20 m then passthrough
  20 m, c=16, server under `scripts/gpu_lock.sh` with `FREETOKEN_SCHEDULER_INVARIANT=warn`.
  It refuses to start below 26 GiB `MemAvailable`, holds a **load-phase** floor
  (`SOAK_RAM_LOAD_ABORT_GIB`, default 0.8) armed on the READY poll, and TERMs the server below
  `SOAK_RAM_ABORT_GIB` (2) while serving; `SOAK_HOST_RAM_RESERVE_GB` overrides
  `--host-ram-reserve-gb`. Everything lands in `runs/<tag>/` (gitignored). `SOAK_PHASES=""` runs
  the disconnect probe alone — both endpoint shapes, ~90 s of GPU. Grade with `analyze.py` (logs
  **and** `stats_*.json`) and `gaps.py`. **The current baseline to diff against is §AB.**
  Contract/e2e: `scripts/switchyard_e2e.sh contract|soak|agents` (`docs/switchyard.md` §8).
- **Oracle**: three phases in `docs/oracle.md` §"The three commands" — A FreeToken (you start
  the server; include `--enable-cache-report`), B llama.cpp (`--n-cpu-moe 14` at 131K/262K, **23**
  at 524K), C compare (CPU only). Sweep dimension is *length*, never depth.
  `--target-prompt-tokens 1044480` at the 1M rung. Verify prompt identity on CPU with
  `record --build-only` before taking the lock. Phase A's `ft serve` needs an absolute
  `.venv/bin/ft` inside a wrapper script — `gpu_lock.sh` does not run through the venv shim.
- **Ticket harnesses**: `scripts/gpu_lock.sh benchmarks/decode16/phaseE2.sh <outdir>` (~22 min);
  `benchmarks/bench_moe_prefill_gemm.py --variant tree deint fused prepass prepass2 --grid shipped
  --verify` (~1 min; prints the weight-byte floor, `%HBM`, padded rows and M-block count per
  `BLOCK_M`); `FREETOKEN_GPU_LOCK_WAIT=7200 scripts/gpu_lock.sh
  benchmarks/extend_moe/run_threshold.sh` (~5 min to m=96; the first m whose `m*top_k` crosses 1024
  costs **~22 min of Triton JIT**); `benchmarks/probe_spec_ngram_impl.py --sweep-k ...` (~2 min).
- **Replay / CPU gates**: `uv run benchmarks/scheduler_replay.py --gate` (5 profiles, CPU, ~4 s;
  the `stall_frac` column reproduces §Y5b); `benchmarks/spec_engage_replay.py` for anything about
  the speculation gate; `benchmarks/trace_replay.py` / `trace_to_profile.py` for real traffic. Full
  CPU test step: **118 s** with no GPU job live. CI runs ruff + the CPU test directories + the
  replay gate (`docs/cpu-checks.md`).
- **A/B hatches**: `FREETOKEN_DECODE_{KV_SPLITS,BLOCK_N,NUM_WARPS}`,
  `FREETOKEN_EXTEND_{BLOCK_M,BLOCK_N,NUM_WARPS,NUM_STAGES}`,
  `FREETOKEN_NVFP4_PREFILL_{BLOCK_M,BLOCK_N,BLOCK_KB,GROUP_M,NUM_WARPS,NUM_STAGES}`,
  `FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS` (bucket-boundary A/B, e.g. `=512`),
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
  `oom_score_adj=1000`, reaps the worker tree on exit. The **model load** is the peak — §AA saw
  `MemAvailable` hit 1.1 GiB during a cold one, which is what `SOAK_RAM_LOAD_ABORT_GIB` now bounds.
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
- Serving profile: see the Production launch checklist above. Needle gates go through
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
