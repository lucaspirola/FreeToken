# Nemotron 3.5 Lightning on a 5080 — 16-way Switchyard soak vs. the slot-reclaim fix

2026-09-04 · FreeToken at `3ac79ec` (working tree clean; the slot-reclaim fix under
test is `dcb617a`) · Switchyard `switchyard-server` / `switchyard-soak` built
2026-09-04 01:21 · RTX 5080 16 GiB, WSL2, 33 GiB host RAM (30 GiB available at
launch), GPU held exclusively through `scripts/gpu_lock.sh` for the whole server
lifetime · `piro-board-embedder.service` stopped.

**Verdict: FAIL** — but not on the bug this run was meant to retest.

* The 3F slot-reclaim fix **holds**. The pool reached its 96/96 pre-crash signature
  **41 times** with zero `LinearStatePool exhausted`; the previous attempt died on the
  *first* one, at t≈3 min.
* The `/health` 503 + bounded-shutdown fixes **hold**. `/health` answered 503 with the
  dead worker's name 11 s after the death, the process exited instead of wedging in
  *"Waiting for background tasks to complete"*, and **no soak interval reported
  `STALLED`** — in-flight requests were cancelled rather than hanging to the 600 s
  client timeout.
* A **different** fatal ended the run at t≈611 s: KV *page* exhaustion raised from
  `CacheManager.committed_pages_required`, killing the scheduler process. Diagnosis in
  §4; it is the same failure shape as 3F (a transient resource shortage raised as a
  fatal instead of applying back-pressure), one currency over.

---

## 1. Exact commands

Server (P2 profile from `docs/nemotron.md` plus the Switchyard serving-compliance flags
from `docs/switchyard.md` §1), started under the GPU lock and held for the whole run:

```bash
systemctl --user stop piro-board-embedder.service     # was already inactive

# scratchpad/soak/serve.sh
FREETOKEN_PIN_BUDGET_GB=17 \
uv run ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 127.0.0.1 --port 1919 \
  --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536 \
  --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0 \
  --attention-backend triton --moe-backend offload --moe-cache-auto \
  --moe-cache-policy lfu \
  --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 \
  --enable-cache-report --served-model-name nemotron-3.5-lightning \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder \
  --force-nonempty-content --max-output-tokens 16384

# scratchpad/soak/run.sh, launched as:
FREETOKEN_GPU_LOCK_WAIT=300 scripts/gpu_lock.sh scratchpad/soak/run.sh
```

Soak (through the router the harness starts on :4000):

```bash
scripts/switchyard_e2e.sh soak --base-url http://127.0.0.1:1919 \
  --model nemotron-3.5-lightning --duration 20m --workdir <scratch>/soakA
# -> switchyard-soak --base-url http://127.0.0.1:4000 --model switchyard/passthrough \
#      --duration 20m --concurrency 16 --max-output-tokens 256 --prompt-bytes 16384 \
#      --context-window-tokens 131072 --max-error-rate 0 --request-timeout 600 \
#      --scenario prefix-reuse --scenario growing-conversation --scenario tool-call-burst \
#      --scenario large-tool-catalog --scenario long-context
```

Deviation from the previous (crashed) attempt's `serve.sh`, deliberate: `--moe-pageable-gpu`
dropped and `--host-ram-reserve-gb` 3 → 6, per `docs/nemotron.md` P2 and the handover host
rule — at `FREETOKEN_PIN_BUDGET_GB=17` every expert layer is pinned, so pageable-GPU only
costs the decode CUDA graphs. Everything else is identical.

The stage route and the 10 m resilience set were queued behind this run and **never
executed**: the upstream was already dead, so soak B was aborted after 11 s.

## 2. Timeline and duration

| | |
|---|---|
| server start → `{"status":"ok"}` | **26 s** (weights 3 s, 18.3 GB expert banks 14 s, KV 262 144 tokens = 0.80 GiB, 2.19 GiB free VRAM after init, graphs captured at bs 1–4) |
| soak A wall clock | **1 200.7 s** (ran the full 20 min against a dead upstream after t≈611 s) |
| scheduler death | **t ≈ 611 s** (16:02:16 local, soak interval 11) |
| `/health` first 503 | 16:02:27 — **11 s** after the death |
| API process exit | ~16:02:40, after `Cancel 16 running task(s), timeout graceful shutdown exceeded` |
| driver shutdown of an already-dead server | `shutdown_seconds=0` |
| GPU at end | **0 MiB** |

## 3. Request counts and rates (soak A, `switchyard/passthrough`)

`summary.json`:

| metric | value |
|---|---|
| requests | 13 030 |
| successes | **612** |
| failures | **12 418** |
| error rate | 95.30 % (limit 0 %) |
| error kinds | `http_502` 12 416, `stream_error` 2 |
| health checks / failures | 20 / 0 *(router's `/health`, not FreeToken's)* |
| metrics checks / failures | 20 / 0 |
| invalid-request canaries | 3 / 0 failures |
| detected server restarts | 0 |
| `passed` | **false** — `request error rate 95.3031% exceeded the 0.0000% limit` |

Every one of the 612 successes landed **before** the crash; every failure after it. The
602 s of healthy service were clean:

```
  60s reqs=68  errors=0  p50=10,801 ms  p95=28,367 ms  status=OK
 120s reqs=47  errors=0  p50=22,657 ms  p95=23,706 ms  status=OK
 ...
 600s reqs=48  errors=0  p50=16,829 ms  p95=19,429 ms  status=OK
 660s reqs=962 errors=962  p50=755 ms   <- backend gone; 502 in 755 ms, NOT stalled
 720s..1200s  ~1,270 req/interval, all http_502 at ~755 ms
```

Per-scenario successes before the death: `prefix-reuse` 128, `growing-conversation` 128,
`tool-call-burst` 128, `large-tool-catalog` 114, `long-context` 114.

The 755 ms constant after the crash is the router's connect-refused path — the useful
signal is that it is **not** the 600 s client timeout the pre-fix server produced. No
interval was ever marked `STALLED`.

## 4. Throughput while healthy

From the server's own batch log over the healthy window (90 decode-batch samples,
348 prefill samples):

| | aggregate tok/s | per-stream tok/s |
|---|---|---|
| all decode batches (mean 12.4 running) | median 94.8, mean 120.0 | median 8.19 |
| `#running-req ≥ 12` (n=70) | median 112.3, mean 135.1 | median 7.28 |
| `#running-req == 16` (n=37) | **median 150.5**, mean 163.9, max 513.5 | **median 9.41** |

Prefill: median 1 395 tok/s instant, max 9 428 tok/s (mixed traffic — short prompts drag
the median down; the clean-prefill figure remains ~3 000 tok/s at 131 K).

KV/state occupancy: token usage median 0.58, max 0.97; mamba slots median 85/96, max
**96/96**. KV grew 65 536 → 131 072 → 196 608 → 262 144 tokens and shrank back twice.

Context: the 2B4 decode-only bench recorded 168.2 tok/s aggregate at 16 concurrent with
LFU. 150 tok/s under mixed Switchyard traffic (with prefills, tool catalogs and 118 K
long-context prompts interleaved) is consistent with it.

## 5. The failure — KV page exhaustion raised as a fatal

`scratchpad/soak/server.log:1790`:

```
Prefill batch, #new-seq: 14, #new-token: 7168, #cached-token: 0, token usage: 0.99,
  #mamba-slot: 86/96, mamba usage: 0.90, #running-req: 0, #queue-req: 16, ...
Session auto:switchyard:6ccc3610... expired after idle timeout
Closed session auto:switchyard:6ccc3610...; retained KV is now reclaimable
   (x16 session closures in the same two seconds)
Process freetoken-TP0-scheduler:
Traceback (most recent call last):
  ...
  File "python/freetoken/scheduler/scheduler.py", line 606, in normal_loop
    forward_input = self._schedule_next_batch()
  File "python/freetoken/scheduler/scheduler.py", line 2065, in _schedule_next_batch
    forward_input = self._prepare_batch(batch)
  File "python/freetoken/scheduler/scheduler.py", line 1943, in _prepare_batch
    required = self.cache_manager.committed_pages_required(batch.reqs)
  File "python/freetoken/scheduler/cache.py", line 159, in committed_pages_required
    raise RuntimeError(
RuntimeError: batch needs 6061 pages but only 3605 are physically allocatable and
             0 logical pages remain
ERROR  Backend supervisor: backend worker freetoken-TP0-scheduler exited
ERROR  Backend worker is gone and cannot be restarted; stopping the API server
```

### Reading the numbers

`page_size == 1`, so pages are tokens. `0 logical pages remain` means the growable pool
was **fully committed** at its `--num-tokens 262144` ceiling — this is not a failed
growth, it is genuine exhaustion. Of those 262 144 pages only 3 605 were free or
evictable; the rest were held by live requests and by *locked* prefixes (admitted
requests' matched prefixes plus session leases, which `full_evictable_size` excludes).
The batch wanted 6 061 new pages.

### Why the batch was allowed to exist

`PrefillAdder.try_add_one` (`python/freetoken/scheduler/prefill.py:199`) has **two**
paths and only one of them checks KV availability:

```python
def try_add_one(self, pending_req, chunk_limit=None):
    if self.token_budget <= 0:
        return None
    if chunked_req := pending_req.chunked_req:
        return self._add_one_req(...)            # <-- no availability check at all
    if resource := self._try_allocate_one(pending_req):   # <-- gated
        ...
```

`_try_allocate_one` gates a *fresh* admit on
`estimated_len + reserved_size > cache_manager.available_size`, and charges the whole
remaining prompt plus `output_len` — strictly more conservative than the chunk actually
allocated. A **continuation** of an already-chunked prompt skips that entirely, on the
premise recorded in `_reclaim_for_blocked_prefill` that *"a continuation already owns its
resources"*. It owns its table slot, its GDN state slots and its already-forwarded pages
— but **not** the pages for its next chunk, which are allocated later, in
`allocate_paged`, after `committed_pages_required` has already decided the batch is
impossible.

Two things make this reachable rather than theoretical:

1. `schedule_next_batch` puts continuations at the **head** of `pending_list`
   (`chunked_list + remaining`, `prefill.py:347-351`), so a prefill pass admits ungated work
   *first*.
2. `PrefillAdder` is constructed fresh every pass, so `reserved_size` is a *per-pass*
   reservation. Between two chunks of the same prompt, the pages that prompt still needs
   are invisible to `available_size`, and later fresh admits and decode growth are free
   to consume them.

Only `token_budget` (`--max-prefill-length 8192`) bounds such a batch — which matches the
observed 6 061 exactly.

The trigger conditions in the log are all present in the seconds before the death: token
usage 0.99, `#queue-req: 16`, 16 sessions expiring at once (each `Closed session …;
retained KV is now reclaimable` converts a lease back to evictable, so the shortage was
*transient* — one or two scheduler iterations of back-pressure would have cleared it).

### Why it is fatal rather than back-pressure

`committed_pages_required` is a growth *planner*; when the plan does not fit the logical
pool it raises. Nothing in `normal_loop` catches it, so the scheduler process dies — the
identical anti-pattern the 3F write-up recorded for `LinearStatePool`: *"a
cache-management step must never be able to kill the scheduler"*.

Note the check only runs under `--kv-grow-step-tokens`; without growable KV the same
shortage would surface later, inside `allocate_paged`.

### Proposed fix (not implemented here — it is a scheduler change, not a one-liner)

Give an in-flight chunked prompt a **standing** KV reservation for its unforwarded
remainder, so that (a) `available_size` reflects it across passes and later admissions
cannot spend it, and (b) a continuation whose next chunk cannot be backed is *capped or
deferred* — routed through the existing escalation ladder (evict unlocked prefixes →
spill the LRU idle session lease, i.e. `_reclaim_for_blocked_prefill`, which today
`continue`s past every continuation) instead of raising. The SWA currency already does
exactly this shape in `_add_one_req` via `reserved_swa` + `max_end`; the KV currency
needs the same treatment. `committed_pages_required` should then be unreachable in
anger, and should still never be the thing that kills the process.

## 6. What this run proves about `dcb617a`

| pre-fix behaviour (2026-09-04, `ec54e21`) | this run |
|---|---|
| `LinearStatePool exhausted: need 1, have 0` on the first `mamba usage 1.00` (t≈3 min) | 41 batches at `#mamba-slot: 96/96` / `mamba usage: 1.00`, **0 exhaustions** |
| `/health` answered `{"status":"ok"}` for the whole 9-minute stall | 503 with `backend worker freetoken-TP0-scheduler exited`, 11 s after the death |
| frontend hung 38 min in *"Waiting for background tasks to complete"*, VRAM + 18.3 GB banks held | bounded shutdown, `Cancel 16 running task(s)`, process gone in ~24 s, GPU 0 MiB |
| soak intervals `STALLED`, errors reported as 600 s timeouts | no `STALLED` interval; failures are 755 ms 502s |

The soak's own verdict is FAIL, and the Phase 3 gate (0 errors) is not met — but the
regression it was written for is closed, and the blocker is now a distinct, newly
characterised KV-page-accounting defect.

## 7. Artifacts

Scratchpad (survives Claude restarts, not a WSL restart):
`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/soak/`

* `serve.sh`, `run.sh` — the exact driver
* `server.log` — FreeToken log incl. the traceback at line 1790
* `driver.log`, `health_bad.log` — 62 non-ok `/health` samples, the first at 16:02:27
* `soakA.log`, `soakA/results-switchyard_passthrough/{summary,config}.json`,
  `intervals.csv`, `errors.jsonl`

---

# Rerun against `fad1fc4` (chunked-prefill KV back-pressure)

2026-09-04 16:32–17:29 · FreeToken at `befcde6` (working tree clean; the fix under test
is `fad1fc4`, its parent `c4486b6` is the run above) · same RTX 5080 / WSL2 / 30 GiB
available at launch · GPU held for the whole server lifetime through
`scripts/gpu_lock.sh` · `piro-board-embedder.service` inactive.

**Verdict: FAIL overall — but the defect this rerun was written for is closed.**

* `fad1fc4` **holds**. 50 minutes of load, 1,941 successful requests, **zero**
  `committed_pages_required`, zero `LinearStatePool exhausted`, zero tracebacks. The
  scheduler never died; `/health` answered `{"status":"ok"}` on every one of ~300
  watchdog samples (`health_bad.log` was never created).
* Soak A / `switchyard/passthrough` (20 m) **PASS**, soak B / resilience set (10 m)
  **PASS** — the 10 m set that never ran last time.
* Soak A / `switchyard/stage` (20 m) **FAIL**: 4 `timeout` errors (the soak's 600 s
  client timeout) on `long-context`, and 4 `STALLED` intervals. No crash, no error from
  the server — it simply could not drain the queue.
* Diagnosed in §R5: the *cap* `fad1fc4` added is charged against `reserved_size`, which
  books each admitted request's **whole remaining prompt**, while the batch it gates only
  allocates its **chunk**. One long in-flight prompt therefore reserved the pool away from
  every other continuation in the pass. Fixed (§R6) and re-run: the stage route then
  **passes** with 2.0x the requests.

---

## R1. Exact commands

Identical server line to §1 above (`scratchpad/soak2/serve.sh` is a byte copy of
`scratchpad/soak/serve.sh`), started under the GPU lock and held for all three soaks:

```bash
systemctl --user stop piro-board-embedder.service   # already inactive

FREETOKEN_GPU_LOCK_WAIT=300 scripts/gpu_lock.sh scratchpad/soak2/run.sh
```

`scratchpad/soak2/run.sh` = the previous `run.sh` plus a 5 s resource sampler
(`sample.sh` → `resources.csv`: per-process CPU, RSS, GPU MiB), and with soak A left on
its **default route list**, so the stage route that was never reached last time runs:

```bash
# soak A -- 20 m PER ROUTE: switchyard/passthrough, then switchyard/stage
scripts/switchyard_e2e.sh soak --base-url http://127.0.0.1:1919 \
  --model nemotron-3.5-lightning --duration 20m --workdir <scratch>/soak2/soakA

# soak B -- 10 m, resilience set (context-overflow, failure-pressure,
#           client-cancellation), passthrough
scripts/switchyard_e2e.sh soak --base-url http://127.0.0.1:1919 \
  --model nemotron-3.5-lightning --duration 10m \
  --scenario-set resilience --route switchyard/passthrough --workdir <scratch>/soak2/soakB
```

## R2. Timeline

| | |
|---|---|
| server start → `{"status":"ok"}` | **37 s** |
| soak A `switchyard/passthrough` | 1,207 s, **PASS** |
| soak A `switchyard/stage` | 1,310 s, **FAIL** (`error rate 1.4388% > 0%`) |
| soak B resilience / passthrough | 635 s, **PASS** |
| scheduler deaths | **0** |
| `/health` non-ok samples | **0** (over ~50 min at 10 s) |
| driver stop of a live server | `shutdown_seconds=4` |
| GPU at end | **0 MiB**, no leftover venv python |

## R3. Request counts

| | passthrough (20 m) | **stage (20 m)** | resilience (10 m) |
|---|---|---|---|
| requests | 1,219 | 278 | 448 |
| successes | 1,219 | 274 | 448 |
| failures | **0** | **4** (`timeout`) | **0** |
| error rate | 0 % | 1.4388 % | 0 % |
| p50 / p95 / p99 ms | 8,717 / 62,613 / 121,217 | 32,033 / 392,867 / 600,001 | 15,568 / 88,458 / 89,135 |
| STALLED intervals | 0 | **4** (t=720–960 s) | 1 (the first, warm-up) |
| health / metrics checks failed | 0 / 0 | 0 / 0 | 0 / 0 |
| invalid-request canaries failed | 0 / 3 | 0 / 3 | 0 / 1 |
| detected server restarts | 0 | 0 | 0 |

Per scenario — passthrough: prefix-reuse 256, growing-conversation 242, tool-call-burst
241, large-tool-catalog 240, long-context 240. Stage: prefix-reuse 60, growing-conversation
59, tool-call-burst 59, large-tool-catalog 54, long-context 42 **+ 4 failures, all
long-context**. Resilience: context-overflow 160, client-cancellation 144,
failure-pressure 144.

## R4. Throughput (server batch log)

| phase | decode batches | aggregate tok/s @ `#running-req == 16` | per-stream tok/s | prefill instant tok/s (median) |
|---|---|---|---|---|
| passthrough | 180 | median **161.4**, mean 172.6, max 441.4 (n=77) | **10.09** | 1,496 |
| stage | 134 | median 81.6, mean 80.3 (n=8) | 5.10 | 1,637 |
| resilience | 191 | median **319.5**, mean 313.1 (n=155) | **19.97** | 1,499 |

The passthrough figure (161 tok/s at 16-way, 10.1 per stream) is the like-for-like
comparison with the previous run's 150.5 / 9.41 and with the 2B4 decode-only bench's
168.2 — the fix costs nothing on this workload (1.010 rps vs 1.017 rps pre-fix).
KV grew 65,536 → 131,072 → 196,608 → 262,144 tokens and stayed there.

## R5. Back-pressure: did the new path engage?

**Yes for the reclaim half, and the fatal is gone — but the cap half was too tight.**

* Reclaim: 749 `Released soft session … KV protection (admission pressure)` (the KV-shaped
  reclaim `_reclaim_for_blocked_prefill` drives, which `fad1fc4` newly reaches *for a
  blocked continuation*) plus 186 `(GDN state-slot pressure)`. The previous run logged 52
  and 104 in its 602 healthy seconds; per second that is 0.25/s vs 0.086/s — the demand
  signal fires roughly 3x more often now that a continuation can raise it.
* Pressure survived: 44 batches at `#mamba-slot: 96/96` (`mamba usage 1.00`), token usage
  peaking at 1.00 on the prefill gauge, and **no** `committed_pages_required`. The
  pre-fix run died the first time this combination appeared under a full pool.
* **No spin.** `resources.csv` (5 s samples of every FreeToken venv process): the busiest
  process sits at **median 106 % CPU in every phase** — 106.3 % while passthrough is
  healthy, 106.1 % through the stage collapse (17:04–17:12, when whole 60 s intervals
  completed 0 requests), 108.8 % during resilience. A deferred prefill does **not** burn
  the loop: the pass falls through to decode and the scheduler's cost is unchanged.
* The deferral itself is **not observable in the log** — `_add_one_req` returns `None`
  silently and `/v1/stats` has no scheduler counter. Everything above is circumstantial
  plus the CPU-only A/B in §R6. A `deferred_prefill_chunks` counter on `/v1/stats` would
  make the next soak decisive; ticket it.

### The stage route's failure

Not a crash and not KV exhaustion: **prefill starvation**. Through the collapse the log is
a monotonous run of

```
Prefill batch, #new-seq: 1, #new-token: 512, #cached-token: 0, token usage: 0.49,
  #mamba-slot: 7/96, mamba usage: 0.07, #running-req: 0, #queue-req: 16,
  input throughput (token/s): 1782.21 instant
```

— one lane, 512 tokens per pass, **with half the KV pool free**. Two mechanisms compose:

1. `chunk_limit = adder.token_budget // min(waiting, available_lanes)`
   (`prefill.py:337`, interleaved mode) is **8192 // 16 = 512** whenever 16 requests are
   queued. That is per-lane fair sharing and only pays off if the pass actually admits 16
   lanes. *(Pre-existing; present in the pre-fix run too — 60 of its 348 prefill batches
   are `#new-seq: 1, #new-token: 512`.)*
2. `fad1fc4`'s cap then stopped the pass at the first or second lane — see below. Mean
   lanes per prefill batch in the stage phase: **1.56**.

The stage route is where this bites because it doubles the prefill demand: 2.19 M new
prompt tokens for 278 requests (7,876 new tokens each) against 1.99 M for 1,219 requests
(1,637 each) on passthrough — the `stage_router` posts a classifier call per user turn,
whose prompt (schema appended, thinking off) is a different prefix, so its prefix reuse is
73.6 % against passthrough's 87.6 %. A 118 K-token `long-context` prompt advancing 512
tokens per pass at ~1,780 tok/s needs ~230 passes; sixteen of them do not fit in 600 s.

## R6. Follow-up fix: charge the cap in pages, not in reserved prompt

`fad1fc4` capped the chunk with

```python
kv_pages = max(0, cache_manager.available_size - self.reserved_size) // kv_ps
```

`reserved_size` is the sum over reqs admitted this pass of **`remain_len + output_len`** —
the whole rest of the prompt. That figure is the right one for `_try_allocate_one`'s
admission policy (don't admit a fresh prompt the pool cannot finish), but it is the wrong
currency for this cap, whose only job is to keep `committed_pages_required` satisfiable —
and that check demands exactly the batch's **per-chunk page deltas**. So the first long
continuation booked the whole pool and every peer behind it was capped to nothing.

CPU-only A/B (`scratchpad/soak2/lane_ab.py`, no GPU, 6 chunked continuations, 700 free
pages, a full 6-lane batch needing 600):

| tree | lanes admitted | tokens forwarded |
|---|---|---|
| `c4486b6` (before `fad1fc4`) | 6 / 6 | 600 |
| `befcde6` (`fad1fc4`) | **2 / 6** | **200** |
| `befcde6` + this fix | 6 / 6 | 600 |

The fix adds `PrefillAdder.reserved_pages` — initialised to the decode in-flight tokens
and charged one **page span per admitted chunk** — and caps against that instead. It is
strictly tighter than what `committed_pages_required` tests, so the fatal stays closed;
`tests/scheduler/test_chunked_prefill_kv_backpressure.py` grows a sixth case
(`test_one_long_continuation_does_not_reserve_the_pool_away_from_its_peers`, which fails
with 2 lanes / 12 tokens on the old cap). Files: `python/freetoken/scheduler/prefill.py`,
`tests/scheduler/test_chunked_prefill_kv_backpressure.py`.

### Stage route re-run with the fix (17:37–18:03, `scratchpad/soak3/`)

Same server line, `--route switchyard/stage --duration 20m`:

| | `befcde6` | `befcde6` + fix |
|---|---|---|
| verdict | **FAIL** (1.4388 % errors) | **PASS** (0 errors) |
| requests / successes / failures | 278 / 274 / **4** | **471 / 471 / 0** |
| STALLED intervals | 4 | 1 (t=480 s) |
| p50 / p95 / p99 ms | 32,033 / 392,867 / 600,001 | 29,104 / 200,742 / 257,441 |
| mean lanes per prefill batch | 1.56 | **2.37** |
| new prompt tokens prefilled | 2.19 M (1,814 tok/s) | 2.72 M (1,877 tok/s) |
| prefix reuse (cached / (new+cached)) | 73.6 % | **83.8 %** |
| batches at `#mamba-slot: 96/96` | 27 | 34 |
| pressure episodes (usage ≥ 0.98, queue > 0) | 0 | **13** |
| `committed_pages_required` / tracebacks | 0 | **0** |
| busiest process CPU (median) | 106.0 % | 106.7 % |

2.0x the requests served, the timeouts gone, and the back-pressure path exercised
*harder* (13 near-full-pool episodes, 34 batches at 96/96) without a fatal.

## R7. What is still open

* `chunk_limit = token_budget // waiting` divides the 8,192-token budget by the **queue
  depth**, not by the lanes the pass can actually admit, so 16 queued requests cost a 16x
  smaller chunk even when only two lanes are admitted. Pre-existing, and it is what leaves
  the stage route's p95 at 200 s. Ticket: size the interleave share by the lanes the pass
  will really seat, or floor it (a chunk below ~2 K tokens is pure launch overhead).
* No scheduler-side observability for the deferral: add a `deferred_prefill_chunks` /
  `capped_prefill_chunks` counter to `/v1/stats` so a soak can prove engagement directly.
* `_try_allocate_one` charges a fresh admit `remain_len + output_len` against
  `available_size`, so at 118 K-token prompts only one or two can be in flight in a 262 K
  pool regardless of how much of the prompt this pass would forward. Same "reserve the
  whole prompt" shape as R6, one layer up; worth revisiting for long-context concurrency.

## R8. Artifacts

`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/`

* `soak2/{run.sh,serve.sh,sample.sh}` — driver; `soak2/server.log`, `driver.log`,
  `resources.csv`, `soakA.log`, `soakB.log`, `soakA/results-switchyard_{passthrough,stage}/`,
  `soakB/results-switchyard_passthrough/`
* `soak2/analyze.py` — the batch-log parser behind §R4/§R5; `soak2/lane_ab.py` — the
  CPU-only lane A/B of §R6
* `soak3/` — the stage-route re-run with the fix (same layout)

---

# Run against `81ab30e` (fresh-admit gate: finishability + this chunk)

2026-09-04 21:00–21:54 local · FreeToken at **`81ab30e`** ("Scheduler: gate fresh admits on
finishability + this chunk, continue past refusals"), working tree clean apart from the
untracked `benchmarks/bench_multi_needle.py` · same RTX 5080 / WSL2 host, 30 GiB available at
launch · one `scripts/gpu_lock.sh` holding the GPU for the whole server lifetime ·
`piro-board-embedder.service` inactive · server line byte-identical to §R1
(`scratchpad/soak5/serve.sh`) · **stage route first, then passthrough**, 20 min each at
`--concurrency 16`, default scenario set.

**Verdict: FAIL on both routes.** No crash, no fatal, no traceback — the scheduler simply
stops producing batches for eight to ten minutes at a time, on a pool it has filled to
`token usage: 1.00`, and the soak's 600 s client timeout collects the casualties.

| | previous best (`befcde6` + §R6 fix, `soak3`) | **`81ab30e`** |
|---|---|---|
| stage 20 m | **PASS** 471 req / 0 err / 1 STALLED | **FAIL** 268 req / **15 timeouts** (5.5970 %) / **7 STALLED** |
| passthrough 20 m | (soak2, `befcde6`) PASS 1,219 / 0 / 0 | **FAIL** 720 req / **16 timeouts** (2.2222 %) / **10 STALLED** |
| stage p50 / p95 / p99 ms | 29,104 / 200,742 / 257,441 | 28,662 / **600,001** / 600,002 |
| stage mean lanes per prefill batch | 2.37 | **4.71** |
| stage prefill instant tok/s (median) | 1,877 | **2,938** |
| passthrough decode at `#running-req == 16` | 161.4 agg / 10.09 per stream | 158.4 agg / 9.90 per stream |
| `committed_pages_required` / tracebacks | 0 / 0 | **0 / 0** |

The throughput half of the commit message reproduces: while `81ab30e` is scheduling it seats
**2.0x the lanes** and prefills **1.6x faster** than the tree that passed. It is the *time
spent not scheduling* that fails the soak.

## S1. Exact commands

```bash
systemctl --user stop piro-board-embedder.service    # already inactive
FREETOKEN_GPU_LOCK_WAIT=300 scripts/gpu_lock.sh <scratch>/soak5/run.sh
```

`run.sh` starts `serve.sh` (identical to `scratchpad/soak3/serve.sh`), waits for
`/health` == ok, runs a 10 s upstream-`/health` watchdog and the 5 s per-process resource
sampler, then:

```bash
scripts/switchyard_e2e.sh soak --base-url http://127.0.0.1:1919 \
  --model nemotron-3.5-lightning --duration 20m --concurrency 16 \
  --route switchyard/stage       --workdir <scratch>/soak5/soakStage
scripts/switchyard_e2e.sh soak --base-url http://127.0.0.1:1919 \
  --model nemotron-3.5-lightning --duration 20m --concurrency 16 \
  --route switchyard/passthrough --workdir <scratch>/soak5/soakPass
```

### The aborted first attempt (20:47–20:57), for the record

An earlier launch of the identical driver was killed nine minutes in when the controlling
Claude Code process was killed; the server went down with it (clean FastAPI shutdown in the
log). Its 9 minutes are **clean**: 214 prefill and 52 decode batches, **0** `Traceback`,
**0** `committed_pages_required`, **0** `LinearStatePool exhausted`, **0** oversize
("can never be admitted") warnings, and `health_bad.log` never created. Archived at
`scratchpad/soak4_killed/`. It contains no evidence either way about the stalls, which begin
at t≈10 min.

## S2. Timeline

| | |
|---|---|
| server start → `{"status":"ok"}` | **21 s** |
| soak stage | 1,736 s wall (20 m + drain), **FAIL** (`error rate 5.5970% > 0%`) |
| soak passthrough | 1,217 s, **FAIL** (`error rate 2.2222% > 0%`) |
| scheduler deaths / detected restarts | **0 / 0** |
| `/health` non-ok samples | **0** (40 soak checks + ~5 min of 10 s watchdog; `health_bad.log` absent) |
| metrics / invalid-request canaries | 0 / 0 failures |
| driver stop of a live server | `shutdown_seconds=3` |
| GPU at end | **0 MiB**, no leftover venv python |

## S3. Request counts

| | **stage (20 m)** | **passthrough (20 m)** |
|---|---|---|
| requests / successes / failures | 268 / 253 / **15** | 720 / 704 / **16** |
| error rate | **5.5970 %** | **2.2222 %** |
| error kinds | `timeout` x15 | `timeout` x16 |
| failing scenario | **long-context, all 15** | **long-context, all 16** |
| p50 / p95 / p99 ms | 28,662 / 600,001 / 600,002 | 7,261 / 81,960 / 600,003 |
| requests/s | 0.154 | 0.591 |
| STALLED intervals (of 20) | **7** | **10** |

Per scenario — stage: prefix-reuse 58, growing-conversation 57, tool-call-burst 57,
large-tool-catalog 48, long-context 33 **+15 failures**. Passthrough: prefix-reuse 144,
growing-conversation 144, tool-call-burst 144, large-tool-catalog 144, long-context 128
**+16 failures**.

## S4. Throughput while it is scheduling

| phase | prefill batches | mean lanes/batch | median `#new-token` | prefill instant tok/s (median) | decode batches | decode agg tok/s | per stream |
|---|---|---|---|---|---|---|---|
| stage | 370 | **4.71** | 7,502 | **2,938** | 97 | 41.4 median (90.4 at 16-way, n=10) | 5.65 |
| passthrough | 252 | **5.18** | 3,995 | 1,502 | 100 | 158.0 median (158.4 at 16-way, n=49) | **9.90** |

Prefix reuse: stage 74.7 % (2.03 M new / 5.99 M cached), passthrough 88.5 % (1.08 M / 8.33 M).
KV grew 65,536 → 262,144 in three steps and stayed there. Busiest FreeToken process:
**median 109.0 % CPU** over 588 five-second samples — the same ~106–109 % the healthy runs
show, so the stall is not distinguishable by CPU (see S5 for what it actually is).
Peak GPU 14,415 MiB.

## S5. The failure: the scheduler stops emitting batches for 8–10 minutes at a time

Not a crash and not KV exhaustion-as-a-fatal. **Gaps in the batch log**, measured as the
wall-clock between consecutive `Prefill batch` / `Decode batch` lines:

| phase | gaps > 30 s | total silent time | share of the phase |
|---|---|---|---|
| stage (1,889 s) | 21:11:12 **+492 s**, 21:23:19 **+515 s** | **1,007 s** | **53 %** |
| passthrough (1,326 s) | 21:34:51 **+624 s**, 21:48:00 +72 s | **696 s** | **52 %** |

Every gap opens on the same picture — a pool at 1.00 with work queued and almost nothing
running:

```
21:11:08 Decode batch,  #running-req: 2, #token: 261953, token usage: 1.00, #queue-req: 14
21:11:09 Prefill batch, #new-seq: 1, #new-token: 104, token usage: 1.00, #running-req: 1, #queue-req: 14
                                    <-- 492 s of nothing -->
21:23:18 Decode batch,  #running-req: 1, #token: 262124, token usage: 1.00, #queue-req: 10
21:23:19 Prefill batch, #new-seq: 1, #new-token: 91,  token usage: 1.00, #running-req: 0, #queue-req: 10
                                    <-- 515 s of nothing -->
21:34:51 Prefill batch, #new-seq: 1, #new-token: 1435, token usage: 1.00, #running-req: 0, #queue-req: 16
                                    <-- 624 s of nothing -->
21:45:15 Prefill batch, #new-seq: 16, #new-token: 8192, #cached-token: 1651608, token usage: 0.72, #running-req: 14
```

**The scheduler loop is alive and busy the whole time.** `py-spy dump` on the core process
(pid 2830479) during the 624 s passthrough gap, five samples, four of them identical:

```
Thread 2830479 (active): "MainThread"
    fast_compare_key   (freetoken/kernel/radix.py:20)
    get_match_len      (freetoken/kvcache/radix_cache.py:85)
    _walk              (freetoken/kvcache/hybrid_radix_cache.py:299)
    match_prefix       (freetoken/kvcache/hybrid_radix_cache.py:88)
    match_req          (freetoken/scheduler/cache.py:121)
    _try_allocate_one  (freetoken/scheduler/prefill.py:118)
    try_add_one        (freetoken/scheduler/prefill.py:360)
    schedule_next_batch(freetoken/scheduler/prefill.py:571)
    _schedule_next_batch / normal_loop / run_forever
```

with `--locals` on the `schedule_next_batch` frame:

```
lane_cap: 0        seatable_lanes: 2      reqs: []        refusals: 2
blocked_fresh: False                      stopped_for_lane_cap: False
  _try_allocate_one  chunk_limit: 4096
  match_req          input_len: 117999    (radix walk: prefix_len 8207, match_len 6287)
```

So every pass: walk the pending list, run a **full 118 K-token radix prefix walk per fresh
candidate**, refuse all of them, return `None`, repeat. `_reclaim_for_blocked_prefill` is
called on every `None` (scheduler.py:2076) and returns False — it has nothing left to
release. The one thing it did log in the window is the giveaway:

```
21:44:52 WARNING Cold restore for session ... failed (AssertionError('Eviction did not free enough space.'))
```

Each stall ends only when *something external* frees KV: the automatic session idle timeout
(178 `expired after idle timeout` / `retained KV is now reclaimable` pairs per phase) or the
in-flight long request finally finishing. 21:45:15's recovery batch seats 16 lanes at once
against a pool that has fallen to 0.72.

### Why `81ab30e` produces this and its parent did not

`fad1fc4`+§R6 charged a fresh admit's **whole remaining prompt** against `available_size`, so
the number of long prompts in flight was bounded by what the pool could actually finish; that
is the property §R7 complained about ("caps long-context concurrency at 1–2 in a 262 K pool")
and it is also what kept the pool from being consumed by prompts that cannot complete.
`81ab30e` replaces it with two checks — finishability against `cache_manager.max_size` minus
`inflight_prefill_size`, and this pass's chunk against `available_size`. The finishability
budget is **the whole pool**: it does not subtract the KV held by requests already decoding,
nor the retained/locked session prefixes that `_reclaim_for_blocked_prefill` cannot evict.
Admissions therefore keep arriving while a 118 K-token prompt is mid-prefill and a second one
is mid-decode, the pool reaches 1.00, and from there **no lane can buy its next chunk and no
request can complete to release one**. The engine is not wedged forever — the idle timeout is
the escape hatch — but a soak with a 600 s client limit reads that as 31 timeouts.

Two secondary observations from the same dumps, both filed as tickets in the handover:

* `lane_cap: 0` on this model (`_resolve_max_prefill_seqs` returns 0 unless the checkpoint
  has `gguf_expert_types`), so `stopped_for_lane_cap` can never become True and the
  interleaved rotation `remaining + chunked_list` is **dead code on exactly the profile that
  enables interleaving here**.
* The per-pass cost of a refused pass is `O(queue x prompt)` radix walks: 16 pending
  118 K-token prompts are re-matched from scratch on every scheduling pass that ends in
  `None`. Harmless when a pass admits something; it is the whole CPU budget during a stall.

## S6. What this run proves

* `81ab30e` **fails** the 16-way Switchyard soak on both routes. The gate for item 2 of the
  handover is **not** met at this commit; `befcde6` + the §R6 `reserved_pages` fix remains the
  last tree that passed the stage route.
* The fatal that this whole line of work exists to close stays closed: **0**
  `committed_pages_required`, **0** `LinearStatePool exhausted`, **0** tracebacks, **0**
  oversize warnings, `/health` ok on every sample, clean 3 s shutdown, GPU 0 MiB.
* The commit's throughput claim is real but is not worth its cost: 2.0x lanes and 1.6x
  prefill rate, against 52 % of the wall clock spent emitting no batch at all.
* `benchmarks/scheduler_replay.py` (the replay gate landed in `508ea32`) reported
  "2.49x tokens / 2.14x completions" for this commit. It does not model retained session
  leases, decode residency or the idle timeout, so it cannot see this failure — a live soak
  is still the acceptance test.

## S7. Artifacts

`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/`

* `soak5/{run.sh,serve.sh,sample.sh,split.py,analyze.py}` — driver; `soak5/server.log`,
  `driver.log`, `resources.csv`, `soakStage.log`, `soakPass.log`,
  `soakStage/results-switchyard_stage/`, `soakPass/results-switchyard_passthrough/`,
  `stats_after_stage.json`, `stats_after_pass.json`
* `soak4_killed/` — the 9-minute aborted first attempt (clean; see S1)

---

# Run against `ea7ed7c` (+`acc91e9`) — admission gate against `admissible_size`

2026-09-05 01:22–02:14 local · FreeToken at **`acc91e9`** (tree = `ea7ed7c` "Scheduler:
admission gate against admissible_size; replay models session leases" + `acc91e9` "Triton
decode: size kv_splits from SM count"), working tree **clean** · same RTX 5080 / WSL2 host,
31 GiB `MemAvailable` at launch, embedder service not running, GPU 0 MiB before and after.

**Verdict: FAIL, and worse than `81ab30e`.** The scheduler emitted its last batch **5 m 35 s
into the stage phase** and never emitted another one: 2,616 s of continuous silence across
the rest of stage *and the whole passthrough phase*. `81ab30e` stalled for 8–10 minutes at a
time and recovered; this tree **deadlocks permanently** — the session idle timeout, which was
`81ab30e`'s escape hatch, no longer frees anything.

## T1. Exact commands

```bash
systemctl --user stop piro-board-embedder.service   # host bus not running; service inactive
FREETOKEN_GPU_LOCK_WAIT=300 scripts/gpu_lock.sh <scratch>/soak6/run.sh
```

`soak6/{run.sh,serve.sh,sample.sh,split.py,analyze.py}` are `soak5`'s byte-for-byte with the
scratch path changed; `soak6/gaps.py` is new (the batch-log gap analyzer §S5 was done by
hand). Server line and soak invocations identical to §S1.

## T2. Timeline

| | |
|---|---|
| server start → `{"status":"ok"}` | **41 s** |
| soak stage | 1,540 s, **FAIL** (`error rate 18.1818% > 0%`) |
| soak passthrough | 1,200 s, **FAIL** (`error rate 100.0000% > 0%`) |
| **last batch of the entire run** | **01:29:48**, 5 m 35 s into stage |
| scheduler deaths / detected restarts | 0 / 0 |
| `/health` non-ok samples | **0** (40 soak checks + the 10 s watchdog; `health_bad.log` absent) |
| metrics / invalid-request canaries | 0 / 0 failures |
| driver stop of a live (wedged) server | `shutdown_seconds=15` |
| GPU at end | **0 MiB**, no leftover venv python |

## T3. Request counts

| | **stage (20 m)** | **passthrough (20 m)** |
|---|---|---|
| requests / successes / failures | 176 / 144 / **32** | 32 / **0** / **32** |
| error rate | **18.1818 %** | **100 %** |
| error kinds | `timeout` ×32 | `timeout` ×32 |
| failing scenarios | long-context 16, prefix-reuse 16 | growing-conversation 16, prefix-reuse 16 |
| p50 / p95 / p99 ms | 30,704 / 600,001 / 600,002 | 600,001 / 600,002 / 600,002 |
| requests/s | 0.114 | 0.027 |
| STALLED intervals (of 20) | **12** | **18** (+2 DEGRADED) |

Comparison of the three trees on the **stage** route:

| | `befcde6`+§R6 (pass) | `81ab30e` | **`ea7ed7c`+`acc91e9`** |
|---|---|---|---|
| verdict | **PASS** | FAIL | **FAIL (deadlock)** |
| requests / errors | 471 / **0** | 268 / 15 | **176 / 32** |
| STALLED intervals | 1 | 7 | **12** |
| p95 ms | 200,742 | 600,001 | 600,001 |
| mean lanes / prefill batch | 2.37 | 4.71 | **6.57** |
| longest batch-log gap | — | 624 s | **2,616 s and never recovers** |

## T4. Throughput during the 335 s it was scheduling

The whole run produced **204 batch lines**, all between 01:24:13 and 01:29:48.

| | stage (01:24:13–01:29:48) |
|---|---|
| prefill batches | 162 |
| mean lanes / prefill batch | **6.57** (median 4) |
| median `#new-token` | 4,164 |
| prefill instant tok/s | median **2,092**, mean 2,811, max 9,109 |
| new tokens prefilled | 692,416 (2,067 tok/s over the window) |
| prefix reuse | 68.2 % (1,481,658 cached / 2,174,074) |
| decode batches | 42 |
| decode aggregate tok/s | median 75.4, max **176.5** |
| decode per-stream tok/s | median 6.53, max 22.3 |
| decode at `#running-req` 16 | n=5, agg median 80.1, per-stream 5.01 |
| batches at `#mamba-slot: 96/96` | 14 |
| busiest FreeToken process | **median 109.0 % CPU** (550 samples) — same as every other run |
| peak GPU | 14,250 MiB |

**Lanes keep going up and the run keeps getting worse**: 2.37 → 4.71 → 6.57 mean lanes per
prefill batch across the three trees, 0 → 15 → 32 errors. Seating more lanes is not the
metric.

**`acc91e9` is not visible in a 16-way aggregate, and this run cannot measure it.** Only
5 decode batches at 16 lanes exist (against 59 in the `81ab30e` run), at a different median
context, and the engine was starved for 92 % of the run. Taken at face value the 16-way
aggregate is *lower* (80.1 vs 137.1 tok/s median) — do not read that as a regression; read it
as "not measured". The mechanism also predicts a small effect here: the decode grid is
`batch × head_blocks × kv_splits`, so at batch 16 it is already SM-filling and the split-count
fix (a **single-stream** 83 → 145 tok/s win at 131K) has little left to buy. A clean 16-way
decode number needs a tree that schedules.

## T5. The failure: 14 chunked prefills deadlock the pool against each other

Not a crash, not a fatal, not a livelock that ends. The last 47 s before the freeze
(abridged; `#cached-token: 0` marks a continuation, `#running-req` is the decode set):

```
01:29:01 Prefill batch, #new-seq: 9,  #new-token: 6471, token usage: 0.45, #running-req: 9, #queue-req: 7
01:29:05 Prefill batch, #new-seq: 9,  #new-token: 8192, token usage: 0.57, #running-req: 7, #queue-req: 9
01:29:11 Prefill batch, #new-seq: 13, #new-token: 8192, token usage: 0.74, #running-req: 3, #queue-req: 13
01:29:17 Prefill batch, #new-seq: 14, #new-token: 8192, token usage: 0.89, #running-req: 2, #queue-req: 14
01:29:44 Prefill batch, #new-seq: 14, #new-token: 8192, token usage: 0.94, #running-req: 0, #queue-req: 15
01:29:48 Prefill batch, #new-seq: 13, #new-token: 7243, token usage: 1.00, #running-req: 0, #queue-req: 16
                              <-- 2,616 s of nothing; the run ends here -->
```

`py-spy dump --locals` on the scheduler process (pid 2830479 → here 3613985), 9 samples
across the 4 min and the 44 min marks of the same stall, is unambiguous and **identical at
both**:

```
_add_one_req  (prefill.py:195)   cached_len: 23846  remain_len: 8189  chunk_size: 585
                                 kv_ps: 1  kv_pages: 0  max_kv_end: 23846
try_add_one   (prefill.py:303)
schedule_next_batch (prefill.py:551)
      lane_cap: 0   seatable_lanes: 14   available_lanes: 14   reqs: []   chunked_list: []
      refusals: 0…15   index: 0…15   is_continuation: True   blocked_fresh: False→True
      inflight_prefill: 222538          (identical in every sample, 40 minutes apart)
PrefillAdder.__init__  token_budget: 8192  reserved_size: 222538  reserved_pages: 0
```

`kv_pages = max(0, available_size - reserved_pages) // kv_ps` is **0 on the pass's very first
candidate** (`index: 0`, `reserved_pages: 0`), so `available_size == 0`: the pool is 100 %
locked. Every pending entry is a **continuation**, every one defers at `chunk_size <= 0`, the
pass returns `None`, `_reclaim_for_blocked_prefill` finds nothing to release, repeat forever.

**The state that produces it.** 14 chunked prefills are in flight. They have forwarded
≈ 237.8 K tokens (the last `/v1/stats` before the report froze: `used_pages 237819 /
262144`) and still owe **222,538** tokens — a combined footprint of ≈ 460 K against a
**262,144-token pool, 1.76×**. Nothing is decoding (`#running-req: 0`), so nothing can
complete and hand a page back; no continuation can buy the next page it needs to reach
completion. The `#queue-req: 16` behind them are correctly refused (`blocked_fresh: True`) —
too late.

**Why the escape hatch is gone.** In `81ab30e` the stall was a *fresh-admit* over-commit
against decoding KV, and the session idle timeout eventually released a retained prefix. Here
the over-committed parties are the in-flight prefills themselves: 112 sessions expired by idle
timeout during the run and it changed nothing, because the KV those leases used to hold had
already been converted into these prefills' forwarded pages. The 16
`Eviction did not free enough space` / `Cold restore for session … failed` warnings all land
at **01:51:33–34**, the instant the stage clients disconnect and the passthrough clients
arrive: 16 new sessions try to restore into a pool that has nothing to give.

**What the fix did and did not fix.** `ea7ed7c`'s gate is correct about what it measures. A
fresh admit is charged its whole remaining footprint against
`admissible_size = available_size + idle-reclaimable lease tokens`, and the adder is seeded
with the unforwarded tail of every in-flight prefill — which is why, once the pool is full,
`blocked_fresh` goes True and no *new* prompt gets in. But the check is only ever applied
**at the moment of admission, against a capacity that later admissions also spend**: an idle
lease's tokens are counted as admissible for prompt A, A is admitted but only forwards one
8 K chunk, and on the next pass the same lease tokens are counted again as admissible for
prompt B. Nothing ever re-validates the invariant that the *set* of already-admitted,
not-yet-finished prefills can all finish. Fourteen of them fit through that door one at a
time and then owned the pool.

The rule the previous three fixes each restated, again: **a resource budget must subtract what
the pool has already promised, not only what it has already given.** `81ab30e` failed to
subtract the KV held by decoders; `ea7ed7c` subtracts that, and fails to subtract the
*reclaimable capacity it has already promised to prefills that have not finished*.

## T6. Markers — the fatals all stay closed

| | count |
|---|---|
| `committed_pages_required` | **0** |
| `LinearStatePool exhausted` | **0** |
| `Traceback (most recent call last)` | **0** |
| oversize (`can never be admitted`) warnings | **0** |
| `Eviction did not free enough space` | **16** (all at 01:51:33–34, the phase boundary) |
| `Restored cold session` (success) | **0** |
| `expired after idle timeout` | 112 |
| `KV grew` | 3 (65,536 → 262,144) |
| `/health` non-ok | 0 |

## T7. What this run proves

* **Item 2 of the handover is still open.** `ea7ed7c`+`acc91e9` fails the 16-way Switchyard
  soak on both routes, worse than `81ab30e`. `befcde6` + the §R6 `reserved_pages` fix remains
  the only tree that has passed the stage route.
* The failure mode has moved one layer down: from *fresh admits over-committing against
  decoders* to *concurrently chunked prefills over-committing against each other*, and it is
  now **permanent** rather than self-healing.
* `benchmarks/scheduler_replay.py` passes this commit (that is the gate `ea7ed7c` was built
  against, `stall_usage_p50 = 1.0` on `81ab30e` and clean here) and the live soak still fails.
  The replay's own "still only checkable on hardware" list — spill/restore cost of a reclaimed
  lease, GDN slots, real per-pass CPU — did not include *this*: it does not model a set of
  chunked prefills whose summed footprint exceeds the pool. Ticket 11 stands, extended.
* Everything the residency/fatal work closed stays closed (T6), and shutdown of a fully
  wedged server is still clean (15 s, GPU 0 MiB).

## T8. Fix direction (not implemented — scheduler.py/prefill.py are another agent's files)

Bound the **live** set, not the arriving one. Two candidates, both cheap:

1. **Admit against the residual, and keep it.** When a fresh admit is granted, record its full
   remaining footprint as a standing reservation held for the life of the prefill, and require
   `sum(reservations) + this admit ≤ admissible_size` on every pass. This is exactly the
   `reserved_size` seed, promoted from a per-pass quantity to a persistent one — the property
   `fad1fc4`+§R6 had by accident (it charged the whole prompt against `available_size`) and
   that §R7 asked to relax for long-context concurrency. The relaxation is what broke it.
2. **Cap concurrent chunked prefills** by `pool_size / expected_prompt_size`, or simply refuse
   a fresh admit whenever `inflight_prefill_size + this_prompt > admissible_size`. The 14
   lanes here would have been ~5.

Either way the acceptance test is a live 16-way soak on **both** routes, and the analyzer must
report the trailing silence after the last batch line, not only the gaps between two lines —
this run's `gaps >= 30 s` count inside the stage phase is **zero**, because the deadlock is
not *between* two batch lines.

## T9. Artifacts

`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/`

* `soak6/{run.sh,serve.sh,sample.sh,split.py,analyze.py,gaps.py}` — driver + analyzers
* `soak6/{server.log,driver.log,resources.csv,soakStage.log,soakPass.log,phase_stage.log,
  phase_pass.log}`, `soak6/soakStage/results-switchyard_stage/`,
  `soak6/soakPass/results-switchyard_passthrough/`, `stats_after_{stage,pass}.json`
* `soak6/pyspy_stall1.txt` (4 min into the stall), `soak6/pyspy_stall2.txt` (30 min into the
  same stall, `inflight_prefill` byte-identical)

---

# Run against `4a99e34` (standing reservation + `max_chunked_prefills` + invariant) — **PASS**

2026-09-05 04:42–05:29 local · FreeToken at **`4a99e34`**, working tree **clean** · same
RTX 5080 / WSL2 host, 31 GiB available at launch, `piro-board-embedder` not running.

Tree under test: scheduler = the `d685e99` gate restored, **plus** `b030c7f`'s standing
reservation (`PrefillManager._standing_reservation` seeded into `PrefillAdder.reserved_size`),
`max_chunked_prefills = 8`, and the `_check_finishability` invariant; `ff470e7` server
disconnect-abort; `acc91e9` decode launch config; `4a99e34` prefill `BLOCK_M` register cap.

**Verdict: PASS on both routes — the first tree since `befcde6`+§R6 to pass, and it passes
harder.** 0 errors on 2,070 requests, 0 STALLED on stage (1 on passthrough), **0
finishability-invariant warnings**, zero fatals, zero trailing silence, and every throughput
axis up on §R4/§R6.

## U1. Exact commands

```bash
# scratchpad/soak7/serve.sh — the soak6 line verbatim + the invariant, under the GPU lock
FREETOKEN_PIN_BUDGET_GB=17 FREETOKEN_SCHEDULER_INVARIANT=warn \
scripts/gpu_lock.sh scratchpad/soak7/serve.sh   # ft serve, port 1919, P2 profile
#   --max-running-requests 16 --elastic-initial-requests 4 --kv-grow-step-tokens 65536
#   --num-tokens 262144 --max-seq-len-override 131072 --kv-cache-dtype q8_0
#   --attention-backend triton --moe-backend offload --moe-cache-auto --moe-cache-policy lfu
#   --memory-ratio 0.85 --max-prefill-length 8192 --host-ram-reserve-gb 6 ...

# scratchpad/soak7/run.sh: stage 20 m, then passthrough 20 m, default scenario set, c=16
scripts/switchyard_e2e.sh soak --base-url http://127.0.0.1:1919 \
  --model nemotron-3.5-lightning --duration 20m --concurrency 16 \
  --route switchyard/{stage,passthrough}
# -> switchyard-soak ... --max-output-tokens 256 --prompt-bytes 16384
#      --context-window-tokens 131072 --max-error-rate 0 --request-timeout 600
#      --scenario prefix-reuse --scenario growing-conversation --scenario tool-call-burst
#      --scenario large-tool-catalog --scenario long-context
```

Server TERMed directly (not the lock wrapper) so the graceful path is what gets timed.

## U2. Result

| | stage | passthrough |
|---|---|---|
| verdict | **PASS** | **PASS** |
| requests / successes / failures | **470 / 470 / 0** | **1600 / 1600 / 0** |
| error rate | 0.0000 % | 0.0000 % |
| STALLED intervals | **0** | 1 (t = 390 s) |
| p50 / p95 / p99 ms | **24,283 / 145,840 / 230,183** | **7,527 / 32,906 / 83,354** |
| health checks / failures | 20 / 0 | 20 / 0 |
| invalid-request canaries / failures | 3 / 0 | 3 / 0 |
| detected server restarts | 0 | 0 |
| scenario failures | none (5/5 scenarios) | none (5/5 scenarios) |

Per-scenario successes — stage: prefix-reuse 95, growing-conversation 95, tool-call-burst 95,
large-tool-catalog 95, long-context 90. Passthrough: 320 each.

## U3. Invariant, fatals and the deadlock signature

| check | stage | passthrough | whole run |
|---|---|---|---|
| `finishability invariant violated` (`=warn`) | 0 | 0 | **0** |
| `committed_pages_required` | 0 | 0 | **0** |
| `LinearStatePool exhausted` | 0 | 0 | **0** |
| `Traceback (most recent call last)` | 0 | 0 | **0** |
| `can never be admitted` (oversize) | 0 | 0 | **0** |
| `Eviction did not free enough space` | 0 | 0 | **0** |
| `ERROR` / `CRITICAL` lines | 0 | 0 | **0** |

The invariant is evaluated **before every scheduling pass** with `FREETOKEN_SCHEDULER_INVARIANT
=warn` in the server env and never fired: across ~3,141 prefill passes the admitted set was
always finishable against `available_size`. `ea7ed7c` violated it on 566 stage passes.

**Trailing silence (the §T deadlock signature) — the analyzer now reports it** (`gaps.py`, in
both `soak6/` and `soak7/`; leading + trailing silence against the driver's phase window):

| | stage | passthrough |
|---|---|---|
| phase window | 1,445 s | 1,296 s |
| batch lines | 2,276 | 1,285 |
| leading silence (phase start → first batch) | 40 s (2.8 %) | 1 s (0.1 %) |
| **trailing silence (last batch → phase end)** | **1 s (0.1 %)** | **1 s (0.1 %)** |
| gaps ≥ 30 s between batch lines | **0** | 1 (54 s) |
| scheduling wall clock | **1,404 s (97.2 %)** | 1,240 s (95.7 %) |

For scale: `ea7ed7c` had 2,616 s of trailing silence and emitted its last batch 5 m 35 s into
a 50-minute run; `81ab30e` spent 52 % of each phase emitting no batch.

The one 54 s gap (passthrough, 05:09:06 → 05:10:00, usage 0.80, queue 1) is **not** a
scheduler stall: it is a session-residency burst — ten `Spilled soft session … 118,062
tokens` (0.45–0.54 GiB each, 5 to RAM then 5 to disk at 0.94–1.95 GiB/s) with their
`Released … KV protection (admission pressure)`, and the batch that ends it restores
**589,680 cached tokens across 6 fresh lanes**. Work, not silence. The passthrough STALLED
interval (t = 390 s, zero completions in one 60 s window) sits in a different window where
batches were flowing throughout.

## U4. Throughput, and how it compares

Server batch log, per phase (`soak7/analyze.py` on `phase_{stage,pass}.log`):

| | stage | passthrough |
|---|---|---|
| decode batches | 185 | 235 |
| mean `#running-req` | 8.06 | 11.28 |
| decode aggregate tok/s (all batches) | median 81.2, mean 87.2, max 311.5 | median 143.4, mean 147.1, max 445.5 |
| **decode aggregate tok/s @ `#running-req == 16`** | **96.8** (n=15) | **177.5** (n=93, mean 182.8, max 445.5) |
| **decode per-stream tok/s @ 16** | **6.05** | **11.09** |
| prefill batches | 2,091 | 1,050 |
| prefill instant tok/s | median 1,851, mean 2,135, max 9,181 | median 1,838, mean 2,766, max 9,020 |
| new prompt tokens | 2,397,852 | 2,263,387 |
| effective new-token prefill rate (Σ`#new-token` / Σ chunk wall) | **1,830 tok/s** | **1,879 tok/s** |
| prefill wall clock | 1,310 s (91 % of phase) | 1,205 s (93 %) |
| prefix reuse (cached / (new+cached)) | **85.0 %** (13.57 M cached) | **88.8 %** (17.90 M cached) |
| mean / median / max lanes per prefill batch | 1.83 / 1 / 16 | 3.53 / 3 / 16 |
| batches at `#mamba-slot: 96/96` | 36 | 3 |
| pressure episodes (usage ≥ 0.98, queue > 0) | 5 | 0 |

Against the last passing tree (§R6, `befcde6` + `reserved_pages`) and the §R4 decode table
(`fad1fc4`):

| stage route | §R6 | **4a99e34** | Δ |
|---|---|---|---|
| requests / errors / STALLED | 471 / 0 / 1 | 470 / 0 / **0** | = / = / **−1** |
| p50 / p95 / p99 ms | 29,104 / 200,742 / 257,441 | **24,283 / 145,840 / 230,183** | **−17 % / −27 % / −11 %** |
| decode agg @ 16 (§R4) | 81.6 (n=8) | **96.8** (n=15) | **+18.6 %** |
| decode per-stream @ 16 (§R4) | 5.10 | **6.05** | **+18.6 %** |
| prefill instant tok/s median (§R4) | 1,637 | **1,851** | **+13.1 %** |
| prefix reuse | 83.8 % | **85.0 %** | +1.2 pt |
| mean lanes per prefill batch | 2.37 | 1.83 | **−23 %** |
| new prompt tokens | 2.72 M | 2.40 M | −12 % |

| passthrough route | §R4 (`fad1fc4`) | **4a99e34** | Δ |
|---|---|---|---|
| requests / errors | 1,219 / 0 | **1,600 / 0** | **+31 %** |
| decode agg @ 16 | 161.4 (n=77) | **177.5** (n=93) | **+10.0 %** |
| decode per-stream @ 16 | 10.09 | **11.09** | **+9.9 %** |
| prefill instant tok/s median | 1,496 | **1,838** | **+22.9 %** |
| p95 ms | — | 32,906 | — |

`acc91e9` + `4a99e34` are visible here in a way they were not in the `ea7ed7c` run (which was
starved 92 % of the wall clock): decode is up ~10 % at 16 lanes on passthrough and ~19 % on
stage, prefill median instant throughput up 13–23 %. The 16-way aggregate of 177.5 tok/s
beats the 168 tok/s recorded on 2026-09-04 for the LFU 16-way configuration.

**Lanes went DOWN and that is the correct trade.** Mean lanes per prefill batch is 1.83
against §R6's 2.37 — the standing reservation makes an in-flight prefill keep costing
admission until it finishes, so fewer fresh prompts are seated at once. It bought 0 STALLED
intervals and a 27 % lower p95 on the route that used to time out. Reading the series with
the §S/§T lesson: 2.37 → 4.71 → 6.57 lanes with 0 → 15 → 32 errors, and now **1.83 lanes with
0 errors and the best latency of any run**.

## U5. `max_chunked_prefills = 8` — no evidence it ever bound (and it cannot be proven)

The cap is a silent `continue` in `schedule_next_batch`; there is no log line and no
`/v1/stats` counter, so "never binds" is **not directly observable**. The strongest available
evidence, log-only:

* A prefill pass with `#cached-token > 0` necessarily admitted a **fresh** request that pass
  (continuation chunks contribute 0 to the cached counter — the hit is booked once at
  admission), which proves `chunked_inflight < 8` at that moment.
* Such passes: **282 of 2,091 (stage)** and **279 of 1,050 (passthrough)**, spaced
  **median 2 s** apart (mean 5.0 / 4.6 s, p95 21 / 14 s, max 73 / 102 s).

So the count of concurrent chunked prefills dropped below 8 every ~2 s throughout both
phases; any window in which the cap could have bound is bounded by 73 s (stage) / 102 s
(passthrough), and no symptom (no gap, no invariant warning, no queue growth) appears in
those windows. Consistent with the knob's design note — the standing reservation is what
holds the set finishable and the cap is belt-and-braces. **Ticket (new, see below): expose
`chunked_prefills_inflight` and a `fresh_admits_blocked_by_cap` counter so the next soak
can answer this directly instead of by inference.**

## U6. Disconnect-abort (`ff470e7`)

* **Live probe, after the two phases, same server**: a 60,000-word (~60 K token) non-streaming
  `/v1/chat/completions` was posted on a raw socket and the socket shut down 6 s later, mid
  prefill. `/v1/stats.requests.active` went 0 → **1** while it was in flight and back to
  **0 seven seconds after the close** — the disconnect poll fired, the `AbortMsg` reached the
  scheduler and the request's slot and forwarded KV were freed. Pre-`ff470e7` that request
  would have held them for the life of the server.
* **No spurious aborts of healthy clients**: 2,070 requests, **0 failures, 0 error records,
  0 dropped error records**, `error_kinds: {}` on both routes, and 3/3 invalid-request
  canaries still behaving. A spurious abort cuts a live client's response, which the soak
  grades as an error; there were none.
* **Count of aborts during the soak: not measurable.** `Scheduler._process_one_msg` logs
  `"Aborting request %d"` at **debug**, and `StatsTracker` keeps only a live `_aborting` set,
  no cumulative counter. Ticket below. What can be said: no client disconnected voluntarily
  (0 errors, 0 timeouts), so the expected count is 0 and nothing in the run contradicts it.

## U7. Host behaviour

* No spin: busiest FreeToken process at **median 109.9 % CPU (stage) / 107.0 % (passthrough)**
  over 497 five-second samples — the same ~106 % every previous run recorded while healthy.
* GPU 14.1 GiB median, **15.8 GiB peak**; top-process RSS peak 24.2 GB, no host OOM.
* `/health` `ok` on all 40 checks plus the driver's own 10 s watchdog (`health_bad.log` never
  created). `KV grew` 3 times in stage (65,536 → 262,144) and stayed there.
* Session residency: 126 (stage) + 270 (passthrough) idle-timeout expiries, **423 cold
  restores, 0 failures**; 1,025 `Released soft session … KV protection` (291 admission +
  85 GDN in stage, 734 admission in passthrough).
* **Graceful shutdown in 3 s**, GPU back to **0 MiB**, no leftover venv processes.

## U8. Still open after this run

1. **§R7 ticket 1 is what now caps the stage route**, and it is measurable:
   `chunk_limit = token_budget // waiting` produces the starvation signature
   (`#new-seq: 1`, `#new-token ≤ 512`, `#queue-req ≥ 8`) on **1,278 of 2,091 stage prefill
   passes (61 %)** and 200 of 1,050 passthrough passes (19 %). A 118 K-token prompt advancing
   512 tokens per pass is why stage p95 is still 146 s. Size the interleave share by the lanes
   the pass will actually seat, or floor the chunk at ~2 K tokens.
2. **No observability for the two admission bounds.** Neither the standing reservation's
   refusals nor `max_chunked_prefills` leaves a trace; §U5 had to infer. Add
   `chunked_prefills_inflight`, `fresh_admits_blocked_by_cap` and `deferred_prefill_chunks`
   to `/v1/stats` (the §R5 ticket, still open).
3. **No abort counter.** `"Aborting request %d"` is debug-level and `StatsTracker` has no
   cumulative `aborted`; a soak cannot count disconnect-aborts. One counter on `StatsTracker`
   plus a `/v1/stats` field closes it.
4. **761 `Discarded cold session …: client tokens diverge at 3, before the first stored state
   boundary`** (INFO; `soak6` logged 179). Benign here — a reused auto-session id whose new
   prompt diverges at token 3 — but it is the same message the handover's item-3 note flags
   as *also* being emitted when `session.spill` is invalid after a capacity eviction. The
   one-line `.valid` check would let the two cases be told apart in a log.
5. `benchmarks/scheduler_replay.py` (ticket 11) still needs the §T set-of-chunked-prefills
   case; this tree passes both the replay and the live soak, which is necessary, not
   sufficient.

## U9. Artifacts

`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/`

* `soak7/{run.sh,serve.sh,sample.sh,split.py,analyze.py,gaps.py}` — driver + analyzers;
  `gaps.py` now reports leading/trailing silence against the driver's phase window and warns
  at ≥ 120 s of trailing silence (the same file was updated in `soak6/`)
* `soak7/{server.log,driver.log,resources.csv,soakStage.log,soakPass.log,phase_stage.log,
  phase_pass.log,stats_after_{stage,pass}.json,stats_{before,after}_probe.json}`
* `soak7/soakStage/results-switchyard_stage/`, `soak7/soakPass/results-switchyard_passthrough/`

---

# Run against 13af13d (2026-09-05) — PASS on both routes; the §R7 starvation signature is gone

Tree: `812bc57` (seatable-lanes chunk divisor) + `32cc504` (fork/main Ada merge) + `52a6503`
(CPU-safe capability probe) + `193da80` + `13af13d` (batch_memcpy probe stream ordering).
Same P2 serving line and same soak invocation as §U, `FREETOKEN_SCHEDULER_INVARIANT=warn`,
server under `scripts/gpu_lock.sh`.

**The drivers now live in the repo**, not in a session scratchpad: a WSL OOM restart at 08:59
destroyed a whole in-flight soak's artifacts along with `scratchpad/soak7/`. See §V8.

## V1. Exact commands

```bash
benchmarks/switchyard_soak/run.sh 13af13d 20m       # writes runs/13af13d/
#   -> gpu_lock.sh benchmarks/switchyard_soak/serve.sh   (the §U1 serve line, verbatim)
#   -> switchyard_e2e.sh soak --duration 20m --concurrency 16 --route switchyard/{stage,passthrough}
#   -> disconnect-abort probe, then TERM the ft serve python directly
uv run python benchmarks/switchyard_soak/split.py   runs/13af13d
uv run python benchmarks/switchyard_soak/analyze.py runs/13af13d/phase_{stage,pass}.log
uv run python benchmarks/switchyard_soak/gaps.py    runs/13af13d/phase_stage.log 30 <start> <end>
```

`run.sh` refuses to start below 26 GiB `MemAvailable` (the OOM lesson) and writes everything
under `benchmarks/switchyard_soak/runs/<tag>/`.

## V2. Result

| | stage | passthrough |
|---|---|---|
| verdict | **PASS** | **PASS** |
| requests / successes / failures | **492 / 492 / 0** | **1904 / 1904 / 0** |
| error rate | 0.0000 % | 0.0000 % |
| STALLED intervals | **0** | **0** |
| p50 / p95 / p99 ms | 29,820 / **109,395** / 149,081 | 6,888 / **24,580** / 46,695 |
| health / metrics checks, failures | 20 + 20, 0 | 20 + 20, 0 |
| invalid-request canaries / failures | 3 / 0 | 3 / 0 |
| detected server restarts | 0 | 0 |
| scenario failures | none (5/5) | none (5/5) |
| error records | 0 | 0 |

Per-scenario successes — stage: prefix-reuse 103, growing-conversation 103, tool-call-burst
101, large-tool-catalog 94, long-context 91. Passthrough: 384 / 384 / 384 / 384, long-context 368.

## V3. Against §U (`4a99e34`) — every headline moves the right way

| | §U stage | §V stage | §U pass | §V pass |
|---|---|---|---|---|
| requests | 470 | **492** (+4.7 %) | 1,600 | **1,904** (+19 %) |
| errors / STALLED | 0 / 0 | 0 / 0 | 0 / 1 | **0 / 0** |
| p95 ms | 145,840 | **109,395** (−25 %) | 32,906 | **24,580** (−25 %) |
| p99 ms | 230,183 | **149,081** (−35 %) | 83,354 | **46,695** (−44 %) |
| **starvation signature** | 1,278/2,091 = **61 %** | **0/603 = 0.0 %** | 200/1,050 = 19 % | **0/599 = 0.0 %** |
| mean lanes / prefill batch | 1.83 | 3.43 | 3.53 | 4.92 |
| decode agg tok/s @ `#running-req == 16` | 96.8 | 99.9 | 177.5 | **190.6** |
| per-stream tok/s @ 16 | 6.05 | 6.24 | 11.09 | 11.91 |
| effective new-token prefill rate | 1,830 | **2,310** (+26 %) | 1,879 | 2,008 (+7 %) |
| prefix reuse | 85.0 % | 82.3 % | 88.8 % | 89.6 % |
| prefill instant tok/s (median) | 1,851 | **2,799** | 1,838 | 1,547 |
| trailing silence | 1 s (0.1 %) | 1 s (0.1 %) | 1 s (0.1 %) | 2 s (0.2 %) |
| scheduling wall clock | 97.2 % | **99.8 %** | 95.7 % | **99.8 %** |

**§R7 ticket 1 is closed.** `chunk_limit = token_budget // waiting` produced the starvation
signature (`#new-seq: 1`, `#new-token ≤ 512`, `#queue-req ≥ 8`) on 61 % of stage passes in §U;
dividing by the lanes the pass will actually *seat* (`812bc57`) drops it to **zero on all 1,202
prefill passes of this run**, on both routes. Stage median `#new-token` is 5,689 against §U's
regime of 512-token crawls, and stage p95 falls 36 s.

**On the lane count.** Lanes rose 1.83 → 3.43 (stage) and 3.53 → 4.92 (passthrough). Passthrough
is inside the 4.7–6.6 band the *failing* trees occupied, so it was checked rather than assumed:
those trees (§R6 4.71 / §R7 6.57) hit that band on the **stage** route with 15 and 32 client
errors and p95 200.7 s. This run has 0 errors, 0 STALLED, 0 invariant violations and the best
latency ever recorded on either route, and stage — the route the band was measured on — sits at
3.43, below §R6's 4.71. More lanes is still not the metric; the metric is errors and p95, and
both improved. See §V7 for what to watch.

The one number that is *down* is passthrough prefill instant tok/s (1,838 → 1,547 median).
That is a per-chunk rate, and this run seats more lanes per pass with a smaller median
`#new-token` (3,010) and 89.6 % prefix reuse; the goodput that pays for it went **up** 19 % in
requests and 7 % in effective new-token rate. Do not read the per-chunk median as throughput.

## V4. Invariant, fatals, deadlock signature

| check | stage | passthrough | whole run |
|---|---|---|---|
| `finishability invariant violated` (`=warn`) | 0 | 0 | **0** |
| `committed_pages_required` | 0 | 0 | **0** |
| `LinearStatePool exhausted` | 0 | 0 | **0** |
| `Eviction did not free enough space` | 0 | 0 | **0** |
| oversize `can never be admitted` | 0 | 0 | **0** |
| `Traceback (most recent call last)` | 0 | 0 | **0** |
| ERROR / CRITICAL lines | 0 | 0 | **0** |
| `/health` non-ok | 0 | 0 | **0** |

Zero invariant warnings across 1,202 prefill passes — `812bc57` moves no admission gate, and the
live run agrees with the replay.

**Deadlock signature (§T):** leading silence 2 s / 1 s, **trailing silence 1 s / 2 s**, scheduling
wall clock **99.8 % of both phase windows** (§U: 97.2 % / 95.7 %). Stage has **0 gaps ≥ 30 s**.
Passthrough has exactly one, 31 s at 09:41:50, and it is *not* a stall: the window holds 291
non-batch lines — a burst of soft-session spills (0.63 GiB and 0.16 GiB records, RAM and disk
tiers) with `KV protection (admission pressure)` releases, the same signature as §U's 54 s gap.
`#queue-req: 11` with `#running-req: 3` at its start: the pool was making room, not stuck.

**`#mamba-slot: 96/96` behaves.** 60 of 761 stage batch lines and 32 of 866 passthrough lines run
at full GDN occupancy, longest consecutive run **7 batches** on each route. Inside those windows
requests still complete (`200 OK` interleaved) and the pool drains through soft-session spill +
release; `LinearStatePool exhausted` stays at 0. Full occupancy is a working state here, not a
cliff.

## V5. Disconnect-abort (`ff470e7`) — verified again

`/v1/stats.requests.active` **0 → 1 → 0, back to 0 five seconds after the socket close** on a
~60 K-token non-streaming request dropped mid-prefill (§U measured 7 s). No spurious aborts:
0 client failures on 2,396 requests and 3/3 invalid-request canaries correct on both routes.
The abort *count* is still unmeasurable (ticket 12).

## V6. Host behaviour

* Busiest FreeToken process: **median 101.0 % CPU** (no spin), max 1,348 % (model load /
  spill bursts across workers).
* GPU **13.9 GiB median, 15.8 GiB peak**; RSS peak **23.9 GB**; host `MemAvailable` bottomed at
  **5.1 GiB** (median 7.5) — comfortable for this run, but see §V8.
* Sessions: 441 cold restores (passthrough) with **0 failures**, 309 idle expiries,
  1,186 `Released soft session … KV protection`. `KV grew` 3 times in stage (65,536 → 262,144)
  and stayed there; `KV shrank` 0.
* **Graceful shutdown in 4 s, GPU back to 0 MiB**, no leftover venv processes.
* 826 `client tokens diverge at 3` INFO lines (§U: 761) — still the benign reused-auto-session
  case, still indistinguishable from a stale-spill discard without the item-3 `.valid` fix.

## V7. Still open after this run

1. **Ticket §R7-1 is closed; watch the lane count instead.** The chunk divisor is the fix, but
   lanes are now a *free* variable — nothing caps how many prompts one pass seats except the
   pools. Record mean lanes every soak; if it climbs past ~5 on the **stage** route while errors
   or p95 move, that is the §R6/§R7 failure mode returning.
2. Tickets **8, 9, 10, 12** from §U8 are untouched by this run and still open (oversize prompts
   have no client rejection path; `stopped_for_lane_cap` is dead code on this profile; a refused
   prefill pass costs `O(queue × prompt)` radix walks; the admission bounds and the abort path
   have no counters).
3. **`--moe-prefill-hit-d2d` is off in the P2 serve profile** (`moe_prefill_hit_d2d=False` in the
   run's `ServerArgs`), so this soak exercised **no** `cudaMemcpyBatchAsync` path and cannot
   confirm the `13af13d` probe latched True. The probe fix is covered by
   `tests/moe/test_prefill_hit_d2d.py::test_batch_memcpy_probe_survives_busy_ambient_stream`
   instead, which calls `load_batch_memcpy()` behind ~0.5 s of queued ambient-stream work.
   A soak that actually grades hit-D2D has to pass the flag.
4. `benchmarks/scheduler_replay.py` (ticket 11) still needs the §T set-of-chunked-prefills case.

## V8. Artifacts — now in the repo

`benchmarks/switchyard_soak/` (tracked): `run.sh`, `serve.sh`, `sample.sh`, `split.py`,
`analyze.py` (throughput, occupancy, lanes, **starvation signature**, markers), `gaps.py`
(gaps + leading/**trailing** silence vs the driver's phase window, WARN at ≥ 120 s).
`runs/` is gitignored; this run is `runs/13af13d/{driver.log,server.log,resources.csv,
soakStage.log,soakPass.log,phase_{stage,pass}.log,stats_after_soak{Stage,Pass}.json,
stats_{before,after}_probe.json,soakStage/,soakPass/}`.

**Why they moved.** The 08:59 WSL OOM restart took `/tmp/claude-1000/.../scratchpad/soak7/`
with it — drivers, analyzers and a running soak's logs, all unrecoverable. A benchmark harness
that only exists in a session scratchpad is one host event away from having never existed.
`sample.sh` now also records host `MemAvailable`, and `run.sh` refuses to start below 26 GiB.
