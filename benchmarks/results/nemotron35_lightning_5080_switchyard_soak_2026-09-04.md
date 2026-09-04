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
