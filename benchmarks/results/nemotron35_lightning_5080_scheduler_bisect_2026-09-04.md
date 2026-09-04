# Scheduler bisect — where the stage-route prefill starvation came from

2026-09-04 · CPU only, no GPU, no model load · replay harness
`scratchpad/bisect/stage_replay.py` · run in the throwaway worktree
`.claude/worktrees/agent-a868193445df5dd41` (left detached at `d685e99`; `main`'s
working tree untouched, nothing committed, nothing stashed).

Companion to `nemotron35_lightning_5080_switchyard_soak_2026-09-04.md` (§R5–R7),
which diagnosed the starvation on hardware. This run answers the follow-up
question: **which of the 101 unpushed commits (25 of them touching
`python/freetoken/scheduler/`) introduced it, and what else is still wrong.**

**Verdict:** the starvation is **one commit**, `f3c3ac4` *"Add persistent agent
sessions and fair prefill lanes"* (2026-08-29). None of the three fixes that
landed today addresses it, and the pending third fix (`lanes.diff`) is a
**no-op** on this traffic. Against the upstream merge-base the scheduler now
prefills **2.5x fewer tokens** and completes **2.2x fewer requests** on the
stage-route traffic shape.

---

## 1. The replay

`schedule_next_batch` is driven directly — real `PrefillManager`,
`CacheManager`, `TableManager`, `DecodeManager`, real radix prefix cache — with
the soak's serving geometry:

| | |
|---|---|
| pool | 262 144 pages, `page_size` 1, committed in 65 536-page steps (`--num-tokens` / `--kv-grow-step-tokens`) |
| concurrency | 16 closed-loop clients (`--max-running-requests 16`) |
| prefill budget | 8 192 tokens (`--max-prefill-length`) |
| decode burst | 32 forwards between prefill passes (`Scheduler._growable_decode_burst`) |
| output | 256 tokens (`--max-output-tokens` on the soak client) |
| `interleave_chunks` | **True** — `bool(kv_grow_step_tokens and max_running_req > 1)`, `scheduler.py:166` |
| `max_batch_seqs` | **0** — `_resolve_max_prefill_seqs` requires `gguf_expert_types is not None`; the NVFP4 Nemotron is not GGUF |

Two traffic profiles, both seeded and deterministic, 20 000 forwards each:

* **stage** — the soak's measured per-scenario mix (`long-context` 118 K tokens,
  `large-tool-catalog` ~9 K, `growing-conversation` ~6 K, `tool-call-burst` ~4 K,
  `prefix-reuse` ~2 K), 75 % shared prefix per conversation family so the radix
  cache really matches (soak measured 73.6 %).
* **pressure** — the steady state a closed 16-client loop drifts into once the
  long prompts start outliving the short ones (90 % long-context).

The synthetic clock uses the soak's own rates (prefill 1 800 tok/s instant,
decode 0.10 s/batch), so wait-to-first-chunk is directly comparable with the
soak's `p95 = 392 867 ms` / four 600 s timeouts. It lands at 402 s p95 — the
right order of magnitude, from a pure-CPU replay.

Metrics per run: lanes per prefill batch, prefill budget utilisation, the chunk
limit handed to `_add_one_req`, wait-to-first-chunk, completions, the
`committed_pages_required` margin (`allocatable - needed`; negative would raise),
and **why each pass stopped admitting** (`schedule_next_batch` breaks at the
first refusal, so a pass has at most one).

---

## 2. Per-commit table — stage profile

Every unpushed commit that touched `scheduler/prefill.py` or `scheduler/cache.py`,
plus the merge-base with upstream (`bd372b6`, `origin/main`, 2026-08-24), plus the
pending `lanes.diff` on top of `d685e99`. 20 000 forwards, seed 7.

| commit | | lanes/batch | budget util | chunk limit p50 | tokens prefilled | completed | wait→1st chunk p50 / p95 (s) | pass stopped at | fatal |
|---|---|---|---|---|---|---|---|---|---|
| `bd372b6` | merge-base (upstream) | 1.44 | **98.8 %** | n/a | **7 103 059** | **404** | 155 / 395 | budget 852, fresh-gate 76 | — |
| `60dd5cf` | Grow KV in stable 64K segments | 1.44 | 98.8 % | n/a | 7 103 059 | 404 | 155 / 395 | budget 852, fresh-gate 76 | — |
| `54650ad` | Elastic multi-agent KV scheduling | 1.43 | 98.8 % | n/a | 7 189 572 | 404 | 152 / 356 | budget 864, fresh-gate 85 | — |
| **`f3c3ac4`** | **persistent agent sessions and fair prefill lanes** | **2.18** | **14.3 %** | **512** | **2 814 602** | **181** | 208 / 402 | **fresh-gate 2 404** | — |
| `fc442a1` | Optimize concurrent GGUF prefill | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `e47eabb` | elastic agents / pageable graphs | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `140370c` | cold agent session spill | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `cefa4bd` | Group only small GGUF prefills | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `1184c4d` | Nemotron-H Mamba-2 Triton SSD | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `dcb617a` | Reclaim Mamba state slots (3F fix) | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `5c8e964` | Restore longest session prefix | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `1f2de67` | Export prompt hidden states | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `fad1fc4` | **fix 1** — KV back-pressure on continuations | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `d685e99` | **fix 2** — cap in pages, not reserved prompt | 2.18 | 14.3 % | 512 | 2 814 602 | 181 | 208 / 402 | fresh-gate 2 404 | — |
| `d685e99` + `lanes.diff` | **fix 3** (pending) — chunk share | **2.18** | **14.3 %** | **512** | **2 814 602** | **181** | 208 / 402 | fresh-gate 2 404 | — |
| `d685e99`, `interleave_chunks=False` | control | 1.43 | 98.8 % | n/a | **7 189 572** | **404** | 152 / 356 | budget 864, fresh-gate 85 | — |

## 3. Per-commit table — pressure profile

| commit | lanes/batch | budget util | tokens prefilled | completed | wait→1st chunk p50 (s) | min `committed_pages_required` margin (prefill / decode) | fatal |
|---|---|---|---|---|---|---|---|
| `bd372b6` … `54650ad` | 1.07 | **99.6 %** | **8 577 078** | **83** | 1 146 | 34 027 / 33 773 | — |
| `f3c3ac4` … `d685e99` | 1.93 | **12.1 %** | 5 000 774 | 60 | 955 | **2 916 / 2 726** | — |
| `d685e99` + `lanes.diff` | 1.93 | 12.1 % | 5 000 774 | 60 | 955 | 2 916 / 2 726 | — |
| `d685e99`, `interleave_chunks=False` | 1.07 | 99.6 % | 8 577 078 | 83 | 1 146 | 34 027 / 33 773 | — |

The control row is the load-bearing one: **turning `interleave_chunks` off at
`d685e99` reproduces the merge-base numbers byte for byte** in both profiles.
Every other change in the 101 unpushed commits is throughput-neutral on this
workload. The regression is that one flag and the chunk-share arithmetic behind it.

---

## 4. Attribution

### (a) The ungated continuation admit — **pre-existing upstream, not yours**

`PrefillAdder.try_add_one` takes the `pending_req.chunked_req` branch and calls
`_add_one_req` with no availability check of any kind. That shape is already in
the merge-base `bd372b6` (`prefill.py:191–205` there; `prefill.py:239` today) —
it is upstream code, present before the first of the 101 commits. **No unpushed
commit introduced it.** `fad1fc4` is what finally gated it.

What the unpushed work *did* do is multiply its exposure. Before `f3c3ac4` the
ungated path ran at most once or twice per pass (the head lane ate the whole
8 192-token budget and the pass ended: 852 of the 928 pass-ending refusals over
878 prefill batches are `budget_exhausted`). After `f3c3ac4` the pass tries to
seat every queued lane, so
the unchecked path runs on every continuation in the queue — which is the
`#new-seq: 14` batch that killed the scheduler in §5 of the soak.

### (b) The whole-prompt reservation division — **two distinct sites**

* **As the chunk cap** (`kv_pages = (available_size - reserved_size) // kv_ps`):
  introduced by **`fad1fc4`**, *today*, and fixed the same day by **`d685e99`**.
  Neither is one of the 44/101 pre-existing commits. Confirmed by pickaxe:
  `git log -S reserved_size -- scheduler/prefill.py` over
  `bd372b6..d685e99` returns exactly `{fad1fc4, d685e99}`.
* **As the admission gate** (`_try_allocate_one`:
  `if estimated_len + self.reserved_size > available_size: return None`, with
  `reserved_size += remain_len + pending_req.output_len`): **pre-existing at the
  merge-base** — `prefill.py:63/66` and `:162` at `bd372b6`, `:82/85` and `:205`
  today. Unmodified by any unpushed commit. This is the §R7 third bullet, and
  §5 below shows it is now the *dominant* limiter.

### (c) The `token_budget // waiting` divisor — **`f3c3ac4`**

```
commit f3c3ac4  Add persistent agent sessions and fair prefill lanes   2026-08-29
+            if self.interleave_chunks:
+                waiting = len(self.pending_list) - index
+                chunk_limit = max(1, adder.token_budget // waiting)
```

`fc442a1` (2026-08-30) later rewrote it to
`token_budget // min(waiting, available_lanes)` — but `available_lanes` falls back
to `waiting` unless `max_batch_seqs` is non-zero, and `_resolve_max_prefill_seqs`
returns non-zero only for GGUF expert models. For the soak's NVFP4 Nemotron
`max_batch_seqs == 0`, so `fc442a1` is a no-op: the replay's numbers at `f3c3ac4`
and `fc442a1` are identical to the digit.

### Did the merge-base have any of them?

| | at `bd372b6` (upstream) |
|---|---|
| (a) ungated continuation admit | **yes** |
| (b) whole-prompt reservation as the *chunk cap* | no (`fad1fc4`, today) |
| (b′) whole-prompt reservation as the *admission gate* | **yes** |
| (c) `token_budget // waiting` | **no** — `interleave_chunks` does not exist |

---

## 5. Remaining risks, ranked

### R-1 — `_try_allocate_one`'s whole-prompt reservation is now the binding constraint. None of the three fixes touches it.

**Evidence (stage, `d685e99`):** **2 404 of 2 408** prefill passes (99.8 %) end
because `_try_allocate_one` refused a fresh admit. In **2 364 of those 2 404
(98.3 %)** the chunk the pass would actually have forwarded fits: median headroom
`available_size - reserved_size` = **23 155 pages** against a chunk limit whose
median is **512** tokens. The refusal is driven entirely by the median **115 189**-token
whole prompt plus its `output_len`.

Under the pressure profile it is **5 065 of 5 065 (100 %)**, and the chunk would
have fit in **100 %** of them.

Consequence: long-context concurrency is pinned at 2 (`lanes_p50 = 2`,
`lanes_max = 4` under pressure) no matter how much of the prompt this pass would
forward, exactly as §R7 predicted — and with the chunk share also dividing by
queue depth, budget utilisation collapses to **12–14 %**.

### R-2 — `break` on the first refusal drops a median of 11 admissible lanes per pass.

`schedule_next_batch` ends the pass at the first request it cannot seat
(`prefill.py:379`, `break  # We cannot add more requests`) instead of skipping it.
At each of the 2 404 refusals the pass abandoned a median of **13** queued
requests, of which a median of **11** were admissible by the pools' own
accounting (continuations, which the gate never applies to, plus fresh prompts
that fit). **23 628** admissible lane-slots dropped across one 20 000-forward run.

Because R-1 makes the refusal happen on a *long* prompt near the head, the
short-prompt tail of the queue is repeatedly skipped — head-of-line blocking on
top of the reservation problem.

### R-3 — `lanes.diff` does not bind on this traffic. It is a no-op here.

Applied on top of `d685e99`, the pending chunk-share fix produces **byte-identical
output** in both profiles: same 2 814 602 tokens, same 181 completions, same
`lanes_mean` 2.181, same chunk-limit distribution (p50 512, min 512, max 4 475).

Why: `admissible_lanes` charges each lane a table slot (fresh only) and **one KV
page**, and deliberately does not model the whole-prompt admission reservation
("a prefix hit can only make this lane cheaper", `cached_len = 0`). With ~23 000
free pages and a queue that is mostly continuations, it returns ≥ `waiting`, so
`min(waiting, admissible_lanes) == waiting` and the divisor is unchanged. The
docstring's own failure case — "sixteen queued requests cost a 16x smaller chunk
even in a pass whose pools can seat two lanes" — is real, but the reason only two
lanes are seated is R-1, which `admissible_lanes` does not consult.

The fix is not wrong; it is aimed at the wrong currency for this workload. It
would bind where table slots or GDN state slots are the shortage (the
`mamba-slot 96/96` regime), not where the whole-prompt KV reservation is.

### R-4 — Net regression against upstream is 2.5x on prefill throughput.

| | merge-base | `d685e99` | ratio |
|---|---|---|---|
| stage: tokens prefilled | 7 103 059 | 2 814 602 | **0.40x** |
| stage: requests completed | 404 | 181 | **0.45x** |
| stage: budget utilisation | 98.8 % | 14.3 % | 0.14x |
| pressure: tokens prefilled | 8 577 078 | 5 000 774 | 0.58x |
| pressure: requests completed | 83 | 60 | 0.72x |

The interleave does buy something — wait-to-first-chunk p50 improves under
pressure (1 146 s → 955 s) because no single prompt monopolises the budget — but
it is paid for with 40 % of the goodput, and the p95 does not improve on stage
(395 s → 402 s). On this workload the "fair lanes" change is a net loss on both
axes it was meant to trade between.

### R-5 — The documented lane rotation is dead code, in two commits.

`prefill.py:388–391`:

```python
self.pending_list = (
    remaining + chunked_list
    if self.interleave_chunks and stopped_for_lane_cap
    else chunked_list + remaining
)
```

`stopped_for_lane_cap` is only ever set inside `if lane_cap and len(reqs) >= lane_cap`,
and `lane_cap = max_batch_seqs = 0` for every non-GGUF model. The rotation branch
therefore **never executes** on the soak's model, and the comment above it
("Interleaved mode rotates unfinished lanes behind requests that did not run this
pass") does not describe the running behaviour. Continuations are always
re-queued at the head.

It was dead in `f3c3ac4` too, differently: the original guard was
`if self.interleave_chunks and not remaining`, and when `remaining` is empty the
two branches are equal by construction — a tautological no-op. `fc442a1`
replaced one dead form with another.

Low severity today (heads-first is the safe ordering given the `break`), but the
code claims a fairness property it does not provide.

### R-6 — The decode-batch `committed_pages_required` margin shrank ~12x, but never went negative.

Minimum `allocatable - needed` over 20 000 forwards, pressure profile:
merge-base **33 773** pages on decode batches, `d685e99` **2 726**. Prefill
batches: 34 027 → 2 916. Still positive everywhere, in every commit, in every
profile — no `committed_pages_required` fatal reproduced at any commit. The
headroom that used to be an order of magnitude is now four decode bursts wide.

### R-7 — A prompt larger than the KV pool livelocks instead of erroring (pre-existing).

With `--num-tokens` reduced to 131 072 and a 131 327-token prompt, every commit
including the merge-base enters a state where `schedule_next_batch` returns
`None` forever with work outstanding: `_try_allocate_one` can never satisfy
`estimated_len + output_len <= available_size`, and there is no rejection path.
In the real server the pass falls through to `_reclaim_for_blocked_prefill` and
the request hangs to the client timeout. Not reachable in the soak's
configuration (`--num-tokens 262144` > `--max-seq-len-override 131072`), and not
introduced by any unpushed commit — recorded because the replay hits it
instantly and it is a one-line admission check away.

---

## 6. What this replay cannot say

* **Non-hybrid.** It drives the plain `radix` cache: no `LinearStatePool`, no GDN
  state slots, no SWA, no session leases. The soak's `mamba-slot 96/96` pressure
  and `reserve_mamba_slots` escalation are not modelled. (The stage-route
  collapse itself logged `#mamba-slot: 7/96`, so the starvation is faithfully
  reproduced without them — but `dcb617a`, `5c8e964`, `140370c` and `91b22b6` are
  effectively untested here, which is why they show as no-change rows.)
* **The pre-`fad1fc4` fatal did not reproduce.** `continuation_chunk_cap` is 0 in
  every run at every commit: `fad1fc4`'s cap never bound, because R-1 refuses the
  fresh admits long before enough continuations accumulate to outrun the pool.
  Pools from 131 072 to 262 144 and three traffic profiles were swept. This
  replay can therefore neither confirm nor deny residual risk inside `fad1fc4`
  and `d685e99`; the soak already showed empirically that they close the fatal.
* **Closed-loop traffic.** Commits that serve more requests also see different
  arrivals. Forwards (20 000) and seed are held fixed, not the request stream, so
  "tokens prefilled" and "completed" are goodput at equal scheduler work — which
  is the intended comparison — not a fixed-workload latency measurement.

---

## 7. Suggested order of attack (not implemented)

1. Charge the fresh-admit gate the **chunk** this pass will forward plus a
   floor for the request's own decode, not `remain_len + output_len`. This is
   R-1 and it is worth ~2.5x on this workload. It is the same "reserve the whole
   prompt" mistake `d685e99` already fixed one layer down, so the argument is
   already written.
2. `continue` past a request the pass cannot seat instead of `break`ing, with a
   bound so the scan stays O(queue). This is R-2, and it is what makes (1) safe
   for the short-prompt tail.
3. Only then re-evaluate `lanes.diff`: once (1) lands, `admissible_lanes` and the
   real seating limit finally agree, and the chunk share starts doing what its
   docstring claims. Land it with a chunk floor (~2 K tokens) as §R7 suggests.
4. Delete or repair the `stopped_for_lane_cap` rotation (R-5), and add the
   `deferred_prefill_chunks` / `capped_prefill_chunks` counter §R5 asked for —
   the replay had to monkey-patch `try_add_one` to learn why a pass stopped,
   which is exactly the observability gap that ticket describes.

## 8. Artifacts

`/tmp/claude-1000/-home-lucas-ai-FreeToken/f4e2e9e3-f4f5-40d0-9980-b3b09d1ef47d/scratchpad/bisect/`

* `stage_replay.py` — the replay (`--profile stage|pressure|fanout`,
  `--diagnose`, `--pool`, `--no-interleave`)
* `run_all.sh` — the per-commit driver; `final_stage.jsonl`, `final_pressure.jsonl`
* `run_pressure.sh`, `run_tight.sh` — the pool sweep behind R-6/R-7;
  `pressure.jsonl`, `tight.jsonl`
