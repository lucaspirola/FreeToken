# `--speculative ngram` — implementation and measurement (Nemotron-3.5-Lightning-30B-A3B-NVFP4, RTX 5080)

Ticket 3 of `benchmarks/results/nemotron35_lightning_5080_ngram_spec_2026-09-05.md` §6, built on
`89b632b` now that ticket 1 (`..._extend_moe_2026-09-05.md`) has made a short extend forward cost
~30 ms instead of ~290. Host: RTX 5080 16 GB, WSL, `FREETOKEN_PIN_BUDGET_GB=17`,
`--moe-backend offload --moe-cache-auto --nvfp4-backend triton --kv-cache-dtype q8_0`,
`max_running_req=1`.

> **Verdict: SHIP, opt-in — a small win, well short of the projection, and the two reasons
> why are both measured.**
>
> The machinery is right: a self-check that replays the accepted prefix and compares it with
> what the verify forward itself wrote agrees to **0.000e+00** on both the recurrent block and
> the conv window, on every step it ran. The n = 8 precision gate does its job — code and prose
> draft on 0.5–0.7 % of steps and come out at **1.03x / 1.02x**, where the literature's n = 3
> would have cost 12–14 %.
>
> But the copy class lands at **1.01–1.11x across runs (best estimate ~1.05x)**, not the
> projected 1.63x, and the shortfall decomposes cleanly:
> **(a) the draft rate is 0.079 against the offline replay's 0.353**, because engagement is
> decided one token stale and a burst is therefore entered one step late (§7); and
> **(b) an end-to-end verify step costs 42–52 ms against the ~30 ms extend forward inside it**,
> so ~40 % of it is drain, batch preparation and a 46-launch eager commit (§5.1). Both are
> ordinary optimisation work with quantified upside, and neither is a design flaw.
>
> Caveat 1 — **speculation is not token-identical to non-speculative greedy decoding, and cannot
> be on this engine.** A verify step computes its logits with the *extend* kernels and commits
> its state with the *SSD scan*, where a decode step uses the graphed decode kernels and the
> recurrent step; those are different reduction orders. With the control arm reproducing the
> baseline **exactly** (`off == off2` on 4/4 prompts), the speculative arm diverges at token
> 40–71 of 1 023 on three of them and is identical on the fourth. §4 is the measurement that
> proves this is not a bug.
>
> Caveat 2 — **at 131K context a verify step costs ~10x a decode step**, so `k + 1 = 9` cannot
> pay for it. A measured break-even gate (§6) shuts speculation off once it has priced itself,
> but the pricing costs two verify steps, which on a 78-token generation IS the whole −11 %.
> Long-context speculation needs a wider draft or a cheaper verify step, not a better drafter.

---

## 1. What a step does

After an ordinary decode step a request sits in one fixed shape: `cached_len == L` (the recurrent
and KV state cover `tokens[:L]`), `device_len == L + 1`, and `tokens[L]` is the token just
sampled and not yet forwarded. A verify step:

1. **drafts** `k` tokens — the continuation of the most recent earlier occurrence of the trailing
   8-gram of *this request's own* prompt + output — and stages them into `token_pool` at
   `L+1 .. L+k` (the forward reads its ids from the device pool, not from the host tensor);
2. runs **one extend forward** over the `m = k + 1` positions `L .. L+k`, keeping **every** logits
   row (`Batch.logits_indices`), greedy-argmaxed in `Engine.spec_verify_forward` — no
   `complete_one`, no sampler, always eager;
3. **accepts** the longest prefix of the draft the argmax agrees with, plus the bonus token the
   first disagreeing row predicts, so a step emits `accepted + 1` tokens and is never *less*
   productive than a plain decode step;
4. **commits**: the accepted prefix into the live Mamba-2 state (§3), the rejected positions' KV
   pages back to the allocator (§2), one `DetokenizeMsg` per accepted token with EOS / stop-string
   / length truncation applied inside the run.

Rejected tokens never enter the host token list, so they cannot reach the prefix cache: the radix
insert boundary is `req.cached_len`, and this path only ever advances it by the accepted count. A
verify batch is drained by the speculative decoder itself, so no `cache_req` runs on it at all.

## 2. KV rollback — the one place a leak was real

`allocate_paged` allocates `[page_ceil(cached_len), page_ceil(device_len))` and nothing in the tree
ever removed a suffix. A verify step widens `device_len` by `k`, so on a partial rejection the
pages of the rejected positions are allocated, unreferenced, and re-allocated by the next step:
**`k - j` pages leaked per partial rejection**, which on a copy burst is a page per two tokens.

`CacheManager.free_spec_tail(req, keep_len=cached_len, alloc_len=L+m)` returns them, restoring the
invariant the allocator assumes — *pages exist exactly up to `page_ceil(cached_len)`*. `keep_len` is
`cached_len`, not `device_len`: the newly emitted token's page must be allocatable by the next
step, exactly as after an ordinary decode step.

## 3. Mamba-2 state — the design held, with one correction

The go/no-go's §5 design is what shipped: never advance the live state speculatively. The verify
forward's `fla.cache_indices` points at a scratch slot pre-loaded with the live state, so
`mamba2_prefill`'s scatter and `causal_conv1d_varlen`'s conv write both land there; each mixer
records its own scan inputs (`x`, `dt`, `B`, `C`, `conv_in`) for the `m` positions; after the
acceptance count is known, one varlen SSD scan per layer replays the first `n = accepted + 1`
positions from the **live** slot, and the conv window slides by the same `n`. A full acceptance
skips the replay and copies the scratch slot back.

**Correction to the design: not the ping-pong pair.** §5 proposed reusing "the per-request
`mamba_ping_pong` pair already reserved by the scheduler". Those two slots are not spare — one
holds the tool-call anchor freeze (`snapshot_toolcall_anchor`) and the other is the destination of
the next radix chunk snapshot; overwriting either corrupts a prefix-cache donation. Speculation
allocates its own `Req.spec_scratch_slot`, returned by `_free_req_slots` with the rest.

**And the pool's free-list is normally empty.** In steady-state decode a request holds live + 2
ping-pong and the radix tree owns every donated snapshot, so `num_free_slots == 0` is the *common*
case. The first measured run had the drafter firing and the verify step silently declining for
exactly this reason. The decoder now escalates to tier 2 (`ensure_mamba_slots` — LRU eviction of
unlocked tree snapshots) but deliberately **not** tier 3 (`reserve_mamba_slots`, which spills a
session lease): checkpointing an idle conversation to fund an optimisation is not a trade this
feature gets to make.

### 3.1 The commit is bit-exact — measured, not argued

`FREETOKEN_SPEC_CHECK_COMMIT=<n>` replays *all* `m` positions into a spare slot (forcing the replay
path past the full-acceptance shortcut) and compares against what the verify forward left in the
scratch slot. Same kernels, same initial state, same tokens, so they must agree:

```
spec commit self-check: recurrent |d|max=0.000e+00 conv |d|max=0.000e+00 (m=9)
```

**0.000e+00 on both, on every one of the 20 steps it was run on.** This is the measurement that
separates "the commit is wrong" from "the extend kernels are a different reduction order than the
decode kernels" — a greedy diff against the non-speculative arm cannot tell those apart, and
without it §4's divergence would have been unattributable.

## 4. Greedy equivalence — agreement, not identity, and why identity is unavailable

Three arms per prompt in one model load, warm prefix tree and warm expert cache before the first
timed arm: **off**, **on**, and **off2** — a second non-speculative arm run *last*. Without off2 an
"on != off" verdict cannot distinguish speculation from run-to-run nondeterminism.

| prompt | off == off2 | on == off | first divergence |
|---|---|---|---|
| code (1 023 tok) | **yes** | no | token 71 |
| prose (1 023 tok) | **yes** | no | token 40 |
| copy (1 023 tok) | **yes** | no | token 58 |
| needle 131K (78 tok) | **yes** | **yes** | — |

The engine is deterministic (off == off2 on 4/4). Speculation changes the stream on 3 of 4, and
the reason is structural, not a defect:

- the accepted and bonus tokens are argmaxed from **extend-path** logits, where the non-speculative
  arm argmaxes **decode-path** logits — the same computation in a different order;
- the committed recurrent state comes from a **chunked SSD scan** over `n` tokens where the
  non-speculative arm's comes from `n` sequential **recurrent decode steps**.

Both are float-noise perturbations to a state the model then carries forward, so a near-tie flips
within a few tens of tokens — the same failure mode the 2026-09-04 extend-tile and 2026-09-05
extend-MoE changes were held to, and the same standard: agreement, not bitwise equality. The 131K
needle answered **identically** in both arms and recalled the passcode in both.

**This is a property of multi-token verification on this engine, not of this drafter.** MTP, EAGLE
and every other speculative scheme would carry it, because none of them can make the extend path
bit-identical to the graphed decode path.

## 5. Throughput and acceptance

One model load, 1 023 output tokens (78 for the needle), greedy, natural stop, the three prompt
classes of the go/no-go plus a 130,904-token needle. Three arms per prompt — **off**, **on**, and
**off2**, a second non-speculative arm run *last*. `draft_rate` is the fraction of scheduler steps
that issue a draft; `lambda` is tokens emitted per scheduler step. Shipped code (break-even gate
enabled):

| class | prompt | off | **on** | off2 | verify steps | draft_rate | accept_rate | tok/verify | lambda | **speedup** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 87 | 135.8 | **140.4** | 136.8 | 5 / 988 | 0.005 | 0.825 | 7.6 | 1.035 | **1.03** |
| prose | 66 | 138.3 | **141.9** | 141.9 | 7 / 957 | 0.007 | 0.250 | 2.7 | 1.069 | **1.02** |
| copy | 1 129 | 135.8 | **136.9** | 133.6 | 54 / 685 | 0.079 | 0.798 | 7.1 | 1.495 | **1.01** |
| needle | 130 904 | 87.3 | 77.3 | 86.9 | 2 / 55 | 0.036 | 0.625 | 6.0 | 1.418 | **0.89** |

tok/s at bs=1. Run-to-run spread on the non-speculative arms is 1.6–3.5 %, so single-digit
percentages here are at the edge of resolution; the drafter statistics (`verify_steps`, `lambda`)
are reproducible to within one step across runs and are the more reliable signal.

**The precision gate works, and that is the load-bearing result.** Code and prose draft on
0.5–0.7 % of steps — the offline study predicted 0.4–0.2 % — so they cannot regress, which is
exactly what n = 8 was chosen for. The offline sweep showed n = 3 costs 12–14 % on these two.

**An ungated control run** (same binary, break-even gate removed) reached **147.2 tok/s on the
copy class against 133.1 / 134.4 for its own off / off2 arms — 1.11x** — with drafter statistics
within one step of the gated run (55 verify steps, lambda 1.493). Since the gate declined only 5
of 59 copy-class drafts, the two runs did essentially the same work, and the gap between 1.11x and
1.01x is run-to-run variance on a quantity whose true value is somewhere in between. **Treat the
copy-class win as ~1.05x with a spread of several points, not as 1.11x.**

### 5.1 Where the projection went — two measured causes, both fixable

The go/no-go projected 1.63x from `lambda / (draft_rate * cost_ratio + (1 - draft_rate))`. Both
inputs came out worse than assumed:

| quantity | projected | measured | why |
|---|---:|---:|---|
| `draft_rate` (copy) | 0.353 | **0.079** | engagement is decided one token stale, so a burst is entered one step late (§7) |
| verify step cost | ~30 ms (the forward) | **42–52 ms** | the forward is ~30 ms; the rest is drain + batch prep + a 46-launch eager commit |

The verify cost is backed out of the copy row: 685 steps of which 631 are plain, so
`631 x 7.4 ms + 54 x V = 7.48 s` gives **V ≈ 52 ms, ~7x a decode step** where the extend forward
alone is 3.6–4.0x (the extend-MoE write-up measured 27.7–30.2 ms at m = 1..8). **~40 % of a verify
step is not the forward.** The commit alone issues 46 eager kernel launches (23 SSD scans + 23 conv
window writes) plus four `.contiguous()` copies per layer.

Put the measured numbers back into the projection: `1.495 / (0.079 x 7 + 0.921) = 1.03`. The model
is consistent with the measurement — it was the inputs that were optimistic, not the arithmetic.

## 6. Long context — a verify step costs ~10x, and the gate is measured

The needle row is a real regression and the arithmetic says why. At 131K a decode step is ~11.5 ms;
the speculative arm spent ~115 ms more than the baseline while saving 10 decode steps, so its 2
verify steps cost **~118 ms each, ~10x a decode step.**

That follows from the shape of the work: a verify step's extend attention reads the whole KV
history **once per query token**, where a decode step reads it once. The 4.4x the go/no-go
projected was a short-context ratio, and break-even needs `accepted + 1 > verify/decode`, which at
10x is out of reach for `k + 1 = 9`.

**The fix is not a context-length flag.** The decoder estimates both terms online — the gap
between consecutive peeks that took the ordinary path *is* one overlapped decode step, and a
verify step times itself — and drafts only while `emitted x 1.25 > verify_ms / decode_ms`, with one
probe every 16 gated steps so a closed gate stays falsifiable. Context length, KV dtype, attention
backend and acceptance all move that ratio; a threshold would have to be re-tuned for each.

**Two tuning facts, both learned the expensive way** (each cost a GPU run):

- **The two sides need different estimators.** `verify_ms` is a running **minimum**: it is sampled
  a few dozen times at most and the first sample pays Triton autotuning for a shape nothing else
  uses — as an EWMA that one-off read 11.35 where the steady state is 4–5, closed the gate, and
  then starved it of the samples that would have reopened it. `decode_ms` is an **EWMA**: it is
  sampled hundreds of times and the scheduler loop's gap is not uniform, so its minimum sits far
  below a real decode step — as a floor it inflated the ratio and gated out **264 of 278**
  copy-class drafts, turning +11 % into −0.3 %.
- **The gate must be hard to close and easy to leave open**, because a false close costs a real
  win and a false open costs one verify step per 16. It therefore requires two timed verify steps
  and a 25 % margin. With those, it declines **5 of 59** copy-class drafts and **14 of 16** at
  131K — which is the intended shape.

**What the gate cannot fix: the price of the measurement.** Two verify steps at 131K is ~236 ms,
and on a 78-token generation that IS the whole −11 %. Over a long generation it amortises to
nothing; over a short one it does not. A context-aware *prior* on the first estimate would remove
it, at the cost of the threshold this design set out to avoid.

## 7. Engagement: why it runs drained, and what that costs

A drafter needs every emitted token before it can index the next n-gram, so a verify step cannot
overlap with its own successor. Running the whole decode loop drained would cost ~30 % on the
99.3–99.5 % of code/prose steps that never draft, so the engagement decision is made **before** the
drain, using the one-token-stale token list: `peek()` is one dict lookup, and only a hit pays for
the drain.

The cost of that hysteresis is the **first step of every burst**: a burst that begins at position
`p` is only entered at `p + 1`, because the stale key at `p` is the key from `p - 1`. With bursts
averaging ~2 verify steps on the copy class, that is roughly a factor of 4 in draft rate — the
0.079-against-0.353 gap of §5.1. Ticket 2.

## 8. Files

- `python/freetoken/scheduler/spec_ngram.py` — `NgramDrafter`, `accepted_count`, `SpecStats`,
  `SpecNgramDecoder` (engagement, verify, commit, rollback, emission, break-even gate).
- `python/freetoken/models/nemotron_h/spec_scan.py` — `SpecScanCapture`: per-layer scan-input
  recording, the accepted-prefix commit, and `replay_error` (§3.1).
- `python/freetoken/models/nemotron_h/model.py` — one `batch.spec_capture` hook in the mixer.
- `python/freetoken/core.py` — `Batch.logits_indices` / `Batch.last_indices` / `Batch.spec_capture`,
  `Req.spec_scratch_slot`.
- `python/freetoken/layers/embedding.py`, `kernel/triton/nvfp4_linear.py`, and the four per-model
  LM heads — route through `Batch.last_indices`.
- `python/freetoken/engine/engine.py` — `Engine.spec_verify_forward`.
- `python/freetoken/scheduler/cache.py` — `free_spec_tail`, and the scratch slot in
  `_free_req_slots`.
- `python/freetoken/scheduler/scheduler.py` — the `peek()` hook in both loops, construction,
  elastic slot remap.
- `python/freetoken/scheduler/status.py` — `generated_tokens` so a verify step's tokens reach the
  decode-throughput log.
- `python/freetoken/scheduler/config.py`, `python/freetoken/server/args.py` — `--speculative`,
  `--spec-ngram-n`, `--spec-draft-len`, `--no-spec-adaptive`.
- `tests/scheduler/test_spec_ngram.py` — 26 CPU tests; `tests/e2e/test_spec_ngram_equivalence.py`
  — CUDA, `needs_weights`.
- `benchmarks/probe_spec_ngram_impl.py`; `benchmarks/switchyard_soak/{run,serve}.sh` gained
  `SOAK_PHASES` / `SOAK_PROBE` / `SOAK_EXTRA_ARGS` for the A/B.

## 9. Reproduction

```
# three prompt classes + a 131K needle, off / on / off2, one model load
FREETOKEN_PIN_BUDGET_GB=17 PYTHONPATH=python .venv/bin/python \
  benchmarks/probe_spec_ngram_impl.py --model <lightning> --moe-cache-auto \
  --max-tokens 1024 --needle-max-tokens 256 --out spec.json
# (run it through scripts/gpu_lock.sh from a wrapper script, redirected to a file)
# add FREETOKEN_SPEC_CHECK_COMMIT=20 for the state self-check of §3.1

# 16-way passthrough soak, spec off then on
SOAK_PHASES=passthrough SOAK_PROBE=0 benchmarks/switchyard_soak/run.sh spec_off 10m
SOAK_PHASES=passthrough SOAK_PROBE=0 SOAK_EXTRA_ARGS="--speculative ngram" \
  benchmarks/switchyard_soak/run.sh spec_on 10m
```

Do not pipe `scripts/gpu_lock.sh`: its exit trap `pkill -9`s its own process group and kills the
reader. Redirect and grep the file.

## 10. Still open — tickets

Ordered by measured upside.

1. **~40 % of a verify step is not the forward** (§5.1): ~52 ms end-to-end against a ~30 ms extend
   forward. The commit issues 46 eager kernel launches (23 SSD scans + 23 conv-window writes) and
   four `.contiguous()` copies per layer, and `_prepare_batch` rebuilds pinned staging tensors for
   a one-request batch. Batching the per-layer commit into one launch, and reusing the staging
   buffers, is the single biggest lever: at the copy class's `draft_rate` 0.079, taking the verify
   step from 7x to 4x a decode step moves the projection from 1.03x to **1.12x**.
2. **The burst-entry hysteresis costs a factor of ~4 in draft rate** (§7): 0.079 measured against
   0.353 offline. A stickiness latch — stay engaged for a few steps after a verify step even when
   `peek()` misses — recovers the entry step at the price of a few drained plain steps per burst.
   Decidable in one run: `draft_rate` and tok/s on the copy class at latch 0 / 2 / 4. With ticket 1
   this is worth roughly another 1.2x on the copy class; without it, much less.
3. **Long context needs a wider draft or a cheaper verify step.** Break-even at 131K needs
   `accepted + 1 > ~10` and the ceiling is `k + 1 = 9`. `--spec-draft-len 16` reaches 17 and the
   copy class accepts 80 %, but the verify step's own attention cost grows with `m` too. One sweep
   of `--spec-draft-len` 4 / 8 / 16 / 24 on the 131K needle answers it, and the break-even gate
   makes it safe to try.
4. **The 16-way soak's tail is unresolved** (§11). p50 and request count are flat; p95 and p99 are
   not, and one 10-minute pair with visibly different session/marker mixes cannot separate that
   from variance. Re-run at the reference 20-minute phase length.
5. **Batched (bs > 1) speculation.** `_make_write_tuple` and the drain loop are one token per
   request, and the verify step is single-request by construction here. A batched verify also needs
   the acceptance count per row and a per-row state commit.
6. **A graph-captured fixed-width verify forward** (ticket 2 of the go/no-go). At short context
   about half of the ~30 ms forward is now Mamba-2, not MoE.
7. **Sampling (non-greedy) speculation** needs `Sampler.prepare` to repeat-interleave its
   per-request parameter rows by `k`, plus the modified-rejection acceptance rule.
8. **The drafter indexes the whole prompt on first engagement** — ~0.1–0.2 s at 131K, amortised
   over a long generation but visible as one slow step. Chunk it across steps if it ever matters.

## 11. 16-way soak — aggregate flat, tail unresolved

`benchmarks/switchyard_soak/run.sh` at concurrency 16, passthrough route, 10 minutes per arm, one
server boot per arm, `SOAK_EXTRA_ARGS="--speculative ngram"` on the second. Speculation engages
only when exactly one request is running and nothing is queued, so at 16-way it is close to a no-op
by construction — the point of the run is that nothing else broke.

| | spec off | spec on |
|---|---:|---:|
| requests / errors | 961 / **0** | 956 / **0** |
| latency p50 | 6 691 ms | 6 960 ms |
| latency p95 | 29 550 ms | 36 001 ms |
| latency p99 | 41 098 ms | 95 058 ms |
| mean #running-req | 12.08 | 12.34 |
| lanes per prefill batch | 4.98 | 5.18 |
| starvation signature | 0 / 304 | 0 / 292 |
| mamba slots at 100 % | 6 batches | **0 batches** |
| `linear_exhausted` / `invariant_violated` / traceback | 0 | 0 |

Both arms **PASS** with zero errors, p50 within 4 %, request count within 0.5 %, and no new failure
markers — in particular no `LinearStatePool exhausted`, which is the marker the tier-2 scratch-slot
escalation of §3 could plausibly have produced.

**The tail is not settled by this pair.** p95 +22 % and p99 +131 % is a large gap, but the two runs
also saw materially different traffic — `discarded_cold` 172 vs 126, `released_admission` 257 vs
471, `session_expired` 0 vs 48 — and the soak that established this harness's reference numbers
used **20-minute** phases for exactly this reason. There is a plausible mechanism (a verify step is
a drained scheduler-loop stall, and at long context it is ~118 ms) but also a plausible null, and
one 10-minute pair distinguishes them at neither. Ticket 4.
