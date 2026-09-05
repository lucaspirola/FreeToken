# Making `--speculative ngram` pay — the verify step, the engagement decision, and the draft length

Follow-up to `nemotron35_lightning_5080_ngram_spec_impl_2026-09-05.md`, which shipped
`--speculative ngram` at **1.01–1.05x on the copy class** and named the two measured reasons
(§5.1, §7, tickets 1–2 and 6). This is those tickets, closed. Host: RTX 5080 16 GB, WSL,
`FREETOKEN_PIN_BUDGET_GB=17`, `--moe-backend offload --moe-cache-auto --nvfp4-backend triton
--kv-cache-dtype q8_0`, `max_running_req=1`, greedy, 1 023 output tokens.

> **Verdict: the verify step is 34 % cheaper (54.0 -> 35.6 ms), and with the draft length
> raised to 16 the copy class projects to 1.88x against 1.11x for the shipped code — measured
> on a fixed transcript, because the end-to-end arms cannot resolve it.** Three changes and
> two corrections:
>
> 1. **One SSD scan for all 23 layers instead of 23.** Mamba-2 heads are independent and every
>    Nemotron-H mixer has the same `(head_dim, state_size, heads_per_group)`, so the layer axis
>    folds onto the head axis: 23 x 64 heads is one 1 472-head sequence, `A` and `dt_bias`
>    concatenate, and `D` is dropped because it feeds only the scan output the commit discards.
>    **~280 kernel launches -> ~11, 7.12 -> 0.45 ms of host time, and bit-exact** (0.000e+00 on
>    the recurrent block and the conv window at eight (m, n) shapes, measured weightlessly).
> 2. **The verify batch is built from its own fixed shape** rather than through
>    `Scheduler._prepare_batch` — 0.80 -> 0.34 ms, and one `Sampler.prepare` that a greedy
>    verify forward never reads is gone.
> 3. **Engagement is decided post-drain**, via a superset predictor that cannot miss a burst
>    entry (§3). It is provably non-lossy and worth ~2 % of λ.
>
> **Correction 1 — ticket 2 ("burst-entry hysteresis costs a factor of ~4 in draft rate") is
> wrong, and the previous write-up's own numbers were the artifact.** Replaying the baseline
> transcript through both peeks (§8) puts the stale-exact peek at draft rate 0.484 / λ 4.59 and
> the superset peek at 0.484 / 4.67 — a 2 % gap, not 4x. The 0.079-against-0.353 that motivated
> the ticket was **stream variance**: the shipped arm's own output simply did not spend much of
> its 1 024-token window copying. The new peek is still the right code (it is strictly better
> and provably cannot miss), but it is not the lever.
>
> **Correction 2 — a single copy-class arm cannot measure this feature to better than a
> factor of two.** Speculation perturbs its own token stream, and the copy prompt's model
> output opens with a few hundred tokens of reasoning before the verbatim copy starts, so where
> that transition lands inside the 1 024-token window decides the draft rate. Measured spread
> across arms that differ in nothing but the stream: **1.04x to 1.67x**. §8 is therefore the
> load-bearing measurement — a fixed-transcript replay with the measured per-step costs — and
> §1 reports the end-to-end arms as the range they are.
>
> **The graph-captured verify forward (ticket 6) is a measured NO-GO**, and the measurement is
> the point: at m = 9 the eager forward's *host launch path is 30.6 ms and its GPU time is
> 36.4 ms*, so the Python already runs underneath the GPU and a graph has almost nothing to
> recover. At 131K it is 31.0 ms of host against 91.8 ms of GPU.

---

## 1. Throughput — one model load, four arms per class

`benchmarks/probe_spec_ngram_impl.py --variants v0 v1`: **off**, **on/v0** (the shipped
2026-09-05 code paths), **on/v1** (all three changes), **off2** (a second non-speculative arm,
run last). Same weights, same warm prefix tree, same warm expert cache. `v0` and `v1` are the
same binary with three booleans flipped, so the comparison carries no build difference.

| class | prompt | off | **on/v0** | **on/v1** | off2 | v0 speedup | **v1 speedup** |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 87 | 139.1 | 138.8 | 134.8 | 141.7 | 1.00x | 0.97x |
| prose | 66 | 146.5 | 146.4 | 143.2 | 142.3 | 1.00x | 0.98x |
| **copy** | 1 129 | 138.9 | **144.9** | **210.5** | 134.0 | 1.04x | **1.52x** |
| needle | 129 921 | 131.9 | 107.2 | 110.8 | 127.3 | 0.81x | 0.84x |

tok/s at bs=1, 1 023 output tokens (79 for the needle). `off` vs `off2` spread is 1.9 / 2.9 /
3.7 / 3.6 %.

**Read the copy row as a sample from a wide distribution, not as the number.** A second session
(same binary, arms `v0 v1 v1b v1c k12 k16`) put the copy class at:

| arm | tok/s | draft rate | λ | tokens/verify |
|---|---:|---:|---:|---:|
| off / off2 | 138.4 / 133.7 | — | — | — |
| v0 (k = 16) | 226.2 | 0.355 | 6.17 | 15.5 |
| v1 (k = 16), and two byte-identical repeats | 158.8 / 164.1 / 168.5 | 0.055 | 1.70 | 13.6 |
| v1 (k = 12) | **231.1** | 0.416 | 5.75 | 12.4 |

A third session ran the copy class **first** instead of third, with the shipped defaults and
nothing else changed: `off` 138.4, **`v1` 138.1**, `off2` 138.9 — and its drafter statistics
(draft rate 0.0365, λ 1.205, 4.94 tokens per verify step) are *byte-identical* to the `n8k8` arm
of the sweep session, which also ran the copy class first. **The same configuration reads 210.5
tok/s when the copy class runs third in the session and 138.1 when it runs first**, reproducibly
in both cases, because the accumulated prefix-cache state changes where the model's reasoning
preamble ends and the verbatim copy begins.

The three repeats of one configuration return **identical drafter statistics** — the engine is
deterministic and the arm is exactly reproducible — and their tok/s spread is 1.8 %, which is
the measurement noise. What is *not* stable is the comparison between configurations: `v1` at
k = 12 drafted on 41.6 % of steps and `v1` at k = 16 on 5.5 %, because their token streams
diverge in the model's reasoning preamble and one of them reached the verbatim-copy phase inside
the 1 024-token window while the other barely did. **Any single copy-class arm is measuring that
lottery as much as it is measuring the drafter**, which is why §8 replays a fixed transcript
instead.

**Code and prose do not move, and cannot.** Both drafted on 0.2–0.6 % of steps in every arm
(1–5 verify steps in 1 023 tokens), so speculation has ~4 steps in which to change anything:
4 verify steps at 41.6 ms replacing ~20 decode steps at 7.3 ms is **+20 ms on a 7.3 s run,
0.3 %.** The 0.97x/0.98x readings are the control spread, not the feature — which is what the
n = 8 precision gate exists to guarantee, and it still holds with the superset peek (§8 confirms
it on a fixed transcript: 0.99x at every k).

**The needle still regresses, for the reason the previous write-up gave** (§6 there): the
break-even gate correctly refuses 14–16 of ~18 drafts at 131K, but pricing itself costs two
verify steps at ~90 ms, and on a 79-token generation that *is* the whole −16 %. §7.2 shows that
no draft length fixes this, because the verify/decode ratio there is ~12.

## 2. Where a verify step goes — measured per phase, before and after

`SpecStats` now carries a wall-clock breakdown (`/v1/stats["scheduler"]["spec"]["cost_ms"]`),
plus CUDA-event GPU time for the forward and the commit. The forward pair is read on the argmax
sync that a verify step already pays; the commit pair is read on the *next* verify step, so the
breakdown adds no synchronisation of its own. Copy class, 100 / 106 verify steps:

| phase | on/v0 | **on/v1** | what changed |
|---|---:|---:|---|
| draft + stage | 0.01 | 0.01 | — |
| batch preparation | 0.80 | **0.34** | fixed-shape prep, no `Sampler.prepare` |
| forward (host launch) | 50.60 | **30.63** | unchanged code; see below |
| argmax sync | 1.31 | 4.43 | the residue of the forward the host did not cover |
| **state commit** | **1.24** | **0.18** | 23 SSD scans + 23 conv writes -> 1 + 1 |
| emission | 0.03 | 0.02 | — |
| **end-to-end step** | **53.98** | **35.60** | |
| forward, GPU (CUDA event) | 54.72 | **36.39** | |
| commit, GPU (CUDA event) | 1.31 | **0.19** | |
| drain (charged by the scheduler) | 0.15 | 0.15 | |

Read the two forward rows together: in `v1` the host issues the forward in 30.6 ms and the GPU
takes 36.4 ms, so **the launch path already hides underneath the GPU** and the step's wall clock
(35.6 ms) is the GPU time. `v0`'s forward rows are inflated by the previous step's 23-scan commit
still draining on the same stream when the event is recorded — which is itself the point: a
280-launch commit does not only cost its own 1.2 ms of host time, it delays the next forward.

The drain is 0.15 ms, not the 30 % the design feared, because inside a copy burst the previous
step was itself a drained verify step: there is nothing left in flight to wait for. The drain
only costs on the step that *enters* a burst.

## 3. Post-drain engagement — a superset predictor, not a latch

A drafter needs every emitted token before it can index the next n-gram, so a verify step cannot
overlap with its successor and the loop must be drained to run one. Draining every step costs
~30 % on the 99+ % of code/prose steps that never draft, so the engagement decision has to be
made **before** the drain — with a token list that is one token short of the one the verify step
will query.

The shipped code asked the *exact* question of that stale list: "is the trailing 8-gram indexed?"
A burst that begins at position `p` has its key completed by token `p` itself, so the stale answer
at `p` is the answer for `p − 1`: **the burst is entered one step late.**

The fix is to ask a question the stale list *can* answer. The key the verify step will use is
`(T[p−7 .. p−1], T[p])`; its first n − 1 tokens are already known. So the drafter keeps, next to
the n-gram index, the set of **(n−1)-prefixes** of every indexed n-gram, and the pre-drain peek
tests membership there. That is a strict superset of the exact test — no draftable step can be
missed — and the exact test still decides, after the drain, inside `run_step`, which declines and
falls through to the ordinary path when the prediction does not hold.

The prefix set stores `hash(key[:-1])`, not the tuple: ~8 MB and one C-level hash per token at a
131K prompt, against a second tuple table. `tests/scheduler/test_spec_ngram.py` checks the
superset property over a random stream and pins the burst-entry case.

**Measured on a fixed transcript (§8), which is the only way to see it without the stream
lottery: draft rate 0.484 -> 0.484, λ 4.59 -> 4.67, per-token acceptance 0.935 -> 0.957.**
About 2 %, not the 4x ticket 2 projected — because a burst is only entered late *once*, and
inside a burst the loop is already drained, so the stale peek is not stale at all after the first
step. **Ticket 2's premise was a stream artifact**: the 0.079 draft rate that motivated it came
from an arm whose own output barely reached the copy phase, and the same shipped code measures
0.353–0.484 on arms that do.

The change is still the right code — it is strictly better, it removes an avoidable asymmetry,
and it makes the engagement decision say what it means — but it is a 2 % lever, and this write-up
says so rather than inheriting the projection. The false-positive cost is small and measured:
0 extra drained steps on the copy class, and 2–10 out of ~1 000 on code and prose, worth 0.1–0.6 %
even when each is charged a full 3 ms drain.

## 4. One SSD scan for the whole model

The commit replays the accepted prefix into the live state slot. It ran per layer: 23
`mamba2_prefill` calls, each with its own chunk metadata, `index_select` gather, `.contiguous()`
copies and `index_copy_` scatter, plus 23 conv-window writes — **~280 kernel launches to advance
9 tokens' worth of state.**

Mamba-2 heads are independent, and every Nemotron-H mixer has the same `head_dim`, `state_size`
and heads-per-group, so **the layer axis is a valid head axis**: 23 layers × 64 heads is one
1 472-head sequence and 23 × 8 groups is 184 groups, with the same 8-heads-per-group mapping the
kernel already assumes. `A` and `dt_bias` are per-head and concatenate; `D` is a skip connection
on the scan *output*, which the commit discards, so it is dropped. The recurrent pool is indexed
as `[layers * slots, H, P, N]` so the 23 per-layer states gather and scatter in one pair of calls,
and the 23 conv windows slide in one `cat`.

**Bit-exact, measured twice.** Weightlessly (`benchmarks/check_spec_fused_commit.py`, 23
synthetic layers, no model load):

```
m=9  n=1,2,5,8   rec |d|max=0.000e+00   conv |d|max=0.000e+00
m=17 n=1,2,5,16  rec |d|max=0.000e+00   conv |d|max=0.000e+00
fused:      host 0.454 ms/commit, wall 0.591 ms/commit
per-layer:  host 7.120 ms/commit, wall 7.151 ms/commit
```

Zero, not "within tolerance" — the fused scan and the 23 separate scans reduce over the same
axes per head, and folding independent heads together does not change any reduction. The
`FREETOKEN_SPEC_CHECK_COMMIT` self-check is now a *stronger* gate than before for the same
reason: the verify forward runs the scan per layer, so a fused replay that reproduces it to 0
has proved the fold.

And in situ, with real weights, through `FREETOKEN_SPEC_CHECK_COMMIT=25`, which replays all `m`
positions into a spare slot and compares against what the verify forward — which runs the scan
**per layer** — actually wrote:

```
spec commit self-check: recurrent |d|max=0.000e+00 conv |d|max=0.000e+00   x25 (m = 5 and m = 9)
```

25 of 25 at exactly zero. That is the gate: the forward's 23 separate scans and the commit's one
fused scan agree bit for bit.

The per-layer path is kept and is what runs if a model's mixers are not uniform
(`SpecScanCapture._commit_per_layer`); the fused plan declines rather than guessing.

One trap the weightless check caught that a model load would have hidden: the fused plan is cached
(the concatenated `A` / `dt_bias` and the two index vectors are model constants), and the first
cache key did not include the *weights* — a second set of mixers was served the first set's `A`.
The key now carries `A.data_ptr()`, which changes when a `load_state_dict` rebuilds the cached
`-exp(A_log)`.

## 5. Building the verify batch from its own shape

A verify batch is the most predictable batch the engine builds: one request, prefill phase,
extend length `k + 1`, never multimodal, never chunked, never SWA, and the same shape on every
step of a burst. `Scheduler._prepare_batch` spends most of its work for it on branches that
cannot apply and on pinned staging tensors rebuilt at an identical shape — including a full
`Sampler.prepare`, whose per-request parameter rows this path never reads (the verify forward is
greedy by construction and runs no sampler at all).

`SpecNgramDecoder._prepare_verify` keeps the page allocation, the positions, the page-table
gather and the two metadata builders. Positions and the per-token request row come off persistent
device buffers (`arange + L` into a preallocated slot), so preparation is four kernel launches and
no host->device staging, and the recurrent metadata is cached by `(extend width, state slot)` —
every tensor in it is a function of those two integers, and the mid-chunk snapshot metadata is
asserted absent rather than assumed (`k + 1` never reaches a 128-token chunk boundary).

Measured: **0.80 -> 0.34 ms per verify step.** It falls back to `_prepare_batch` whenever
`--kv-grow-step-tokens` is on, where the batch can change the KV geometry.

## 6. The graph-captured verify forward — a measured NO-GO

Ticket 6 of the previous write-up proposed capturing a fixed-`(1, k+1)` verify forward, on the
theory that an eager forward on this checkpoint is ~27 ms of Python (measured at bs=1 in
`..._decode16_2026-09-05.md` §4c: 33.9 ms eager vs 6.88 ms graphed).

**The breakdown says the theory does not transfer.** At m = 9 the verify forward's host launch
path is **30.6 ms** and its GPU time is **36.4 ms**; at 131K it is **31.0 ms of host against
91.8 ms of GPU**. The Python already runs underneath the GPU, and the end-to-end step (35.6 ms)
equals the GPU time — so a graph can recover at most the difference, which is negative at short
context and heavily negative at long. A 9-token extend routes ~9x more distinct experts than a
1-token decode step, so its cost is expert-cache bytes over PCIe, exactly like the 16-lane decode
step whose graph was worth 4 ms instead of the projected 27.

This is the same lesson as `..._decode16_2026-09-05.md`: **a fixed per-step host cost does not
carry over to a step whose GPU term is longer than the host term.** It is recorded here so the
ticket is closed by a number rather than left open as plausible.

## 7. Draft length and n-gram width — the sweep, and what it can and cannot decide

`--sweep-k 4 8 12 16 --sweep-n 6 8 10`, twelve speculative arms per class, each preceded by `off`
and followed by `off2`. Two model loads: code and prose in one, copy and the 131K needle in a
second (the first run died on the instrumentation itself — a CUDA event pair read on the next
verify step is not always complete inside a tight burst, and `elapsed_time` raises rather than
waits; it is guarded with `query()` now).

**Read the throughput columns of this section through §1**: the per-arm tok/s on the copy class is
subject to the stream lottery. What the sweep measures reliably is the *cost* curve (a kernel
property) and the needle's verdict (where every arm regresses, so the ordering does not matter);
the acceptance-vs-k question is answered on a fixed transcript in §8.

### 7.1 The verify forward's cost is linear in the draft length

The clean result of the sweep is the cost curve, because it is a property of the kernel and not
of the token stream. Copy class, GPU time of the verify forward (CUDA events, 20–120 samples per
arm), against a ~7.3 ms graphed decode step:

| k | m = k + 1 | verify forward, GPU | per emitted token at full acceptance |
|---:|---:|---:|---:|
| 4 | 5 | 33.0–35.1 ms | 6.6–7.0 ms |
| 8 | 9 | 36.4–41.1 ms | 4.0–4.6 ms |
| 12 | 13 | 44.0–47.9 ms | 3.4–3.7 ms |
| 16 | 17 | 56.4–56.9 ms | 3.3–3.4 ms |

End to end a verify step is **33 / 36 / 45 / 55 ms** at k = 4 / 8 / 12 / 16, i.e.
`step ≈ 26 + 1.8·k`, against a 7.3 ms decode step. **k = 4 does not pay**: at *full* acceptance
it already costs a decode step per token, so any rejection is a loss. Everything above it gets
cheaper per token, and §8 — which measures acceptance on a fixed transcript rather than on twelve
diverging ones — shows the acceptance a longer draft needs actually holds on copy-heavy traffic
(0.93–0.99 per token all the way to k = 16, 15.7 of 17 tokens per verify step).

### 7.2 Long context: no draft length reaches break-even (ticket 3, answered)

| arm | 131K needle tok/s | tokens per verify step |
|---|---:|---:|
| off / off2 | 125.0 / 123.4 | — |
| best speculative (n=8, k=8) | 103.9 | 7.0 |
| widest draft (n=8, k=16) | 100.3 | 9.3 |
| n=10, k=12 | 106.4 | 9.0 |

**All twelve arms regress**, from −15 % to −36 %. A verify step at 131K is ~92 ms of GPU against a
~7.6 ms decode step — a ratio of ~12 — and `k + 1 = 17` reaching 9.3 accepted tokens still does not
clear it, exactly as the break-even arithmetic says. The gate does its job (`declined_uneconomic`
15–30 of ~50 peeks), but the two verify steps it spends measuring are ~180 ms, and on a 79-token
generation that is the whole regression. **Ticket 3 is closed as a no: long-context speculation
needs a cheaper verify step, not a wider draft.**

### 7.3 n = 6 does not break code and prose, and does not help either

| n | code, tok/s range over k | prose, tok/s range over k |
|---:|---|---|
| 6 | 131.6 – 142.9 | 141.7 – 144.4 |
| 8 | 135.3 – 144.1 | 144.4 – 148.7 |
| 10 | 138.5 – 140.4 | 144.2 – 147.3 |
| — | off 136.3 / off2 139.5 | off 143.9 / off2 143.3 |

Every arm is inside or adjacent to the 2–3 % control spread. n = 6 does roughly double the draft
rate on code (0.8–1.4 % against 0.3–2 %) without a throughput cost, because the break-even gate
absorbs the extra drafts (`declined_uneconomic` 15–54 at n = 6, 0 at n = 10). **There is no reason
to move off n = 8**: it is the only setting that is neutral on code and prose *without* leaning on
the gate.

## 8. The measurement that is not a lottery — replaying a fixed transcript

Speculation perturbs its own token stream, so a copy-class arm measures "where did this run start
copying" as much as it measures the drafter (§1). Under **greedy** decoding that is avoidable:
the accepted length of a prompt-lookup draft is a deterministic function of the greedy
continuation, so replaying the **non-speculative** transcript through the shipped `NgramDrafter`
gives exact acceptance with no divergence at all — the same trick the go/no-go used, now applied
to the engagement policy and with the *measured* per-step costs instead of a projection.

`benchmarks/spec_engage_replay.py` steps the recorded `off` stream position by position,
models the drain state exactly as the scheduler does (after a verify step the loop is drained, so
the next peek is not stale), and charges `decode = 7.3 ms` and the §7.1 verify curve. `v0` is the
shipped path (stale-exact peek, `+18.4 ms` per verify step — the measured v0−v1 gap of §2); `v1`
is the superset peek at the optimised step cost.

**Copy class** (1 129-token prompt, 1 023-token baseline output):

| n | k | **v0** speedup | **v1** speedup | v1 draft rate | v1 λ | v1 tokens/verify | v1 acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 0.70x | 1.04x | 0.631 | 3.34 | 4.7 / 5 | 0.943 |
| 8 | 8 | 1.11x | **1.61x** | 0.484 | 4.67 | 8.6 / 9 | 0.957 |
| 8 | 12 | 1.35x | **1.80x** | 0.396 | 5.47 | 12.3 / 13 | 0.948 |
| 8 | 16 | 1.48x | **1.88x** | 0.339 | 5.98 | 15.7 / 17 | 0.934 |
| 6 | 16 | 1.36x | 1.75x | 0.377 | 5.85 | 13.9 / 17 | 0.856 |
| 10 | 16 | 1.55x | **1.95x** | 0.316 | 5.98 | 16.8 / 17 | 0.986 |

**Code and prose, every (n, k), with a 3 ms drain charged on every false positive:** n = 8 and
n = 10 are **0.99x** at every draft length; n = 6 costs **6–7 %** on code. That is the same
verdict the offline go/no-go reached about n = 3, one rung up, and it is why the default stays at
**n = 8**.

Three things this settles that the throughput arms could not:

1. **The per-step work is worth ~1.45x on the copy class by itself** (1.11x -> 1.61x at the
   shipped k = 8), which is consistent with the 54.0 -> 35.6 ms step of §2 and with both
   end-to-end sessions.
2. **`--spec-draft-len 16` is worth another ~1.17x on top** (1.61x -> 1.88x at n = 8), because
   acceptance on verbatim-copy traffic does not decay with the draft length — 15.7 of 17 tokens
   are kept — while the verify step only grows by 1.8 ms per drafted token. The default is left
   at 8 in this pass: raising it doubles the price of the break-even gate's two probe steps at
   long context (§7.2), and that trade wants its own confirming run. **Set
   `--spec-draft-len 16` for copy-heavy agent traffic.**
3. **k = 4 is a regression under the shipped step cost** (0.70x) and barely neutral under the new
   one (1.04x). The adaptive halving in `_SpecState.note` walks toward exactly that region after a
   dead draft; it is bounded at 1 and recovers on a full acceptance, but it is the one knob here
   that can make speculation cost more than it saves.

## 9. Files, reproduction, and what is left

Changed:

- `python/freetoken/models/nemotron_h/spec_scan.py` — `_commit_fused` (one varlen SSD scan and
  one conv slide for the whole model), `_FusedPlan` + its cache, `_commit_per_layer` kept as the
  reference and the fallback for non-uniform mixers.
- `python/freetoken/scheduler/spec_ngram.py` — `NgramDrafter.could_match` and the (n−1)-prefix
  hash set; `peek(stale=...)`; `SpecNgramDecoder._prepare_verify` / `_buffers` / `_fla_metadata`;
  `SpecStats.cost_ms` and the CUDA-event instrumentation; `post_drain` / `fused_commit` /
  `fast_prep` (env-overridable, all on by default).
- `python/freetoken/scheduler/scheduler.py` — `peek(stale=last_data is not None)` in the
  overlapped loop, `stale=False` in the non-overlapped one, and the drain charged to `SpecStats`.
- `benchmarks/probe_spec_ngram_impl.py` — `--variants`, `--sweep-k`, `--sweep-n`.
- `tests/scheduler/test_spec_ngram.py` — 30 CPU tests, including the superset property over a
  random stream, the burst-entry case, and the verify-prep metadata cache.

Reproduction:

```
# per-phase cost and the v0/v1 arms, one model load
FREETOKEN_PIN_BUDGET_GB=17 PYTHONPATH=python .venv/bin/python -u \
  benchmarks/probe_spec_ngram_impl.py --model <lightning> --moe-cache-auto \
  --max-tokens 1024 --needle-max-tokens 256 --variants v0 v1 --out spec.json
# the (n, k) grid
  ... --sweep-k 4 8 12 16 --sweep-n 6 8 10
# the fused commit, weightless, ~40 s of GPU
PYTHONPATH=python .venv/bin/python -u benchmarks/check_spec_fused_commit.py
# the fixed-transcript engagement replay, CPU only
.venv/bin/python benchmarks/spec_engage_replay.py <lightning> spec.json 3
```

(Run the GPU ones through `scripts/gpu_lock.sh` from a wrapper that redirects its own stdout and
uses `python -u`; do not pipe `gpu_lock.sh`.)

Still open, by upside:

1. **Raise `--spec-draft-len` to 16 by default** — §8 projects 1.88x against 1.61x on copy and
   0.99x on code/prose, but it doubles the cost of the break-even gate's probe steps at long
   context. One confirming session with the gate's long-context behaviour instrumented.
2. **A cheaper long-context verify step** is the only thing that makes speculation pay above
   ~64K: the ratio is ~12x there and no draft length reaches it (§7.2). The extend attention
   reads the whole KV history once per query token; a fused multi-query extend kernel that reads
   it once for all `m` is the shape of the fix.
3. **Batched (bs > 1) speculation** — unchanged from the previous write-up.
4. **Sampling (non-greedy) speculation** — unchanged.
5. **The 16-way soak tail** — unchanged, ticket 4 there.
