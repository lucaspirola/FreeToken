# Prompt-lookup (n-gram) speculative decoding — Nemotron-3.5-Lightning-30B-A3B-NVFP4, RTX 5080

Backlog item *"prompt-lookup (n-gram) speculative decoding for agent-session decode"*, run
2026-09-05 at `a25e954`. Host: RTX 5080 16 GB, WSL, `FREETOKEN_PIN_BUDGET_GB=17`,
`--moe-backend offload --moe-cache-auto --nvfp4-backend triton --kv-cache-dtype q8_0`.

> **Verdict: NO-GO as the engine stands, but not for the reason the plan expected — and the
> blocker is a bug-shaped fixed cost, not the feature.**
>
> Acceptance is *excellent* on exactly the traffic the item was filed for: on a copy-heavy
> agent tool-output prompt an 8-gram drafter reaches **λ = 3.6 accepted tokens per step at
> 93–97 % per-token acceptance**, while code and prose stay within ±0.5 % of neutral. Mamba-2
> state rollback, which the item flagged as the hard part, is **not** hard — §5 gives an exact
> design costing ~5 MiB of cached activations and no state copy at all.
>
> What kills it is that **this engine has no cheap multi-token forward**. The only path that
> can carry k > 1 query tokens for a running request is the prefill/extend path, and that path
> costs **290 ms of host time per forward, flat from 1 to 32 tokens**, of which **267 ms is the
> MoE layers** (§4). Against a 6.9–8.0 ms graphed decode step that is **36–42×**, so break-even
> would need λ ≈ 40 accepted tokens per verify step against a ceiling of k + 1 ≤ 17.
>
> Fix the extend path's fixed MoE cost first (§6). With it fixed, the projection from the
> measurements here is **1.74× on copy-heavy agent traffic and 1.00× elsewhere** — well clear
> of the 1.25× bar that Phase 4 (MTP) failed.

---

## 1. Method — both decisive quantities are measurable without implementing anything

Two numbers decide this, and neither needs a verify step to exist:

1. **λ(n, k), the mean accepted length.** Under greedy decoding a prompt-lookup drafter is
   verified against exactly the greedy continuation, so its acceptance distribution is a
   *deterministic function of an ordinary greedy transcript*. `benchmarks/probe_ngram_spec.py`
   records real transcripts as token ids (the offline `LLM` returns ids; the HTTP API has no
   logprobs and would have forced a retokenization approximation) and
   `benchmarks/ngram_spec_analysis.py` replays the drafter over them.
   `tests/benchmarks/test_ngram_spec_analysis.py` covers the replay, including the invariant
   that speculation changes the step count and never the token count.

2. **The verify step's cost.** A verify step *is* an m-token extend on a running request: same
   extend attention kernel, same varlen conv + chunked SSD scan, same expert routing over m
   consecutive tokens, same eager launch. The prefix cache reproduces that shape with no engine
   change — resend a cached prefix plus m fresh tokens.

Three prompt classes, greedy (`temperature 0`), natural stop, 1 023 output tokens each:

| class | prompt | what it is |
|---|---:|---|
| `code` | 87 tok | "write `ringbuf.py` + pytest tests" — novel code generation |
| `prose` | 66 tok | a 700-word essay — novel prose |
| `copy` | 1 129 tok | `sample.py` pasted in, "rename one function, output the complete file" — the canonical agent tool-output shape |

Measured decode baseline in the same process: **143–149 tok/s at bs=1**, i.e. **6.88–8.03 ms
per graphed decode step**, matching `docs/nemotron.md`'s 143.2 tok/s.

## 2. Acceptance — n-gram lookup works, and the right n is much larger than the literature's

`λ` is tokens emitted per scheduler step; `draft_rate` is the fraction of steps that issue a
draft at all (no match ⇒ an ordinary decode step at no extra cost); `accept_rate` is per drafted
token. `speedup@c` uses the flat model `T(m)/T(1) = 1 + c·(m−1)`; `speedup@measured` uses the
routing-derived cost of §3.

**n = 3–5 (the standard prompt-lookup setting) is the wrong choice here:**

| n | class | k | draft_rate | accept_rate | λ | speedup@c=0.63 |
|---:|---|---:|---:|---:|---:|---:|
| 5,4,3 | code | 4 | 0.116 | 0.227 | 1.105 | **0.856** |
| 5,4,3 | prose | 4 | 0.082 | 0.180 | 1.059 | **0.878** |
| 5,4,3 | copy | 4 | 0.598 | 0.800 | 2.915 | 1.162 |

A 3-gram fires on 12 % of code steps and is right 23 % of the time; the wasted verify steps cost
more than the accepted tokens buy, so **code and prose regress 12–14 %**.

**n = 8–12 turns the drafter into a precision gate, and that is the whole design:**

| n | class | k | draft_rate | accept_rate | λ | speedup@c=0.63 | speedup@measured |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | code | 8 | 0.004 | 0.500 | 1.016 | 0.996 | **1.004** |
| 8 | prose | 8 | 0.002 | 0.312 | 1.005 | 0.995 | **0.999** |
| 8 | copy | 8 | 0.353 | **0.926** | **3.615** | **1.300** | **1.738** |
| 12 | copy | 8 | 0.325 | 0.977 | 3.540 | 1.341 | 1.775 |
| 12 | code | 8 | 0.002 | 0.375 | 1.006 | 0.996 | 1.000 |

Full n × k sweep is reproducible with `ngram_spec_analysis.py`; the shape is monotone — raising
n costs a little λ on `copy` and buys neutrality everywhere else, and it saturates by n ≈ 10.

**The rule this establishes: when verification is expensive, draft for precision, not recall.**
The published prompt-lookup setting (n = 3, maximize hit rate) assumes a nearly free verify step,
which is true on a GPU-resident dense model and false on an offload MoE. Here the recall a
3-gram adds is bought with verify steps that are 40–80 % likely to be wasted, and that trade is
negative. n = 8 keeps ~93 % of the copy-class λ while dropping the code/prose draft rate by 30×.

Adaptive k (shrink to `accepted+1` after a partial rejection, grow by 2 after a full accept)
was measured and is **not** worth it at n ≥ 8: it lowers λ on `copy` (3.19 vs 3.62 at k=8)
because a single early rejection collapses k for the following steps, and the precision gate has
already removed the regressions it exists to prevent.

## 3. Expert routing over consecutive tokens — measured, not assumed

Task 2B4's MTP go/no-go used a bs=2 decode step (two *different* AIME problems) as its 2-token
verify proxy and explicitly left open that *"consecutive tokens within one sequence are more
correlated than two different AIME problems"*. That is now measured: hook the router, prefill a
recorded transcript, and take sliding-window unions of the top-6 sets over the generated
positions. Distinct experts touched per MoE layer, mean over 23 layers × 1 023 positions:

| m (tokens in the step) | code | prose | copy | independent positions | 2B4 (two problems) |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |
| 2 | 10.25 | 10.11 | 10.56 | 11.30 | 11.61 |
| 4 | 17.46 | 17.05 | 18.24 | 20.60 | — |
| 9 | 31.75 | 30.64 | 33.23 | 38.80 | — |
| 17 | 48.04 | 46.56 | 50.02 | 58.97 | — |

Consecutive tokens **are** more correlated, but only modestly: at m = 2 they share ~1.5 of 12
possible experts (13 %) against ~0.7 (6 %) for random positions in the same stream and ~0.4 (3 %)
for two different problems. So 2B4's 1.63× was a fair upper bound and the true same-stream
figure is ≈ 1.51×. The growth is sublinear enough to matter at width: the 9th token of a verify
step adds only ~2.7 new experts per layer, not 6.

Anchoring the cost on 2B4's two directly measured points (6.00 experts → 6.98 ms, 11.61 → 11.38 ms
⇒ 0.784 ms per extra expert per layer) gives the `speedup@measured` column of §2. This is the
right functional form for this hardware: dense layers at m ≤ 16 are weight-bound and flat in m,
while routed experts cost per *distinct* expert fetched.

## 4. The blocker — the extend path costs 290 ms per forward, and 267 ms of it is the MoE

The cost model above assumes a verify step differs from a decode step only in the work it does.
It does not. Timed inside `model.forward` (`perf_counter` on the host, CUDA events on the stream;
both agree, so the stream is waiting on the host), same process, same weights, CUDA graphs off so
the two paths are compared like for like:

| forward | tokens | host ms | GPU-event ms |
|---|---:|---:|---:|
| **decode** path | 1 | **33.9** | — |
| **extend** path | 1 | **303.6** | 310.6 |
| extend path | 2 | 300.3 | 307.9 |
| extend path | 8 | 307.9 | 313.7 |
| extend path | 17 | 311.7 | 318.4 |
| extend path | 32 | 314.2 | — |

**Flat in token count from 1 to 32.** It is a fixed per-forward cost, and per-mixer attribution
locates it precisely:

| m | forward host ms | mamba | attention | **moe** |
|---:|---:|---:|---:|---:|
| 1 | 290.2 | 17.7 | 1.8 | **267.8** |
| 4 | 290.3 | 19.6 | 1.8 | **265.8** |
| 9 | 289.7 | 17.8 | 1.7 | **267.2** |
| 17 | 289.1 | 17.7 | 1.7 | **266.8** |

267 ms across 23 MoE layers is **11.6 ms per MoE layer per forward, independent of how many
tokens that forward carries**. A 1-token extend routes 6 experts per layer = 138 fetches, so this
is ~1.94 ms per expert fetch — *not* bandwidth (138 × 5.35 MiB = 738 MiB, which at the measured
52.9 GB/s PCIe gather would be 14 ms) but per-fetch host cost: the prefill MoE path plans and
issues its expert gather per layer per forward and does not reuse the decode expert cache.

This is invisible in normal operation because it is hidden behind the GPU work of a large chunk:
an 8 192-token prefill chunk does ~861 ms of GPU work, so 290 ms of host work runs ahead of it
and never appears. It becomes the whole cost the moment a forward carries few tokens — which is
exactly what a verify step is. (It also bounds `--max-prefill-length` from below: below ~3K
tokens per chunk the GPU work no longer covers the host work and prefill goes host-bound.)

**Arithmetic.** Verify step 290 ms against a 6.88 ms graphed decode step = **42×**; against the
eager 33.9 ms decode forward = 8.6×. Break-even needs λ ≥ 42 accepted tokens per verify step.
The ceiling is k + 1, and k > 16 is pointless (λ saturates near 4.7). **No drafter can pay this.**

## 5. Mamba-2 state rollback — solved, and it is not the expensive part

Recorded because it is the part the item expected to be hard, and because the ticket in §6 needs
it ready. Measured/derived from `kvcache/linear_state_pool.py` and Nemotron's geometry
(H=64, P=64, N=128, G=8, conv_kernel=4, 23 Mamba layers): **46.81 MiB of state per slot**, 98.3 %
of it the fp32 recurrent block, only 828 KiB the conv window.

The three options the item listed:

- **(a) snapshot before verify.** `LinearStatePool.copy_from(src, dst)` already exists (two
  strided D2D copies covering all 23 layers) at ~100 µs per direction — cheap, but on a partial
  rejection it only restores the *pre-verify* state, so the accepted prefix must then be
  re-forwarded. Rejected: a second forward is the expensive thing.
- **(b) per-position states in scratch, commit the accepted row.** Exact, and the `track_dst` /
  `track_h_row` machinery from Phase 3G already does "copy one per-chunk state row into an
  arbitrary slot". But the accepted count is only known after the LM head, so all 23 layers'
  per-token state blocks must stay live: 23 × m × 2 MiB = **414 MiB at m = 9**, ≈ 77 expert cache
  slots. Rejected: violates "no extra VRAM beyond k-token activations".
- **(c) recompute the Mamba state for the accepted prefix.** Correct, and cheap once refined.

**Chosen: (c), refined — never advance the state speculatively, and defer the commit.**

1. The verify forward points `fla.cache_indices` at a **scratch slot** (the per-request
   `mamba_ping_pong` pair already reserved by the scheduler), so `mamba2_prefill`'s
   `state_source.index_copy_` and `causal_conv1d_varlen`'s conv write land there. The live slot
   is never touched, so there is nothing to roll back.
2. Each Mamba mixer caches its own scan inputs for the m verify positions — `x`, `dt`, `B`, `C`
   and `conv_in`. Per layer per token that is 8 192 + 128 + 2 048 + 2 048 + 12 288 B ≈ 24.7 KiB,
   so **23 layers × 9 tokens ≈ 5.1 MiB total** — 80× smaller than option (b), and it is literally
   "k-token activations".
3. After the sampler yields the accepted count j, a commit pass runs one varlen SSD scan per
   layer over the first j cached positions, starting from the **live** slot's state, and writes
   the conv window with the existing Phase 3G `index_copy_` of `conv_in[j-3:j]`. The SSD scan is
   1.068 ms per layer per 8 192-token chunk, i.e. ~0.13 µs per token per layer, so at j ≤ 8 the
   commit is 23 launches of pure overhead: **~0.2–0.5 ms**, ~3–7 % of a decode step.

KV needs no rollback at all: `CacheManager.allocate_paged` is already k-general (it allocates
from `cached_len` to `device_len`), and rejected positions simply keep their pages and are
overwritten by the next step.

So the state question costs ~5 MiB and ~0.3 ms. It is not what stops this.

## 6. Ticket — what has to change, in order

1. **`extend`-path MoE: reuse the decode expert cache and stop paying a fixed per-layer gather
   cost.** 11.6 ms per MoE layer per forward, independent of tokens (§4). This is the whole
   blocker, it is worth far more than speculative decoding (it also caps how small a prefill
   chunk can usefully be), and it is measurable in one run with
   `probe_ngram_spec.py --layer-profile`.
2. **A graph-captured fixed-width verify forward** at `(bs, m)`, or at minimum a lean eager
   extend. If step 1 lands and a verify step costs what the eager decode forward costs
   (33.9 ms host ⇒ ~4.9× a graphed decode step), the copy class already projects **1.52×**. If it
   also reaches the routing-implied 4.06× of §3, the projection is **1.74×**.
3. **Then** build the feature: `--speculative ngram`, `--spec-ngram-n` (default **8**, not 3),
   `--spec-draft-len` (default 8), greedy-only in v1, no adaptive k, the §5 deferred-commit state
   handling, `Req.complete_n`, k rows through `ParallelLMHead`/`Sampler`, one `DetokenizeMsg` per
   accepted token, and EOS/stop truncation inside an accepted run.

**Do not build step 3 before step 1.** The measurements above are the gate, and they are
reproducible in ~10 GPU minutes.

## 7. Relationship to the Phase 4 (MTP) NO-GO

Phase 4 was refused on a projected 0.96–1.34× from a 1.63× verify-step cost. Two corrections
from this run:

- **The 1.63× was measured on the wrong path.** It came from a bs=2 *decode* step, which uses the
  cached decode MoE path. A real verify step takes the *extend* path, which costs 290 ms — 42×,
  not 1.63×. MTP's projection was ~25× too optimistic about its own verify step, and the flag
  would have been built against a number that did not describe the thing being built.
- **Its open question is answered.** Consecutive tokens share ~13 % of their experts, not the 3 %
  two different problems share (§3), so 1.63× was a genuine upper bound — but the correction is
  to 1.51×, far too small to have changed that decision.

The standing conclusion is unchanged and now better supported: **multi-token verification of any
kind — MTP, n-gram, EAGLE — is blocked on the same thing, and it is the offload MoE's per-forward
gather cost, not the drafter and not the Mamba state.**

## 8. Reproduction

```
# transcripts (greedy, token ids) for the three prompt classes
FREETOKEN_PIN_BUDGET_GB=17 PYTHONPATH=python scripts/gpu_lock.sh .venv/bin/python \
  benchmarks/probe_ngram_spec.py --model <lightning> --out-dir <dir> --moe-cache-auto

# acceptance replay + projections (CPU only)
python3 benchmarks/ngram_spec_analysis.py <dir>/transcripts.jsonl --n 8 --k 1 2 4 8 \
  --routing <dir>/routing.json

# expert routing over consecutive tokens
... probe_ngram_spec.py --routing-from <dir>/transcripts.jsonl --widths 1 2 4 9 17

# verify-step cost, decode vs extend path (add FT_PROBE_EAGER=1 to time the decode forward)
... probe_ngram_spec.py --verify-cost-from <dir>/transcripts.jsonl --verify-base-tokens 800

# per-mixer attribution of the extend forward
... probe_ngram_spec.py --verify-cost-from <dir>/transcripts.jsonl --layer-profile
```

Caveats: transcripts are 1 023 tokens at short context, not 131K — the 131K acceptance and the
16-way aggregate check were not run, because §4 removes anything they could decide. λ on the copy
class should if anything rise with context (more prompt to match against). `--layer-profile`
serializes the mixers on the host; the totals sum to the model forward, so the attribution holds,
but the absolute numbers are ~1 % lower than the unhooked forward.
