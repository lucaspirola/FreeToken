
## 2026-09-05 (making `--speculative ngram` pay — a superset predictor and one scan for 23 layers)
- **When a decision must be made before the data arrives, ask a question the stale data can
  answer.** Engagement had to be decided pre-drain (draining every step costs ~30 %), and the
  shipped code asked the *exact* n-gram question of a one-token-stale list — which is the answer
  for the previous position, so every copy burst was entered one step late (draft rate 0.079
  against an offline 0.353). The fix is not a latch or a heuristic: the key the next step will use
  is `(known n-1 tokens, unknown next token)`, so testing membership of the **(n−1)-prefix set** is
  a *strict superset* that cannot miss a draftable step, and the exact test still runs post-drain
  and declines. Draft rate **0.353 -> 0.505**, λ **3.62 -> 4.88**, copy class **1.04x -> 1.52x**.
- **Independent heads mean the layer axis is a head axis.** The verify commit ran 23
  `mamba2_prefill` calls, one per Mamba-2 layer — ~280 kernel launches to advance 9 tokens. Every
  Nemotron-H mixer has the same `(head_dim, state_size, heads_per_group)`, so 23 x 64 heads
  concatenate into one 1 472-head sequence (`A` and `dt_bias` concatenate with them; `D` feeds only
  the discarded scan output and is dropped). **7.12 -> 0.45 ms of host time and bit-exact
  (0.000e+00) at eight (m, n) shapes** — because folding independent heads together changes no
  reduction. Ask "which axis of this loop is the kernel already batching over?" before accepting a
  per-layer loop.
- **Validate a kernel-shape change weightlessly before booking the model load.** 60 lines of
  synthetic mixers and a fake pool proved bit-exactness AND measured the 7.12 -> 0.45 ms in ~40 s
  of GPU. It also caught a real bug the model load would have hidden: the fused-plan cache was
  keyed on the pool and layer ids but not on the *weights*, so a second set of mixers was served
  the first set's concatenated `A`. The key now carries `A.data_ptr()`.
- **A per-step host cost that hides under the GPU is not a graph opportunity.** Ticket "graph the
  verify forward" rested on the bs=1 decode measurement (33.9 ms eager vs 6.88 ms graphed). At
  m = 9 the verify forward is **30.6 ms of host launch against 36.4 ms of GPU**, and at 131K it is
  31.0 against 91.8 — the Python already runs underneath. Close the ticket with the two numbers
  instead of leaving it plausible. (Same shape as the 16-lane decode finding; the rule generalises:
  **measure the host and GPU legs before promising a launch-path fix.**)
- **A verify batch is the one batch whose shape never changes — build it that way.** The general
  `_prepare_batch` spent its work on inapplicable branches, re-pinned staging tensors at an
  identical shape, and ran a full `Sampler.prepare` that a greedy verify forward never reads.
  0.80 -> 0.34 ms/step. Cache what is a function of `(extend width, state slot)`, and *assert* the
  invariant that makes the cache legal (here: `k + 1` never crosses a 128-token chunk boundary, so
  the mid-chunk snapshot metadata is always absent).
- **`for name in ...` inside `for name, ids, params in cases:` silently collapses a whole sweep.**
  The probe wrote every prompt class into `results["classes"][<variant name>]`, so a 4-class JSON
  came out with one key. The log had everything; the artifact had one row. **Never reuse a loop
  variable of an enclosing loop** — and check the artifact's row count against the run's, not just
  that it parsed.
- **Report the per-phase cost with the throughput.** Adding `cost_ms` (draft / prep / launch /
  sync / commit / emit, plus CUDA-event GPU time for the forward and the commit) turned "~40 % of
  a verify step is not the forward" into a table that named the 1.24 ms commit and the 0.80 ms
  prep — and showed that the *drain* the design was built around costs 0.15 ms inside a burst,
  because the previous step was itself drained. Free to collect: the forward events ride the
  argmax sync the step already pays, and the commit events are read on the next step.
