# Seeding the break-even gate from narrow probes — measured, and a NO-GO as it stands

`FREETOKEN_SPEC_GATE_SEED=1` against the shipped gate, one model load, copy class + 131K
needle, arms `off / v1 / seed / off2`. Host: RTX 5080 16 GB, WSL, tree at `efa37da` plus the
seeded-gate patch, `FREETOKEN_PIN_BUDGET_GB=17`, `--moe-backend offload --moe-cache-auto
--nvfp4-backend triton --kv-cache-dtype q8_0`, `max_running_req=1`, greedy, n=8, k=8,
adaptive on, 1 023 output tokens (256 max for the needle).

> **Verdict: do NOT default it on. The mechanism does exactly what it was built to do and
> still does not buy the thing it was built for.** At 131K the seeded arm ran the two narrow
> probes (m = 2 then m = 4), fitted them, primed the gate — and then ran a full-width verify
> step anyway, spending **226 ms in verify steps against the shipped arm's 182 ms**. The
> cause is the one flagged before the run: the fit prices `verify_ms`, but `emit` is still
> sitting on its optimistic `max_k + 1 = 9` prior, so `_pays_off` stays open until a
> full-width step has fed the accepted-length EWMA. Seeding the cost side alone cannot close
> a gate whose other side is unmeasured.

## 1. The numbers

| class | arm | tok/s | vs off | out tok | verify steps | drafted | acc rate | tok/verify | `declined_uneconomic` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| copy (1 129) | off | 133.81 | — | 1 023 | — | — | — | — | — |
| | **v1** | 124.11 | 0.928x | 1 023 | 13 | 90 | 0.367 | 3.54 | 152 |
| | **seed** | 139.06 | 1.039x | 1 024 | 43 | 283 | 0.689 | 5.54 | 64 |
| | off2 | 138.63 | 1.036x | 1 023 | — | — | — | — | — |
| needle (129 855) | off | 87.14 | — | 70 | — | — | — | — | — |
| | **v1** | 80.36 | 0.922x | 70 | 2 | 16 | 0.750 | 7.00 | 7 |
| | **seed** | 83.98 | 0.964x | **96** | 3 | 12 | 0.417 | 2.67 | 23 |
| | off2 | 89.18 | 1.023x | 70 | — | — | — | — | — |

`control_identical=True` on both classes — the engine is deterministic and the off/off2 pair
is a real control. Control spread: **3.6 %** on copy, **2.3 %** on the needle.

## 2. The mechanism fired exactly as designed — and that is the problem

The needle's seeded arm drafted **12 tokens over 3 verify steps**, which at k=8 decomposes one
way only: **1 + 3 + 8**. Two narrow probes at the `_SEED_WIDTHS` of `(1, 3)`, then a
full-width step. So the probes ran, the fit ran, the gate was primed with zero full-width
samples — and a full-width verify step still followed, because `_pays_off` reads

```
state.emit * _GATE_MARGIN > state.verify_ms / state.decode_ms
```

and `emit` was untouched at 9.0. With the fitted cost (~82 ms) against a ~11 ms decode step
the ratio is ~7.9, and `9.0 x 1.25 = 11.25` clears it comfortably. Only after the full-width
step fed `emit` did the gate shut — visible as `declined_uneconomic` **23 against v1's 7**.

Charged in milliseconds, which is the only currency this ticket cares about:

| arm | verify steps | mean step | **total spent pricing the gate** |
|---|---:|---:|---:|
| v1 (shipped) | 2 | 91.0 ms | **182 ms** |
| seed | 3 | 75.4 ms | **226 ms** |

The seeded arm spent **44 ms more**, not the ~75 ms less the design projected. The two narrow
probes are cheap exactly as predicted — the mean step cost fell 91.0 → 75.4 ms because two of
the three steps were narrow — but a third step was one too many.

## 3. Why the tok/s column must not be read as the verdict

The needle's 0.922x → 0.964x looks like the seed recovering half the regression. It is not
attributable, for three independent reasons, and this is the more useful result of the run.

**(a) The arms generated different amounts.** The seeded arm emitted **96 tokens** where every
other arm emitted 70 — the streams diverge at token 56. Speculation is not token-identical to
non-speculative decoding on this engine (a property of the architecture, not a bug), so a
perturbed stop point is expected; it just means the two rows are not the same experiment.

**(b) The non-speculative decode rate itself moved 18 % between arms.** Backing the verify
steps out of each arm's wall clock leaves the plain decode cost, which every arm pays on the
identical code path:

| needle arm | plain decode | over |
|---|---:|---:|
| off | 11.47 ms/token | 70 tok |
| off2 | 11.21 ms/token | 70 tok |
| v1 | 12.30 ms/token | 56 tok |
| **seed** | **10.42 ms/token** | 88 tok |

10.42 to 12.30 ms on a step that no arm changes. That spread is larger than the entire effect
being attributed to the seed, and it is measured over 56–88 samples.

**(c) The copy class drew opposite lottery tickets.** `v1` drafted on 13 steps at 0.367
acceptance; `seed` drafted on 43 at 0.689. That is the documented copy-class variance — arms
of the *same* binary have spanned 1.04x to 1.67x, because the prompt's output opens with a
reasoning preamble and where the verbatim copy starts inside the 1 024-token window decides
the draft rate. `v1`'s 0.928x and `seed`'s 1.039x are two draws, not a comparison.

One further trap in this run's table, worth recording so the next reader does not repeat it:
**`v1`'s `cost_ms` is not comparable to `seed`'s.** `v1` ran first among the speculative arms
and amortised the one-off Triton autotune of the verify shape over only 13 steps, reading
total 64.4 ms / `gpu_forward` 64.6 ms. `seed`, over 43 steps, reads 37.0 / 38.4 — which is the
already-published steady state (35.6 / 36.4). Neither number says anything about seeding.

## 4. Verdict and what would actually close it

**`FREETOKEN_SPEC_GATE_SEED` stays off.** It is correct, it is cheap, its unit tests pin the
fit against both measured operating points, and it is worth keeping in the tree — but on the
evidence it costs one extra verify step at long context and buys nothing measurable.

The fix is not more probe tuning; it is that **both sides of the gate need seeding or
neither.** Two candidates, in order of how clearly correct they are:

1. **Prime `emit` from the probes' own acceptance.** The narrow probes already produce it (the
   needle's three steps accepted 5 of 12). Under greedy decoding a probe that *rejects* has
   found the divergence a full-width draft would have found at the same position, so its
   accepted count is a true full-width sample — the seeded path already uses this rule, but
   only when a probe rejects, and here both probes saturated. Widening `_SEED_WIDTHS` so at
   least one probe is long enough to usually reject (say `(3, 6)`) would feed `emit` at the
   price of a slightly dearer second probe. Cheap to try, and the same run answers it.
2. **Let the cost side close the gate alone when it is overwhelming**, i.e. compare against
   the ceiling `max_k + 1` rather than the `emit` estimate while `emit` is still a prior. This
   is a genuine policy change and should not be made without the fixed-transcript replay.

**And do not re-run this experiment as end-to-end arms.** Two to three verify steps on a
79-token generation cannot resolve a 4 % effect against an 18 % baseline spread. The authority
here is `benchmarks/spec_engage_replay.py` — replaying the fixed non-speculative transcript
through both gate policies, CPU only, no model load, no lottery — which is what settled the
draft-rate question in `..._ngram_spec_fast_2026-09-05.md` §8. Extend it with a
gate-policy axis before booking another GPU slot.

## 5. Reproduction

```bash
# one model load, copy + 131K needle, arms off / v1 / seed / off2 (~90 s of GPU)
# (through scripts/gpu_lock.sh from a wrapper that redirects its OWN stdout and uses -u;
#  never pipe gpu_lock.sh -- its exit trap pkill -9s its own process group)
FREETOKEN_PIN_BUDGET_GB=17 PYTHONPATH=python .venv/bin/python -u \
  benchmarks/probe_spec_ngram_impl.py \
  --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-cache-auto --only copy --variants v1 seed \
  --max-tokens 1024 --needle-max-tokens 256 --out spec_seed.json
```

`--only copy` is deliberate: the needle case is appended automatically and `--only needle`
would fault on the prompt table. Note that this runs the copy class *first* in the session,
which is its low-draft-rate ordering.

Raw artifacts for this run (gitignored / scratch): `spec_seed.json`, `seed_ab.log`.

## 6. Files

- `python/freetoken/scheduler/spec_ngram.py` — `_SEED_WIDTHS`, `_fit_verify_ms`,
  `_SpecState.seeding/seed_probes/verify_seeded`, the `_budget` clamp, the probe branch in
  `run_step`, `SpecStats.seed_probe_steps/seed_fits`, `FREETOKEN_SPEC_GATE_SEED` (default 0).
- `tests/scheduler/test_spec_ngram.py` — 8 CPU tests: the fit against both measured operating
  points, the one-vs-two-probe argument as an assertion, the degenerate-input refusals, the
  `_budget` width sequence, the three-step chain, and the two `emit` rules.
- `benchmarks/probe_spec_ngram_impl.py` — `gate_seed` on every variant plus the `seed` arm,
  and `seed_probe_steps` / `seed_fits` in `_stats` (they were dropped from the JSON on the run
  above, so §2's probe decomposition is arithmetic on `drafted_tokens`, not a counter).

## Verification
`ruff check .` clean; `pytest tests/scheduler` 269 passed. GPU at 0 MiB at the end of the
session.
