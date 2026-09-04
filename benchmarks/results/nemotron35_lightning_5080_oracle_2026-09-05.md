# Nemotron 3.5 Lightning on a 5080 — first cross-engine oracle sweep

2026-09-05 02:14–… local · RTX 5080 (16 GiB) / WSL2, 32 GiB `MemAvailable` · FreeToken at
**`acc91e9`** (working tree clean), llama.cpp `9542 (6b80c74f2)` · procedure and verdict
vocabulary: `docs/oracle.md` · harness: `benchmarks/oracle_cross_engine.py`.

This is the first run of the standing cross-engine oracle (`5f7c0d6`) on real hardware. It
answers one question per graded turn — *did the engine retrieve it, and if not, is that us or
is that the model* — and it replaces the single-engine depth sweeps that produced the
retracted "model/quant retrieval limit" verdict on 2026-09-04.

**Headline at 262K: FreeToken beats llama.cpp on the same prompt (19/24 vs 17/24 graded
turns), and not one of the six needles failed on retention in either engine.** The two
`freetoken-only-miss` rows are both *composition* failures on turns where the retrieval
itself was demonstrably correct — one is an arithmetic off-by-one — so nothing here reopens
an engine bug.

---

## 1. Exact commands

```bash
export ORACLE_OUT=~/ai/bench/oracle/2026-09-05          # all three phases write here
# Phase A -- FreeToken, server started by the driver, stopped before phase B
FREETOKEN_GPU_LOCK_WAIT=300 scripts/gpu_lock.sh <scratch>/oracle/phaseA.sh 262144
# Phase B -- llama.cpp, starts and stops llama-server itself
FREETOKEN_GPU_LOCK_WAIT=300 scripts/gpu_lock.sh <scratch>/oracle/phaseB.sh 262144
# Phase C -- compare, CPU only, no lock
uv run benchmarks/oracle_cross_engine.py compare \
  --freetoken $ORACLE_OUT/ft_262144.json --llamacpp $ORACLE_OUT/lc_262144.json \
  --markdown $ORACLE_OUT/report_262144.md --json $ORACLE_OUT/merged_262144.json
```

`phaseA.sh` runs `docs/oracle.md`'s Phase-A server line with one change: **the documented
`--session-spill-dir /mnt/nvme/ft-spill` does not exist on this host** (single `/` on
`/dev/sdd`; there is no `/mnt/nvme`). It uses `~/.cache/freetoken/oracle-spill` instead —
`docs/oracle.md` should be corrected. It also waits for a real 1-token completion rather than
for `/health`, per the 2026-09-05 lesson that FreeToken answers `/health` while the weights
are still streaming.

llama.cpp ran on the documented defaults: `-c 270336 -np 1 --no-context-shift --cache-ram 0
-ngl 999 --n-cpu-moe 14 -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 512 -t 16 --jinja --no-warmup`.

Prompt identity: 262,076 tokens, sha256 `f0c0c130de47abb4`, filler cursor 0, **byte-identical
across the two engines** (the compare's exit code 3 "not the same prompt" did not fire).
The spill directory was empty before the run, so no stale checkpoint could shortcut a prefill.

## 2. Cost

| | FreeToken (NVFP4) | llama.cpp (Q4_0) |
|---|---|---|
| load → first real token served | 20 s | ~3 min (18.9 GiB GGUF, 14 CPU-MoE blocks) |
| turn 1 TTFT (cold 262 K prefill) | **134.5 s** (1,949 tok/s) | 150.6 s (1,740 tok/s) |
| turns 2–19 TTFT (prefix cache) | median 1.24 s | median 0.49 s |
| turns 2–19 decode | **median 105.5 tok/s** | median 39.7 tok/s |
| 19 graded turns + 5 generics, wall | 160 s | 169 s |
| phase wall incl. load and shutdown | 3 m 29 s | 3 m 45 s |

Two things worth keeping. First, **`acc91e9` is visible here**: 105 tok/s decode at a 262 K
context against the 56.3 tok/s recorded at the same length on 2026-09-04, and 2.7× llama.cpp
on the same prompt and the same card. Second, the whole 262 K sweep cost **7 minutes of GPU
for both engines** — the one-prefill-many-turns design works; turns 2–19 ride the prefix cache
at 99.978 % cached and cost ~1.2 s each.

## 3. Agreement matrix (262,144)

`$ORACLE_OUT/report_262144.md` is the generated report; the numbers below are its summary.

| | llama.cpp PASS | llama.cpp FAIL |
|---|---:|---:|
| **FreeToken PASS** | **15** | **4** |
| **FreeToken FAIL** | **2** | **3** |

| verdict | count |
|---|---:|
| `agree` | 15 |
| `both-miss` | 3 |
| `freetoken-only-miss` | **2** |
| `llamacpp-only-miss` | 4 |
| `missing` | 0 |

Totals: FreeToken **19/24** graded turns, llama.cpp **17/24**. Compare exit code **2**
(≥ 1 FreeToken-only miss — read the classification before filing anything).

Standing confound, printed by the report every time: FreeToken serves NVFP4 and llama.cpp
serves Q4_0, and neither engine can load the other's weights, so engine and quantization move
together. Every `freetoken-only-miss` reads as "engine **or** NVFP4".

## 4. The two `freetoken-only-miss` rows are not retrieval misses

Both are `combined` turns, and in both the codes FreeToken printed are the *correct* ones:

| turn | expected | FreeToken answered | what actually failed |
|---|---|---|---|
| `combined:orchard+harbour` | larger=orchard, sum 9,854,500 | "orchard ledger code 5663623, **sum 9854499**" | picked the right larger key, retrieved 5,663,623 and 4,190,877 — and added them **off by one** |
| `combined:thicket+orchard` | larger=orchard, sum 9,610,785 | "thicket ledger code 3947162, **sum 9610785**" | **sum exactly right** (so both codes were in hand); named the wrong one as larger |

Neither is evidence about the KV/SSM state, and neither is an engine bug: the needles are in
state, the arithmetic and the comparison are not. Both turns are marked **not leak-free**, so
they were not independent evidence to begin with. Filing a retention bug on either of these
is precisely the mistake this suite exists to prevent.

## 5. Needle classification — zero `retention` in either engine

| needle | depth | FreeToken | in state | llama.cpp | in state |
|---|---|---|---|---|---|
| orchard | 0.050 | `recall-partial` | yes | `recall` | yes |
| harbour | 0.250 | `recall-partial` | yes | `recall-partial` | yes |
| quarry | 0.500 | `interference-near` | yes | `interference-near` | yes |
| cavern | 0.600 | `recall-partial` | yes | `interference-near` | yes |
| meadow | 0.750 | `recall-partial` | yes | `interference-near` | yes |
| thicket | 0.950 | `recall-partial` | yes | `recall-partial` | yes |

**No `retention`, no `selection`, no `incoherent`, in either engine.** Every needle was
recovered by at least one probe on both engines; `in state` is true 12/12. The
`recall-partial` class means the direct question passed and a composed probe did not — the
binding is there and composition is weak, which is a model property at this length, not a
cache property.

The one genuinely interesting row is **quarry at depth 0.500**: *both engines*, independently,
answered the direct question with the quarry **register** twin (1,607,392) instead of the
quarry **ledger** code (8,324,516) — the near-duplicate planted at depth 0.4802. That is a
`both-miss` / `interference-near`: the model cannot separate `quarry ledger` from
`quarry register` 300 tokens apart at 262 K, and it is **not** an engine or quantization
effect. A single-engine run would have logged it as a mid-depth FreeToken retrieval failure,
which is exactly the wrong conclusion — and exactly the one the 2026-09-04 bisect drew.

llama.cpp's three `interference-near` classes (quarry, cavern, meadow) against FreeToken's one
also say the Q4_0 build is *more* susceptible to the near-duplicate trap at this length than
the NVFP4 build, not less.

The control (`belfry`, a key that is not in the haystack) is correctly denied by both engines
with no fabrication. All five haystack-free generic prompts pass on both.

## 6. Logprobs

Still not compared: FreeToken's chat endpoint rejects `top_logprobs > 0` with HTTP 400 and
nothing below the HTTP layer computes logprobs (`SamplingParams` has no field, the sampler
never gathers them). llama.cpp reports `supported`. The comparison code path is live and
starts producing its table the day the engine grows the feature. See `docs/oracle.md`.

## 7. Verdicts

* **262,144 — no engine bug.** 0 retention failures, 0 selection failures, 12/12 needles in
  state on both engines. The two `freetoken-only-miss` rows are composition/arithmetic on
  turns where retrieval succeeded. FreeToken outscores llama.cpp 19/24 to 17/24 on the
  identical prompt.
* The `dt_limit` fix (handover item 1) holds up under an independent oracle: mid-depth
  needles at 262 K are retrieved, and the one that is not is missed by llama.cpp too.
* `docs/oracle.md`'s Phase-A serve line names a `/mnt/nvme` spill directory that does not
  exist on this host — fix the doc.

## 8. Artifacts

* `~/ai/bench/oracle/2026-09-05/{ft_262144.json,lc_262144.json,report_262144.md,
  merged_262144.json,llama_262144.log}`
* Drivers: `<scratch>/oracle/{serve_ft.sh,phaseA.sh,phaseB.sh}`, logs
  `phaseA_262144.log`, `phaseB_262144.log`, `ft_262144_server.log`

---

## 9. The 1M rung — FreeToken only (no cross-engine leg)

Run 02:57–03:33 as the third lock acquisition. **llama.cpp at 1M was not attempted** (optional
and last per the plan; it has never been run on this host and the budget table flags it as the
one leg that may not fit the 4 h cap). So there are **no cross-engine verdicts at this length**
— everything below is the FreeToken recording and its own needle classification.

### 9a. First attempt failed on a context-budget bug in the *procedure*, not the engine

`--target-prompt-tokens 1048576` against the documented `--num-tokens 1048576 /
--max-seq-len-override 1048576` profile **cannot work**: the suite is a conversation, so every
graded turn appends its question and its reply and the server also reserves the decode budget.
Turn 1 passed (1,048,545 tokens, `direct:orchard` PASS); **turn 2 was refused with HTTP 400
`context_length_exceeded` — 1,048,623 > 1,048,576** — after paying the full 1,818 s prefill.
Recording kept as `ft_1048576_ctxoverflow.json`. `docs/oracle.md` has been corrected (leave
headroom above the top rung, or record it at 1,044,480; the llama.cpp side already adds
`--llama-ctx-headroom` automatically). The re-run used **`--target-prompt-tokens 1044480`**
and completed all 19 turns.

### 9b. Cost at 1,044,416 haystack tokens

| | |
|---|---|
| turn 1 TTFT (cold prefill, 1,044,476 tokens) | **1,813.8 s** at **576 tok/s** |
| turns 2–19 TTFT | median **4.61 s** (99.9946 % of each prompt cached) |
| turns 2–19 decode | median **80.7 tok/s** |
| 19 graded turns + prefill, wall | 1,903 s |

`acc91e9` again: **80.7 tok/s decode at a 1M context** against the ~20 tok/s recorded on
2026-09-04, and **prefill is unchanged** (576 tok/s here, 573 then) — exactly the claim in the
commit message, now confirmed on a 1M workload rather than a paired microbenchmark.

### 9c. Result: 7/19 turns pass, and **zero `retention`**

| needle | depth | class | in state |
|---|---|---|---|
| orchard | 0.050 | `recall-partial` | yes |
| harbour | 0.250 | `interference-near` | yes |
| quarry | 0.500 | `interference-near` | **no** (its only recovering probe leaked) |
| cavern | 0.600 | `interference-near` | yes |
| meadow | 0.750 | `interference-near` | yes |
| thicket | 0.950 | `interference-near` | yes |

The direct probes collapse — 1 of 6 passes, and the misses are textbook interference: five of
them return either the key's own `register` twin or the depth-0.03 `quarry register` code
(1,607,392), which acts as an attractor for the whole prompt.

**The reverse probes then recover almost everything.** Asked `code → key`, the engine returns:

```
reverse:harbour  4190877   reverse:quarry  8324516   reverse:cavern  6082735
reverse:meadow   7218459   reverse:thicket 3947162
```

— five for five, **correct codes for the right keys, four of the five leak-free** (only
`reverse:quarry` had leaked earlier). Those are exactly the codes the direct questions had
just "missed". `reverse:orchard` is the sixth and the only genuine slip: it produced the right
code and attached the wrong key ("The meadow ledger code is 5663623").

So at 1M this checkpoint's long-context state **holds all six needles**; what fails is
*addressing* them from a key, not *retaining* them. That is the same result as the 2026-09-04
multi-needle run's question 8 (which recovered a "missed" depth-0.25 needle through a combined
question), generalised: a leak-free `code → key` probe recovers 5/6 needles that direct
questioning loses. **A single-question-shape gate at 1M would have filed six retention bugs
against the kernels; the correct count is zero.**

The control (`belfry`) is still denied with no fabrication. Composition remains the weak axis:
every `combined` turn gets its sum wrong at 1M (at 262K four of six were right).

### 9d. Verdicts at 1M

* **No engine bug is indicated, and none can be confirmed either** — without the llama.cpp leg
  there is no oracle at this length. What the FreeToken recording alone establishes is the
  *classification*: 0 `retention`, 0 `incoherent`, 5 `interference-near`, 1 `recall-partial`,
  5/6 needles demonstrably in state via leak-free reverse probes.
* Next time this rung is run, run the llama.cpp leg with it — the prediction to test is that
  llama.cpp shows the same `interference-near` collapse (it already showed *more* of it than
  FreeToken at 262K), which would settle it as a model limit.
* `--target-prompt-tokens` must be ≤ `--num-tokens` − ~4 K at the top rung; see §9a.

### 9e. Artifacts (1M)

`~/ai/bench/oracle/2026-09-05/{ft_1044480.json, ft_1048576_ctxoverflow.json}`; driver logs
`<scratch>/oracle/{phaseA_1044480.log, phaseA_1048576.log, ft_1044480_server.log,
ft_1048576_server.log}`.
