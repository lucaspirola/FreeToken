# Nemotron 3.5 Lightning on a 5080 — first cross-engine oracle sweep

2026-09-05 02:14–14:00 local · RTX 5080 (16 GiB) / WSL2, 32 GiB `MemAvailable` · llama.cpp
`9542 (6b80c74f2)` · procedure and verdict vocabulary: `docs/oracle.md` · harness:
`benchmarks/oracle_cross_engine.py`.

Two FreeToken checkpoints appear here: §§1–9 (262K, and the 1M FreeToken-only leg) ran at
**`acc91e9`**; §§10–12 (the 1M llama.cpp attempt and the 524K cross-engine rung) ran at
**`2a139ad`**. The intervening commits are prefill/scheduler work — `2a139ad`'s NVFP4 MoE
prefill GEMM is bit-exact by construction — so the §9 recording is still the 1M reference and
was not re-recorded. **§12 is the verdict.**

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

---

## 10. The 1M llama.cpp leg — attempted three times, **does not fit this card**

Run 12:52–13:39 as three lock acquisitions at FreeToken `2a139ad`. The prompt was verified
identical to the FreeToken 1M recording *before* any GPU work: `record --build-only` at
`--target-prompt-tokens 1044480 --filler-cursor 0` reproduces
sha256 `38f517c3e3b06ae3…`, byte-for-byte §9's haystack. So the leg was runnable in principle;
it is the card that refuses.

`llama-server -c 1052672` (the harness's `target + --llama-ctx-headroom 8192`; llama.cpp then
caps the slot at the 1,048,576 training context) reserves **essentially the whole 16 GiB card
at every `--n-cpu-moe` setting** — `nvidia-smi` reports 15,956–15,960 MiB used / 18–22 MiB free
at `--n-cpu-moe` 14, 20 *and* 23. Prompt-processing throughput is then a monotone function of
how much VRAM the *weights* leave behind, measured on the first 4,096-token chunk:

| run | weights on GPU | first 4,096-token chunk | vs the healthy chunk |
|---|---:|---:|---:|
| `-c 270336 --n-cpu-moe 14` (the 262K reference) | 8,424 MiB | **2.05 s** | 1.00× |
| `-c 1052672 --n-cpu-moe 14` | 8,424 MiB | 27.33 s | 13.3× |
| `-c 1052672 --n-cpu-moe 20` | 4,313 MiB | 11.64 s | 5.7× |
| `-c 1052672 --n-cpu-moe 23` (**all** MoE on host) | 2,258 MiB | 3.87 s | 1.9× |

`--n-cpu-moe 23` is the floor: the GGUF's routed experts are 685.1 MiB per MoE block over 23
blocks (`gguf.GGUFReader`, 18,015 MiB of tensors total), and at 23 there is nothing left to
move to host RAM. That run held **~3.45 s per 4,096-token chunk (≈1,190 tok/s) to about
570K tokens** and then degraded linearly — **+11.5 s of chunk cost per further 4,096-token
chunk**, i.e. the written KV had outgrown what stays resident and each chunk started paying
PCIe for the whole prefix:

```
n_tokens   589,824  593,920  598,016  602,112  606,208  610,304  614,400  618,496  622,592
chunk (s)     5.0     25.0     39.7     54.2     65.4     76.2     87.4     97.3    107.5
```

Extrapolating that slope over the 103 chunks still owed at 622,592 tokens gives
**≈72,700 s ≈ 20 h for the remaining prefill alone**, against the lock's 4 h cap
(`FREETOKEN_GPU_LOCK_MAX_HOLD`) — a 5× overrun, and that is before a single graded turn. The
attempt was killed at 622,592 tokens / 1,151 s. Artifacts kept as
`llama_1044480_vramwall.log` and `lc_1044480_vramwall_partial.json` (1 row, the turn-1 error).

**This is a host limit, not a llama.cpp defect and not a FreeToken result.** FreeToken serves
the same 1M prompt on the same card because NVFP4 weights plus a q8_0 KV for 1M is ~3.5 GiB of
KV against a much smaller resident weight set, and because its MoE offload streams experts per
step instead of pinning them. The comparison simply cannot be made at 1M on a 16 GiB card.
Per the plan's fallback rule, the rung was moved to **524,288 on both engines**.

## 11. The 524K rung — the cross-engine oracle where direct addressing collapses

This is the interesting rung, and it is the one that settles §9's open question. **The direct
probes collapse between 262K and 524K on *both* engines, and the reverse probes survive on
both.** 524K is therefore the cheapest length at which the 1M phenomenon is reproducible, and
it *is* reproducible in llama.cpp.

Prompt identity: 524,204 haystack tokens, 524,264-token turn-1 prompt, sha256
`72683f24c68885d1`, filler cursor 0, byte-identical across the two engines (compare did not
raise exit 3). Both legs ran `--no-generic`. FreeToken at `2a139ad`; llama.cpp `9542
(6b80c74f2)` with `-c 532480 … --n-cpu-moe 23` (23 rather than the 262K run's 14 — see §10).

### 11a. Cost

| | FreeToken (NVFP4) | llama.cpp (Q4_0) |
|---|---|---|
| turn 1 TTFT (cold 524K prefill) | **228.2 s** at **2,297 tok/s** | 544.5 s at 963 tok/s |
| turns 2–19 TTFT | median **2.46 s** | median 0.94 s |
| turns 2–19 decode | median **85.3 tok/s** | median 23.6 tok/s |
| 19 graded turns, wall | **325 s** | 619 s |
| phase wall incl. load and shutdown | 7 m 23 s | 12 m 08 s |

`2a139ad` is plainly visible: **2,297 tok/s of 524K prefill against the 1,064 tok/s recorded at
the same length on 2026-09-04** (2.16×, the NVFP4 MoE prefill GEMM), and 2.4× llama.cpp's
prefill on the identical prompt — where at 262K the two engines were within 12 % of each other.
Decode is 3.6× llama.cpp.

### 11b. Agreement matrix (524,288)

| | llama.cpp PASS | llama.cpp FAIL |
|---|---:|---:|
| **FreeToken PASS** | **8** | **0** |
| **FreeToken FAIL** | **2** | **9** |

| verdict | count |
|---|---:|
| `agree` | 8 |
| `both-miss` | **9** |
| `freetoken-only-miss` | 2 |
| `llamacpp-only-miss` | 0 |
| `missing` | 0 |

Totals: FreeToken **8/19**, llama.cpp **10/19**. Compare exit code **2**.

### 11c. Pass rate by question *shape* — this is the whole result

| | direct | combined | reverse | control | total |
|---|---|---|---|---|---|
| FreeToken 262K | 5/6 | 2/6 | 6/6 | 1/1 | 14/19 |
| llama.cpp 262K | 3/6 | 2/6 | 6/6 | 1/1 | 12/19 |
| **FreeToken 524K** | **1/6** | 0/6 | **6/6** | 1/1 | 8/19 |
| **llama.cpp 524K** | **2/6** | 1/6 | **6/6** | 1/1 | 10/19 |
| FreeToken 1M | 1/6 | 0/6 | 5/6 | 1/1 | 7/19 |

Both engines lose the `key → code` direction between 262K and 524K (FreeToken 5→1,
llama.cpp 3→2) while `code → key` stays **6/6 on both**. That is §9's classification —
*addressing, not retention* — now confirmed by a second engine and a second quantization.

### 11d. The four `both-miss` direct probes return the **same wrong code** in both engines

| probe | expected | FreeToken answered | llama.cpp answered |
|---|---|---|---|
| `direct:quarry` | 8324516 | "quarry ledger code is **1607392**" | "**1607392** …" |
| `direct:cavern` | 6082735 | "cavern ledger code is **3518470**" | "**3518470** The cavern ledger code is 3518470." |
| `direct:meadow` | 7218459 | "meadow ledger code is **8043961**" | "**8043961** The meadow ledger code is 8043961." |
| `direct:thicket` | 3947162 | "thicket ledger code is **5290638**" | "**5290638** The thicket ledger code is 5290638." |

Every one of those wrong codes is that key's own `register` near-duplicate. Two independent
engines, two different quantizations, two different attention and SSM implementations, and the
*identical* wrong number four times over. A single-engine run would have filed four retention
bugs against the kernels here; the correct count is zero. The composition failures behave the
same way — `combined:meadow+thicket` produces the byte-identical wrong sum 13,334,599 in both.

### 11e. Needle classification

| needle | depth | FreeToken | in state | llama.cpp | in state |
|---|---|---|---|---|---|
| orchard | 0.050 | `recall-partial` | yes | `recall-partial` | yes |
| harbour | 0.250 | `interference-cross` | no | `recall-partial` | yes |
| quarry | 0.500 | `interference-near` | yes | `interference-near` | yes |
| cavern | 0.600 | `interference-near` | yes | `interference-near` | yes |
| meadow | 0.750 | `interference-near` | yes | `interference-near` | yes |
| thicket | 0.950 | `interference-near` | yes | `interference-near` | yes |

**Zero `retention`, zero `selection`, zero `incoherent` on either engine**, and all six needles
recovered by their reverse probe on both. Five of six classes agree exactly.

### 11f. The two `freetoken-only-miss` rows

* `direct:harbour` (leak-free) — FreeToken returned the *orchard* code 5,663,623 for the
  harbour key: `interference-cross`, not a lost needle (`reverse:harbour` recovers 4,190,877
  leak-free on the same server two turns later). It is the one row where llama.cpp holds a
  direct probe that FreeToken loses. Worth noting: this is **turn 2**, and turn 2 is also the
  one turn whose TTFT was 50.0 s against 2.4 s for turns 3–19 — a partial prefix re-prefill.
  The server log rules out a lost prefix as the cause: turn 2 matched
  `#cached-token: 524287` of a 524,342-token prompt, so it re-forwarded 55 tokens, not 524K —
  the 50 s went somewhere else (session-spill checkpoint or expert-cache re-warm, not
  established here). It is the single concrete lead this rung produces and it is cheap to
  re-probe (re-run 524K with `--filler-cursor 65`).
* `combined:orchard+harbour` (not leak-free) — FreeToken had *both* codes right and summed
  5,663,623 + 4,190,877 to 9,851,000 instead of 9,854,500. Arithmetic, not retrieval; the same
  class as the two 262K `freetoken-only-miss` rows in §4.

Both read as "engine **or** NVFP4" under the standing confound.

## 12. Verdict: is there a 1M engine defect in FreeToken?

**No — and the 524K oracle is what licenses saying so at 1M.** The 1M leg's suspicious result
was that FreeToken answers only 1 of 6 direct `key → code` questions while leak-free reverse
`code → key` probes recover 5 of 6, which could equally have been an addressing defect in the
engine or a property of the model at that length. There is no way to run the llama.cpp oracle
at 1M on a 16 GiB card (§10: ≈20 h of prefill against a 4 h cap, with every MoE block already
on host RAM), so the question was settled one rung down, at the length where the same collapse
first appears. At 524K the collapse is *identical in llama.cpp*: direct 2/6 there against
FreeToken's 1/6, reverse 6/6 on both, nine of nineteen turns `both-miss`, and four of the six
direct probes return the same near-duplicate `register` code **byte-for-byte in both engines**.
Zero `retention` and zero `selection` classes on either side; every needle demonstrably in
state. The direct-addressing collapse is therefore a property of Nemotron 3.5 Lightning at
half a million tokens and beyond — the near-duplicate twin planted half a haystack away wins
the `key → code` lookup — and not of FreeToken's KV, SSM state, prefix cache or scheduler. The
only FreeToken-specific residue at 524K is `direct:harbour`, an `interference-cross` on the one
turn that also paid a 50 s partial re-prefill, and a wrong sum on a turn whose two codes were
both retrieved correctly. Neither is a retention or selection failure, and neither is grounds
to open a kernel bug. **Recommendation: close the 1M direct-addressing ticket as
model-limited.**

Two operational findings fall out of this rung and are *not* about recall:

1. **`docs/oracle.md`'s Phase-A serve line omits `--enable-cache-report`, which disarms the
   runbook's own re-prefill check.** Every FreeToken turn in the 524K recording reports
   `cached_tokens: 0`, which `docs/oracle.md` says means "the prefix match broke and the run is
   measuring re-prefill" — it did not. `openai_api.py:937-939` returns 0 unless
   `config.enable_cache_report` is set, and `:955` then omits `prompt_tokens_details`
   altogether, so *flag off*, *genuine zero* and *field absent* are indistinguishable on the
   wire. The scheduler's own counters in the server log show the truth:
   `#new-token: 55, #cached-token: 524287` on turn 2 and the same shape through turn 19.
   Nothing on the `cached_tokens` code path changed in `acc91e9..2a139ad`; the 2026-09-04
   profile that reported it correctly simply carried the flag (`docs/nemotron.md:125` has it,
   the oracle's 1M profile does not). **Fix applied to `docs/oracle.md`: the Phase-A serve line
   now carries `--enable-cache-report`.**
2. **`docs/oracle.md`'s budget table row for llama.cpp at 1M is now answered: it does not fit,
   for VRAM, not for time.** See §10 for the numbers and the `--n-cpu-moe 23` floor.

### 13. Artifacts (524K and the 1M attempt)

* `~/ai/bench/oracle/2026-09-05/{ft_524288.json, lc_524288.json, report_524288.md,
  merged_524288.json, llama_524288.log}`
* 1M llama.cpp attempt: `~/ai/bench/oracle/2026-09-05/{llama_1044480_vramwall.log,
  lc_1044480_vramwall_partial.json}`
* Drivers and logs: `<scratch>/oracle/{phaseA_524288.sh, phaseB_1M.sh, phaseB_524288.sh,
  phaseA_524288.log, phaseB_1044480.log, phaseB_524288.log, ft_524288_server.log,
  chunks_1M_ncpumoe23.txt}`
