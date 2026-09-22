# Bounded expert mirror — status 2026-09-22 (lever 1 landed)

Supersedes the 2026-09-18 status entirely. That document's two headline claims
are both contradicted by measurement in this tree:

* "graphs must stay off" — they are ON, and have been since the boundary
  restore moved to the scheduler's batch boundary. Every arm below ran with
  decode CUDA graphs enabled and 0 coverage faults.
* "-8.3 GiB RAM (20.2 -> 11.9 GiB cgroup)" — that was `memory.current`, which
  counts page cache. On the metric that answers the question it is not a saving
  at all; see **Host RAM** below.

## What the branch is

A bounded pinned host pool (`MirrorExpertPool`) holding the complement of the
GPU expert cache instead of the whole model, with a device-side Triton swap
kernel turning each step's misses into H2D admissions, D2H writebacks and D2D
relocations. The invariant is `on_gpu(id) OR in_pool(id)` for every expert:
design, sizing and failure modes are in `plan.md`.

## Fixed in this round (all committed on exp/exclusive-expert-ram)

1. **`7c3fa1b` rows are found through the model's own NVFP4 spec.** The pool had
   Nemotron's key format and layer list baked in; a model with no published
   spec is now refused rather than guessed at. This is what lets Ornith run at
   all.
2. **`f007a7c` prefill is assembled from residency, not from the checkpoint.**
   Mirrored prefill ran through `materialize_layer`, which invalidated every
   resident and forced a full coverage rebuild from disk at every
   prefill->decode transition — 3683 rows, 19.3 GiB of disk per request,
   measured as 8.04 s of TTFT on an 8K prompt whose baseline prefill is 0.34 s.
   Coverage already says a prefill layer never needs the disk. **TTFT 8K:
   8.04 s -> 0.21 s, which is FASTER than the baseline's 0.34 s** (resident
   experts reach the buffer by D2D instead of crossing PCIe).
3. **`e2ac473` decode is kept out of the prefill buffer.** A usage sentinel
   cannot express this: LFU ranks by owner frequency first and an empty slot
   loads the minimum, so the empty buffer region won every admission. The first
   fixed arm died with "expert 73 (layer 0) is neither a GPU resident nor in
   the pool". Fixed with a victim floor in the kernel. The same commit makes
   the mirror refuse the ungated admission kernel, which it had silently
   required all along.
4. **`1a51330` an admitted expert keeps its host row.** The kernel freed the
   admission's source row unconditionally, so every expert that reached the GPU
   lost its host copy and paid a D2H on its next eviction forever. Free
   evictions were 1.4% of swaps at 1800 rows and 16.9% with the WHOLE model
   mirrored — host RAM bought nothing, which is why the first capacity sweep
   came out flat. **Free evictions are now 23.6% / 43.4% / 69.5% at 2100 /
   2500 / 2944 rows**, and decode tracks them.

## Decisions (owner, 2026-09-22)

* **Memory ratio 1.00 everywhere.** `--memory-ratio` is a fraction of FREE
  VRAM; the owner's tuner measured 1.00 working on this GPU and chose it for
  every host. Set in `~/.config/freetoken/serve.env`, in
  `scripts/serve-default.sh` and as `measure.sh`'s default.
* **The piro-board embedder is stopped for measurements and is NOT restarted**
  by any agent. Restarting it is the owner's separate decision.
* **Production will use smaller KV ceilings, but testing must confirm 1M
  works** (Nemotron; Ornith's model ceiling is 256K).
* **Ornith must be measured on three KV lanes**: q8_0/q8_0, q8_0 K + q6_0 V,
  q6_0 K + q5_0 V.

## Phase 0 — the baseline of record: empty GPU, memory ratio 1.00

Conditions, which are part of every number: **GPU empty** (`nvidia-smi` 0 MiB
before each arm, embedder stopped), `--memory-ratio 1.00`, **thinking OFF**
(`probe_decode.py` sends `enable_thinking: false`), decode is **probe pass 2**,
spare port 1920, one arm at a time, 2026-09-22, commit of this message.
Baseline ran **first and last** so host drift over the sweep is visible instead
of assumed. Arena is 2173 slots in all three arms.

| arm | host RAM | pool | decode 8K/32K/80K tok/s | TTFT 8K/32K/80K s |
|---|---|---|---|---|
| baseline, whole model (opening) | 18.26 GiB | 2944 rows / 15.41 GiB | 191.9 / 190.8 / 183.3 | 0.65 / 2.90 / 9.31 |
| **auto pool** | **12.90 GiB** | **1893 rows / 9.91 GiB** | **164.9 / 160.3 / 152.3** | 0.68 / 3.01 / 9.68 |
| baseline, whole model (closing) | 18.22 GiB | 2944 rows / 15.41 GiB | 195.8 / 190.8 / 178.5 | 0.63 / 2.83 / 9.43 |

The two baselines agree to 0.04 GiB of host RAM and within 4% of decode, so the
sweep did not drift under the auto arm.

Against the mean of the two baselines the auto pool costs **15–16% of decode**
(164.9 vs 193.9 at 8K, 152.3 vs 180.9 at 80K) and saves **5.34 GiB of host
RAM** (12.90 vs 18.24). 0 coverage faults, 0 starved writebacks, 5376 swaps,
1524 retained rows, **28.4% free evictions**, exactly one CUDA graph capture,
no traceback (the one `ERROR` in the journal is the end-of-arm SIGTERM,
status 143).

**What raising the ratio to 1.00 bought (lever 3), against the same pair
measured at 0.91 on 2026-09-22 00:1x:** the arena grows 1923 -> 2173 slots, so
the pool — which is sized for what the arena *cannot* hold at the 1M ceiling —
shrinks 2144 -> 1893 rows (11.23 -> 9.91 GiB). Host RAM 14.21 -> 12.90 GiB and
decode 151.2 -> 164.9 tok/s at 8K, narrowing the gap from 20% to 15%. Lever 3
therefore buys RAM *and* speed at once; it is the cheapest of the four and it
is now done.

**The coverage-floor log prints (it never did before).** Added in `f96f296`, it
was dead: `bank_row_bytes` is a LIST of per-bank row bytes (the multiplication
raised `TypeError`) and the dataclass field `arena_step_slots` is `None` until
`__post_init__` resolves it, and a `debug_rank0` handler swallowed both. Now
resolved through `cache.arena_layout` and `sum(...)`, with the handler raised to
`warning_rank0` so a future breakage is visible instead of silent:

    Mirror pool: the coverage floor is 1696 of 2173 arena slots,
    leaving 477 slots (2.50 GiB) the growable KV may still take

That line is the diagnosis for the 1700-row arm that killed a server at 80K:
the arena had stopped at its floor and the growable KV had nowhere to go.

**One honest anomaly, recorded rather than averaged away:** pass-1 TTFT at 32K
and 80K was 71–176 s in *both* the baseline and the auto arm (pass 2: 2.9 and
9.3 s). That is the cold expert-bank build on the first large prefill after a
start, not a pool effect — it appears equally in the arm that has no pool. Pass
2 is the TTFT of record for this reason. Phase 3 (1M) exercises KV growth
properly and will say whether anything else hides in it.

## Lever 1 — decode gets the whole arena, and the buffer pays nothing for it

Under the pool, decode ran on 1917 of 2173 arena slots: the prefill double
buffer kept the first 2*E = 256 for itself, fenced off by a victim floor
(`e2ac473`). The whole-model baseline shares those slots freely, because with
every expert in RAM overwriting a slot loses nothing. That missing 12% of
residents was the largest measured cause of the decode gap.

The floor is gone. `_invalidate_prefill_buffer` now writes a buffer half's
occupants back to the pool before the fill overwrites them
(`mirror_kernels.writeback_buffer_occupants`, one launch per half), and
`mirror_warm_start` seats residents from slot 0. The coverage invariant
`on_gpu(id) OR in_pool(id)` is unchanged: a retained duplicate is a free
eviction, a sole copy is written into a free-stack row exactly as a decode
victim would be, and reserve exhaustion increments the same `starved` counter
rather than dropping an expert.

That alone cost TTFT: 0.68 -> 1.47 s at 8K (+116%), because every prefill then
evicted the 256 decode residents sitting in the half it was about to use, and
each sole copy paid a 5.36 MiB writeback. The cause was visible in the record
rather than guessed: pass-1 TTFT was unchanged (0.19 -> 0.25 s) and only pass 2
regressed, and pass 1 runs on a fresh server whose buffer slots are still
EMPTY. So the fix targets exactly that: **an admission landing in the buffer
region always retains its source pool row** (one condition in
`_resolve_swaps_kernel`), making every occupant a duplicate by construction, so
invalidation is pure free eviction with no PCIe transfer. It cannot starve the
reserve, because it only withholds a push onto the free stack.

Conditions as always: GPU empty, ratio 1.00, thinking OFF, decode and TTFT from
probe pass 2, port 1920, 2026-09-22. Four baselines bracket the sweep.

| arm | host RAM | decode 8K/32K/80K | TTFT 8K/32K/80K | free evict |
|---|---|---|---|---|
| baseline, whole model (x4) | 18.22-18.37 | ~194 / ~190 / ~179 | 0.64 / 2.86 / 9.46 | — |
| auto pool (Phase 0) | 12.90 | 164.9 / 160.3 / 152.3 | 0.68 / 3.01 / 9.68 | 28.4% |
| + lever 1 | 12.86 | 171.7 / 169.5 / 162.5 | 1.47 / 4.79 / 10.18 | 64.9% |
| **+ forced retention** | 12.86-13.46 | **177.0 / 175.9 / 162.8** | **0.66 / 3.00 / 9.66** | 95.4% (WRONG, see Correction) |

**The decode gap to the whole model is 15-16% -> 7-9%**, TTFT is inside the
+-5% gate (and better than the Phase 0 auto arm), residents are 2173 of 2173,
and the coverage floor fell 1696 -> 1440 slots, handing the growable KV 1.34
GiB more headroom (2.50 -> 3.84 GiB) -- which is what Phase 3's 1M proof needs.
0 coverage faults, 0 starved, one graph capture, no coverage-lost errors.

**Lever 1 also delivered most of lever 2.** Free evictions went 28.4% -> 64.9%
from the write-back alone (it CREATES a RAM copy for experts that had none),
then -> 95.4% (WRONG, see the Correction section) with forced retention. Lever 2's own target was >50%, and only
4.6% of evictions still pay a writeback, so teaching the LFU victim search to
read `pool_row_of_id` must be re-priced against this arm before it is built.

### Correctness: the pool serves the same model, byte for byte

Counters are not evidence (a stale free-row publish once served the WRONG
experts at 0 faults). Both arms ran `recall.py` at 21K/120K/240K and the
seven-type battery at 21K/120K, same settings, same seeds:

* recall: **3 of 3 codes at every size on both arms**, `finish_reason` stop,
  27 completion tokens -- no truncation.
* battery: **14 of 14 answers byte-identical** between the bounded pool and the
  whole model in RAM at temperature 0, including the two wrong ones.

So both arms score 5/7 at 21K and 3/7 at 120K, and neither failure is the
pool's: `counting` is answered 5 and 4 against a true 7 by BOTH arms (the
model's own limit at length), and every other miss is `finish_reason=length` --
the server's `--max-output-tokens 16384` clipping a thinking answer that had
already found the fact. The multi-hop question emitted 125119 characters of
reasoning without reaching an answer. Phase 7 raises that cap; the battery is
re-run once at the raised cap rather than twice.

### Two measurement artefacts, recorded rather than averaged away

* **A one-off growable-KV commit can contaminate a probe pass.** Arm `lever1b`
  reported TTFT 3.94 s at 8K while its 32K/80K were clean; the journal shows
  `Committed growable KV through 131072 tokens; MoE slots 2173 -> 2056` landing
  inside that window -- the same pathology behind the old 46.4 tok/s outlier.
  Two passes reduce but do not eliminate it. A contaminated number is repeated,
  never averaged: `lever1c` is the number of record.
* **Host RAM varies ~0.56 GiB between identical arms** (`lever1b` 12.90 vs
  `lever1c` 13.46, same code), more than the ~0.2 GiB previously claimed.
  Report it as a range across repeats.

### Prefill once, ask many: only below the resting KV size

The plan's evaluation strategy assumes a long haystack is prefilled once and
every question then hits the radix prefix cache. That holds only up to
`--kv-grow-step-tokens`. On every request completion the scheduler shrinks the
growable KV back to exactly one grow step (`scheduler.py:2271`,
`initial = min(cm.num_pages, step)`) and gets there by calling the radix tree's
real LRU eviction (`scheduler.py:2344, 2473-2489`) before decommitting.
Measured: `cached_tokens` 19840 at a 19927-token haystack (under the 65536
floor) and **0** at 112298 (above it), ~17 s of re-prefill per question.

`--pin-prefix-max-tokens` does NOT fix this and did nothing at all here:
pinning is gated on `pinning_enabled = is_hybrid and pin_prefix_min_tokens > 0`
(`scheduler/cache.py:579-581`) with `is_hybrid = (type == "hybrid_radix")`
(`cache.py:93`), and this model resolves to `cache_type='radix'`. Even under a
hybrid cache the auto-pin only fires for a prefix shared by two session keys,
never one session repeating its own haystack.

Each commit/release cycle also swings the expert arena (MoE slots 2088 <-> 2048
per question) and invalidates the shrunk slots (`offload_cache.py:1281-1283`),
so they return cold -- churn on top of the re-prefill. Needle and recall arms
must therefore set `--kv-grow-step-tokens` at least as large as the biggest
haystack, and any per-question timing taken without that is prefill-bound.

## Lever 2 — prefer victims that already have a pool row

Arms `nemotron-tiebreak-off` / `nemotron-tiebreak-on`, 2026-09-22, empty GPU
(0 MiB), ratio 1.00, auto pool, KV q8_0/q8_0, two probe passes, **thinking OFF**,
128 generated tokens. The only difference between the arms is
`FREETOKEN_MIRROR_TIEBREAK`; the pool stays attached in both, so exactly one thing
changes. Decode of record is pass 2.

| | tie-break off | tie-break on |
|---|---|---|
| decode free-eviction rate | 53.4% | **74.1%** |
| decode 8K / 32K / 80K tok/s | 170.3 / 168.3 / 162.6 | 177.7 / 174.6 / 158.6 |
| TTFT 8K / 32K / 80K s | 0.71 / 4.33 / 9.79 | 0.68 / 3.02 / 12.14 |
| swaps | 4617 | 4713 |
| buffer free evictions | 1941 | 1792 |
| retained rows | 2373 | 3208 |
| host RAM GiB | 12.93 | 12.92 |
| coverage faults / starved | 0 / 0 | 0 / 0 |

**What it bought:** +20.7 percentage points of free evictions, clearing the plan's
>50% target. Writebacks fall from ~2152 to ~1221, about 4.9 GiB of VRAM->RAM
traffic avoided across the arm. Retained duplicates rise 2373 -> 3208, which is the
mechanism: preferring copy-having victims leaves more experts holding a pool row.

**What it did not buy, honestly:** decode is 8K +4.3%, 32K +3.7%, 80K **-2.5%**.
Two of three up, one down, all within a few percent on a single arm pair. The
eviction-rate change is far outside noise; the decode effect is not resolvable
from one pair and is NOT claimed as a win. TTFT at 80K (9.79 -> 12.14) moved more
than the +-5% gate, but TTFT at this size is the noisiest number in the harness
and pass 1 is contaminated by the bank build -- it needs a repeat before it means
anything.

**Rejected approach:** a separate per-slot `has_copy` array maintained alongside
the eviction, warm-start and seed paths. `pool_row_of_id` is already the
authoritative device-resident residency map, so reading it directly leaves nothing
to keep in sync -- one fewer invariant that can silently rot and serve wrong
experts at zero fault count.

**Design note that mattered:** the extra frequency bucket is added BEFORE the
sentinel mask (`2147483647`). Applied after, it would increment the sentinel and
overflow it negative, making masked-out slots the most attractive victims.

### Correction to an earlier claim in this document

An earlier revision of this file said lever 2 was worth "at most ~0.39 percentage
points" and was effectively spent after lever 1. **That was wrong, and wrong
because of the counter bug above:** the ceiling was computed against the inflated
95.4% free-eviction rate, which left no headroom by construction. Against the true
53.4%, lever 2 recovers 20.7 points.

## Lever 4 — the reserve: 2E is the winner, E breaks coverage

Arms `nemotron-reserve-{3e,2e,1e}`, 2026-09-22, empty GPU (0 MiB), ratio 1.00,
auto pool, KV q8_0/q8_0, sizes 8K/80K/713K, two probe passes, **thinking OFF**.
Each arm's journal records `Mirror pool: reserve resolved to N rows
(FREETOKEN_MIRROR_RESERVE_ROWS='N')`, so every row below is attributable.

| reserve | pool rows / GiB | host RAM GiB | 713K | outcome |
|---|---|---|---|---|
| 3E = 384 (default) | 1893 / 9.91 | 13.18 | 0 starved, 0 faults | works |
| **2E = 256** | 1765 / 9.24 | **12.24** | 0 starved, 0 faults | **works, -0.94 GiB** |
| E = 128 | 1637 / 8.57 | — | never reached | **coverage lost, refuses to serve** |

**E is too small, and it failed the right way.** 33 seconds in, on the FIRST 8K
prefill -- not under 713K pressure:

    RuntimeError: bounded expert mirror lost coverage before prefill: expert 161
    (layer 1) is neither a GPU resident nor in the pool, so this layer cannot be
    assembled.

The server answered 503 and stopped rather than assembling the layer from
whatever happened to be resident. That is the correctness guarantee holding under
a real under-provisioning: wrong weights would have produced fluent tokens with
every fault counter at zero, which is the failure mode this branch exists to
prevent. (It does take the backend worker down rather than failing the one
request -- the known open defect, unchanged.)

**2E is accepted**: 0.94 GiB of pinned host RAM recovered, 0 starved write-backs
at 713K, and confirmed at 1M as the plan requires -- arm `nemotron-reserve-2e-1m`,
reserve verified as 256 rows in its journal:

| | value |
|---|---|
| prompt tokens | 1,000,032 (pass 2), 1,000,030 (pass 1) |
| TTFT p1 / p2 | 1162.74 s / 1438.29 s |
| decode p1 / p2 | 76.9 / 72.9 tok/s |
| host RAM | **12.26 GiB** |
| coverage faults / starved | 0 / 0 |
| graph captures | 1 |

This is the best 1M configuration measured on this branch: 1M context in
**12.26 GiB** of host RAM, against 12.94 at the 3E default. Both probe passes
completed, so the teardown fix holds at the smaller reserve too.

**It also refines the pass-2 finding.** This arm ran 8K and 1M only and lost 5% of
decode on pass 2 (76.9 -> 72.9, TTFT +24%). The five-size arm, which ran 713K
first, collapsed to 34.1 with TTFT +78%. So the degradation scales with how much
large-context work the server has already done, not with the request being a
repeat -- which points at accumulated pool/arena churn or radix-tree growth rather
than anything about repetition. Still undiagnosed; the next discriminating step is
a fresh server serving one 1M request twice with nothing else in between.

### The reserve does not change eviction behaviour -- which gave us a noise floor

3E and 2E recorded **identical** `swaps` (7116), free-eviction rate (0.618) and
buffer free evictions (1792). The reserve sets pool capacity, not victim choice,
so the two arms performed provably the same transfer work. Their measured decode
still differed:

| | 3E | 2E |
|---|---|---|
| decode 80K | 168.7 | 154.1 |
| decode 713K | 102.3 | 107.6 |

Same work, **8.7% apart at 80K**. That is this harness's run-to-run noise floor,
measured from the inside rather than assumed. Consequences, applied throughout
this document: lever 2's decode deltas (+4.3% / +3.7% / -2.5%) sit well inside it
and are NOT a claimed win; decode differences between reserve arms carry no
signal, so the reserve's real effects are RAM and starvation, both measured
cleanly. A single arm pair cannot resolve a decode change smaller than ~9% at
80K -- that needs repeats, which is why contaminated arms are repeated rather
than averaged.

### The knob



`FREETOKEN_MIRROR_RESERVE_ROWS` selects the pool's reserve; unset reproduces
`default_reserve_rows` = 3 * num_experts = 384 on Nemotron (E = 128 experts per
layer, 23 MoE layers, 2944 experts total) byte-identically. The value is resolved
once, in one place, and passed to BOTH `plan_capacity(reserve=...)` and
`MirrorExpertPool(reserve_rows=...)`: if only one of them got the override, the
planner would size the pool against a different reserve than the pool enforces. A
test fails if a future edit re-splits them. An invalid value raises instead of
falling back, because a silent fallback produces a valid-looking arm sized against
the wrong reserve -- a wrong number that looks right is worse than a crash.

`measure.sh` gained `FT_RESERVE` (and `FT_TIEBREAK`), both stripped from the host's
`serve.env` like every other knob the sweep owns: without the strip, a value left
in the owner's file would apply to every arm and a 3E/2E/E sweep would be three
identical arms wearing three names. Sweep still to run.

## Correction (2026-09-22): the free-eviction rate was overstated

**Every free-eviction rate measured after lever 1 landed is wrong, including the
95.4% in commit `1ed6372` and the 77.5% in commit `67ec22a`.** Those commit
messages cannot be rewritten; this section is the correction of record.

`mirror_stats()` computes `free_eviction_rate = free_evictions / swaps`. Lever 1
added `_writeback_buffer_kernel`, which evicts the occupants of a prefill-buffer
half and incremented the shared `free_evictions` counter (stats slot 1) -- but a
buffer invalidation is not an admission, so it never incremented `swaps`
(slot 0, written only by `_resolve_swaps_kernel`). The numerator therefore
counted two populations and the denominator one.

It was caught because the rate went **above 1.0**: arm `nemotron-lever2` reported
`free_eviction_rate 1.121` from `swaps 4713`, `free_evictions 5285`,
`writebacks 1732` -- note `5285 + 1732 = 7017`, which cannot be reconciled with
4713 swaps under any reading.

Fixed by giving buffer evictions their own counter (stats slot 6, surfaced as
`buffer_free_evictions`), so `free_eviction_rate` is decode-only and bounded by 1,
and the prefill-buffer work is visible instead of hidden inside a decode metric.
Two lever-1 tests asserted the old behaviour and were corrected; each now also
asserts that a buffer invalidation leaves the decode counter untouched, which is
the invariant that actually broke. The stats tensor grew 6 -> 7, which surfaced a
second unpack site in `_mirror_assert_coverage` that the test suite caught.

**What this does and does not invalidate.** Decode tok/s, TTFT, host RAM, arena
slots, pool rows, coverage faults and starved write-backs are all measured
independently and are unaffected -- every performance conclusion in this document
stands. Only the free-eviction percentages are wrong. Phase 0's 29.7% predates
lever 1 and is sound. Lever 1's qualitative claim (forced retention makes nearly
all buffer evictions free) is confirmed by the D2H byte counts and the tests, not
by the broken ratio. Corrected rates are re-measured in the
`nemotron-tiebreak-off` / `nemotron-tiebreak-on` arms below.

## Phase 3 — 1M proven on Nemotron, and the teardown defect it exposed

Arm `nemotron-1m-fixed`, 2026-09-22, commit of this section. Conditions: empty GPU
(`nvidia-smi` 0 MiB before the start, embedder stopped and not restarted), memory
ratio 1.00, auto pool (`--moe-mirror-host-rows -1`), KV lane q8_0/q8_0, two probe
passes, **thinking OFF** (`probe_decode.py` sends `enable_thinking: false`),
128 generated tokens per probe. Pool 1893 rows / 9.91 GiB, arena 2173 slots,
coverage floor 1440 of 2173 (733 slots / 3.84 GiB left for the growable KV).

| prompt | TTFT p1 (s) | TTFT p2 (s) | decode p1 | decode p2 |
|---|---|---|---|---|
| 8K | 0.18 | 1.27 | 180.4 | 171.7 |
| 80K | 174.67 | 21.24 | 157.4 | 172.6 |
| 240K | 60.18 | 158.88 | 135.6 | 152.3 |
| 713K | 463.67 | 1214.62 | 96.4 | 109.3 |
| 1M | 1322.79 | 2349.16 | 80.4 | 34.1 |

Host RAM 12.94 GiB, peak 14.22 GiB. 0 coverage faults, 0 starved write-backs,
exactly one CUDA-graph capture, 10417 swaps (free-eviction rate 77.5% as recorded,
WRONG -- see the Correction section), and no
`cuMemSetAccess` / `Traceback` / backend-worker-death line anywhere in the journal.

**1M is proven.** The KV grew stepwise to the ceiling with the arena yielding slots
(`Committed growable KV through 1048576 tokens (3.23 GiB physical); MoE slots 1552 -> 1512`)
against the 1440 floor. The pool serves this in 12.94 GiB of host RAM, against
~18.3 GiB for the whole-model configuration at 80K.

> **RETRACTED AND DISPROVEN (2026-09-22).** An earlier revision of this section,
> and the message of commit `67ec22a`, said the whole-model configuration "cannot
> reach this context on this GPU at all". **That was never measured, and arm
> `nemotron-baseline-1m` has now shown it is false.** Whole-model mode reaches 1M
> without difficulty. The claim was an argument wearing a measurement's clothes:
> every baseline arm before it ran 8K/32K/80K and nothing else. The real
> comparison is below.

### 1M, both configurations measured

Arms `nemotron-baseline-1m` (rows=0, whole model) and `nemotron-reserve-2e-1m`
(auto pool, reserve 2E, levers 1-4). Empty GPU, ratio 1.00, two passes,
**thinking OFF**, 128 generated tokens.

| | whole model | **pool, all levers** | delta |
|---|---|---|---|
| 1M reached | yes | yes | -- |
| host RAM | 18.06 GiB | **12.26 GiB** | **-5.80 GiB (-32%)** |
| TTFT 1M p1 / p2 | 1229.34 / 1482.77 s | 1162.74 / 1438.29 s | pool ~5% faster |
| decode 1M p1 / p2 | 86.5 / 84.9 | 76.9 / 72.9 | pool 11-14% slower |

So the pool's case at 1M is **32% less host RAM for 11-14% less decode**, with
prefill slightly in the pool's favour -- it leaves more VRAM for the KV to grow
into. Reachability is not part of the case, and saying otherwise was wrong.

### This also kills a hypothesis about the pass-2 slowdown

The baseline has no pool at all, yet its 1M TTFT degrades from 1229.34 to 1482.77
on pass 2 (+21%) -- essentially the pool's +24% (1162.74 -> 1438.29). The "second
large request is slower" effect is therefore **not** pool or arena churn: it is
present, at the same magnitude, with the mirror entirely absent. That eliminates
one of the three candidates and leaves the growable-KV/radix layer
(`_evict_growable_prefix_pages`, radix-tree growth at a 1M-token tree) and VMM
fragmentation across repeated commit/uncommit ladders. Decode degradation does
differ (baseline 86.5 -> 84.9, -1.8%; pool 76.9 -> 72.9, -5.2%), so the decode
component may still have a pool term; the TTFT component does not.
A second, independent arm (`nemotron-1m-only`) measured 1,000,030 prompt tokens at
TTFT 1125.64 s / 888.0 prefill tok/s / 76.5 decode tok/s — its pass 1 agrees with
this arm's pass 1 (80.4) to within noise.

### The defect: NOT_READY on the arena give-back killed the server

1M prefill and decode were never the problem. After the request the scheduler shrinks
the growable KV and `_arena_grow` re-maps the just-released VA; `cuMemSetAccess`
answered `CUDA_ERROR_NOT_READY`, the exception propagated, and the backend worker died
(`Backend worker is gone and cannot be restarted`). That is why the `nemotron-1m-only`
arm reported `prompt_tokens: 0` for probe pass 2.

`_arena_shrink` already drains the device before it unmaps (step (b)); `_arena_grow`
had no matching sync. **The asymmetry predates lever 1**, and `engine.py:2145` already
issues a full-device synchronize before the arena branch — so lever 1 is exonerated.
Fix, both Python (no C++ rebuild):

* `torch.cuda.synchronize(self.device)` at the top of `_arena_grow`, symmetric with
  the shrink path.
* `VMMTensor.commit_ranges` / `uncommit_ranges` retry `CUDA_ERROR_NOT_READY` six times
  with a device synchronize and exponential backoff (~1.3 s total). NOT_READY means
  "ask again", not "this failed"; every other error still raises on the first attempt,
  so an out-of-memory cannot be retried into a stall. Tests assert all three behaviours
  (`tests/kernels/test_vmm_tensor.py`).

Reproduce: `FT_NAME=nemotron FT_ROWS=-1 FT_RATIO=1.00 FT_SIZES="8000 80000 240000 713000 1000000" bash tasks/exclusive-expert-ram/measure.sh nemotron-1m-fixed`

### Open finding: the second very large request is ~2x slower than the first

Not a pool defect, not diagnosed yet, and recorded because it changes how the tables
must be read. Within this arm, with the same prompt on the same server:

| | prefill tok/s pass 1 | pass 2 |
|---|---|---|
| 713K | 3117.6 | 642.1 |
| 1M | 862.4 | 473.9 |

`#cached-token: 0` on every pass-2 prefill batch, so this is a full re-prefill with no
prefix reuse (expected above `--kv-grow-step-tokens`) — but that explains only why it
is not *fast*, not why it is *half as fast*. Throughput degrades within the request,
so it is not queueing. Candidates not yet discriminated: pool/arena churn after pass 1
has consumed the warm-start duplicates; radix-tree growth and
`_evict_growable_prefix_pages` cost at a 1M-token tree; VMM fragmentation across
repeated commit/uncommit ladders.

**Consequence for the protocol.** The plan's rule is that the decode of record is
pass 2. That rule stands at 8K/32K/80K, where pass 2 is equal or better and where every
lever comparison in this document was measured. At >= 240K, pass 2 is systematically
pessimistic from this effect, so both passes are reported above and neither is quietly
preferred. A single-request 1M figure taken on a freshly started server is the honest
one for "what does 1M cost": TTFT ~1126-1323 s, decode ~76-80 tok/s.

## Superseded: measured 2026-09-21 at ratio 0.91, on a SHARED GPU

**Every decode number in this section is void.** The piro-board embedder held
4.3 GB of VRAM throughout, and `--memory-ratio` is a fraction of FREE VRAM, so
these arms sized themselves to a smaller card than the table implies — they run
10–17% low. They are kept because the *relative* findings (retention works,
free evictions track decode, the mirror knob is real) are what motivated the
four levers. The numbers of record are in Phase 0 above.

### The 2026-09-21 table (void for decode, see above)

Warm arms, one at a time alone on the host, spare port 1920, same
`--memory-ratio 0.91`, graphs on, 1M KV ceiling. Full table:
`results/sweep.tsv`; metric definitions and two superseded ones:
`results/README.md`.

| arm | pool | decode 8K/32K/80K tok/s | TTFT 8K/32K/80K s | free evict |
|---|---|---|---|---|
| baseline (whole model pinned) | — | 172.6 / 169.5 / 159.2 | 0.34 / 2.92 / 11.19 | — |
| mirror 2100 rows | 10.99 GiB | 136.2 / 137.4 / 120.5 | 0.21 / 3.03 / 11.23 | 0.236 |
| mirror 2500 rows | 13.08 GiB | 140.8 / 140.9 / 55.2 | 0.21 / 3.02 / 9.95 | 0.434 |
| mirror 2944 rows (whole model) | 15.41 GiB | 149.4 / 147.7 / 131.1 | 0.21 / 3.03 / 9.66 | 0.695 |

0 coverage faults and 0 starved writebacks in every arm.

* **Decode**: the mirror costs 13% at full capacity (149.4 vs 172.6), down from
  25% before retention. Two causes, not one: the remaining 30% of evictions
  still pay a D2H, and the mirror gives 2*E = 256 GPU slots to the prefill
  buffer, so it decodes from 1667 residents against the baseline's 1923 — 13%
  fewer, which is the size of the remaining gap.
* **TTFT** is at or better than the baseline everywhere.
* **The 55.2 tok/s at 80K on the 2500 arm is not explained** and is not noise
  to be averaged away; it needs a repeat before any conclusion rests on that
  row.

## Host RAM — re-measuring at a defined point

The first attempt at this table drew a wrong conclusion and it is worth
recording why. Arms were sampled at different points in their life -- some at
readiness, one after an 80K request had grown the KV arena, which on WSL2 can
be host-backed -- and the pinned region came out at 18.1 GiB for pools of both
10.99 and 15.41 GiB. I read that as a fixed arena the pool is served from,
i.e. a knob that moves nothing. Sampling the floor capacity refuted it:

    mirror 1700 rows (pool 8.90 GiB), at readiness:  9.11 GiB pinned
    baseline        (banks 15.41 GiB), at readiness: 15.49 GiB pinned

The knob works. At its floor the mirror pins 6.4 GiB less than pinning the
whole model (total RSS 12.9 GiB against 19.3), which is the saving this branch
exists for. The apparent flatness was a measurement artefact of my own making.

`measure.sh` now samples host RAM with the server ready and nothing served yet
(`ram_gib`, `pinned_gib`) and again after the probe (`ram_after_80k_gib`), so
model residency and KV growth are separate numbers. The full curve is being
re-measured on that basis -- two baselines, first and last, so host drift under
a 15-minute sweep is visible rather than assumed.

## Not yet done

Work is tracked as the phases of `~/.claude/plans/steady-baking-bird.md`
(levers 1-4, 1M, Ornith across three KV lanes, long generation, needles).
Phase 0 is done and is the section above.

* **Lever 2** — RE-PRICE FIRST. Its premise was that ~70% of evictions pay a
  writeback; after lever 1 only 4.6% do, past lever 2's own >50% target. Decide
  against the lever-1 arm rather than building to a stale number.
* **Lever 4** — the pool reserve is 384 experts (3*E, 2.0 GiB), sized before
  retention existed; sweep 3E/2E/E.
* Ornith-1.5-35B-A3B-NVFP4: the same curve, on the same harness
  (`sweep-ornith.sh`), on three KV lanes. Not optional, not deferred.
* Long-context verification: 21K / 240K / 713K / 1M recall plus the seven-type
  needle battery (`needles.py`), `acceptance.sh` R3 + R6.
* Long generation (`long_gen.py`): every probe so far decodes 128 tokens, so
  the branch has never been measured on the workload these models ship for.

### Open defects (flagged, not fixed)

* A refused growable-KV commit kills the scheduler worker instead of failing
  the request. (The *transient* NOT_READY case is fixed — see Phase 3 — but a
  genuinely refused commit still takes the worker down.)
* The second very large request on a server is about half the prefill speed of
  the first; see the Phase 3 open finding. Undiagnosed.
* `MirrorExpertPool.close()` unregisters the pinned banks but does not drop
  `self.banks`.
* The NVFP4 review's fixes are written but held as a patch, not yet on the
  branch (they were developed in the main checkout while arms were running).
* `nemotron-r-baseline` is contaminated (GPU microbenchmarks ran beside it) and
  must be repeated; `results/README.md` says so where the number lives.

## Rules this work runs under

* Server only via systemd, never from an agent shell; spare port 1920, never
  1919. No torch pytest and no GPU benchmark beside a live arm — doing that
  cost one baseline (80K TTFT 9.36 -> 11.19 s) and killed the 1700 arm outright
  ("growable KV refused an unsafe VMM commit: need 0.46 GiB free, have 0.18").
* Commit on this branch; no merge, no push.
