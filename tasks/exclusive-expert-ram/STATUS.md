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
| **+ forced retention** | 12.86-13.46 | **177.0 / 175.9 / 162.8** | **0.66 / 3.00 / 9.66** | **95.4%** |

**The decode gap to the whole model is 15-16% -> 7-9%**, TTFT is inside the
+-5% gate (and better than the Phase 0 auto arm), residents are 2173 of 2173,
and the coverage floor fell 1696 -> 1440 slots, handing the growable KV 1.34
GiB more headroom (2.50 -> 3.84 GiB) -- which is what Phase 3's 1M proof needs.
0 coverage faults, 0 starved, one graph capture, no coverage-lost errors.

**Lever 1 also delivered most of lever 2.** Free evictions went 28.4% -> 64.9%
from the write-back alone (it CREATES a RAM copy for experts that had none),
then -> 95.4% with forced retention. Lever 2's own target was >50%, and only
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
  the request.
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
