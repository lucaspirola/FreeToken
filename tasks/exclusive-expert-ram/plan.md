# Bounded host mirror for the NVFP4 expert cache — design

Branch `exp/exclusive-expert-ram`, worktree
`/home/lucas/ai/FreeToken-wt/exclusive-expert-ram`.

**The branch name is older than the design it now carries.** It started as
"exclusive expert RAM", a disk-backed pool behind
`FREETOKEN_EXCLUSIVE_EXPERT_RAM=1` in which a GPU miss could fall through to the
checkpoint. That design is **REJECTED** — see "Rejected" at the bottom — and
nothing in the tree implements it. What exists is a bounded pinned host mirror.

## Objective

Serve the same models on less host RAM than the default profile's
whole-model-in-RAM residency, at the highest decode throughput that costs
buys — measured, on this tree, against a warm baseline taken the same day on
the same port. Two models, neither optional:

| | Nemotron-3.5-Lightning-30B-A3B-NVFP4 | Ornith-1.5-35B-A3B-NVFP4 |
|---|---|---|
| model_type | `nemotron_h` | `qwen3_5_moe` |
| MoE layers x experts | 23 x 128 = 2944 rows | 40 x 256 = 10240 rows |
| experts | ungated (relu2) | gated (silu, gate\|up fused) |
| expert row | 5.36 MiB | 1.688 MiB |
| expert banks | 15.41 GiB | 16.88 GiB |
| context | 1M | 256K |

Ornith's row and bank sizes are read off the checkpoint, not the config: one
expert is gate+up+down packed NVFP4 (524288 B each) plus three e4m3 block-scale
planes (65536 B each) and three f32 globals = 1769484 B. Its `config.json` has
neither `decoder_sparse_step` nor `mlp_only_layers`, so all 40 layers are MoE,
which is what `models/qwen3_5_moe/weight.py` assumes (`layer_to_bank` is the
identity). That file's key regex was checked against this checkpoint's
`weight_map`: it matches 92160 tensors (40 x 256 x 3 projections x 3 kinds) and
excludes the 768 `mtp.layers.*` expert tensors of the MTP head, which is not
served. So the mirror reaches Ornith through the model's own spec, with no
Nemotron geometry anywhere in the pool.

## What it is

The default profile pins every expert bank in host RAM, so a GPU cache miss is
a plain H2D. That is fast and costs the whole model in RAM. The mirror keeps a
**bounded pinned pool** holding only what the GPU does not, plus slack.

**The coverage invariant is the whole design:**

    for every expert id:  on_gpu(id)  OR  in_pool(id)

It is what makes a GPU miss a plain H2D and never a disk read, which is what
keeps decode CUDA graphs capturable. Every sizing rule below is derived from
it, and every failure mode is a way of breaking it.

* `MirrorExpertPool` (`moe/mirror_pool.py`) owns the pinned rows and reads the
  checkpoint with O_DIRECT at load and at host boundaries only — never on a
  decode step.
* The residency maps live on the device; the Triton swap kernel
  (`moe/mirror_kernels.py`) turns each step's misses into H2D admissions, D2H
  writebacks for victims with no pool row, and D2D relocations for experts that
  are already resident in another slot.
* Rows are located through the model's own `Nvfp4ExpertSourceSpec`
  (`models/nvfp4_banks.py`, exported per model as `NVFP4_EXPERT_SOURCE_SPEC`),
  so the pool has no key format, layer-type list or gating assumption of its
  own. A model that has not published a spec is **refused**, not guessed at.

## Sizing, all derived from coverage

* `plan_capacity(L, E, final_gpu_slots)` = `L*E - final_gpu_slots + reserve`.
  Sized for the KV **ceiling**, where the GPU cache is smallest and the host
  side must be largest: growing a pinned pool later costs ~762 ms/GiB.
* `default_reserve_rows(E)` = `3 * E`. One layer so a writeback always has a
  landing row the same step's upload is not reading, one for staging, and one
  of decode-burst slack (measured: 4401 starved writebacks with only two).
  The planner and the pool must agree on this number, because the arena floor
  is priced against it.
* `min_gpu_slots` = `total - capacity + reserve`, i.e. the GPU residents
  coverage needs. A saturated pool (`capacity >= total`) is exempt. The KV
  arena may not shrink the expert cache below it; `_grow_runtime_kv_arena`
  folds it into the floor **before** the shrink, and adds the buffer region
  below, because `min_gpu_slots` counts residents while the floor counts slots.
* `prefill_buffer_slots(E)` = `2 * E`: the head of the cache, which the prefill
  double buffer owns outright under the mirror. See below.

## What capacity buys: duplicates

Coverage only requires `capacity >= complement + reserve`. Everything above
that floor is spare, and what the spare rows do decides whether host RAM is
worth anything here at all.

A row is a **duplicate** when its expert is also on the GPU. Coverage does not
need it — the GPU copy already satisfies the invariant — but it makes that
expert's eviction **free**: the victim already has a host copy, so the swap is
a pure H2D admission with no D2H writeback behind it. The free-eviction rate is
therefore the decode cost of the mirror, and duplicates are the only lever on
it.

The kernel keeps duplicates by *retaining* the row an admission read from
instead of freeing it (`mirror_kernels._resolve_swaps_kernel`). Expert weights
are read-only on the GPU, so a retained row stays a valid copy for as long as
its expert is resident, and an expert that just arrived from the pool is the
likeliest to go back out. Retention consumes free rows and nothing returns
them, so it stops at `retain_floor` — the pool's own reserve, which bounds one
launch's writebacks by construction. Steady state is then

    owned = complement + duplicates,  duplicates = capacity - complement - reserve

which is the line the RAM x decode curve is measured along. Freeing the row
unconditionally, which is what shipped, pins `duplicates` at whatever warm
start seeded and lets admissions consume even that: measured 1.4% free
evictions at 1800 rows and 16.9% with the whole model mirrored, i.e. no curve
at all.

## Slot regions under the mirror

    [0, 2E)            prefill double buffer
    [2E, cache_size)   rest of the arena

This used to read "prefill's alone": a victim floor fenced `[0, 2E)` out of
decode admission entirely, because evicting an occupant to make room for a
prefill layer would drop an expert's only copy. That floor was removed at
commit `1ed6372` — it cost 256 of 2173 arena slots on Nemotron for no coverage
benefit once the writeback exists (below). Decode now admits into the buffer
region exactly like any other slot; warm start seats residents from slot 0,
holding nothing back for prefill (`test_warm_start_seats_the_whole_arena`).

What keeps coverage instead: `_prefetch_split_mirror` writes an occupant back
to its pool row (or recognises it already has one — a retained duplicate needs
no D2H) before the buffer overwrites it, so the expert that was sitting in the
buffer slot is never left without a copy. `test_decode_may_admit_into_the_prefill_buffer_and_prefill_still_covers`
drives decode admissions into `[0, 2E)` on purpose, over a sweep of random
routing, then runs a full prefill sweep over the result and checks every
expert is still either a GPU resident or a pool row — the exact hole the old
victim floor existed to close, now closed by the writeback instead.

## Prefill

Prefill runs through the overlap double buffer and reads **no checkpoint bytes
at all**. Coverage is exactly the statement that this is possible: each of a
layer's experts is either a decode resident — gathered slot -> buffer D2D, no
PCIe — or has a pool row, copied host -> buffer H2D. One
`fast_index_copy_multi_jit` launch each, on the copy stream
(`_prefetch_split_mirror`).

The alternative, `materialize_layer`, reinstalls the layer into the LRU slots
and invalidates every other resident. That empties the mirror and forces a full
coverage rebuild from disk at every prefill->decode transition. Measured, that
is 19.3 GiB per request and 8.0 s of TTFT on an 8K prompt whose baseline
prefill takes 0.34 s. It is the single largest thing that was wrong here.

## Hard requirements

* **`FREETOKEN_EXPERT_ARENA=1`.** Only the gated `_v2` admission kernel
  publishes `victim_ids`/`prior_ids`, and the swap kernel needs the displaced
  expert's identity to know whether its bytes still exist. On the ungated
  kernel every writeback is skipped and coverage is lost with the fault
  counters still reading zero. `attach_mirror_pool` refuses.
* Native NVFP4 experts, the triton backend, single-rank GPU offload, no CPU or
  pageable routing.
* A published `NVFP4_EXPERT_SOURCE_SPEC` for the model, agreeing with the
  config about gating.

## How a failure shows up

`resolve_swaps` cannot raise from inside a Triton kernel, so it counts:
`violations` (admission with no host copy) and `starved` (a writeback with no
free row). `mirror_fault_check()` turns either into a `RuntimeError` at the
scheduler's batch boundary — output past that point is wrong, not merely slow.

The counters are a **backstop for the sizing, not a detector**: a stale
free-row publish once served wrong experts with `violations` still at 0 (see
`_mirror_publish_free_rows`). Correct sizing is what keeps coverage, so every
capacity change must also be checked against long-context output, not counters.

## Rejected

**Disk-backed exclusive pool** (`FREETOKEN_EXCLUSIVE_EXPERT_RAM`,
`ExclusiveExpertPool`, the original `plan.md`). A GPU miss could fall through
to the checkpoint, which puts an O_DIRECT read on the decode path. That breaks
CUDA graphs — a replay runs no host code — so decode would have had to go
eager. The bounded mirror gives the same RAM lever with the disk contact moved
to load and host boundaries, and graphs stay on. No code for it remains.

## Where the numbers are

`tasks/exclusive-expert-ram/results/sweep.tsv`, one row per arm. Each arm is
produced by `tasks/exclusive-expert-ram/measure.sh` (one arm at a time, alone on
the host, warmed before measuring, on a spare port — never :1919) into its own
`$ARM-record.json`; `table.py` then rebuilds the TSV from every record with a
fixed column list. `results/README.md` records what the RAM column means and the
two definitions that were tried and discarded. `STATUS.md` reads the curve.
