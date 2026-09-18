# Mirror expert pool — status 2026-09-18

## Where it works
- Unit invariants: `tests/moe/test_mirror_pool.py` 6/6.
- Real model, short/medium context: answers correct (`42`, counting),
  **131-142 tok/s** vs 176.4 baseline and 19.9 for the disk-backed version,
  cgroup **11.86 GiB** vs 20.20 baseline (**-8.3 GiB**), CUDA graphs ON,
  0 coverage faults, **82.4 % of evictions free** (duplicates working).

## Open defect (blocks long context)
`_mirror_stage_layer` assumes the prefill materialize writes into a per-layer
slot window at `lru_slot_range(layer)`. It does not: for a uniform cache the
range is the whole cache for every layer, and the kernel always installs expert
`e` into slot `e`. Consecutive materializes therefore reuse slots 0..E-1 and
invalidate same-layer owners anywhere in the cache:

    after warm : [0 1 2 3 4 5 8 9 10 11 12 16 17 18 19 20]
    after mat(0): [0 1 2 3 4 5 6 7 10 11 12 16 17 18 19 20]
    after mat(1): [8 9 10 11 12 13 14 15 -1 -1 -1 16 17 18 19 20]
    after mat(2): [16 17 18 19 20 21 22 23 -1 -1 -1 -1 -1 -1 -1 -1]

Consequences: the GPU half-empties across prefill, the staging math books the
wrong rows, and `swap_smoke.py` reports byte mismatches from decode step 1.
Symptom on the real model: a 96K-token completion degenerates
("Hass Hass Hass") once KV growth shrinks the arena.

## Next step
Rework `_mirror_stage_layer` around the kernel's real behaviour (`slot == expert
id` within `[begin, begin+E)`, shared by every layer) instead of a private
window, then re-run:

    cd ~/ai/FreeToken && FREETOKEN_EXPERT_ARENA=1 .venv/bin/python \
      ~/ai/FreeToken-wt/exclusive-expert-ram/tasks/exclusive-expert-ram/swap_smoke.py

It must print PASS before any server run is meaningful.

## Running things
- Worktree is now durable at `~/ai/FreeToken-wt/exclusive-expert-ram`
  (the old `/tmp` one was lost to the reboot; all commits survived).
- The worktree has no built CUDA extensions; run from the main checkout's venv
  (the `.so` files were copied in, they are build artifacts, not versioned).
- Experiment server: unit `freetoken-swap-mirror`, port 1920, env
  `FREETOKEN_MIRROR_EXPERT_RAM=1`. Never the production unit.
- GPU: the piro-board embedder (`llama-server`, 4.1 GiB) respawns after being
  killed; 11.8 GiB remain free, which is enough.
