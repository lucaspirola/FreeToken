# Mirror expert pool — status 2026-09-18 (end of session)

## Architecture (as designed with the owner)
Bounded pinned host mirror; a GPU miss is served from RAM, the displaced expert
is written back to RAM. No disk on the hot path, so CUDA graphs stay ON. Where
`gpu_slots + mirror_rows > L*E` the slack holds duplicates, and evicting a
duplicated expert costs no writeback.

## Measured on the real model (port 1920, 21K-token prompt)
- RAM: cgroup **11.9 GiB** vs 20.2 baseline -> **-8.3 GiB**
- decode: 131-142 tok/s vs 176.4 baseline (disk-backed version was 19.9)
- TTFT 0.15 s vs 0.36 baseline
- free_eviction_rate 0.28-0.82 (duplicates work)
- short prompts: correct ("42")

## Bugs found and fixed (all with repros)
1. prefill materialize published neither victims nor prior owners -> the
   non-size-class kernel needed the same instrumentation as the sized one.
2. "already in place" cannot be read from id_of_slot (overwritten first);
   prior_ids is now a separate signal from victim_ids.
3. slot relocation corrupted slot_for_id of the overwritten owner -> removed;
   the materialize reinstalls the layer anyway.
4. D2D relocation issued AFTER the uploads read admitted bytes -> moved before.
5. free stack was rebuilt from the HOST map while the kernel owns the DEVICE
   map -> rows still in use were published free (236 wrong experts -> 0).
6. `_mirror_refill_uncovered` had the same host/device divergence across an
   arena shrink (0 -> 72 -> 237 wrong across three shrinks -> now 0).
7. prefill cannot hold the decode coverage invariant (it would need L*E-E =
   2816 rows, 14.7 GiB); it now drops evictions and coverage is re-established
   once at the prefill->decode boundary.

## Open defect
`starved_writebacks = 4017` on a 21K-token request (16025 swaps). Raising the
reserve from 2 to 3 layers barely moved it (4401 -> 4017), so the free stack is
LEAKING in the real workload, not undersized. Every starved writeback loses an
expert's only copy; the answer is now grammatical but still wrong
("the user is asking for the access code" instead of SIERRA-7741).

Not reproduced locally yet: stack depth stays exactly stable in every synthetic
case tried (40-step decode, 23 layers/token, 64-token prefill batches, three
arena shrinks). The real run differs in scale (2944 experts, top-6, 23 layers,
21K-token prefill) and in having real KV growth.

## Next step
Instrument `free_count` per step on the live server (log it from
copy_missing_mirror behind an env flag) and find where a pop is not matched by a
push. Suspect: `publish_freed_rows` runs per copy_missing, but `resolve_swaps`
pops from a stack that a concurrent prefill path may have rebuilt.

## Running things
- Worktree: `~/ai/FreeToken-wt/exclusive-expert-ram` (durable; /tmp one was lost
  to the reboot, all commits survived). `main` untouched at eb21dc6.
- The worktree has no built CUDA extensions; `.so` files are copied in from the
  main checkout (build artifacts, not versioned).
- Tests:
    cd ~/ai/FreeToken-wt/exclusive-expert-ram && PYTHONPATH=python \
      ~/ai/FreeToken/.venv/bin/python -m pytest tests/moe/test_mirror_pool.py -q
    cd ~/ai/FreeToken && FREETOKEN_EXPERT_ARENA=1 .venv/bin/python \
      ~/ai/FreeToken-wt/exclusive-expert-ram/tasks/exclusive-expert-ram/swap_smoke.py
- Server: unit `freetoken-swap-mirror`, port 1920, `FREETOKEN_MIRROR_EXPERT_RAM=1`.
- GPU: the piro-board embedder (llama-server, 4.1 GiB) respawns when killed;
  11.8 GiB free is enough.
