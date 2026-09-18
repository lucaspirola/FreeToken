# Mirror expert pool — status 2026-09-18 (end of session, round 2)

## What is DONE and verified (do not re-do)
- Architecture works: bounded pinned mirror, GPU<->RAM swap, disk only at
  load/boundaries. -8.3 GiB RAM (20.2 -> 11.9 GiB cgroup).
- Eager decode CORRECT end-to-end: 21K recall = "SIERRA-7741", 74K recall
  cites the planted code with KV-arena growth mid-request, 0 coverage faults,
  0 starved writebacks. Same-seed greedy vs baseline: 2/3 byte-identical, 1
  coherent paraphrase (slot-order float reduction; not corruption).
- Tests: unit 3/3, tests/scheduler 380/380, swap_smoke PASS (prefill sweeps +
  decode + arena shrinks, byte-exact), baseline re-run from this tree correct
  at full speed with graphs (shared-path changes are harmless).
- Numbers recorded: benchmarks/results/nemotron35_lightning_5080_exclusive_2026-09-18.md

## Known issue #1 (the only correctness blocker): graphs must stay off
State of diagnosis (2026-09-18, end of round 2 -- supersedes earlier notes):
- DISPROVEN: "replay reads the previous token's slot ids". The router is
  INSIDE the captured region (model.forward is captured whole), so topk_ids
  are recomputed as fresh EXPERT ids from the current token's hidden states on
  every replay. There is no staleness at the LRU's entry.
- PROVEN (tasks/exclusive-expert-ram/graph_race_repro.py, no server needed):
  60 pure replays of a captured (ensure_experts + copy_missing) step produce
  91 wrong expert bytes. Stats anomaly that points at the mechanism:
  swaps=36, free_evictions=36, writebacks=0, coverage_faults=20. Over 36
  admissions into a 17-slot cache, ~19 victims MUST have been unmirrored
  GPU-residents (their mirror rows were freed at their own admission), so
  writebacks=0 means the replayed kernel read pool_row_of_id[victim] >= 0 --
  a stale/incorrect mirror map inside the captured chain. Not yet root-caused.
- LANDMINE found on the way: a coverage violation inside a replay copies
  POOL ROW 0 into the target slot and counts it on device; the host-side hard
  error never runs in graph mode, so it is SILENT wrong bytes. Whatever the
  primary fix is, the violation path must be made graph-safe (no row-0 copy).
- REJECTED fix shapes (do not retry): priming topk_ids with slot ids (breaks
  the LRU's expert-id entry contract; also solved a non-problem); a
  capture_pos side-table (degenerates to exactly the miss set).

Next diagnostic step: instrument the repro to dump, for one failing replay,
the full device state the resolve kernel read (pool_row_of_id, free stack,
id_of_slot, prior/victim) and compare against the same step run eagerly with
identical routing -- the divergence point is the capture-safety defect.

## Known issue #2 (perf, not correctness): warm start at the boundary## Known issue #2 (perf, not correctness): warm start at the boundary
One full checkpoint re-read (~4 s) per prefill->decode transition
(`_mirror_needs_coverage`). Optimization: warm start currently re-reads ALL
rows; it could keep rows whose (expert, bytes) are still valid — compare
against the device ownership map and skip re-reading experts whose row already
holds them. Cuts the 4 s to well under 1 s in the steady case.

## Follow-up list (owner-approved, in order)
1. Graph race: second-buffer design above; validate with graph_race_repro.py
   (must print PASS), then server + 21K recall + 127-tok decode probe.
2. Warm start incremental reuse.
3. 200K/600K prompts, acceptance.sh R3/R6, promote swap_smoke + repros into
   tests/moe/, then merge decision.

## Running things (unchanged from round 1)
- Worktree: ~/ai/FreeToken-wt/exclusive-expert-ram (durable). main at eb21dc6.
- Build extensions are copied .so files from the main checkout (not versioned).
- Test commands:
    cd ~/ai/FreeToken && FREETOKEN_EXPERT_ARENA=1 .venv/bin/python \
      ~/ai/FreeToken-wt/exclusive-expert-ram/tasks/exclusive-expert-ram/swap_smoke.py
    .../graph_race_repro.py   (must print PASS == graphs-safe... see file doc)
    pytest: PYTHONPATH=python ~/ai/FreeToken/.venv/bin/python -m pytest tests/moe/ -q
    (run from the worktree)
- Server: transient unit freetoken-swap-mirror, port 1920,
  FREETOKEN_MIRROR_EXPERT_RAM=1. NEVER the production unit.
- GPU: piro-board embedder (llama-server, ~4 GiB) respawns if killed; 11.8 GiB
  free is enough.
