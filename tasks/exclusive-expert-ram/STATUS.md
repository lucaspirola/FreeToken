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
ROOT CAUSE FOUND (2026-09-18, round 2 -- final, proven by direct state dumps):
- A prefill sweep leaves GPU = last layer only, mirror = 0 rows. In EAGER mode
  the prefill->decode boundary flag (_mirror_needs_coverage) is consumed by
  the first ensure_experts, which runs a full mirror_warm_start (host + disk),
  restoring the pool. In GRAPH mode the decode step is a pure replay: host
  code runs only at CAPTURE, so the boundary warm start NEVER runs after any
  later prefill. The replay then miss-admits every routed expert with no
  mirror row: coverage_faults grow 6 per replay and the GEMM reads pool row 0
  (silent wrong bytes -- the host-side hard error cannot run in a replay).
- Everything about the earlier "stale topk slot ids" theory is DISPROVEN:
  the router is captured inside the graph and recomputes expert ids per
  replay; the swap/priming machinery itself is replay-safe (verified: writeback
  descriptors, free stack, ownership maps all correct across pure replays).
- Independent landmine (fix regardless of mode): the resolve kernel copies
  POOL ROW 0 on a coverage violation "to keep the descriptor well-formed".
  Eager raises; a replay cannot. The row-0 copy must go -- leave the slot
  holding stale bytes and let the violation counter trip a scheduler-level
  check, or write a canary.

Fix shape (not yet implemented): the boundary restore must be host-visible
BEFORE any replayed decode step can be admitted. Options:
  a) scheduler-side: after a prefill batch completes, if mirror is on, run
     cache.reset() / warm start eagerly at the batch boundary (host code runs
     there -- forward_batch is called per batch). Cheapest, reuses existing
     tested code, ~4 s once per prefill (see #2 to shrink that).
  b) device-signalled: a device flag the scheduler polls (one .item() per
     step) that forces one eager decode step (graphs skip for one token,
     boundary code runs, replay resumes).
Option (a) is boring and likely right: the scheduler knows when a decode
phase follows a prefill phase; hook that transition in engine/scheduler, not
inside the cache.

## Known issue #2 (perf, not correctness): warm start at the boundary## Known issue #2 (perf, not correctness): warm start at the boundary## Issue #2 (perf, minor): warm start at the boundary -- ANALYZED, deprioritized
One full checkpoint re-read (~4 s) per prefill->decode transition. Breakdown:
~2175 rows re-fill the EMPTY GPU cache (a prefill sweep clears every resident
except the last layer), ~770 fill the mirror complement. The GPU re-fill is
irreducible without a design change (e.g. letting prefill materializes write
victims back into free rows until the stack runs dry -- saves <= 25 % of reads,
re-introduces drain-adjacent machinery). Within a 50 K-500 K prefill (tens of
seconds to minutes), 4 s is noise. Revisit only if short-prefill workloads
matter.

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
