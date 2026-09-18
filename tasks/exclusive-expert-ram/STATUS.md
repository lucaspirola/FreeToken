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
Root cause now PROVEN, not hypothesized. `Engine.forward_batch` decodes through
`GraphRunner.replay` — a pure graph replay. Host code (`ensure_experts`) runs
exactly once at CAPTURE. The captured LRU kernel rewrites `topk_ids` from
expert ids to slot ids IN PLACE on every replay. So:
  - token 1 (capture routing): correct
  - token 2..N (replay): the buffer still holds token N-1's SLOT ids; the new
    token's routing (written by GraphCaptureBuffer.copy_from into the static
    buffer) is never consumed, because the reader expects EXPERT ids at entry.
  91 wrong experts over 60 pure replays reproduced in
  tasks/exclusive-expert-ram/graph_race_repro.py (no server needed).

Why the baseline does NOT have this problem: with whole-model host residency
`copy_missing` sources row = expert id, and slot ids written by the LRU are
consumed only within the same captured step. The mirror does not change this;
the defect is that the buffer's CONTENT semantics differ between capture
(expert ids) and steady-state replay (previous step's slot ids). The baseline
is bit-correct here only because its GEMM... (to verify if re-enabling graphs:
diff a baseline graph-mode run's topk_ids handling against the mirror's — the
suspicion is the baseline hides it because its expert ids are ALSO valid
source rows; the mirror's pool-row indirection makes it visible).

Fix attempts made and REJECTED (do not retry these shapes):
1. Prime kernel writing slot ids for resident experts at replay start: breaks
   the LRU contract (it reads EXPERT ids at entry to compute misses), and -1
   for absent experts corrupts miss admission. The buffer cannot carry both
   semantics.
2. `capture_pos_of_expert` side table maintained by resolve_swaps: the
   cleared-positions are exactly the misses, which the LRU must see.

Promising directions (not attempted):
- A second persistent buffer: `copy_from` writes raw expert ids into
  `routing_buf`; the captured step starts with a small kernel
  routing_buf -> topk_ids (pure copy, identity), then LRU consumes topk_ids as
  at capture. Cost: one extra buffer copy per layer per replay (trivial).
  Requires: hooking GraphCaptureBuffer.copy_from / the model's decode entry to
  write routing_buf instead of topk_ids, and adding the copy kernel into the
  captured region before the first MoE layer.
- Or: make ensure_experts idempotent on already-slot ids (treat value as
  expert id iff < num_experts? slot ids are >= 0 too... needs a tag bit).
  Fragile; prefer the explicit second buffer.

## Known issue #2 (perf, not correctness): warm start at the boundary
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
