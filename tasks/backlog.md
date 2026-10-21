# Backlog

Work that is done on a branch but waits for a condition before it reaches main.

## feat/cold-prefix-share (worktree ~/ai/FreeToken-wt/cold-prefix)

- What: the cold session store reuses other sessions' checkpoints by prefix, and spilled
  records carry per-file size and checksum. Study: `.work/prefix-cache-study/README.md`
  (untracked), from the Flare prefix-cache-offload design.
- Blocked on: host RAM. The tests in `tests/scheduler` are written but not run, because
  torch tests must not run beside the live model.
- To land it:
  1. Stop the server (`ft-down`).
  2. In the worktree, run `uv run --no-sync pytest -q tests/scheduler`.
  3. If the tests pass, merge into main and restart the unit (`ft-up`).
- Added 2026-10-21. Production stays on main as it is until then.
