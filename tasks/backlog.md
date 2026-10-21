# Backlog

Work that is done on a branch but waits for a condition before it reaches main.

## feat/cold-prefix-share (worktree ~/ai/FreeToken-wt/cold-prefix)

- What: the cold session store reuses other sessions' checkpoints by prefix, and spilled
  records carry per-file size and checksum. Study: `.work/prefix-cache-study/README.md`
  (untracked), from the Flare prefix-cache-offload design.
- Commits: ebb54d6c (cross-session lookup), a4197150 (checksums, manifest version 3).
- Tested 2026-10-21 with the server stopped: `PYTHONPATH=python pytest -q tests/scheduler` in
  the worktree gave 483 passed, 1 skipped. Not yet run on the live model.
- Waiting for: the owner's decision to update production, with enough host RAM.
- To land it:
  1. Merge into main. Version 2 checkpoints on disk are not adopted after the restart, so
     their sessions prefill once.
  2. Restart the unit (`ft-down`, then `ft-up`).
  3. Watch the log for `from the checkpoint of session` and `restores_corrupt`.
- Added 2026-10-21. Production stays on main as it is until then.
