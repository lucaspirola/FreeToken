# Fork shape rules — proposal for `CLAUDE.md`

Written by the S9 worker (reorganisation plan, `2026-09-22-refactor-plan-final.md`,
section 5, step S9). This is a proposal: the owner decides whether and how to fold
it into `CLAUDE.md`. Nothing here is self-enacting and no agent edits `CLAUDE.md`
from this file.

## The rules the plan already names (section 5, S9)

These are quoted from the plan verbatim, as the baseline the owner asked for:

1. **New behaviour in new modules.** A fork feature gets its own file
   (`moe/residency.py`, `engine/growable_kv.py`, ...), not a scattered set of edits
   inside an upstream file's body.
2. **≤ ~20 lines of hooks per upstream file per feature.** If a feature needs more
   than that inside a file upstream owns, the feature is probably not a hook, it is
   a fork of that file's logic — split it out instead.
3. **No import-time env constants.** `FREETOKEN_EXPERT_ARENA`-style module-level
   `os.environ` reads (`offload_cache.py:151`, `offload_kernels.py:32` before S7)
   bake a value in at import time, before `server/args.py` has had a chance to
   parse flags, alias env vars or validate anything. They also make the same
   process behave differently depending on import order.
4. **Env vars resolved in `args.py` only.** Every `FREETOKEN_*` knob becomes a
   config field there (with the env var kept only as a documented alias for
   backward compatibility), never read directly from a deep module.
5. **No tests that parse upstream class bodies.** `ast`-parsing `Engine`'s or
   `Scheduler`'s source to assert a private method exists (`test_growable_kv_
   transaction_source.py` before S7) breaks on every refactor that moves code
   without changing behaviour, and proves nothing about behaviour. Test the
   public seam (the controller class, its return values) instead.
6. **No arm names, dates or "lever N" in library code.** `python/` is not where a
   measurement's identity lives; that belongs in `tasks/exclusive-expert-ram/`
   (env files, result records, journals). A grep for a lever name or a date string
   in `python/freetoken` should always return nothing.
7. **Sync on a fixed cadence.** Before every measurement phase, and at least
   weekly regardless, run `scripts/upstream-sync-check.sh` and file the result.
   Waiting for "a big rewrite is coming, better sync now" is what produced the S0
   conflict load.

## What S0 taught, worth adding

### 8. A clean auto-merge is not a safe merge

The single most expensive fact from S0: `git merge`'s silence is not evidence of
correctness. When the fork's side of a file is byte-identical to the merge-base
(the fork simply hasn't touched that file recently), any upstream rewrite of that
file wins outright with **zero** conflict markers, however much fork-owned code it
deletes (`Fp8PerTensorLinear` — a whole class, `EngineConfig.nvfp4_backend`, and
~8 more, see `sync-check-2026-09-22.txt`'s replay). The same silence happens
*inside* a file that has other, unrelated conflicts: one hunk can be conflict-free
and still delete fork code, while a human's attention is entirely spent on the
hunks that *did* raise a marker.

Rule: **a rewrite of an upstream file is not "clean" because git says so.**
`scripts/upstream-sync-check.sh`'s silent-loss detector is a cheap, textual,
intentionally over-inclusive check for exactly this — every warning it prints
deserves a human look, even the false positives (a symbol renamed in the same
window). It cannot see a loss that keeps its name but loses a branch inside its
body (the `_expert_gemm` nvfp4 dispatch case): that class of loss is only caught
by the GPU checkpoints (plan section 7.2/7.3) and by code review that reads the
whole diff of an upstream file the fork also touches, not just the conflicting
hunks.

### 9. Rule 1 (new modules) is also the fix for rule 8

Every one of the ~10 S0 losses lived inside an upstream-owned file the fork had
patched in place (`layers/moe.py`, `engine/config.py`, `models/config.py`,
`models/qwen3_5_moe/weight.py`, a triton kernel file). None of them lived in a
fork-owned module like `moe/mirror_pool.py` or `moe/residency.py` — those cannot
be silently overwritten because upstream has no file at that path to overwrite.
Rule 1 is not just cleanliness; it is the actual containment for rule 8's risk.
Where rule 1 cannot apply (a genuine behavioural change to upstream logic, not an
addition beside it), that file is now a standing merge-conflict liability and
should be named as such somewhere durable (a comment at the top of the file
pointing at the owning feature, so upstream-sync-check's warning has a place to
point back to).

### 10. The budget is a ratchet, not a thermostat

`upstream-sync-check.sh`'s conflict budget starts at 0 and grows only by a dated,
reasoned comment in the script itself — never by a flag, an environment variable,
or a one-off "skip it this time." The same discipline should extend to the
silent-loss warnings once the detector has a track record: a warning that is
confirmed to be a real rename, not a loss, is worth a short-lived allowlist entry
with the same dated-comment shape, not a change to the detector's sensitivity.

### 11. A rewritten upstream file with a large diff window hides everything

The S0 replay's noisiest single file was `models/qwen3_5_moe/weight.py`
(dozens of warnings): a file that has drifted far from upstream accumulates so
much fork-only surface that any single real loss is buried in noise from
legitimate divergence. Rule 2 (≤ ~20 hook lines per feature) is the preventive
measure; once a file is already this divergent, the practical fix is the one S8
converges on for this exact file — extract the model-specific loading logic
behind a narrow, testable interface (`nvfp4_expert_row_layout`,
`check_experts`), so upstream's edits to the surrounding file have nothing
fork-owned left inside them to silently delete.

## Not proposed here

Whether to make `upstream-sync-check.sh`'s weekly run a CI job, a pre-merge hook,
or a manual habit is the owner's call, not this proposal's; the script works
identically in all three, being read-only and side-effect-free.
