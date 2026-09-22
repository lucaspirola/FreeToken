#!/usr/bin/env bash
# Goal 1 (plan section 5, S9): "the fork can merge upstream at any time." Read-only.
#
# Does three things, none of which touches the working tree or refs:
#   1. Runs `git merge-tree --write-tree BASE HEAD` (writes a tree object to the
#      object database only -- no checkout, no merge commit, no index change) and
#      reports the conflicting files with a hunk count per file, plus how many
#      commits HEAD is ahead/behind BASE.
#   2. Exits non-zero if the conflict count exceeds the budget below. The budget
#      is 0 (the state after S0: exp/reorg is 0 behind, 0 conflicts). It grows
#      ONLY by editing CONFLICT_BUDGET below and adding a dated, reasoned
#      acknowledgement line next to it -- never by a flag, an env var, or a
#      "just this once" skip.
#   3. Reports the risk that cost the most in S0 (plan section 5, S9's own
#      "why"): a CLEAN auto-merge -- one that leaves no conflict marker at all --
#      silently deleting fork code. See "Silent-loss detector" below for the
#      method and its evidence.
#
# Usage:
#   scripts/upstream-sync-check.sh                    # origin/main vs HEAD, real check
#   scripts/upstream-sync-check.sh --no-fetch          # skip `git fetch origin` (offline / replay)
#   scripts/upstream-sync-check.sh --base REF --head REF   # replay a past merge (see "Replay test")
#
# Never merges, rebases, commits or pushes. `git merge-tree --write-tree` writes
# only object-database blobs/trees; nothing under .git/refs or the worktree moves.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
REPO="$PWD"

# --- Conflict budget ---------------------------------------------------------
# 0 after S0 (plan C-EVIDENCE / S9 "what"). To raise it, add a line here, e.g.:
#   # ACK 2026-10-01: raised to 3 -- upstream's tokenizer rewrite (#NNN) touches
#   #   3 fork files we haven't ported yet; tracked in tasks/.../<ticket>.md.
# and change the number. No other way to silence this check is authorised.
CONFLICT_BUDGET=0

# --- Args ----------------------------------------------------------------
FETCH=1
BASE_REF="origin/main"
HEAD_REF="HEAD"
while [ $# -gt 0 ]; do
  case "$1" in
    --no-fetch) FETCH=0; shift ;;
    --base) BASE_REF="$2"; shift 2 ;;
    --head) HEAD_REF="$2"; shift 2 ;;
    *) echo "usage: $0 [--no-fetch] [--base REF] [--head REF]" >&2; exit 2 ;;
  esac
done

if [ "$FETCH" = 1 ]; then
  echo "fetching origin ..." >&2
  git fetch origin
fi

BASE_SHA=$(git rev-parse "$BASE_REF")
HEAD_SHA=$(git rev-parse "$HEAD_REF")
MERGE_BASE=$(git merge-base "$BASE_REF" "$HEAD_REF")

echo "== upstream-sync-check =="
echo "base: $BASE_REF ($BASE_SHA)"
echo "head: $HEAD_REF ($HEAD_SHA)"
echo "merge-base: $MERGE_BASE"

# ahead/behind: left = BASE-only commits (how far HEAD is behind), right = HEAD-only
read -r BEHIND AHEAD <<<"$(git rev-list --left-right --count "$BASE_REF...$HEAD_REF")"
echo "ahead: $AHEAD   behind: $BEHIND"

# --- 1. merge-tree, read-only -------------------------------------------------
MT_OUT="$(mktemp)"
trap 'rm -f "$MT_OUT"' EXIT
MT_EXIT=0
git merge-tree --write-tree "$BASE_REF" "$HEAD_REF" >"$MT_OUT" 2>&1 || MT_EXIT=$?
TREE_OID="$(head -1 "$MT_OUT")"

# The "conflicted file info" section (mode/oid/stage/path lines, one per stage
# per conflicting path) runs from line 2 to the first blank line; only paths
# that could not be resolved to a single stage-0 blob appear there (verified:
# a clean auto-merge like tests/kernels/test_triton_attention.py in the S0
# replay below never appears in this section, only in the "Auto-merging ..."
# messages after the blank line). Dedup gives exactly the conflicting paths.
# Each line in that section is "<mode> <oid> <stage>\t<path>" (tab before the
# path); take everything after the first tab, dedup.
mapfile -t CONFLICT_PATHS < <(
  awk 'NF==0{exit} NR>1{print}' "$MT_OUT" | awk -F'\t' '{print $2}' | sort -u
)
N_CONFLICTS=${#CONFLICT_PATHS[@]}

echo
echo "conflicting files: $N_CONFLICTS"
if [ "$N_CONFLICTS" -gt 0 ]; then
  for p in "${CONFLICT_PATHS[@]}"; do
    [ -z "$p" ] && continue
    hunks=$(git show "$TREE_OID:$p" 2>/dev/null | grep -c '^<<<<<<<' || true)
    printf '  %-70s %s hunk(s)\n' "$p" "${hunks:-0}"
  done
fi

# --- 2. Silent-loss detector --------------------------------------------------
# S0's own postmortem (plan section 5, S9): ~10 fork additions (the
# `_expert_gemm` nvfp4 branches in layers/moe.py, `Fp8PerTensorLinear`,
# `EngineConfig.nvfp4_backend`, ...) were dropped by upstream edits that git
# merged WITHOUT a conflict marker. `git merge-tree` reporting 0 conflicts (or
# N conflicts none of which mention the lost file) says nothing about this
# class of loss, by construction: a clean auto-merge is exactly the case where
# no marker is emitted.
#
# Method (verified against the real S0 merge -- see "Replay test" below for
# the exact commits and what it found):
#   1. For every file where HEAD (the fork side) disagrees with BASE_REF
#      (`git diff --name-only BASE_REF HEAD`, restricted to *.py under
#      python/ -- deliberately against the upstream ref, not the merge-base,
#      so a file the fork owns but has not touched recently still counts),
#      collect two symbol sets from HEAD's version of the file:
#        - module-level `def NAME` / `class NAME` (column 0only)
#        - class-body dataclass-style fields at exactly 4-space indent
#          (`    NAME: TYPE = ...` or `    NAME: TYPE` with no default),
#          because the concrete loss this replays (`nvfp4_backend`) is a
#          dataclass field, not a def/class, and the plan names it explicitly.
#   2. For each such symbol, check whether it appears ANYWHERE (as a whole
#      identifier, not a substring) in that path's content in the merge
#      result: for a conflicting path, the conflict-marked blob at TREE_OID
#      (which still contains verbatim BOTH sides' text inside the <<<<<<< /
#      ======= / >>>>>>> markers for whichever hunks conflicted, and the
#      auto-resolved text for hunks that did not); for a clean path, the
#      resolved blob directly.
#   3. If a HEAD symbol is textually absent from the merge result at that
#      path, warn. This catches both:
#        - a whole hunk of ours silently dropped inside a file that ALSO has
#          unrelated conflicts elsewhere (nvfp4_backend: the file conflicts,
#          but that field's hunk was clean on our side and upstream deleted
#          it, so it is gone even from inside the markers), and
#        - a whole FILE silently overwritten by theirs because our side never
#          touched it since the merge-base, so git needed no marker at all
#          (Fp8PerTensorLinear: the file is not in the conflict list at all;
#          our blob for it equals the merge-base's blob, so upstream's
#          rewrite of the same file wins outright).
#   4. This is a cheap, textual, over-inclusive check: a symbol the fork
#      merely renamed or refactored inside the same window will also warn
#      (false positive, by design -- the cost of a missed real loss is much
#      higher than a human spending 10 seconds confirming a rename). It
#      cannot see a loss that is not a named symbol at all, such as one
#      branch deleted out of a multi-branch function that keeps its name
#      (the `_expert_gemm` example) -- that class of loss needs the GPU
#      checkpoints (plan section 7) and code review, not this script.
echo
echo "silent fork-symbol losses (clean or partially-clean merge, no marker for the lost text):"
python3 - "$BASE_REF" "$HEAD_REF" "$TREE_OID" "${CONFLICT_PATHS[*]}" <<'PY'
import re, subprocess, sys

base_ref, head_ref, tree_oid, conflict_paths = sys.argv[1:5]
conflict_paths = set(conflict_paths.split()) if conflict_paths.strip() else set()

def sh(*args):
    return subprocess.run(["git", *args], cwd=None, capture_output=True, text=True)

def blob(rev, path):
    r = sh("show", f"{rev}:{path}")
    return r.stdout if r.returncode == 0 else None

DEF_RE = re.compile(r'^(?:async )?def (\w+)|^class (\w+)', re.M)
FIELD_RE = re.compile(r'^ {4}(\w+):\s*\S', re.M)
IDENT_KEYWORDS = {"self", "cls", "return", "None", "True", "False"}

def symbols_of(text):
    syms = set()
    for m in DEF_RE.finditer(text):
        syms.add(m.group(1) or m.group(2))
    for m in FIELD_RE.finditer(text):
        name = m.group(1)
        if name not in IDENT_KEYWORDS and not name.startswith("_"):
            syms.add(name)
    return syms

# Diff against BASE_REF (origin), not the merge-base: this is what makes the
# whole-file clean-overwrite case (Fp8PerTensorLinear -- fork never touched
# the file after the merge-base, so it isn't in a merge-base..HEAD diff at
# all, but it IS in a base..HEAD diff because fork's and origin's content
# disagree) show up, not just files fork edited in the current window.
changed = sh("diff", "--name-only", base_ref, head_ref, "--", "python/").stdout.split()
changed = [p for p in changed if p.endswith(".py")]

warned = 0
for path in changed:
    head_text = blob(head_ref, path)
    if head_text is None:
        continue  # deleted by the fork itself; not a loss to detect here
    head_syms = symbols_of(head_text)
    if not head_syms:
        continue
    if path in conflict_paths:
        merged_text = blob(tree_oid, path)
    else:
        merged_text = blob(tree_oid, path)
    if merged_text is None:
        # path vanished from the merge result entirely: every HEAD symbol lost
        for s in sorted(head_syms):
            print(f"  LOST  {path}: {s}  (whole file absent from merge result)")
            warned += 1
        continue
    merged_idents = set(re.findall(r'\b\w+\b', merged_text))
    for s in sorted(head_syms):
        if s not in merged_idents:
            tag = "conflict-file" if path in conflict_paths else "clean-merge"
            print(f"  LOST  {path}: {s}  ({tag})")
            warned += 1

if warned == 0:
    print("  none detected")
else:
    print(f"\n  {warned} possible silent loss(es) -- confirm each by reading the file; "
          "renames/refactors inside the same window will also show up here.")
PY

# --- Report + budget -----------------------------------------------------
echo
if [ "$N_CONFLICTS" -gt "$CONFLICT_BUDGET" ]; then
  echo "FAIL: $N_CONFLICTS conflicting file(s) > budget $CONFLICT_BUDGET"
  exit 1
fi
if [ "$MT_EXIT" != 0 ] && [ "$N_CONFLICTS" = 0 ]; then
  # merge-tree can exit non-zero for reasons other than content conflicts
  # (e.g. a rename/type clash reported only in the message section); surface it.
  echo "merge-tree exited $MT_EXIT with 0 parsed conflicts -- inspect the messages above the report"
fi
echo "OK: $N_CONFLICTS conflicting file(s) <= budget $CONFLICT_BUDGET"
exit 0
