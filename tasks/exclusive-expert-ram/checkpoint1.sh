#!/usr/bin/env bash
# GPU checkpoint 1 of the reorganisation (plan 2026-09-22-refactor-plan-final.md, section 7.2),
# run on the post-S0 commit. The OWNER runs this (plan 9.4: the owner starts servers for
# checkpoints). Agents prepare it, read the results, and never start it.
#
#   tasks/exclusive-expert-ram/checkpoint1.sh            # all four arms, in order
#   tasks/exclusive-expert-ram/checkpoint1.sh preflight  # only the checks, starts nothing
#
# Arms, serial, each alone on the GPU (measure.sh refuses otherwise):
#   ck1-whole        whole model in RAM, 8K/32K/80K, then needles 21K/120K + recall
#                    21K/120K/240K: the correctness REFERENCE, re-recorded on this commit
#   ck1-mirror-1m    record config (auto pool, reserve 2E=256), 8K + 1M, same needles/recall:
#                    compared field-for-field with ck1-whole
#   ck1-mirror       record config, 8K/80K/713K: against nemotron-reserve-2e-record.json
#   ck1-whole-close  whole model again, 8K/32K/80K: closes the bracket (C-EMPTY-GPU)
#
# Every arm serves THIS tree's python/ through the server venv (FT_VENV) -- no uv sync --
# and its record carries the commit. Roughly 2.5-3 h in total; the 1M arm alone is ~50 min.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO="$PWD"
HERE="$REPO/tasks/exclusive-expert-ram"
OUT="$HERE/results"
export FT_VENV=/home/lucas/ai/FreeToken/.venv
export FT_RATIO=1.00
POST_TIMEOUT=10800

die() { echo "checkpoint1: $*" >&2; exit 1; }

# The host lock every agent worker takes around torch pytest (flock -w). Held for the whole
# checkpoint, so no test suite can start beside a live arm, and an arm cannot start while a
# suite runs (three concurrent torch suites coincided with the host going down once).
LOCK=/home/lucas/.cache/freetoken/gpu-host.lock
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "waiting for a worker's test run to finish (host lock held) ..."
  flock 9
fi

preflight() {
  [ -z "$(git -C "$REPO" status --porcelain -- python)" ] || die "python/ has uncommitted edits: commit first, a number must be tied to a commit"
  echo "code: $(git -C "$REPO" log --oneline -1)"
  systemctl is-active --quiet freetoken-serve && die "the production unit freetoken-serve is running: ft-down first"
  systemctl --user list-units --state=active --no-legend 'ft-measure-*' | grep -q . && die "an ft-measure unit is still up"
  pgrep -f 'pytest' >/dev/null && die "a pytest is running (a worker?): wait for it"
  if systemctl --user is-active --quiet piro-board-embedder 2>/dev/null; then
    echo "stopping piro-board-embedder for the measurement (it is NOT restarted afterwards)"
    systemctl --user stop piro-board-embedder
    sleep 5
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = "0" ] || { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; die "GPU holds ${used} MiB, must read 0"; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || die "MemAvailable ${avail} GiB < 22"
  echo "preflight ok: GPU 0 MiB, MemAvailable ${avail} GiB, no server, no pytest"
}

journal() { journalctl --user -u "ft-measure-$1" -o cat --no-pager > "$OUT/$1-journal.txt" || true; }

NEEDLES="NEEDLES_THINK_MAX_TOKENS=65536 NEEDLES_OUT=\"$OUT/\$ARM_NAME-needles.json\" python3 $HERE/needles.py 21000 120000; python3 $HERE/recall.py 21000 120000 240000"

arm() {  # name, then env assignments
  local name=$1; shift
  preflight
  echo "=== arm $name ==="
  env "$@" FT_POST_TIMEOUT=$POST_TIMEOUT "$HERE/measure.sh" "$name"
  journal "$name"
}

preflight
[ "${1:-}" = "preflight" ] && exit 0

arm ck1-whole       FT_ROWS=0                                     FT_POST="${NEEDLES//\$ARM_NAME/ck1-whole}"
arm ck1-mirror-1m   FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000 1000000" FT_POST="${NEEDLES//\$ARM_NAME/ck1-mirror-1m}"
arm ck1-mirror      FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000 80000 713000"
arm ck1-whole-close FT_ROWS=0

for a in ck1-whole ck1-mirror-1m ck1-mirror ck1-whole-close; do
  for r in R3 R6; do
    printf '%s %s: ' "$a" "$r"
    FREETOKEN_LOG="$OUT/$a-journal.txt" bash "$REPO/benchmarks/switchyard_soak/checks/acceptance.sh" "$r" \
      > "$OUT/$a-acceptance-$r.txt" 2>&1 && echo PASS || echo "FAIL (see $a-acceptance-$r.txt)"
  done
done
echo "checkpoint 1 arms done. The embedder stays stopped. Tell Claude; it compares the records."
