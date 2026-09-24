#!/usr/bin/env bash
# GPU checkpoints of the reorganisation (plan 2026-09-22-refactor-plan-final.md, section 7.2).
# Run by the orchestrating agent as a systemd transient unit (owner, checkpoint 1: "You run
# it"), never from an agent shell.
#
#   tasks/exclusive-expert-ram/checkpoint1.sh [ck1|ck2]            # all four arms, in order
#   tasks/exclusive-expert-ram/checkpoint1.sh [ck1|ck2] preflight  # only the checks
#
# Arms (named <label>-*), serial, each alone on the GPU (measure.sh refuses otherwise):
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
CK="${1:-ck1}"
case "$CK" in ck[0-9]) shift || true ;; *) CK=ck1 ;; esac

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
  # A running python -m pytest (not a flock merely WAITING for the lock we hold, whose
  # command line also contains 'pytest' -- matching that would abort on a queued worker).
  pgrep -f '^[^ ]*python[0-9.]* -m pytest' >/dev/null && die "a pytest is running outside the host lock: find it"
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
echo "checkpoint label: $CK"

# ARMS picks and orders arms (default: the original four). whole-1m is the whole model at
# 8K + 1M: the same-commit 1M reference compare_box.py judges mirror-1m against.
for a in ${ARMS:-whole mirror-1m mirror whole-close}; do
  case "$a" in
    whole)       arm $CK-whole       FT_ROWS=0 FT_POST="${NEEDLES//\$ARM_NAME/$CK-whole}" ;;
    whole-1m)    arm $CK-whole-1m    FT_ROWS=0 FT_SIZES="8000 1000000" ;;
    mirror-1m)   arm $CK-mirror-1m   FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000 1000000" FT_POST="${NEEDLES//\$ARM_NAME/$CK-mirror-1m}" ;;
    mirror)      arm $CK-mirror      FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000 80000 713000" ;;
    whole-close) arm $CK-whole-close FT_ROWS=0 ;;
    *) die "unknown arm $a" ;;
  esac
done

# R3 from acceptance.sh. Its R6 checks the production system unit freetoken-serve (down by
# design during measurements) and read /proc/0/limits on checkpoint 1; for an arm only its
# journal half applies, plus the memlock limit the user manager gives transient units.
r6_arm() {
  local j="$1" start
  start=$(grep -n "ServerArgs(model_path" "$j" | tail -1 | cut -d: -f1)
  ! tail -n +"$start" "$j" | grep -qi "settled pageable" &&
    ! tail -n +"$start" "$j" | grep -qi "mlock.*fail" &&
    [ "$(systemctl --user show -p DefaultLimitMEMLOCK --value)" = infinity ] &&
    echo "R6(arm) ok: no pageable fallback, no mlock failure, user DefaultLimitMEMLOCK=infinity"
}
for a in $CK-whole $CK-whole-1m $CK-mirror-1m $CK-mirror $CK-whole-close; do
  [ -f "$OUT/$a-journal.txt" ] || continue
  printf '%s R3: ' "$a"
  FREETOKEN_LOG="$OUT/$a-journal.txt" bash "$REPO/benchmarks/switchyard_soak/checks/acceptance.sh" R3 \
    > "$OUT/$a-acceptance-R3.txt" 2>&1 && echo PASS || echo "FAIL (see $a-acceptance-R3.txt)"
  printf '%s R6(arm): ' "$a"
  r6_arm "$OUT/$a-journal.txt" > "$OUT/$a-acceptance-R6.txt" 2>&1 && echo PASS || echo "FAIL (see $a-acceptance-R6.txt)"
done
# The gate twice: needles/recall vs the same-commit whole arm, decode pass by pass vs the
# same-commit whole arms (compare_box.py, the owner's gate) and vs the record
# (compare_records.py).
python3 "$HERE/compare_needles.py" "$OUT" $CK-whole $CK-mirror-1m > "$OUT/$CK-needles-compare.txt" 2>&1 || true
python3 "$HERE/compare_box.py" "$OUT" $CK > "$OUT/$CK-box-compare.txt" 2>&1 || true
python3 "$HERE/compare_records.py" "$OUT" $CK > "$OUT/$CK-records-compare.txt" 2>&1 || true
cat "$OUT/$CK-needles-compare.txt" "$OUT/$CK-box-compare.txt" "$OUT/$CK-records-compare.txt"
echo "checkpoint $CK arms done. The embedder stays stopped."
