#!/usr/bin/env bash
# Two gate re-checks before the dynamic-transient merge (coordinator, 2026-09-24):
#   1. 8K decode, pool vs whole, with a window long enough to measure: whole and mirror
#      alternated three times each, 8K prompt, 512 decode tokens, passes 1 and 2. The gate
#      is the median of each pair >= 91% pass by pass (compare_recheck.py). The probe
#      records the first inter-chunk gap (gap1_ms), so a stall right after the first token
#      (the dynamic-headroom release) shows as its own number, in or out of the window.
#   2. The 1M RAM point: mirror-1m at 8K + 1M once, ram_gib measured the same way as the
#      12.26 GiB reference (MemAvailable before start minus at ready). Before it, the live
#      host processes and MemAvailable are recorded (<label>-host-before.txt). Nothing on
#      the host is stopped.
#   RC_LABEL  arm prefix (default rc); RC_OUT results dir (default results/recheck-local)
#   RC_PARTS  "8k ram" (default both)
#   RC_SIZE   prompt size(s) of the alternated part (default 8000; "8000 80000" runs both per arm)
#   RC_RESERVE mirror reserve rows (default 256 = Nemotron's 2E; set empty for the pool's own
#             default, 3E, e.g. on a 256-expert model where 256 is below the 2E minimum)
# Waits for MemAvailable >= 23 GiB and 0 MiB on the GPU before each arm [agent practice].
# Run as a systemd transient user unit (--setenv=PATH), never from an agent shell.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
L="${RC_LABEL:-rc}"
OUT="${RC_OUT:-$HERE/results/recheck-local}"
mkdir -p "$OUT"
quiet() {
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 60; done
  echo "host quiet at $(date -Is): $(grep MemAvailable /proc/meminfo)"
}
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
one() {  # name, then env assignments for measure.sh
  local name=$1; shift
  quiet
  flock 9
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$used" -gt "${GPU_IDLE_MIB:-0}" ]; then echo "GPU holds $used MiB, skipping $name"; flock -u 9; return; fi
  env "$@" "$HERE/measure.sh" "$name" || echo "$name measure exit $?"
  journalctl --user -u "ft-measure-$name" -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$name-journal.txt" || true
  mv "$HERE"/results/$name-* "$OUT/" 2>/dev/null || true
  flock -u 9
  sleep 30
}
for part in ${RC_PARTS:-8k ram}; do
  case "$part" in
    8k)
      for i in 1 2 3; do
        one "$L-whole-$i"  FT_ROWS=0 FT_SIZES="${RC_SIZE:-8000}" FT_GEN=512
        one "$L-mirror-$i" FT_ROWS=-1 FT_RESERVE="${RC_RESERVE-256}" FT_SIZES="${RC_SIZE:-8000}" FT_GEN=512
      done ;;
    ram)
      {
        echo "date $(date -Is)"
        grep -E "MemTotal|MemFree|MemAvailable|^Cached|Shmem:" /proc/meminfo
        echo "--- processes by RSS (top 25)"
        ps -eo pid,rss,etime,comm --sort=-rss | head -26
        echo "--- user units active"
        systemctl --user list-units --state=active --no-legend --type=service | cut -c1-100
      } > "$OUT/$L-host-before.txt"
      one "$L-mirror-1m" FT_ROWS=-1 FT_RESERVE="${RC_RESERVE-256}" FT_SIZES="8000 1000000" ;;
  esac
done
python3 "$HERE/compare_recheck.py" "$OUT" "$L" "${RC_SIZE:-8000}" > "$OUT/$L-compare.txt" 2>&1 || true
cat "$OUT/$L-compare.txt"
echo "recheck done"
