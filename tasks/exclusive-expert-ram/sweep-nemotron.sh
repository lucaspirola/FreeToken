#!/usr/bin/env bash
# Host RAM x decode curve for Nemotron-3.5-Lightning-30B-A3B-NVFP4.
#
# Capacity points, in mirror rows (5.36 MiB each, 2944 rows = the whole model):
#   ~1700  the bottom stop: 1277 complement + 384 reserve, 39 duplicates left
#    2100  near what auto-sizing picks
#    2500  half the spare budget spent on duplicates
#    2944  the whole model mirrored -- the most RAM this design can hold
# The baseline (rows=0) pins the whole model AND keeps no pool, so it is not
# the same as the 2944 arm. The server logs the geometry of each arm ("Mirror
# pool: N rows pinned ... up to D duplicates"); the geometry file records it.
#
# One arm at a time, each alone on the host. The RAM column is the MemAvailable
# delta at readiness (see measure.sh), never memory.current and never anon.
# Nothing else may touch the GPU while an arm is live -- a microbenchmark run
# beside an arm contaminated a baseline and killed the 1700 arm outright.
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."

run() {
  local arm="$1" rows="$2"
  echo "=============== $arm (rows=$rows) $(date +%H:%M:%S)"
  for _ in $(seq 10); do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 22 ] && break
    sleep 20
  done
  FT_ROWS="$rows" timeout 1800 tasks/exclusive-expert-ram/measure.sh "$arm" 2>&1 | tail -40
  systemctl --user reset-failed "ft-measure-$arm" 2>/dev/null
  sleep 15
}

# Two baselines, first and last: the sweep takes ~15 minutes and a baseline
# that only appears at one end cannot show whether the host drifted under it.
for arg in "$@"; do
  case "$arg" in
    baseline*) run "nemotron-$arg" 0 ;;
    *)         run "nemotron-$arg" "$arg" ;;
  esac
done
echo "=============== nemotron sweep done $(date +%H:%M:%S)"
