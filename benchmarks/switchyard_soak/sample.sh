#!/usr/bin/env bash
# Per-5s sampler over every FreeToken-venv python process: total CPU%, the busiest single
# pid and its CPU% (a spinning scheduler shows ~100), that pid's RSS, GPU MiB, and host
# MemAvailable (the 2026-09-05 WSL OOM restart is why the last column exists).
set -uo pipefail
SP="$1"
OUT="$SP/resources.csv"
echo "ts,total_cpu_pct,top_pid,top_cpu_pct,top_rss_mb,gpu_mib,nprocs,mem_avail_gb" > "$OUT"
HZ=$(getconf CLK_TCK)
declare -A PREV
prev_t=0
while true; do
  now=$(date +%s.%N)
  total=0; top_pid=""; top_cpu=0; nprocs=0
  for p in $(pgrep -f '^/home/lucas/ai/FreeToken/\.venv/bin/python' 2>/dev/null); do
    [ -r /proc/$p/stat ] || continue
    nprocs=$((nprocs+1))
    j=$(awk '{ s=$0; sub(/^[0-9]+ \(.*\) /, "", s); split(s, f, " "); print f[12]+f[13] }' /proc/$p/stat 2>/dev/null)
    [ -z "$j" ] && continue
    pj=${PREV[$p]:-}
    if [ -n "$pj" ] && [ "$prev_t" != 0 ]; then
      c=$(awk -v dj=$((j-pj)) -v dt="$now" -v pt="$prev_t" -v hz="$HZ" 'BEGIN{d=dt-pt; if(d>0) printf "%.1f", 100*dj/hz/d; else print 0}')
      total=$(awk -v a="$total" -v b="$c" 'BEGIN{printf "%.1f", a+b}')
      if awk -v a="$c" -v b="$top_cpu" 'BEGIN{exit !(a>b)}'; then top_cpu=$c; top_pid=$p; fi
    fi
    PREV[$p]=$j
  done
  prev_t=$now
  rss=""
  [ -n "$top_pid" ] && rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$top_pid/status 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  ma=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  echo "$(date -Is),$total,${top_pid:-},${top_cpu},${rss:-},${gpu:-},$nprocs,$ma" >> "$OUT"
  sleep 5
done
