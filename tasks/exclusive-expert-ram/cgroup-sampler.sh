#!/usr/bin/env bash
# Samples the memory of every active ft-measure-* unit, measured per unit instead of by
# ram_gib, which is a host-wide MemAvailable delta and on a busy host also counts other
# processes (coordinator 2026-09-24).
# Every 10 s it appends one row per unit:
#   * the unit cgroup's memory.current and memory.peak;
#   * the cgroup's memory.stat anon, file, unevictable and shmem;
#   * the largest process in the cgroup (the server): VmRSS, RssAnon, RssFile, RssShmem;
#   * MemAvailable;
#   * ready: 1 once the journal shows "API server is ready";
#   * probes_done: lines in the arm's probe.jsonl, so a row can be tied to the request that
#     had just finished.
# The rows for "at ready" and "after request k" are the first rows where ready becomes 1
# and where probes_done becomes k (summarize_cgroup.py).
#   CG_OUT      output TSV (required)
#   CG_RESULTS  space-separated results dirs to look for <arm>-probe.jsonl
# Per process (every PID in the unit's cgroup), into CG_OUT.procs.tsv every 10 s:
#   pid, ppid, threads, Rss and Anonymous from /proc/PID/smaps_rollup, and the process's
#   distinct thread names (they tell the scheduler from the tokenizer/detokenizer).
# At ready and whenever probes_done changes, CG_OUT.maps/<arm>-<point>-<pid>.txt holds the
# process's Anonymous memory summed per mapping name (smaps), which shows where a growth sits.
# Stops when the file CG_OUT.stop exists.
set -u
O="${CG_OUT:?}"; RES="${CG_RESULTS:-}"
[ -s "$O" ] || printf 'ts\tunit\tready\tprobes_done\tcg_current\tcg_peak\tcg_anon\tcg_file\tcg_unevict\tcg_shmem\tpid\tvm_rss\trss_anon\trss_file\trss_shmem\tmemavail\n' > "$O"
declare -A READY LAST
P="$O.procs.tsv"; M="$O.maps"; mkdir -p "$M"
[ -s "$P" ] || printf 'ts\tunit\tprobes_done\tpid\tppid\tthreads\trss\tanon\tthread_names\n' > "$P"
mapsum() {  # Anonymous KiB per mapping name, largest first
  awk '/^[0-9a-f]+-[0-9a-f]+ /{n=($6==""?"[anon-unnamed]":$6)} /^Anonymous:/{a[n]+=$2} /^Rss:/{r[n]+=$2}
       END{for(k in a) if(a[k]>=1024) printf "%10d KiB anon %10d KiB rss  %s\n", a[k], r[k], k}' "/proc/$1/smaps" 2>/dev/null | sort -rn | head -25
  echo "--- largest single mappings by Anonymous"
  awk '/^[0-9a-f]+-[0-9a-f]+ /{h=$1" "$6} /^Anonymous:/{if($2>=4096) printf "%10d KiB  %s\n", $2, h}' "/proc/$1/smaps" 2>/dev/null | sort -rn | head -12
}
while [ ! -e "$O.stop" ]; do
  for U in $(systemctl --user list-units --no-legend --state=active 'ft-measure-*' | awk '{print $1}'); do
    CG=/sys/fs/cgroup$(systemctl --user show -p ControlGroup --value "$U")
    [ -r "$CG/memory.current" ] || continue
    if [ "${READY[$U]:-0}" = 0 ] && journalctl --user -u "$U" -o cat --no-pager 2>/dev/null | grep -q "API server is ready"; then READY[$U]=1; fi
    arm=${U#ft-measure-}; arm=${arm%.service}; n=0
    for d in $RES; do [ -f "$d/$arm-probe.jsonl" ] && n=$(grep -c . "$d/$arm-probe.jsonl"); done
    best=0; bpid=0
    for p in $(cat "$CG/cgroup.procs" 2>/dev/null); do
      r=$(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null); r=${r:-0}
      [ "$r" -gt "$best" ] && { best=$r; bpid=$p; }
    done
    point=""
    if [ "${READY[$U]:-0}" = 1 ] && [ "${LAST[$U]:-x}" != "$n" ]; then point="ready-p$n"; LAST[$U]=$n; fi
    for p in $(cat "$CG/cgroup.procs" 2>/dev/null); do
      [ -r /proc/$p/smaps_rollup ] || continue
      tn=$(cat /proc/$p/task/*/comm 2>/dev/null | sort | uniq -c | sort -rn | head -6 | awk '{printf "%s*%s ", $2, $1}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$arm" "$n" "$p" \
        "$(awk '/^PPid/{print $2}' /proc/$p/status)" "$(awk '/^Threads/{print $2}' /proc/$p/status)" \
        "$(awk '/^Rss:/{print $2*1024}' /proc/$p/smaps_rollup)" "$(awk '/^Anonymous:/{print $2*1024}' /proc/$p/smaps_rollup)" "$tn" >> "$P"
      [ -n "$point" ] && mapsum $p > "$M/$arm-$point-$p.txt"
    done
    st() { awk -v k="$1" '$1==k{print $2}' "$CG/memory.stat"; }
    ps_() { awk -v k="$1:" '$1==k{print $2*1024}' /proc/$bpid/status 2>/dev/null; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$arm" "${READY[$U]:-0}" "$n" \
      "$(cat $CG/memory.current)" "$(cat $CG/memory.peak 2>/dev/null)" "$(st anon)" "$(st file)" "$(st unevictable)" "$(st shmem)" \
      "$bpid" "$(ps_ VmRSS)" "$(ps_ RssAnon)" "$(ps_ RssFile)" "$(ps_ RssShmem)" \
      "$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)" >> "$O"
  done
  sleep 10
done
