#!/usr/bin/env bash
# RAM-gate re-run: exp/reorg 31b8efb, checkpoint1.sh ARMS=mirror-1m (label ck4g), cgroup sampler,
# host load, page-cache (mincore) snapshots before / ready / after 1M p1 / end.
set -u
SP=/tmp/claude-1000/-home-lucas-ai-FreeToken/abf74d75-4e36-4cde-a055-495e4c1687c8/scratchpad/ram
WT=/home/lucas/ai/FreeToken-wt/reorg; X=$WT/tasks/exclusive-expert-ram; R=$X/results; L=ck4g; C=$R/$L-local
U=ft-measure-$L-mirror-1m.service
mkdir -p $C; H=$C/hostload.txt; DIRS=$(cat $SP/dirs.txt)
snap() {  # host state
  { echo "--- $(date -Is) $1"; uptime; grep -E "MemTotal|MemFree|MemAvailable|^Cached|Dirty|Shmem:" /proc/meminfo
    nvidia-smi --query-gpu=memory.used,clocks.sm,utilization.gpu,temperature.gpu --format=csv,noheader
    echo "top CPU:"; ps -eo pcpu,pid,comm --sort=-pcpu | sed -n 2,8p
    echo "top RSS (KiB):"; ps -eo rss,pid,comm --sort=-rss | sed -n 2,9p; } >> $H
}
quiet() { awk -v a="$(awk '/MemAvailable/{print $2/1048576}' /proc/meminfo)" '{exit !($1 < 3.0 && a >= 23)}' /proc/loadavg; }
echo "wait-for-quiet start $(date -Is)" >> $H
for i in $(seq 0 20); do
  snap "poll $i"
  if quiet; then echo "QUIET at poll $i" >> $H; break; fi
  [ $i = 20 ] && { echo "NEVER QUIET in 60 min: running anyway" >> $H; break; }
  sleep 180
done
snap "before start"
python3 $SP/pcache.py $C/pcache-before.tsv $DIRS 2>> $H
rm -f $C/cg.tsv.stop
systemctl --user reset-failed ft-cgsample-g ft-ck4g 2>/dev/null
systemd-run --user --unit=ft-cgsample-g --setenv=PATH="$PATH" --setenv=CG_OUT=$C/cg.tsv "--setenv=CG_RESULTS=$R $C" $X/cgroup-sampler.sh
systemd-run --user --unit=ft-ck4g --setenv=PATH="$PATH" --setenv=ARMS=mirror-1m /bin/bash -c "$X/checkpoint1.sh $L > $C/checkpoint.log 2>&1"
# ready -> snapshot; probes_done 2 (1M p1) -> snapshot; hostload every 60 s
got_r=0; got_p=0; t=0
while systemctl --user is-active --quiet ft-ck4g; do
  if [ $got_r = 0 ] && journalctl --user -u $U -o cat --no-pager 2>/dev/null | grep -q "API server is ready"; then
    got_r=1; snap "ready"; python3 $SP/pcache.py $C/pcache-ready.tsv $DIRS 2>> $H; fi
  n=$(wc -l < $R/$L-mirror-1m-probe.jsonl 2>/dev/null || echo 0)
  if [ $got_p = 0 ] && [ "${n:-0}" -ge 2 ]; then got_p=1; snap "after 1M p1"; python3 $SP/pcache.py $C/pcache-p1.tsv $DIRS 2>> $H; fi
  t=$((t+10)); [ $((t % 60)) = 0 ] && snap "tick"
  sleep 10
done
snap "end"; python3 $SP/pcache.py $C/pcache-end.tsv $DIRS 2>> $H
touch $C/cg.tsv.stop
echo "RUNDONE $(date -Is)" >> $H
