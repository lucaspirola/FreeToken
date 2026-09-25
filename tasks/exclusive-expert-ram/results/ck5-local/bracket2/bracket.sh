#!/usr/bin/env bash
# 1M p2 decode bracket on round3 ceb482c after the WSL restart: warm-up (whole, 8K only, fills the CUDA
# compute cache and page cache), then whole / mirror / mirror LFU 256 / mirror / whole back to back,
# 8K + 1M two passes each (the checkpoint's whole-1m/mirror-1m arms without the needles post),
# host load + SM clock every 5 s. Arms run under the host lock, each alone on an empty GPU.
set -u
WT=/home/lucas/ai/FreeToken-wt/round3; X=$WT/tasks/exclusive-expert-ram; R=$X/results; B=$R/ck5-local/bracket2
ST=$B/status.txt
( while [ ! -e $B/stop ]; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader)" >> $B/load5s.txt; sleep 5; done ) &
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
one() {  # name rows sizes envs
  flock 9
  # 17 MiB is the owner's Windows ChatGPT.exe (nvidia-smi.exe, C+G); not ours to stop: <= 32 MiB counts as empty.
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
  echo "$1 start $(date +%T) $(cat /proc/loadavg) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $ST
  ( cd $WT && FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00 FT_ROWS=$2 FT_RESERVE=256 FT_SIZES="$3" \
      FT_ENVS="$4" $X/measure.sh $1 > $B/$1-measure.log 2>&1 ) || echo "$1 exit $?" >> $ST
  journalctl --user -u ft-measure-$1 -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > $B/$1-journal.txt || true
  mv $R/$1-* $B/ 2>/dev/null || true
  echo "$1 done $(date +%T)" >> $ST
  flock -u 9; sleep 30
}
one ck5c-warmup 0 "8000" ""
one ck5c-whole-a 0 "8000 1000000" ""
one ck5c-mirror-a -1 "8000 1000000" ""
one ck5c-mirror-l256 -1 "8000 1000000" "FREETOKEN_LFU_HALVE_STEPS=256"
one ck5c-mirror-b -1 "8000 1000000" ""
one ck5c-whole-b 0 "8000 1000000" ""
git -C $WT checkout -q -- tasks/exclusive-expert-ram/results/sweep.tsv 2>/dev/null
touch $B/stop; echo BRACKETDONE >> $ST
