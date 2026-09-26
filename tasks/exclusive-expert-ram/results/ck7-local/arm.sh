#!/usr/bin/env bash
# One Ornith EXL3 262K arm on :1920 (run it inside a systemd --user unit, never from an agent
# shell), under the GPU host lock, with the ck6o settings (q8_0 KV, ratio 1.00, pin budget 20):
#   arm.sh WORKTREE NAME ROWS "SIZES" OUTDIR [measure.sh env words...]
# ROWS 0 = whole model, -1 = saver (pool default reserve). Leaves NAME-* in OUTDIR, plus the
# journal and acceptance R3; appends the start line (load, GPU, MemAvailable, code) to
# OUTDIR/status.txt.
set -u
wt=$1 name=$2 rows=$3 sizes=$4 out=$5; shift 5
export GPU_IDLE_MIB=32 FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL=$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq FT_NAME=ornith
export FT_EXTRA="--text-model-only --num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith --reasoning-parser qwen3 ${FT_EXTRA_MORE:-}"
export FT_ENVS="FREETOKEN_PIN_BUDGET_GB=20 ${FT_ENVS_EXTRA:-}"
export PROBE_STATS=1
mkdir -p "$out"
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
until [ "$(gpu)" -le $GPU_IDLE_MIB ]; do sleep 5; done
until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
echo "$name start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(gpu)MiB avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader) code=$(git -C $wt log --oneline -1 | cut -c1-8)$(git -C $wt diff --quiet HEAD -- python || echo +dirty)" >> $out/status.txt
( cd $wt && env FT_ROWS=$rows FT_SIZES="$sizes" "$@" $wt/tasks/exclusive-expert-ram/measure.sh $name > $out/$name-measure.log 2>&1 ) || echo "$name measure exit $?" >> $out/status.txt
journalctl --user -u ft-measure-$name -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > $out/$name-journal.txt || true
mv $wt/tasks/exclusive-expert-ram/results/$name-* $out/ 2>/dev/null || true
git -C $wt checkout -q -- tasks/exclusive-expert-ram/results/sweep.tsv 2>/dev/null
FREETOKEN_LOG=$out/$name-journal.txt bash $wt/benchmarks/switchyard_soak/checks/acceptance.sh R3 > $out/$name-acceptance-R3.txt 2>&1
systemctl --user reset-failed ft-measure-$name 2>/dev/null
echo "$name done $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg): $(tail -1 $out/$name-acceptance-R3.txt)" >> $out/status.txt
