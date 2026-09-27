#!/usr/bin/env bash
# kfix gate: a copy of tasks/exclusive-expert-ram/results/ck7-local/chain-r5.sh (the ck8o setup) on this tree,
# label ck9k, results in ./gate, then the 8-dir suite with the model unloaded (./suite, as
# _orch/r5/local-suite/run.sh). Waits for the e2e A/B chain (unit kfix-e2e) first.
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
while systemctl --user is-active --quiet kfix-e2e || systemctl --user is-active --quiet kfix-e2e2; do sleep 60; done
F=$(dirname "$(readlink -f "$0")")
set -u
WT=$(cd "$F/../../.." && pwd); X=$WT/tasks/exclusive-expert-ram; R=$X/results; C=${CK_OUT:-$F/gate}
L=${CK_LABEL:-ck9k}
NG=/home/lucas/ai/FreeToken-wt/harvest/tasks/harvest/belady/natural_gen.py
H=$C/hostload.txt; ST=$C/chain-status.txt
export GPU_IDLE_MIB=32 FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL=$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq FT_NAME=ornith
export FT_EXTRA="--text-model-only --num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith --reasoning-parser qwen3"
export FT_ENVS="FREETOKEN_PIN_BUDGET_GB=20"
mkdir -p $C; rm -f $C/cg.tsv.stop
st() { echo "$* $(date -Is)" >> $ST; }
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
snap() {
  { echo "--- $(date -Is) $1"; uptime; grep -E "MemAvailable|^Cached" /proc/meminfo
    nvidia-smi --query-gpu=memory.used,clocks.sm,utilization.gpu,temperature.gpu --format=csv,noheader
    echo "top CPU:"; ps -eo pcpu,pid,comm --sort=-pcpu | sed -n 2,7p
    echo "top RSS (KiB):"; ps -eo rss,pid,comm --sort=-rss | sed -n 2,7p; } >> $H
}
quiet() { awk -v a="$(awk '/MemAvailable/{print $2/1048576}' /proc/meminfo)" '{exit !($1 < 3.0 && a >= 23)}' /proc/loadavg; }
waitu() { sleep 20; while systemctl --user is-active --quiet "$1"; do sleep 30; done; sleep 20; }
( while [ ! -e $C/cg.tsv.stop ]; do snap tick; sleep 60; done ) &
( while [ ! -e $C/cg.tsv.stop ]; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader)" >> $C/load5s.txt; sleep 5; done ) &
systemctl --user reset-failed ft-cgsample-kf ft-kf-8k 2>/dev/null
systemd-run --user --unit=ft-cgsample-kf --setenv=PATH="$PATH" --setenv=CG_OUT=$C/cg.tsv "--setenv=CG_RESULTS=$R $C" $X/cgroup-sampler.sh
st "chain start $(git -C $WT log --oneline -1 | cut -c1-60)"

NEEDLES() { echo "NEEDLES_THINK_MAX_TOKENS=65536 NEEDLES_OUT=$C/$1-needles.json python3 $X/needles.py 21000 120000; python3 $X/recall.py 21000 120000 240000"; }

exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
one() {  # worktree name rows sizes outdir [measure env words...]
  local wt=$1 name=$2 rows=$3 sizes=$4 out=$5; shift 5
  flock 9
  until [ "$(gpu)" -le $GPU_IDLE_MIB ]; do sleep 5; done
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
  snap "$name start"
  echo "$name start $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) gpu=$(gpu) MiB avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) code=$(git -C $wt log --oneline -1 | cut -c1-8)" >> $ST
  ( cd $wt && env FT_ROWS=$rows FT_SIZES="$sizes" "$@" \
      $wt/tasks/exclusive-expert-ram/measure.sh $name > $out/$name-measure.log 2>&1 ) || st "$name measure exit $?"
  journalctl --user -u ft-measure-$name -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > $out/$name-journal.txt || true
  mv $wt/tasks/exclusive-expert-ram/results/$name-* $out/ 2>/dev/null || true
  git -C $wt checkout -q -- tasks/exclusive-expert-ram/results/sweep.tsv 2>/dev/null
  FREETOKEN_LOG=$out/$name-journal.txt bash $WT/benchmarks/switchyard_soak/checks/acceptance.sh R3 > $out/$name-acceptance-R3.txt 2>&1
  flock -u 9
  st "$name done: $(tail -1 $out/$name-acceptance-R3.txt)"; sleep 30
}

# 0. warm-up; stop if the whole model does not come up
one $WT $L-warm-whole 0 "8000" $C
if [ ! -s $C/$L-warm-whole-probe.jsonl ]; then st "WHOLE WARM-UP FAILED: see $L-warm-whole-journal.txt"; touch $C/cg.tsv.stop; exit 1; fi
one $WT $L-warm-mirror -1 "8000" $C
[ -s $C/$L-warm-mirror-probe.jsonl ] || { st "saver warm-up failed, once more (the transient record is written on a start)"; one $WT $L-warm-mirror2 -1 "8000" $C; }

# 1. 8K/80K whole vs saver alternated x3
systemd-run --user --unit=ft-kf-8k --setenv=PATH="$PATH" --setenv=GPU_IDLE_MIB=32 --setenv=RC_OUT=$C --setenv=RC_LABEL=$L \
  --setenv=RC_PARTS=8k "--setenv=RC_SIZE=8000 80000" --setenv=RC_RESERVE= --setenv=FT_MODEL=$FT_MODEL --setenv=FT_NAME=ornith \
  "--setenv=FT_EXTRA=$FT_EXTRA" "--setenv=FT_ENVS=$FT_ENVS" /bin/bash -c "$X/recheck-local.sh > $C/recheck-8k.log 2>&1"
waitu ft-kf-8k; st "8k/80k done"

# 2. 256K whole, the same-commit needles/recall reference
one $WT $L-whole-256k 0 "8000 256000" $C FT_POST_TIMEOUT=10800 "FT_POST=$(NEEDLES $L-whole-256k)"

# 3. quiet host for the RAM reading, then 256K saver
for i in $(seq 0 20); do
  snap "quiet poll $i"
  if quiet; then st "QUIET at poll $i"; break; fi
  [ $i = 20 ] && { st "NEVER QUIET in 60 min: running anyway"; break; }
  sleep 180
done
one $WT $L-mirror-256k -1 "8000 256000" $C FT_POST_TIMEOUT=10800 "FT_POST=$(NEEDLES $L-mirror-256k)"
python3 $X/compare_needles.py $C $L-whole-256k $L-mirror-256k > $C/$L-needles-compare.txt 2>&1
python3 $R/ck6-local/ck6cmp.py $C $L-whole-256k $L-mirror-256k > $C/$L-256k-compare.txt 2>&1
st "256k compared: $(tail -1 $C/$L-needles-compare.txt); $(tail -1 $C/$L-256k-compare.txt)"

# 4. natural text, whole / saver alternated x2
for k in 1 2; do
  for a in whole:0 mirror:-1; do
    n=$L-nat-${a%%:*}-$k
    one $WT $n ${a##*:} 8000 $C FT_GEN=128 FT_POST_TIMEOUT=5400 \
      "FT_POST=python3 $NG --doc $WT/docs/nemotron.md --doc $WT/docs/cli.md --max-tokens 3500 --out $C/$n-natural.json --texts $C/$n"
  done
done

# 5. /clear replay (same captures and replay.py as ck6o; compared with ck6-local/ornith/ck6o-replay.json)
one $WT $L-replay 0 8000 $C FT_POST_TIMEOUT=1200 "FT_POST=REPLAY_OUT=$C/$L-replay.json python3 $WT/tasks/prefix-reuse/replay.py"

touch $C/cg.tsv.stop
st "GATEDONE"

# 6. suite, 8 dirs, model unloaded (after the last arm released the GPU)
O=$F/suite; mkdir -p $O; flock 9
until [ "$(gpu)" -le $GPU_IDLE_MIB ]; do sleep 5; done
echo "suite start $(date -Is) gpu=$(gpu) $(uptime)" >> $O/status
echo "tree $(git -C $WT log --oneline -1 | cut -c1-120)" >> $O/status
rm -rf $O/ext
( cd $WT && PYTHONPATH=$WT/python TVM_FFI_CUDA_ARCH_LIST=12.0 TORCH_EXTENSIONS_DIR=$O/ext /home/lucas/ai/FreeToken/.venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/engine tests/scheduler tests/kernels tests/kvcache tests/server tests/tokenizer tests/layers > $O/suite.txt 2>&1 )
rc=$?
echo "suite rc=$rc $(tail -1 $O/suite.txt) | illegal: $(grep -c 'illegal memory access' $O/suite.txt) | failed-lines: $(grep -c '^FAILED' $O/suite.txt)" >> $O/status
rm -rf $O/ext; flock -u 9
st "ALLDONE"
