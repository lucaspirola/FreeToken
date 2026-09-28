#!/usr/bin/env bash
# One Ornith EXL3 5.0bpw arm on :1920 for the final numbers (run inside a systemd --user unit, never
# from an agent shell), under the GPU host lock, ratio 1.00, pin budget 20 GiB:
#   arm.sh NAME ROWS "SIZES" [measure.sh env words...]
# ROWS 0 = whole model, -1 = saver (mirror, auto rows, the pool's default reserve 3E).
# Env: CEIL (KV ceiling = --num-tokens = --max-seq-len-override, default 262144), YARN (e.g. 2 ->
#   --rope-yarn-factor 2), FT_KV (measure.sh lane: q8q8 default, q8q6, q6q5, or verbatim flags),
#   NAT=1 (natural text instead of the probe: 8K warm-up probe only, then natural_gen.py, 5 tasks),
#   EXT=1 (after the probe: extend_probe.py, file-read extends at depth; EXT_PAIRS overrides its pairs),
#   TREE (the worktree whose python/ and scripts/ serve, default this one; e.g. the 757caee base).
# Output in this directory's results/: NAME-* (record, probe, stats, geometry, journal, acceptance R3)
# and one start/done line per arm in results/status.txt (load, GPU MiB, MemAvailable, SM clock, code).
set -u
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib"
F=$(dirname "$(readlink -f "$0")"); WT=${TREE:-$(cd "$F/../../.." && pwd)}; out=${OUT:-$F/results}
name=$1 rows=$2 sizes=$3; shift 3
CEIL=${CEIL:-262144}
Y=""; [ -n "${YARN:-}" ] && Y="--rope-yarn-factor $YARN"
export GPU_IDLE_MIB=32 FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL=$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq FT_NAME=ornith
export FT_EXTRA="--text-model-only --num-tokens $CEIL --max-seq-len-override $CEIL --served-model-name ornith --reasoning-parser qwen3 $Y --session-spill-dir $out/spill-$name ${FT_EXTRA_MORE:-}"
export FT_ENVS="FREETOKEN_PIN_BUDGET_GB=20 ${FT_ENVS_EXTRA:-}"
export PROBE_STATS=1
post=()
if [ "${NAT:-0}" = 1 ]; then
  post=(FT_GEN=128 FT_POST_TIMEOUT=5400
        "FT_POST=python3 $F/natural_gen.py --doc $WT/docs/nemotron.md --doc $WT/docs/cli.md --max-tokens 3500 --out $out/$name-natural.json --texts $out/$name")
fi
if [ "${EXT:-0}" = 1 ]; then
  post=(FT_POST_TIMEOUT=3600
        "FT_POST=FREETOKEN_URL=http://127.0.0.1:1920 FREETOKEN_MODEL_NAME=ornith python3 $F/extend_probe.py ${EXT_PAIRS:-} > $out/$name-extend.jsonl")
fi
mkdir -p "$out"
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
until [ "$(gpu)" -le $GPU_IDLE_MIB ]; do sleep 5; done
until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
echo "$name start $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(gpu)MiB avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader) ceil=$CEIL yarn=${YARN:-} kv=${FT_KV:-q8q8} rows=$rows code=$(git -C $WT log --oneline -1 | cut -c1-8)$(git -C $WT diff --quiet HEAD -- python || echo +dirty)" >> $out/status.txt
( cd $WT && env FT_ROWS=$rows FT_SIZES="$sizes" "${post[@]}" "$@" $WT/tasks/exclusive-expert-ram/measure.sh $name > $out/$name-measure.log 2>&1 ) || echo "$name measure exit $?" >> $out/status.txt
journalctl --user -u ft-measure-$name -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > $out/$name-journal.txt || true
mv $WT/tasks/exclusive-expert-ram/results/$name-* $WT/tasks/exclusive-expert-ram/results/$name.env $out/ 2>/dev/null || true
git -C $WT checkout -q -- tasks/exclusive-expert-ram/results/sweep.tsv 2>/dev/null
FREETOKEN_LOG=$out/$name-journal.txt bash $WT/benchmarks/switchyard_soak/checks/acceptance.sh R3 > $out/$name-acceptance-R3.txt 2>&1
systemctl --user reset-failed ft-measure-$name 2>/dev/null
rm -rf $out/spill-$name
echo "$name done $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg): $(tail -1 $out/$name-acceptance-R3.txt)" >> $out/status.txt
sleep 30
