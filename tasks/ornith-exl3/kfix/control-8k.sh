#!/usr/bin/env bash
# Same-day control for the gate's 8K/80K saver-vs-whole ratio (ck9k 8K p2 median 91.1% vs ck8o 95.1%):
# the identical recheck (recheck-local.sh, RC_PARTS=8k, 8000 + 80000, 3 alternated pairs) on kbase
# (b967140, label ck9b) and then on this tree (label ck9n), back to back, after the gate chain.
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
while systemctl --user is-active --quiet kfix-gate; do sleep 60; done
F=$(dirname "$(readlink -f "$0")"); C=$F/gate-control; mkdir -p $C
export FT_MODEL=$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq
FT_EXTRA="--text-model-only --num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith --reasoning-parser qwen3"
for tl in /home/lucas/ai/FreeToken-wt/kbase:ck9b $(cd "$F/../../.." && pwd):ck9n; do
  t=${tl%%:*}; L=${tl##*:}; X=$t/tasks/exclusive-expert-ram
  echo "$L start $(date -Is) code $(git -C $t log --oneline -1 | cut -c1-8) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $C/status.txt
  systemctl --user reset-failed ft-kf-ctl 2>/dev/null
  systemd-run --user --unit=ft-kf-ctl --setenv=PATH="$PATH" --setenv=GPU_IDLE_MIB=32 --setenv=RC_OUT=$C --setenv=RC_LABEL=$L \
    --setenv=RC_PARTS=8k "--setenv=RC_SIZE=8000 80000" --setenv=RC_RESERVE= --setenv=FT_MODEL=$FT_MODEL --setenv=FT_NAME=ornith \
    "--setenv=FT_EXTRA=$FT_EXTRA" "--setenv=FT_ENVS=FREETOKEN_PIN_BUDGET_GB=20" /bin/bash -c "$X/recheck-local.sh > $C/recheck-$L.log 2>&1"
  sleep 20; while systemctl --user is-active --quiet ft-kf-ctl; do sleep 30; done; sleep 20
  git -C $t checkout -q -- tasks/exclusive-expert-ram/results/sweep.tsv 2>/dev/null
  echo "$L done $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $C/status.txt
done
echo CONTROLDONE >> $C/status.txt
