#!/usr/bin/env bash
# abtrees.sh OUT ROUNDS SCRIPT : run SCRIPT in the base tree (757caee, pbase-exl3) and this tree,
# base/new/new/base per round, one GPU-lock hold per process (run via go.sh -> gpu.sh already holds
# the lock: this script must NOT flock again)
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); BASE=/home/lucas/ai/FreeToken-wt/pbase-exl3
OUT=$1; R=$2; S=$3; PY=/home/lucas/ai/FreeToken/.venv/bin/python
for r in $(seq $R); do
  for t in base new new base; do
    [ $t = base ] && T=$BASE || T=$WT
    PYTHONPATH=$T/python $PY $F/$S $t >> $OUT 2>> $OUT.err
  done
done
