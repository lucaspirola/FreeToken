#!/usr/bin/env bash
# Short-extend A/B on the final code (after e2e-chain.sh), b967140 (worktree kbase) vs this tree, Ornith
# EXL3 5.0bpw on :1920 through tasks/ornith-exl3/final/arm.sh of each tree (systemd --user unit; each arm
# takes the GPU host lock). Probe 64/150/300/1000/8000 x 2 passes, 128 tokens (EXL3 extends always run
# the fused prefill: below 256 tokens the in-kernel-decode GEMM), saver base/new/new/base, whole new/base/base/new.
# Before each arm: wait for 1-min load < 3 (60 s polls, 30 min max) [agent practice, as chain2.sh].
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); BASE=/home/lucas/ai/FreeToken-wt/kbase
E=$F/e2e2; mkdir -p $E
st() { echo "$* $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> $E/chain-status.txt; }
quiet() { for i in $(seq 30); do awk '{exit !($1 < 3.0)}' /proc/loadavg && return; sleep 60; done; st "not quiet after 30 min: running anyway"; }
rm -f $E/load.stop
( while [ ! -e $E/load.stop ]; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(nvidia-smi --query-gpu=clocks.sm,utilization.gpu,memory.used --format=csv,noheader)" >> $E/load5s.txt; sleep 5; done ) &
tree() { [ $1 = base ] && echo $BASE || echo $WT; }
arm() {  # arm TREE MODE KIND K
  local t=$1 m=$2 kind=$3 k=$4 rows; rows=$([ $m = whole ] && echo 0 || echo -1)
  local name=kf2-$kind-$t-$m-$k A=$(tree $t)/tasks/ornith-exl3/final/arm.sh
  quiet; st "$name start"
  if [ $kind = probe ]; then $A $name $rows "64 150 300 1000 8000" FT_GEN=128
  else NAT=1 $A $name $rows 8000; fi
  mv $(tree $t)/tasks/ornith-exl3/final/results/$name[-.]* $E/ 2>/dev/null
  st "$name done"
}
st "chain start base=$(git -C $BASE log --oneline -1 | cut -c1-8) new=$(git -C $WT log --oneline -1 | cut -c1-8)"
arm base saver probe 1; arm new saver probe 1; arm new saver probe 2; arm base saver probe 2
arm new whole probe 1; arm base whole probe 1; arm base whole probe 2; arm new whole probe 2
touch $E/load.stop; st "E2EDONE"
