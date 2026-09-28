#!/usr/bin/env bash
# Server ABBA chain, exp/popt-exl3 (snapshot pnew-exl3) vs 757caee (pbase-exl3), :1920, q8_0 KV, 262144, ratio 1.00;
# each arm takes the GPU host lock itself (arm.sh). Rounds (results/abN/):
#   ab1: probe 300/1000/8K/32K/80K + file-read extends 100K:10K, 100K:30K, 200K:30K; saver and whole
#   ab4: Ornith 4.0bpw checkpoint (ultimatechris/...-EXL3-4bpw), saver, probe 8K/32K (8K decode), extends off
#   ab5: combined tree vs exp/reorg (BASE_TREE / NEW_TREE), saver, as ab1 (ROUNDS=5)
#   ab3: decode probe 8K/80K/256K, then natural text (5 tasks, 3500 tokens); saver and whole
# Arm names: b-* = base, n-* = new.  ROUNDS limits the rounds (default "1 4 3").
F=$(dirname "$(readlink -f "$0")"); B=${BASE_TREE:-/home/lucas/ai/FreeToken-wt/pbase-exl3}; P=${NEW_TREE:-/home/lucas/ai/FreeToken-wt/pnew-exl3}   # detached snapshot of the commit under test (the worktree stays editable)
run() { local n=$1 tree=$2 rows=$3 s=$4; shift 4; env TREE=$tree "$@" $F/arm.sh $n $rows "$s"; }
abba() { local tag=$1 rows=$2 s=$3; shift 3
  run b-${tag}1 $B $rows "$s" "$@"; run n-${tag}1 $P $rows "$s" "$@"; run n-${tag}2 $P $rows "$s" "$@"; run b-${tag}2 $B $rows "$s" "$@"; }
for r in ${ROUNDS:-1 4 3}; do
  export OUT=$F/results/ab$r
  case $r in
    1) abba s -1 "300 1000 8000 32000 80000" EXT=1; abba w 0 "300 1000 8000 32000 80000" EXT=1 ;;
    4) until grep -q DLDONE <(journalctl --user -u popt-exl3-dl4bpw -o cat --no-pager 2>/dev/null) \
             || ! systemctl --user -q is-active popt-exl3-dl4bpw; do sleep 30; done
       abba q -1 "8000 32000" MODEL=$HOME/ai/models/Ornith-1.5-35B-A3B-exl3-4.0bpw ;;
    5) # combined tree (NEW_TREE=pcomb-exl3) vs exp/reorg with popt-sched (BASE_TREE=preorg-exl3), saver
       while systemctl --user is-active --quiet popt-exl3-gatec; do sleep 60; done
       abba s -1 "300 1000 8000 32000 80000" EXT=1 ;;
    3) abba sd -1 "8000 80000 256000"; abba wd 0 "8000 80000 256000"
       abba sn -1 "8000" NAT=1; abba wn 0 "8000" NAT=1 ;;
  esac
  echo "AB${r}DONE $(date -Is)" >> $OUT/status.txt
done
