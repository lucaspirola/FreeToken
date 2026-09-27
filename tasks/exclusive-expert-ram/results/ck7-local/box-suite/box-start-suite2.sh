#!/bin/bash
# Replaces box-start-suite.sh: retry starting ft-dev (Vast 52296107) every 5 min for up to 8 h
# (resources_unavailable); once running, ship a thin bundle with exp/reorg-round5 AND exp/r5-seed-b (idle seed),
# run r5fsuite2.sh (both trees), copy the results back into this directory, verify the copies,
# then STOP the box and verify actual_status=exited. The API key is read inside the commands only
# and never printed. Log: _orch/r5/box-start.log.
B=$(dirname "$(readlink -f "$0")"); L=/home/lucas/ai/FreeToken-wt/_orch/r5/box-start.log
R=/home/lucas/ai/FreeToken-wt/round5
api() { curl -s -H "Authorization: Bearer $(cat ~/.config/vastai/vast_api_key)" "$@" https://console.vast.ai/api/v0/instances/52296107/; }
st() { api | python3 -c "import json,sys;d=json.load(sys.stdin);i=d.get('instances',d);print(i.get('actual_status'))"; }
echo "--- box-start-suite2 $(date -Is)" >> $L
git -C $R bundle create /home/lucas/ai/FreeToken-wt/_orch/r5/r5f2-thin.bundle 74edbe8..exp/reorg-round5 74edbe8..exp/r5-seed-b >> $L 2>&1
for i in $(seq 96); do
  s=$(st); echo "$(date +%T) $s" >> $L
  [ "$s" = running ] && break
  api -X PUT -H "Content-Type: application/json" -d '{"state":"running"}' | python3 -c "import json,sys;print(' put:', json.load(sys.stdin).get('error'))" >> $L
  sleep 300
done
[ "$(st)" = running ] || { echo "gave up $(date +%T)" >> $L; exit 1; }
for i in $(seq 30); do timeout 20 ssh -o ConnectTimeout=15 ft-dev true 2>/dev/null && break; sleep 20; done
scp -q /home/lucas/ai/FreeToken-wt/_orch/r5/r5f2-thin.bundle $B/r5fsuite2.sh ft-dev:/root/ >> $L 2>&1
ssh ft-dev 'set -e
src=""; for d in /root/FT-*; do git -C $d cat-file -e 74edbe8^{commit} 2>/dev/null && { src=$d; break; }; done
echo "source repo: $src"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
for t in FT-round5f:exp/reorg-round5 FT-r5seed:exp/r5-seed-b; do
  d=/root/${t%%:*}; b=${t##*:}
  rm -rf $d; git clone -q $src $d; cd $d; git fetch -q /root/r5f2-thin.bundle $b:tested; git checkout -q tested; git log --oneline -1
  cp -p $src/python/freetoken/kernel/*.so python/freetoken/kernel/ 2>/dev/null || true
done
rm -f /root/r5suites.done
chmod +x /root/r5fsuite2.sh; setsid nohup /root/r5fsuite2.sh > /root/r5fsuite2.log 2>&1 < /dev/null & echo started' >> $L 2>&1
echo "suite launched $(date +%T)" >> $L
for i in $(seq 240); do timeout 30 ssh ft-dev test -f /root/r5suites.done 2>/dev/null && break; sleep 60; done
for o in r5fsuite r5seedsuite; do rm -rf $B/$o; mkdir -p $B/$o; scp -q ft-dev:/root/$o/status ft-dev:/root/$o/suite.txt $B/$o/ >> $L 2>&1; done
scp -q ft-dev:/root/r5fsuite2.log $B/ >> $L 2>&1
ok=1; for o in r5fsuite r5seedsuite; do grep -q ALLDONE $B/$o/status 2>/dev/null && [ -s $B/$o/suite.txt ] || ok=0; done
echo "results copied ok=$ok $(date +%T): $(cat $B/r5fsuite/status $B/r5seedsuite/status 2>/dev/null | grep suite | tr '\n' ' ')" >> $L
[ $ok = 1 ] || { echo "copies incomplete: box left running for the worker" >> $L; exit 2; }
api -X PUT -H "Content-Type: application/json" -d '{"state":"stopped"}' | python3 -c "import json,sys;print(' stop put:', json.load(sys.stdin).get('success'))" >> $L
for i in $(seq 30); do s=$(st); [ "$s" = exited ] && break; sleep 20; done
echo "$(date +%T) after stop: actual_status=$(st)" >> $L
