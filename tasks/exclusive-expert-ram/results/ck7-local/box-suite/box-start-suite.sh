#!/bin/bash
# Retry starting ft-dev (Vast 52296107) every 5 min for up to 4 h (its host GPU was taken:
# resources_unavailable); as soon as it runs, ship the round-5 thin bundle + r5fsuite.sh and start
# the suite there (setsid, under /root/gpu.lock inside the script). The API key is read inside
# the commands only and never printed. Log: _orch/r5/box-start.log.
B=$(dirname "$(readlink -f "$0")"); L=/home/lucas/ai/FreeToken-wt/_orch/r5/box-start.log
api() { curl -s -H "Authorization: Bearer $(cat ~/.config/vastai/vast_api_key)" "$@" https://console.vast.ai/api/v0/instances/52296107/; }
st() { api | python3 -c "import json,sys;d=json.load(sys.stdin);i=d.get('instances',d);print(i.get('actual_status'))"; }
for i in $(seq 48); do
  s=$(st); echo "$(date +%T) $s" >> $L
  [ "$s" = running ] && break
  api -X PUT -H "Content-Type: application/json" -d '{"state":"running"}' | python3 -c "import json,sys;print(' put:', json.load(sys.stdin).get('error'))" >> $L
  sleep 300
done
[ "$(st)" = running ] || { echo "gave up $(date +%T)" >> $L; exit 1; }
for i in $(seq 30); do timeout 20 ssh -o ConnectTimeout=15 ft-dev true 2>/dev/null && break; sleep 20; done
scp -q /home/lucas/ai/FreeToken-wt/_orch/r5/r5f-thin.bundle $B/r5fsuite.sh ft-dev:/root/ >> $L 2>&1
ssh ft-dev 'set -e
src=""; for d in /root/FT-*; do git -C $d cat-file -e 74edbe8^{commit} 2>/dev/null && { src=$d; break; }; done
echo "source repo: $src"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
rm -rf /root/FT-round5f; git clone -q $src /root/FT-round5f
cd /root/FT-round5f; git fetch -q /root/r5f-thin.bundle exp/reorg-round5:r5f; git checkout -q r5f; git log --oneline -1
cp -p $src/python/freetoken/kernel/*.so python/freetoken/kernel/ 2>/dev/null || true
chmod +x /root/r5fsuite.sh; setsid nohup /root/r5fsuite.sh > /root/r5fsuite.log 2>&1 < /dev/null & echo started' >> $L 2>&1
echo "suite launched $(date +%T)" >> $L
