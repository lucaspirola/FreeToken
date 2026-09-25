#!/usr/bin/env bash
# WSL decode-headroom cushion probe (exp/decode-headroom 83b6c19 = a2f08c0 + results):
# dhw-np (mirror-np, 8K/80K) and dhw-1m (mirror-1m, 8K/1M) with FREETOKEN_DECODE_MEM_PROBE=1.
set -uo pipefail
S=/tmp/claude-1000/-home-lucas-ai-FreeToken/606da56c-cd31-49f4-adb4-b4ab53d18508/scratchpad
WT=/home/lucas/ai/FreeToken-wt/decode-headroom; H=$WT/tasks/exclusive-expert-ram; O=$S/dhw; mkdir -p $O
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
one() {  # arm sizes extra-envs
  local ARM=$1 SIZES=$2 ENVS="FREETOKEN_DECODE_MEM_PROBE=1 ${3:-}"
  flock 9
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" = 0 ]; do sleep 5; done
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
  echo "=== $ARM $(git -C $WT log --oneline -1 | cut -c1-40) envs=[$ENVS] $(date -Is)"
  FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00 FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="$SIZES" FT_ENVS="$ENVS" "$H/measure.sh" $ARM > $O/$ARM.log 2>&1 || echo "measure exit $?"
  journalctl --user -u ft-measure-$ARM -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > $O/$ARM-journal.txt || true
  mv "$H"/results/$ARM-* $O/ 2>/dev/null || true
  grep '"target"' $O/$ARM.log | cut -c1-160
  grep -E "Decode memory window|KV commit memory" $O/$ARM-journal.txt | cut -c1-240
  flock -u 9; sleep 20
}
one dhw-np "8000 80000"
if grep "Decode memory window" $O/dhw-np-journal.txt | grep -qE "drop 0\.0[0-5]"; then one dhw-128 "8000 80000" FREETOKEN_DECODE_FREE_TARGET_MB=128; fi
one dhw-1m "8000 1000000"
echo DHW-DONE
