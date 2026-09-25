#!/usr/bin/env bash
# Round-2 decode regression A/B (whole model, 300K, passes 1+2): base reorg-next vs round2 default,
# round2 with the triton extend kernel, round2 without chunk snapshots, base again.
set -uo pipefail
S=/tmp/claude-1000/-home-lucas-ai-FreeToken/606da56c-cd31-49f4-adb4-b4ab53d18508/scratchpad
O=$S/dec-ab2; mkdir -p $O
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
one() {  # tree arm envs
  local WT=$1 ARM=$2 ENVS="${3:-}" H=$1/tasks/exclusive-expert-ram
  flock 9
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" = 0 ]; do sleep 5; done
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 20; done
  echo "=== $ARM $(git -C $WT log --oneline -1 | cut -c1-40) envs=[$ENVS] $(date -Is)"
  FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00 FT_ROWS=0 FT_EXTRA="--moe-collect-stats" FT_GEN=512 FT_SIZES="8000 300000" FT_ENVS="$ENVS" "$H/measure.sh" $ARM > $O/$ARM.log 2>&1 || echo "measure exit $?"
  grep '"target"' $O/$ARM.log | python3 -c 'import sys,json; [print(" ", (d:=json.loads(l))["target"], d["pass"], "decode", d["decode_tok_s"], "prefill", d["prefill_tok_s"]) for l in sys.stdin]'
  journalctl --user -u ft-measure-$ARM -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > $O/$ARM-journal.txt || true
  mv "$H"/results/$ARM-* $O/ 2>/dev/null || true
  flock -u 9; sleep 20
}
B=/home/lucas/ai/FreeToken-wt/reorg-next; R=/home/lucas/ai/FreeToken-wt/round2





for r in 1 2; do
  one $B dab2-base-$r
  one $R dab2-r2-$r
  one $R dab2-r2-trit-$r FREETOKEN_EXTEND_BACKEND=triton
  one $R dab2-r2-nosnap-$r FREETOKEN_CHUNK_SNAPSHOTS=0
done
echo DEC-AB2-DONE
