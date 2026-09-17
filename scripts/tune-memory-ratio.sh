#!/usr/bin/env bash
# Find the largest --memory-ratio this host can serve with (i.e. the least free VRAM,
# the most resident experts) and persist it in ~/.config/freetoken/serve.env.
#
#   scripts/tune-memory-ratio.sh [LO] [HI]      defaults LO=0.90 (known good) HI=1.00
#
# The candidate ratio is written to the host env file before each restart: the launcher
# sources it, so a trial tests exactly the written ratio (systemctl set-environment is
# NOT inherited by ExecStart and would be clobbered by serve.env anyway).
# Run as the invoking user; under root (sudo scripts/... or WSL `wsl -u root`) restarts
# go through plain systemctl, otherwise through `sudo -n` (passwordless sudo required).
#
# Part of the installation on every machine (CLAUDE.md "New machine"): run once after
# `sudo scripts/systemd/install.sh`, and again after a driver/VRAM change. Takes ~4 min per
# trial (restart + 8K/80K/256K probes), usually 4-6 trials.
#
# Algorithm: try HI first (zero free VRAM is the goal); if it breaks, bisect between the
# last good and the last bad ratio until they are 0.005 apart (~80 MB on 16 GB). A trial
# PASSES when the unit reaches "API server is ready", captures its CUDA graphs, and serves
# 8K, 80K and 256K-token prompts (KV growth to 256K funded from the expert arena) with no
# Traceback / out-of-memory / CUDA error in the log. The server is left running on the
# winning ratio, which is written to $HOME/.config/freetoken/serve.env
# (FREETOKEN_MEMORY_RATIO=...), the file scripts/serve-default.sh sources.
set -u
LO=${1:-0.90}; HI=${2:-1.00}
if [ "$(id -u)" -eq 0 ]; then SUDO=""; elif sudo -n true 2>/dev/null; then SUDO="sudo -n"; else
  echo "need passwordless sudo (or run me as root)" >&2; exit 1; fi
HERE=$(dirname "$(readlink -f "$0")")
LOG=${FREETOKEN_LOG:-$HOME/.cache/freetoken/logs/ft_serve.log}
ENV_FILE=${FREETOKEN_HOST_ENV:-$HOME/.config/freetoken/serve.env}
OUT=${TUNE_OUT:-$HOME/.cache/freetoken/logs/tune-memory-ratio.tsv}
mkdir -p "$(dirname "$OUT")" "$(dirname "$ENV_FILE")"
echo -e "# $(date -Is)\nratio\tresult\tslots\tfree_after_capture\tdecode80k\tdecode256k\tnote" >> "$OUT"

restart_with() {
  if grep -q '^FREETOKEN_MEMORY_RATIO=' "$ENV_FILE" 2>/dev/null; then
    sed -i "s/^FREETOKEN_MEMORY_RATIO=.*/FREETOKEN_MEMORY_RATIO=$1/" "$ENV_FILE"
  else
    echo "FREETOKEN_MEMORY_RATIO=$1" >> "$ENV_FILE"
  fi
  $SUDO systemctl reset-failed freetoken-serve 2>/dev/null
  $SUDO systemctl restart freetoken-serve
}
run_log() { tail -n +"$1" "$LOG"; }

trial() {  # $1 = ratio -> prints PASS/FAIL, appends a row to $OUT
  local r=$1 before start t0 note slots freecap d80 d256 res
  before=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  restart_with "$r"
  t0=$(date +%s)
  # Wait for THIS start's ServerArgs line (the log appends across starts; the stop of the
  # previous server can take a while, so never trust the last segment before the restart).
  start=""
  until [ -n "$start" ] || [ $(( $(date +%s) - t0 )) -gt 300 ]; do
    sleep 2
    start=$(tail -n +"$((before + 1))" "$LOG" | grep -n "ServerArgs(model_path" | tail -1 | cut -d: -f1)
  done
  if [ -z "$start" ]; then
    echo -e "$r\tFAIL-start\t-\t-\t-\t-\tno ServerArgs line within 300 s" >> "$OUT"; echo FAIL; return
  fi
  start=$((before + start))
  until run_log "$start" | grep -q "API server is ready" || ! systemctl is-active --quiet freetoken-serve \
        || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
  if ! systemctl is-active --quiet freetoken-serve || ! run_log "$start" | grep -q "API server is ready"; then
    note=$(run_log "$start" | grep -m1 -E "out of memory|OOM|CUDA error|Traceback|Error" | cut -c1-80)
    echo -e "$r\tFAIL-start\t-\t-\t-\t-\t$note" >> "$OUT"; echo FAIL; return
  fi
  freecap=$(run_log "$start" | grep -o "Free GPU memory after capturing CUDA graphs: [0-9.]* GiB" | grep -o "[0-9.]* GiB" || true)
  d80=$(timeout 900 python3 "$HERE/probe_decode.py" 8000 2>/dev/null \
        | python3 -c "import sys,json;print(json.loads(sys.stdin.read())['decode_tok_s'])" 2>/dev/null || echo ERR)
  d80k=$(timeout 900 python3 "$HERE/probe_decode.py" 80000 2>/dev/null \
        | python3 -c "import sys,json;print(json.loads(sys.stdin.read())['decode_tok_s'])" 2>/dev/null || echo ERR)
  d256=$(timeout 1200 python3 "$HERE/probe_decode.py" 256000 2>/dev/null \
        | python3 -c "import sys,json;print(json.loads(sys.stdin.read())['decode_tok_s'])" 2>/dev/null || echo ERR)
  slots=$(run_log "$start" | grep -o "MoE cache now [0-9]* slots" | head -1 | grep -o "[0-9]*" || true)
  if systemctl is-active --quiet freetoken-serve \
     && ! run_log "$start" | grep -q -E "Traceback|out of memory|CUDA error" \
     && [ "$d80" != ERR ] && [ "$d80k" != ERR ] && [ "$d256" != ERR ]; then res=PASS; else res=FAIL-run; fi
  note=$(run_log "$start" | grep -m1 -E "out of memory|CUDA error|Traceback" | cut -c1-80)
  echo -e "$r\t$res\t${slots:--}\t${freecap:--}\t$d80k\t$d256\t$note" >> "$OUT"
  echo $res
}

echo "=== trial $HI (top: zero free VRAM) $(date +%T)"
PREV_RATIO=$(sed -n 's/^FREETOKEN_MEMORY_RATIO=//p' "$ENV_FILE" 2>/dev/null | tail -1)
GOOD=""
if [ "$(trial "$HI")" = PASS ]; then
  LO=$HI; GOOD=$HI
else
  while awk -v lo="$LO" -v hi="$HI" 'BEGIN{exit !(hi-lo>0.0051)}'; do
    MID=$(awk -v lo="$LO" -v hi="$HI" 'BEGIN{printf "%.3f", (lo+hi)/2}')
    echo "=== trial $MID (good=$LO bad=$HI) $(date +%T)"
    if [ "$(trial "$MID")" = PASS ]; then LO=$MID; GOOD=$MID; else HI=$MID; fi
  done
fi
echo "=== best memory-ratio: $LO (first failing: $HI) $(date +%T)"

# Persist for scripts/serve-default.sh and leave the server running on it. On total
# failure (nothing passed) restore the pre-tuning ratio rather than persisting an
# untested value, restart once, and exit non-zero.
if [ -z "$GOOD" ]; then
  echo "=== every trial failed; NOT persisting a ratio" >&2
  if [ -n "$PREV_RATIO" ]; then restart_with "$PREV_RATIO"; fi
  exit 1
fi
if grep -q '^FREETOKEN_MEMORY_RATIO=' "$ENV_FILE" 2>/dev/null; then
  sed -i "s/^FREETOKEN_MEMORY_RATIO=.*/FREETOKEN_MEMORY_RATIO=$LO/" "$ENV_FILE"
else
  echo "FREETOKEN_MEMORY_RATIO=$LO" >> "$ENV_FILE"
fi
$SUDO systemctl reset-failed freetoken-serve 2>/dev/null; $SUDO systemctl restart freetoken-serve
echo "wrote FREETOKEN_MEMORY_RATIO=$LO to $ENV_FILE; server restarting on it. Trials:"
cat "$OUT"
