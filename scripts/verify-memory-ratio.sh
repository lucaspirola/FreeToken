#!/usr/bin/env bash
# Verify that this host serves at --memory-ratio 1.00: one start, 8K / 80K / 256K prompts,
# PASS or FAIL, one TSV row. Nothing is bisected and nothing is written to serve.env.
#
#   scripts/verify-memory-ratio.sh                 # the systemd unit freetoken-serve (owner hosts)
#   VERIFY_LAUNCHER=nohup scripts/verify-memory-ratio.sh   # hosts without systemd (rented boxes)
#
# 1.00 is the default and is not tuned per host. The engine reserves the runtime headroom
# itself: the growable arena is filled back around the measured prefill transient, a
# fixed-size start takes the predicted reserve out of its budget. It also logs its
# config-based prediction beside each measurement ("Memory prediction check" lines,
# engine/memory_prediction.py). This script checks that the default works on this GPU and
# model, and reports the prediction lines, so a mismatch shows up here instead of as a long
# prompt's OOM. Run it at install time (CLAUDE.md "New machine") and after a driver, VRAM or
# model change. A FAIL is a bug to fix (or a host that needs an explicit override, decided by
# the owner), not a ratio to search for.
#
# A run PASSES when the server reaches "API server is ready" with one CUDA graph capture,
# serves 8K, 80K and 256K-token prompts (KV growth to 256K on the default profile), and its
# log has no Traceback / out-of-memory / CUDA error. The prediction check is reported
# (WARN lines counted) but does not fail the run: the measurement is what the engine uses.
#
# Launchers:
#   systemd (default)  sudo -n systemctl restart freetoken-serve with FREETOKEN_MEMORY_RATIO
#                      set for this start; the log is ~/.cache/freetoken/logs/ft_serve.log.
#                      The server is started by PID 1, never from this shell (CLAUDE.md: a
#                      server started from an agent shell dies with it). It is left running.
#   nohup              setsid nohup scripts/serve-default.sh on $FREETOKEN_PORT (default 1920),
#                      log $VERIFY_LOG; stopped at the end unless VERIFY_KEEP=1. For rented
#                      boxes (root, no systemd). Launch this script itself detached there too.
#
# Env: VERIFY_RATIO (default 1.00), VERIFY_SIZES (default "8000 80000 256000"),
#      VERIFY_OUT (TSV, default ~/.cache/freetoken/logs/verify-memory-ratio.tsv),
#      FREETOKEN_MODEL / FREETOKEN_EXTRA_ARGS as for serve-default.sh.
set -u
HERE=$(dirname "$(readlink -f "$0")")
RATIO=${VERIFY_RATIO:-1.00}
SIZES=${VERIFY_SIZES:-8000 80000 256000}
LAUNCHER=${VERIFY_LAUNCHER:-systemd}
OUT=${VERIFY_OUT:-$HOME/.cache/freetoken/logs/verify-memory-ratio.tsv}
mkdir -p "$(dirname "$OUT")"
[ -s "$OUT" ] || echo -e "date\thost\tgpu\tmodel\tratio\tresult\tready_s\tcaptures\tfree_after_capture\tdecode\ttransient_pred\ttransient_meas\tprediction_warns\tnote" > "$OUT"

case "$LAUNCHER" in
  systemd)
    PORT=${FREETOKEN_PORT:-1919}
    LOG=${FREETOKEN_LOG:-$HOME/.cache/freetoken/logs/ft_serve.log}
    ;;
  nohup)
    PORT=${FREETOKEN_PORT:-1920}
    LOG=${VERIFY_LOG:-$HOME/.cache/freetoken/logs/verify-memory-ratio-server.log}
    ;;
  *) echo "VERIFY_LAUNCHER must be systemd or nohup" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$LOG")"; touch "$LOG"
before=$(wc -l < "$LOG")
alive() {
  if [ "$LAUNCHER" = systemd ]; then systemctl is-active --quiet freetoken-serve
  else kill -0 "$SPID" 2>/dev/null; fi
}

echo "=== verify memory-ratio $RATIO ($LAUNCHER, port $PORT) $(date +%T)"
if [ "$LAUNCHER" = systemd ]; then
  sudo -n systemctl set-environment "FREETOKEN_MEMORY_RATIO=$RATIO"
  sudo -n systemctl reset-failed freetoken-serve 2>/dev/null
  sudo -n systemctl restart freetoken-serve
else
  FREETOKEN_PORT=$PORT FREETOKEN_MEMORY_RATIO=$RATIO setsid nohup "$HERE/serve-default.sh" >> "$LOG" 2>&1 < /dev/null &
  SPID=$!
fi
t0=$(date +%s)
start=""
until [ -n "$start" ] || [ $(( $(date +%s) - t0 )) -gt 300 ]; do
  sleep 2
  start=$(tail -n +"$((before + 1))" "$LOG" | grep -n "ServerArgs(model_path" | tail -1 | cut -d: -f1)
done
run_log() { tail -n +"$((before + ${start:-1}))" "$LOG" | sed 's/\x1b\[[0-9;]*m//g'; }
until run_log | grep -q "API server is ready" || ! alive || [ $(( $(date +%s) - t0 )) -gt 900 ]; do sleep 5; done
ready_s=$(( $(date +%s) - t0 ))

result=PASS; note=""; decode=""
if [ -z "$start" ] || ! alive || ! run_log | grep -q "API server is ready"; then
  result=FAIL-start
  note=$(run_log | grep -m1 -E "out of memory|OutOfMemory|CUDA error|Traceback|Error" | cut -c1-120)
else
  for s in $SIZES; do
    d=$(FREETOKEN_URL="http://127.0.0.1:$PORT" timeout 1800 python3 "$HERE/probe_decode.py" "$s" 2>/dev/null | tail -1 \
        | python3 -c "import sys,json;print(json.loads(sys.stdin.read())['decode_tok_s'])" 2>/dev/null || echo ERR)
    decode="$decode${decode:+/}$d"
    [ "$d" = ERR ] && { result=FAIL-run; note="probe $s failed"; break; }
  done
  if ! alive || run_log | grep -q -E "Traceback|out of memory|OutOfMemory|CUDA error"; then
    result=FAIL-run
    note=${note:-$(run_log | grep -m1 -E "out of memory|OutOfMemory|CUDA error|Traceback" | cut -c1-120)}
  fi
fi
captures=$(run_log | grep -c "Start capturing CUDA graphs" || true)
[ "$result" = PASS ] && [ "$captures" != 1 ] && { result=FAIL-run; note="captures=$captures"; }
freecap=$(run_log | grep -o "Free GPU memory after capturing CUDA graphs: [0-9.]* GiB" | tail -1 | grep -o "[0-9.]* GiB" || true)
tline=$(run_log | grep -m1 "Memory prediction check: prefill transient" || true)
tpred=$(sed -nE 's/.*predicted ([0-9.]+ GiB).*/\1/p' <<< "$tline")
tmeas=$(sed -nE 's/.*measured ([0-9.]+ GiB).*/\1/p' <<< "$tline")
warns=$(run_log | grep -c "Memory prediction check:.*outside the" || true)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
model=$(run_log | grep -m1 -oE "model_path='[^']*'" | sed -E "s/model_path='(.*)'/\1/" | xargs -r basename)
echo -e "$(date -Is)\t$(hostname)\t$gpu\t$model\t$RATIO\t$result\t$ready_s\t$captures\t${freecap:--}\t${decode:--}\t${tpred:--}\t${tmeas:--}\t$warns\t$note" >> "$OUT"

echo "--- startup memory lines"
run_log | grep -E "Startup memory prediction|Memory prediction check|Prefill headroom|Startup geometry|moe-cache-auto resolved" | cut -c1-400
echo "--- $result (ratio $RATIO, ready ${ready_s}s, captures $captures, decode ${decode:--} tok/s, prediction WARNs $warns)"
[ -n "$note" ] && echo "    $note"
if [ "$LAUNCHER" = systemd ]; then
  sudo -n systemctl unset-environment FREETOKEN_MEMORY_RATIO
  echo "server left running (unit freetoken-serve); a restart picks up serve.env / the 1.00 default again"
elif [ "${VERIFY_KEEP:-0}" != 1 ]; then
  kill -TERM -- -"$SPID" 2>/dev/null || kill "$SPID" 2>/dev/null
  for _ in $(seq 60); do alive || break; sleep 1; done
  alive && kill -KILL -- -"$SPID" 2>/dev/null
fi
echo "TSV: $OUT"
[ "$result" = PASS ]
