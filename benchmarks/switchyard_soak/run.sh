#!/usr/bin/env bash
# 16-way Switchyard soak: stage route 20 m, then passthrough route 20 m, default scenario
# set, c=16, server under scripts/gpu_lock.sh with FREETOKEN_SCHEDULER_INVARIANT=warn.
#
# Usage:  benchmarks/switchyard_soak/run.sh [tag] [duration]
#   tag       output subdirectory under benchmarks/switchyard_soak/runs (default: UTC stamp)
#   duration  per-phase duration passed to the harness (default: 20m)
#
# Everything it writes lands in $OUT; nothing is left in /tmp, because a WSL OOM restart
# (2026-09-05) wiped a whole soak's artifacts out of a scratchpad mid-run.
#
# Hard rules this encodes (tasks/lessons.md):
#   * the server runs under the GPU lock and its output is REDIRECTED, never piped -- the
#     lock's exit trap pkill -9's its own process group, which kills any reader
#   * shutdown TERMs the `ft serve` python directly, so the graceful path is what gets timed
#   * it refuses to start below 26 GiB MemAvailable
set -uo pipefail
REPO=/home/lucas/ai/FreeToken
TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
DUR="${2:-20m}"
PORT="${SOAK_PORT:-1919}"
SP="$REPO/benchmarks/switchyard_soak/runs/$TAG"
HERE="$REPO/benchmarks/switchyard_soak"
mkdir -p "$SP"
exec >"$SP/driver.log" 2>&1
echo "=== driver start $(date -Is)  tag=$TAG duration=$DUR port=$PORT ==="
git -C "$REPO" log -1 --oneline
git -C "$REPO" status --porcelain
free -g | head -2

avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail" -lt 26 ]; then
  echo "ABORT: only ${avail} GiB MemAvailable (< 26); a model load here OOMs the host"
  exit 2
fi

SOAK_PORT="$PORT" "$REPO/scripts/gpu_lock.sh" "$HERE/serve.sh" > "$SP/server.log" 2>&1 &
LOCK=$!
echo "$LOCK" > "$SP/lock.pid"

"$HERE/sample.sh" "$SP" &
SAMP=$!

READY=0
for i in $(seq 1 900); do
  if ! kill -0 "$LOCK" 2>/dev/null; then echo "SERVER_DIED_DURING_STARTUP"; kill "$SAMP" 2>/dev/null; exit 3; fi
  s=$(curl -s -m 3 "http://127.0.0.1:$PORT/health" | tr -d ' \n' || true)
  case "$s" in *'"status":"ok"'*) echo "READY after ${i}s at $(date -Is)"; READY=1; break;; esac
  sleep 1
done
[ "$READY" = 1 ] || { echo "STARTUP_TIMEOUT"; kill -TERM "$LOCK"; kill "$SAMP" 2>/dev/null; exit 4; }
curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo

# The top ft-serve python (its parent is not itself a FreeToken-venv python).
find_srv() {
  for p in $(pgrep -f '^/home/lucas/ai/FreeToken/\.venv/bin/python' 2>/dev/null); do
    ppid=$(awk '{s=$0; sub(/^[0-9]+ \(.*\) /,"",s); split(s,f," "); print f[2]}' /proc/$p/stat 2>/dev/null)
    [ -n "$ppid" ] || continue
    pc=$(tr '\0' ' ' < /proc/$ppid/cmdline 2>/dev/null)
    case "$pc" in *FreeToken/.venv/bin/python*) ;; *) echo "$p"; return;; esac
  done
}
SRV=$(find_srv)
echo "server python pid=$SRV"

( while true; do
    code=$(curl -s -m 5 -o "$SP/.h.json" -w '%{http_code}' "http://127.0.0.1:$PORT/health")
    body=$(tr -d ' \n' < "$SP/.h.json" 2>/dev/null)
    case "$code:$body" in
      200:*'"status":"ok"'*) : ;;
      *) echo "$(date -Is) health_http=$code body=$body" >> "$SP/health_bad.log" ;;
    esac
    sleep 10
  done ) &
HW=$!

run_phase() {  # $1 = route suffix, $2 = phase key (STAGE|PASS), $3 = workdir name
  echo "=== soak $1: $DUR @ $(date -Is) ==="
  echo "$2_START_TS=$(date +%s)"
  "$REPO/scripts/switchyard_e2e.sh" soak \
    --base-url "http://127.0.0.1:$PORT" --model nemotron-3.5-lightning \
    --duration "$DUR" --concurrency 16 --route "switchyard/$1" \
    --workdir "$SP/$3" > "$SP/$3.log" 2>&1
  echo "$1 exit=$?  $(date -Is)"
  echo "$2_END_TS=$(date +%s)"
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/stats" > "$SP/stats_after_${3}.json" || true
  echo
}

# SOAK_PHASES selects which routes run (default: both, in this order). A short single-route
# A/B sets it to just "passthrough".
for phase in ${SOAK_PHASES:-stage passthrough}; do
  case "$phase" in
    stage)       run_phase stage       STAGE soakStage ;;
    passthrough) run_phase passthrough PASS  soakPass ;;
    *) echo "unknown SOAK_PHASES entry: $phase"; exit 5 ;;
  esac
done
curl -s -m 5 -o /dev/null -w 'health_http=%{http_code}\n' "http://127.0.0.1:$PORT/health"

# --- disconnect-abort probe (ff470e7): drop a long prompt mid-prefill, then confirm
# /v1/stats.active returns to 0 (the abort reached the scheduler and freed the slot).
if [ "${SOAK_PROBE:-1}" = 1 ]; then
echo "=== disconnect probe $(date -Is) ==="
curl -s -m 5 "http://127.0.0.1:$PORT/v1/stats" > "$SP/stats_before_probe.json"
python3 - "$PORT" <<'PY'
import json, socket, sys, time, urllib.request
port = sys.argv[1]
body = json.dumps({
    "model": "nemotron-3.5-lightning",
    "messages": [{"role": "user", "content": "word " * 60000 + "\nSummarize."}],
    "max_tokens": 64, "stream": False,
}).encode()
req = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + port.encode() + b"\r\n"
       b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode()
       + b"\r\nConnection: close\r\n\r\n" + body)

def active():
    u = f"http://127.0.0.1:{port}/v1/stats"
    return json.load(urllib.request.urlopen(u, timeout=5))["requests"]["active"]

print("active before probe:", active())
s = socket.create_connection(("127.0.0.1", int(port)), timeout=10)
s.sendall(req)
time.sleep(6.0)
print("active while probe in flight:", active())
s.shutdown(socket.SHUT_RDWR); s.close()
print("probe socket closed at", time.strftime("%H:%M:%S"))
t0 = time.time()
for _ in range(60):
    time.sleep(2)
    a = active()
    if a == 0:
        print("active back to 0 after %.0f s" % (time.time() - t0)); break
else:
    print("active did NOT return to 0:", a)
PY
curl -s -m 5 "http://127.0.0.1:$PORT/v1/stats" > "$SP/stats_after_probe.json"
fi

kill "$HW" 2>/dev/null
kill "$SAMP" 2>/dev/null
echo "=== stopping server $(date -Is) ==="
t0=$(date +%s)
kill -TERM "$SRV" 2>/dev/null
while kill -0 "$LOCK" 2>/dev/null; do sleep 1; done
echo "shutdown_seconds=$(( $(date +%s) - t0 ))"
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af '/home/lucas/ai/FreeToken/\.venv/bin/python' | head
echo "=== driver done $(date -Is) ==="
