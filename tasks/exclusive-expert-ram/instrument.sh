#!/usr/bin/env bash
# The section 7.3 "finer instrument" (plan `2026-09-22-refactor-plan-final.md`,
# sections 5/S6/S7 and 7.3): before and after S6/S7, on the same commit pair, one
# 128-token decode at 8K, measuring (a) the count of GPU kernel launches per
# decode step INCLUDING kernels inside CUDA-graph replays, and (b) bytes-moved
# proxy counters per step from the server's own /v1/stats (mirror_stats() and
# the raw decode-miss counters). Identical launches + identical counters means
# the refactor added no GPU work; the checkpoint bundle (7.2) is too coarse to
# see a 2-3% regression, this instrument is not.
#
# Launch it as a systemd transient unit itself (systemd-run --user ... instrument.sh run
# ...), never from a plain agent shell: CLAUDE.md "Do not start the server from an agent
# shell" -- the harness can kill the shell mid-capture and take the arm's unit with it.
# (An earlier version said "THE OWNER RUNS THIS", citing the plan's agent-authored 9.4;
# the owner chose on 2026-09-23 that the agent runs checkpoints.)
#
# Usage:
#   instrument.sh run   --before /path/to/worktree/at/before-commit \
#                        --after  /path/to/worktree/at/after-commit  \
#                        --label  s6                                 \
#                        [--port 1920] [--size 8000] [--decode-tokens 128]
#
#     Runs the "before" arm, then the "after" arm (never concurrently -- see
#     C-EMPTY-GPU and the host lock below), each producing
#     results/instrument-<label>-{before,after}.txt.
#
#   instrument.sh compare results/instrument-s6-before.txt results/instrument-s6-after.txt
#
#     Diffs two result files; prints IDENTICAL or the differences, exit 0/1.
#     Pure text/JSON diff, no GPU, no lock, safe from anywhere including an
#     agent shell.
#
# What "run" does, per arm (mirrors measure.sh's / checkpoint1.sh's
# conventions -- same host lock, same port, same refusals, same FT_VENV
# pattern for serving an arbitrary worktree's python/ without a uv sync):
#   1. Preflight: refuse port 1919, refuse if freetoken-serve is active, refuse
#      if the GPU is not empty, refuse if MemAvailable < 22 GiB, stop
#      piro-board-embedder if active (never restart it -- checkpoint1.sh's
#      rule, repeated here because this script does not source it).
#   2. Take the host lock (flock, whole run) at
#      /home/lucas/.cache/freetoken/gpu-host.lock, exactly as checkpoint1.sh
#      does, so no torch pytest and no other GPU script can race this.
#   3. Start the arm's server via systemd-run --user (transient unit, port from
#      --port, FT_VENV pointed at the given worktree's python/ through the
#      installed server venv, no uv sync -- serve-default.sh's own contract),
#      wait for readiness exactly as measure.sh does (a real completion, not
#      /v1/models), then warm it up with a couple of throwaway decodes so the
#      captured window is steady-state, not the first-request bank-build
#      slowdown (CLAUDE.md: first requests after a start run at a fraction of
#      the eventual speed for ~3 minutes).
#   4. `nsys launch` primes the unit's process tree with CUDA/NVTX injection
#      BEFORE step 3's readiness wait (injection must be present from process
#      start; a deferred `nsys launch` does not itself begin recording -- see
#      below), then, once warm, `nsys start` begins recording, this script
#      fires exactly one chat completion whose prompt is padded to --size
#      prompt tokens and whose max_tokens is --decode-tokens, waits for it to
#      finish, and `nsys stop` ends recording. This bounds the capture to
#      (padding of one prefill) + (the 128 decode steps) rather than the whole
#      server lifetime, which would drown the decode signal in load-time and
#      idle-poll noise.
#   5. `/v1/stats` is read immediately before nsys start and immediately after
#      nsys stop; the counter DELTA over that window is what is reported (a
#      cumulative counter read once is not comparable across arms whose
#      warmup differed).
#   6. `nsys export --type sqlite`, then `nsys stats --report cuda_gpu_trace
#      --format csv` on the resulting .nsys-rep, then
#      instrument_analyze.py's cluster/summarize is used to turn the raw
#      kernel trace into per-step counts + one representative histogram.
#   7. Everything is written to results/instrument-<label>-<before|after>.txt
#      (JSON body, see WRITE_RESULT below) plus the raw .nsys-rep and the CSV
#      kept alongside for a human to open in the Nsight Systems GUI.
#
# nsys options used, and where they come from (this host, Nsight Systems
# 2025.6.3.541-256337736014v0, verified by reading `nsys profile --help`,
# `nsys launch --help`, `nsys start --help`, `nsys export --help`,
# `nsys stats --help` directly -- not from memory):
#   --trace=cuda,nvtx           only the APIs this instrument reads.
#   --cuda-graph-trace=node     "If 'node' is selected, node activities will be
#                                collected... This may cause significant
#                                runtime overhead." -- the whole point: decode
#                                runs from a captured graph, so the default
#                                ('graph', one opaque launch event per replay)
#                                would report one "kernel" per decode step
#                                regardless of what is inside it.
#   nsys launch / start / stop  a --duration/--delay window can't be aimed at
#                                "exactly the decode window of one specific
#                                request sent at an unpredictable time after
#                                readiness"; the launch-then-start/stop session
#                                protocol can, because start/stop are ordinary
#                                commands this script issues exactly when it
#                                knows the request is about to fire / has
#                                finished (`nsys start --help`: "Delays
#                                collection indefinitely until the nsys start
#                                command is executed for this session" under
#                                --delay -1 / -c none semantics for launch).
#   nsys export --type sqlite   `nsys stats` accepts a .nsys-rep directly too,
#                                but exporting once and reusing the sqlite for
#                                more than one report (cuda_gpu_trace here,
#                                nothing else yet but cheap to add) avoids
#                                re-parsing the raw report per report type.
#   nsys stats --report
#     cuda_gpu_trace --format csv   confirmed columns (read directly from
#                                `.../reports/cuda_gpu_trace.py`'s
#                                `query_stub` on this host): Start:ts_ns,
#                                Duration:dur_ns, CorrId, Grd/BlkX/Y/Z,
#                                Reg/Trd, St/DymSMem, Bytes, Throughput,
#                                Src/DstMemKd, Device, Ctx, GreenCtx, Strm,
#                                Name. `instrument_analyze.py` matches on
#                                Start/Duration/Name only.
#
# What HAS been verified live on this host (no GPU, no torch, no server --
# ordinary shell commands, cleaned up after): `nsys launch --session=NAME ...
# -- sleep N &` really does register a "Launched" session (`nsys sessions
# list`); `nsys start --session=NAME --output=...nsys-rep` really does begin
# and later produce that exact .nsys-rep file; `nsys export --type=sqlite`
# and `nsys stats --report=cuda_gpu_trace --format=csv --output=BASE` really
# do write `BASE_cuda_gpu_trace.csv` (confirmed the naming rule from
# `nsys stats --help`'s "<basename>_<analysis&args>.<output_format>" against
# a real file, not just the docs); a capture with no CUDA activity produces a
# 0-byte CSV and an nsys "SKIPPED: ... does not contain GPU trace data"
# message rather than an error -- both `instrument.sh` and
# `instrument_analyze.py` now check for and explain that case specifically,
# because it is the first thing a first real run against a CPU-only sleep
# would otherwise have hit silently.
#
# What is STILL UNVERIFIED until the owner's first run against the real
# server (all of the above used `sleep`, never a CUDA kernel):
#   - Whether `nsys launch`'s injection survives serve-default.sh's exec chain
#     (a bash script that execs into the actual python server process) and
#     still traces CUDA kernels launched deep inside a systemd-run --user
#     transient unit's process tree (cgroup interactions, injection surviving
#     systemd's own exec) -- verified only for a plain shell + `sleep`.
#   - The actual magnitude of intra-step vs inter-step timing gaps on real
#     hardware, which instrument_analyze.py's clustering heuristic assumes
#     (documented and unit-tested on synthetic data using the plan's own
#     record number, 76.9 tok/s => ~13 ms/token; never run against a real
#     capture, which is the only way to confirm the 10x-median/50us-floor
#     threshold in `_choose_gap_threshold` is right for this GPU/kernel mix).
#   - Whether `--cuda-graph-trace=node`'s overhead ("significant runtime
#     overhead", nsys's own words) distorts the BYTE counters (mirror_stats /
#     decode counters). Those are read from /v1/stats independent of nsys, so
#     they should not be affected by nsys's own overhead, but this has not
#     been measured together.
#   - `--moe-collect-stats` is passed to every arm (FREETOKEN_EXTRA_ARGS
#     below) so the decode_* byte counters are populated; if they still read
#     0 on a real run, check that this flag actually reached the server (a
#     serve.env or FT_EXTRA-equivalent override could shadow it -- see
#     measure.sh's own comment about FREETOKEN_MIRROR_EXPERT_RAM doing exactly
#     that to an earlier sweep).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO_HERE="$PWD"   # this worktree, only used to find instrument_analyze.py
HERE="$REPO_HERE/tasks/exclusive-expert-ram"
OUT="$HERE/results"
mkdir -p "$OUT"
ANALYZE="$HERE/instrument_analyze.py"
PY3=python3

die() { echo "instrument.sh: $*" >&2; exit 1; }

usage() {
  cat >&2 <<'USAGE'
usage:
  instrument.sh run --before DIR --after DIR --label NAME [--port N] [--size N] [--decode-tokens N]
  instrument.sh compare BEFORE_RESULT.txt AFTER_RESULT.txt
USAGE
  exit 2
}

# ---------------------------------------------------------------------------
CMD="${1:-}"; shift || true
case "$CMD" in
  compare)
    [ $# -eq 2 ] || usage
    exec "$PY3" "$ANALYZE" compare "$1" "$2"
    ;;
  run) : ;;
  *) usage ;;
esac

BEFORE_DIR=""; AFTER_DIR=""; LABEL=""; PORT=1920; SIZE=8000; DECODE_TOKENS=128
# The arm's residency is pinned like measure.sh does (the host serve.env carries
# FREETOKEN_MIRROR_* and would otherwise decide it): default = the record config,
# auto pool (-1) with reserve 256 (2E), ratio 1.00. --rows 0 = whole model in RAM.
ROWS=-1; RESERVE=256; RATIO=1.00
NAME="nemotron-3.5-lightning"
while [ $# -gt 0 ]; do
  case "$1" in
    --before) BEFORE_DIR="$2"; shift 2 ;;
    --after) AFTER_DIR="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --decode-tokens) DECODE_TOKENS="$2"; shift 2 ;;
    --model-name) NAME="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --reserve) RESERVE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$BEFORE_DIR" ] && [ -n "$AFTER_DIR" ] && [ -n "$LABEL" ] || usage
[ -d "$BEFORE_DIR/.git" ] || [ -f "$BEFORE_DIR/.git" ] || die "--before $BEFORE_DIR is not a git worktree"
[ -d "$AFTER_DIR/.git" ] || [ -f "$AFTER_DIR/.git" ] || die "--after $AFTER_DIR is not a git worktree"
[ -x "$(command -v nsys)" ] || die "nsys not found on PATH"

FT_VENV_DEFAULT="/home/lucas/ai/FreeToken/.venv"
LOCK=/home/lucas/.cache/freetoken/gpu-host.lock
mkdir -p "$(dirname "$LOCK")"

preflight() {
  [ "$PORT" = "1919" ] && die "refusing: 1919 is the owner's production unit"
  if [ -n "$(ss -H -ltn "sport = :$PORT" 2>/dev/null)" ]; then
    die "refusing: port $PORT is occupied"
  fi
  systemctl is-active --quiet freetoken-serve 2>/dev/null && die "the production unit freetoken-serve is running: ft-down first"
  # only a real pytest process, not a worker's shell queued on the host lock
  pgrep -f '^[^ ]*python[0-9.]* -m pytest' >/dev/null && die "a pytest is running (a worker?): wait for it"
  if systemctl --user is-active --quiet piro-board-embedder 2>/dev/null; then
    echo "stopping piro-board-embedder for the measurement (it is NOT restarted afterwards)"
    systemctl --user stop piro-board-embedder
    sleep 5
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = "0" ] || { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; die "GPU holds ${used} MiB, must read 0"; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || die "MemAvailable ${avail} GiB < 22"
}

# One arm: $1=worktree dir, $2=arm suffix ("before"/"after")
run_arm() {
  local WT="$1" SUFFIX="$2"
  local UNIT="ft-instrument-$LABEL-$SUFFIX"
  local SESSION="ft-instrument-$LABEL-$SUFFIX"
  local REP="$OUT/instrument-$LABEL-$SUFFIX"   # .nsys-rep, .sqlite, .csv share this stem
  local COMMIT
  COMMIT="$(git -C "$WT" rev-parse --short HEAD)$(git -C "$WT" diff --quiet HEAD -- python || echo -dirty)"

  preflight
  systemctl --user reset-failed "$UNIT" 2>/dev/null || true
  local ARMENV="$OUT/instrument-$LABEL-$SUFFIX.env"
  grep -vE '^[[:space:]]*export[[:space:]]+(FREETOKEN_MIRROR_EXPERT_RAM|FREETOKEN_MIRROR_HOST_ROWS|FREETOKEN_MEMORY_RATIO|FREETOKEN_MIRROR_RESERVE_ROWS|FREETOKEN_MIRROR_TIEBREAK)=' \
    "$HOME/.config/freetoken/serve.env" > "$ARMENV" 2>/dev/null || : > "$ARMENV"
  {
    echo "export FREETOKEN_MEMORY_RATIO=$RATIO"
    if [ "$ROWS" = "0" ]; then
      echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    else
      echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"; echo "export FREETOKEN_MIRROR_HOST_ROWS=$ROWS"
      echo "export FREETOKEN_MIRROR_RESERVE_ROWS=$RESERVE"
    fi
  } >> "$ARMENV"

  echo "[$SUFFIX] starting worktree=$WT commit=$COMMIT port=$PORT"
  # nsys launch primes CUDA/NVTX injection into the process tree that
  # serve-default.sh eventually execs into. Deferred start (-y/--delay is
  # overridden by the launch/start protocol): no data is recorded until this
  # script calls `nsys start --session=$SESSION` below, so the model-load and
  # bank-build period is never captured.
  systemd-run --user --unit="$UNIT" --property=OOMScoreAdjust=1000 \
    --setenv=PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" \
    --setenv=FREETOKEN_PORT="$PORT" \
    --setenv=FREETOKEN_HOST_ENV="$ARMENV" \
    --setenv=UV_PROJECT_ENVIRONMENT="$FT_VENV_DEFAULT" \
    --setenv=UV_NO_SYNC=1 \
    --setenv=PYTHONPATH="$WT/python" \
    --setenv=FREETOKEN_EXTRA_ARGS="--moe-collect-stats" \
    -- nsys launch --session="$SESSION" --trace=cuda,nvtx --cuda-graph-trace=node \
       -- "$WT/scripts/serve-default.sh" >/dev/null

  trap 'systemctl --user stop "'"$UNIT"'" >/dev/null 2>&1 || true' RETURN

  local started deadline
  started=$(date +%s); deadline=$(( started + 900 ))
  until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
              -H 'Content-Type: application/json' \
              -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
              "http://127.0.0.1:$PORT/v1/chat/completions")" = "200" ]; do
    if ! systemctl --user is-active --quiet "$UNIT"; then
      journalctl --user -u "$UNIT" -n 40 --no-pager >&2 || true
      die "[$SUFFIX] server failed to start"
    fi
    [ "$(date +%s)" -gt "$deadline" ] && die "[$SUFFIX] timed out waiting for readiness"
    sleep 5
  done
  echo "[$SUFFIX] ready after $(( $(date +%s) - started ))s"

  # Warm-up: two throwaway small decodes, exactly measure.sh's convention, so
  # the captured window is steady-state, not the post-load slow period.
  FREETOKEN_URL="http://127.0.0.1:$PORT" FREETOKEN_MODEL_NAME="$NAME" \
    PROBE_GEN_TOKENS=64 "$REPO_HERE/scripts/probe_decode.py" "$SIZE" >/dev/null 2>&1 || true
  sleep 20
  FREETOKEN_URL="http://127.0.0.1:$PORT" FREETOKEN_MODEL_NAME="$NAME" \
    PROBE_GEN_TOKENS=64 "$REPO_HERE/scripts/probe_decode.py" "$SIZE" >/dev/null 2>&1 || true

  local stats_before stats_after
  stats_before="$(curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/stats" 2>/dev/null || echo '{}')"

  echo "[$SUFFIX] recording: nsys start"
  # -o/--output belongs to `nsys start`, not `nsys stop` (verified: `nsys stop
  # --help` offers only --session/--keep; `nsys start --help` has -o/--output,
  # default 'report%n'). Give it the .nsys-rep path directly so `nsys stop`
  # needs no output argument at all.
  nsys start --session="$SESSION" --output="$REP.nsys-rep"

  echo "[$SUFFIX] firing the measured decode: ${SIZE} prompt tokens, ${DECODE_TOKENS} decode tokens"
  FREETOKEN_URL="http://127.0.0.1:$PORT" FREETOKEN_MODEL_NAME="$NAME" \
    PROBE_GEN_TOKENS="$DECODE_TOKENS" PROBE_PASSES=1 \
    "$REPO_HERE/scripts/probe_decode.py" "$SIZE" > "$OUT/instrument-$LABEL-$SUFFIX-probe.jsonl" || \
    echo "[$SUFFIX] probe reported an error; recording the capture anyway" >&2

  echo "[$SUFFIX] recording: nsys stop"
  nsys stop --session="$SESSION"

  stats_after="$(curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/stats" 2>/dev/null || echo '{}')"

  echo "[$SUFFIX] stopping server"
  systemctl --user stop "$UNIT" || true
  trap - RETURN

  echo "[$SUFFIX] exporting + analysing"
  nsys export --type=sqlite --force-overwrite=true --output="$REP.sqlite" "$REP.nsys-rep"
  nsys stats --report=cuda_gpu_trace --format=csv --output="$REP" "$REP.sqlite" >/dev/null
  # nsys names the per-report csv "<output>_<report>.csv"
  local CSV="${REP}_cuda_gpu_trace.csv"
  [ -f "$CSV" ] || die "[$SUFFIX] expected $CSV from nsys stats, not found"
  # Verified live on this host: a capture with no CUDA activity makes `nsys
  # stats` print "SKIPPED: ... does not contain GPU trace data" and still
  # write a 0-byte CSV -- not a parse failure, a sign the capture window
  # caught nothing (see instrument_analyze.py's parse_cuda_gpu_trace_csv).
  [ -s "$CSV" ] || die "[$SUFFIX] $CSV is empty -- nsys likely reported 'SKIPPED: ... does not contain GPU trace data'; the capture window caught no CUDA activity"

  local KERNEL_JSON
  KERNEL_JSON="$("$PY3" "$ANALYZE" cluster "$CSV" --drop-first 1)"

  "$PY3" - "$LABEL" "$SUFFIX" "$COMMIT" "$stats_before" "$stats_after" "$KERNEL_JSON" \
    "$SIZE" "$DECODE_TOKENS" "$OUT/instrument-$LABEL-$SUFFIX.txt" <<'PY'
import json, sys

label, suffix, commit, stats_before_s, stats_after_s, kernel_json_s, size, decode_tokens, out_path = sys.argv[1:10]

def moe_block(stats_s):
    try:
        return (json.loads(stats_s).get("scheduler") or {}).get("moe") or {}
    except Exception:
        return {}

moe_before = moe_block(stats_before_s)
moe_after = moe_block(stats_after_s)

def delta(key_path, default=0):
    def get(d):
        cur = d
        for k in key_path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, default)
        return cur if isinstance(cur, (int, float)) else default
    return get(moe_after) - get(moe_before)

byte_counters = {
    # mirror_stats() fields (moe/mirror_stats.py): row counts, not literal
    # bytes -- multiply by the checkpoint's row_bytes (S5a's
    # nvfp4_expert_row_layout) for a byte figure; left as counts here because
    # this instrument runs on an arbitrary commit pair and row_bytes is a
    # model/format fact this script has no business hard-coding.
    "mirror_swaps": delta(["mirror", "swaps"]),
    "mirror_free_evictions": delta(["mirror", "free_evictions"]),
    "mirror_writebacks": delta(["mirror", "writebacks"]),
    "mirror_coverage_faults": delta(["mirror", "coverage_faults"]),
    "mirror_starved_writebacks": delta(["mirror", "starved_writebacks"]),
    "mirror_retained_rows": delta(["mirror", "retained_rows"]),
    "mirror_buffer_free_evictions": delta(["mirror", "buffer_free_evictions"]),
    # decode_stat_totals() raw cumulative counters (offload_cache.py), gated
    # behind --moe-collect-stats (serve-default.sh's profile does not set it
    # by default -- if these all read 0 on a real run, that is why: rerun with
    # FREETOKEN_MOE_COLLECT_STATS=1 or the server's --moe-collect-stats flag).
    "decode_missing": delta(["decode", "missing"]),
    "decode_active": delta(["decode", "active"]),
    "decode_fetched": delta(["decode", "fetched"]),
    "decode_calls": delta(["decode", "calls"]),
}

doc = {
    "label": label,
    "arm": suffix,
    "commit": commit,
    "prompt_tokens": int(size),
    "decode_tokens": int(decode_tokens),
    "kernel_launches": json.loads(kernel_json_s),
    "byte_counters": byte_counters,
    "stats_before_raw": moe_before,
    "stats_after_raw": moe_after,
}
with open(out_path, "w") as f:
    json.dump(doc, f, indent=2)
print(json.dumps(doc, indent=2))
PY
  echo "[$SUFFIX] wrote $OUT/instrument-$LABEL-$SUFFIX.txt"
}

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "waiting for a worker's test run / another arm to finish (host lock held) ..."
  flock 9
fi

run_arm "$BEFORE_DIR" before
run_arm "$AFTER_DIR" after

echo
echo "=== compare ==="
"$PY3" "$ANALYZE" compare "$OUT/instrument-$LABEL-before.txt" "$OUT/instrument-$LABEL-after.txt"
