#!/usr/bin/env bash
# One arm of the host-RAM x decode-performance sweep.
#
#   tasks/exclusive-expert-ram/measure.sh ARM
#
# Environment:
#   FT_MODEL   model directory                (default: Nemotron 3.5 Lightning NVFP4)
#   FT_ROWS    mirror rows: unset/0 = baseline (whole model in RAM), -1 = auto, N = N rows
#   FT_PORT    spare port                     (default 1920 -- NEVER 1919, the owner's unit)
#   FT_SIZES   probe prompt sizes             (default "8000 32000 80000")
#   FT_EXTRA   extra ft serve flags, verbatim (last flag wins; e.g. a lower seq-len cap)
#   FT_NAME    served model name for the probe (default nemotron-3.5-lightning)
#
# Every arm runs ALONE: a 28 GiB host cannot hold two of these, and the numbers are
# worthless if a second model is resident. The server runs as a transient --user unit so
# an agent shell dying cannot take it down mid-measurement.
#
# Output: one row appended to tasks/exclusive-expert-ram/results/sweep.tsv, plus the raw
# probe JSON and /v1/stats of the arm beside it.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO="$PWD"

ARM="${1:?usage: measure.sh ARM}"
MODEL="${FT_MODEL:-$HOME/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
ROWS="${FT_ROWS:-0}"
PORT="${FT_PORT:-1920}"
SIZES="${FT_SIZES:-8000 32000 80000}"
NAME="${FT_NAME:-nemotron-3.5-lightning}"
RATIO="${FT_RATIO:-0.91}"   # same for every arm; a sweep that moves two knobs measures neither
UNIT="ft-measure-$ARM"
OUT="$REPO/tasks/exclusive-expert-ram/results"
mkdir -p "$OUT"

[ "$PORT" = "1919" ] && { echo "refusing: 1919 is the owner's production unit" >&2; exit 1; }
if [ -n "$(ss -H -ltn "sport = :$PORT" 2>/dev/null)" ]; then
  echo "refusing: port $PORT is occupied" >&2; exit 1
fi
if systemctl is-active --quiet freetoken-serve 2>/dev/null; then
  echo "refusing: the production unit is running; this arm needs the host RAM" >&2; exit 1
fi
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
[ "$avail" -lt 22 ] && { echo "refusing: only ${avail} GiB available, need >= 22" >&2; exit 1; }

# serve-default.sh SOURCES $HOME/.config/freetoken/serve.env after the inherited
# environment, so that file wins over --setenv. It currently carries
# FREETOKEN_MIRROR_EXPERT_RAM=1, which silently turned the first "baseline" arm
# into a mirror arm (59902 swaps in its /v1/stats). Every arm therefore gets its
# own env file, built from the host's minus the knobs this sweep controls, so the
# arm is what it says it is and the host's file is never edited.
ARMENV="$OUT/$ARM.env"
grep -vE '^[[:space:]]*export[[:space:]]+(FREETOKEN_MIRROR_EXPERT_RAM|FREETOKEN_MIRROR_HOST_ROWS|FREETOKEN_MODEL|FREETOKEN_MEMORY_RATIO)=' \
  "$HOME/.config/freetoken/serve.env" > "$ARMENV" 2>/dev/null || : > "$ARMENV"
{
  echo "export FREETOKEN_MODEL=$MODEL"
  echo "export FREETOKEN_MEMORY_RATIO=$RATIO"
  if [ "$ROWS" = "0" ]; then
    echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"
    echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
  else
    echo "export FREETOKEN_MIRROR_EXPERT_RAM=1"
    echo "export FREETOKEN_MIRROR_HOST_ROWS=$ROWS"
  fi
} >> "$ARMENV"

systemctl --user reset-failed "$UNIT" 2>/dev/null || true

# The RAM number, decided after two wrong ones (see results/README.md):
#
#   memory.current counts page cache, and the arms do not fill it the same way,
#   so two identical baselines differed by 2.8 GiB and bigger pools measured
#   smaller.
#   memory.stat's `anon` misses the baseline ENTIRELY: its expert banks are
#   mmap'd from the safetensors and then cudaHostRegister'd, so 15.4 GiB of
#   unreclaimable residency is charged to `file` (measured: anon 3.02 GiB,
#   file 19.02 GiB).
#
# What the owner actually feels on a 28 GiB host is how much MemAvailable the
# profile takes away. Clean page cache stays available and pinned pages do not,
# which is exactly the distinction both cgroup counters failed to make -- and it
# does not care whether a profile pins anonymous or file-backed pages.
sync
sleep 5
avail_before=$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo)

echo "[$ARM] starting: model=$(basename "$MODEL") rows=$ROWS ratio=$RATIO port=$PORT"
systemd-run --user --unit="$UNIT" --property=OOMScoreAdjust=1000 \
  --setenv=PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" \
  --setenv=FREETOKEN_HOST_ENV="$ARMENV" \
  --setenv=FREETOKEN_PORT="$PORT" \
  --setenv=FREETOKEN_EXTRA_ARGS="${FT_EXTRA:-}" \
  "$REPO/scripts/serve-default.sh" >/dev/null

# From here on the unit must never outlive this script: a leaked server holds the GPU and
# ~20 GiB of host RAM, which blocks every later arm.
trap 'systemctl --user stop "$UNIT" >/dev/null 2>&1 || true' EXIT

# Readiness is NOT /v1/models: the frontend answers that within seconds while the engine is
# still building expert banks, and the first real request then returns 503 (measured). The
# ground truth is a real completion succeeding. The load is 1-3 min; allow 15.
started=$(date +%s)
deadline=$(( started + 900 ))
until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            "http://127.0.0.1:$PORT/v1/chat/completions")" = "200" ]; do
  if ! systemctl --user is-active --quiet "$UNIT"; then
    echo "[$ARM] FAILED to start; last log:" >&2
    journalctl --user -u "$UNIT" -n 40 --no-pager >&2 || true
    exit 1
  fi
  [ "$(date +%s)" -gt "$deadline" ] && { echo "[$ARM] timed out waiting for readiness" >&2; exit 1; }
  sleep 5
done
echo "[$ARM] ready after $(( $(date +%s) - started ))s"

# What the arm actually built. Without this the pool geometry behind a row of
# the table is a guess, and the RAM column is the whole point of the table.
journalctl --user -u "$UNIT" --no-pager 2>/dev/null \
  | grep -E "Mirror (expert RAM|pool)|expert arena|memory ratio|KV cache" \
  > "$OUT/$ARM-geometry.txt" || true
sed -n '1,12p' "$OUT/$ARM-geometry.txt" 2>/dev/null || true

# Warm up BEFORE measuring: the first requests after a start run at a fraction of speed
# until the expert-bank build settles (~3 min per CLAUDE.md). A cold number compared
# against a warm one is the exact distortion this sweep exists to avoid.
FREETOKEN_URL="http://127.0.0.1:$PORT" FREETOKEN_MODEL_NAME="$NAME" \
  PROBE_GEN_TOKENS=64 "$REPO/scripts/probe_decode.py" 8000 >/dev/null 2>&1 || true
sleep 20
FREETOKEN_URL="http://127.0.0.1:$PORT" FREETOKEN_MODEL_NAME="$NAME" \
  PROBE_GEN_TOKENS=64 "$REPO/scripts/probe_decode.py" 8000 >/dev/null 2>&1 || true

echo "[$ARM] measuring"
if ! FREETOKEN_URL="http://127.0.0.1:$PORT" FREETOKEN_MODEL_NAME="$NAME" \
     PROBE_GEN_TOKENS=128 "$REPO/scripts/probe_decode.py" $SIZES > "$OUT/$ARM-probe.jsonl"; then
  echo "[$ARM] probe failed; recording the arm anyway" >&2
fi
cat "$OUT/$ARM-probe.jsonl"

curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/stats" > "$OUT/$ARM-stats.json" 2>/dev/null || true

# memory.current is NOT the number this sweep is about: it counts page cache,
# and the arms do not populate it the same way -- the baseline loads its expert
# banks through ordinary reads while the mirror pool reads with O_DIRECT. Two
# identical baseline arms measured 23.77 and 20.93 GiB that way, and the mirror
# arms came out LOWER the BIGGER the pinned pool got. The number that answers
# "how much host RAM does this profile hold" is anonymous memory: the pinned
# banks are page-locked anonymous pages, so they land in anon/unevictable and
# page cache does not.
sync
sleep 5
avail_after=$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo)

CG=$(systemctl --user show -p ControlGroup --value "$UNIT")
CGDIR="/sys/fs/cgroup$CG"
mem_now=$(awk '/^anon /{print $2}' "$CGDIR/memory.stat" 2>/dev/null || echo 0)
mem_file=$(awk '/^file /{print $2}' "$CGDIR/memory.stat" 2>/dev/null || echo 0)
mem_unevict=$(awk '/^unevictable /{print $2}' "$CGDIR/memory.stat" 2>/dev/null || echo 0)
mem_peak=$(cat "$CGDIR/memory.peak" 2>/dev/null || echo 0)
mem_cur=$(cat "$CGDIR/memory.current" 2>/dev/null || echo 0)
# Attribution for the MemAvailable delta: summed over every process of the arm,
# because the banks live in the scheduler worker, not in the frontend.
rss_kb=0; lck_kb=0
for pid in $(cat "$CGDIR/cgroup.procs" 2>/dev/null); do
  r=$(awk '/^Rss:/{print $2}'    "/proc/$pid/smaps_rollup" 2>/dev/null || echo 0)
  l=$(awk '/^Locked:/{print $2}' "/proc/$pid/smaps_rollup" 2>/dev/null || echo 0)
  rss_kb=$(( rss_kb + ${r:-0} )); lck_kb=$(( lck_kb + ${l:-0} ))
done
gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

RATIO_USED="$RATIO" python3 - "$ARM" "$MODEL" "$ROWS" "$mem_now" "$mem_peak" "$gpu_used" \
  "$OUT/$ARM-probe.jsonl" "$OUT/$ARM-stats.json" "$OUT/sweep.tsv" \
  "$mem_file" "$mem_unevict" "$mem_cur" "$avail_before" "$avail_after" \
  "$rss_kb" "$lck_kb" <<'PY'
import json, os, sys
arm, model, rows, anon, peak, gpu, probe, stats, tsv = sys.argv[1:10]
mfile, unevict, cur, av_before, av_after, rss_kb, lck_kb = sys.argv[10:17]
gib = lambda b: round(int(b) / 2**30, 2)
rec = {"arm": arm, "model": os.path.basename(model), "rows": rows,
       "ratio": os.environ.get("RATIO_USED", ""),
       # THE metric: host RAM this profile takes away from everything else.
       "ram_gib": gib(int(av_before) - int(av_after)),
       # Attribution, so a surprising ram_gib can be explained rather than
       # guessed at. rss/locked are summed over the arm's processes; anon/file
       # are the cgroup's split (the baseline's banks are file-backed mmap).
       "rss_gib": gib(int(rss_kb) * 1024), "locked_gib": gib(int(lck_kb) * 1024),
       "anon_gib": gib(anon), "file_gib": gib(mfile),
       "unevict_gib": gib(unevict), "current_gib": gib(cur),
       "peak_current_gib": gib(peak), "gpu_mib": gpu}
try:
    for line in open(probe):
        d = json.loads(line)
        rec[f"decode_{d['prompt_tokens'] // 1000}k"] = round(d.get("decode_tok_s") or 0, 1)
        rec[f"ttft_{d['prompt_tokens'] // 1000}k"] = round(d.get("ttft_s") or 0, 2)
except Exception as exc:                      # a failed probe must not erase the arm
    rec["probe_error"] = str(exc)
try:
    m = (json.load(open(stats)).get("scheduler", {}).get("moe", {}) or {}).get("mirror")
    if m:
        rec["coverage_faults"] = m.get("coverage_faults")
        rec["starved"] = m.get("starved_writebacks")
        rec["free_evict_rate"] = round(m.get("free_eviction_rate", 0), 3)
except Exception:
    pass
new = not os.path.exists(tsv)
keys = list(rec)
with open(tsv, "a") as f:
    if new:
        f.write("\t".join(keys) + "\n")
    f.write("\t".join(str(rec[k]) for k in keys) + "\n")
print(json.dumps(rec, indent=2))
PY

echo "[$ARM] stopping"
systemctl --user stop "$UNIT" || true
