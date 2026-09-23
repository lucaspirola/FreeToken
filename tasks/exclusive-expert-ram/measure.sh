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
#   FT_POST    command run while the server is STILL UP (recall.py, needles.py):
#              sees FREETOKEN_URL/FREETOKEN_MODEL_NAME, output -> $ARM-post.txt
#   FT_POST_TIMEOUT  seconds for FT_POST (default 3600)
#   FT_VENV    serve THIS tree's python/ through an existing venv instead of letting
#              `uv run` build one here (a fresh worktree has no .venv, and a sync is
#              forbidden): sets UV_PROJECT_ENVIRONMENT=$FT_VENV, UV_NO_SYNC=1 and puts
#              $REPO/python first on PYTHONPATH, so the code of record is this tree's.
#   FT_ENVS    extra NAME=value exports for this arm only (A/B knobs)
#   FT_KV      KV lane by name, recorded with the number so it is attributable:
#              q8q8 (q8_0 K + q8_0 V, the default lane), q8q6, q6q5. Anything
#              else is passed through verbatim as flags. Only these asymmetric
#              pairs are validated (server/args.py), and only on the triton
#              attention backend.
#
# Every arm runs ALONE: a 28 GiB host cannot hold two of these, and the numbers are
# worthless if a second model is resident. The server runs as a transient --user unit so
# an agent shell dying cannot take it down mid-measurement.
#
# Output: tasks/exclusive-expert-ram/results/$ARM-record.json, the raw probe JSON, the
# arm's /v1/stats and its pool geometry from the log. sweep.tsv is then REBUILT from every
# record by table.py -- never appended to here (see that file for why).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO="$PWD"

ARM="${1:?usage: measure.sh ARM}"
MODEL="${FT_MODEL:-$HOME/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
ROWS="${FT_ROWS:-0}"
PORT="${FT_PORT:-1920}"
SIZES="${FT_SIZES:-8000 32000 80000}"
NAME="${FT_NAME:-nemotron-3.5-lightning}"
RATIO="${FT_RATIO:-1.00}"   # same for every arm; a sweep that moves two knobs measures neither
# A KV lane is part of the identity of a number, not a detail of how it was
# taken: the same arm on q6/q5 is a different measurement, so the lane name
# goes into the record and the flags go in LAST (last flag wins over FT_EXTRA).
KV="${FT_KV:-}"
case "$KV" in
  ""|q8q8) KV_FLAGS="" ;;
  q8q6)    KV_FLAGS="--kv-cache-dtype-k q8_0 --kv-cache-dtype-v q6_0" ;;
  q6q5)    KV_FLAGS="--kv-cache-dtype-k q6_0 --kv-cache-dtype-v q5_0" ;;
  *)       KV_FLAGS="$KV" ;;
esac
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
grep -vE '^[[:space:]]*export[[:space:]]+(FREETOKEN_MIRROR_EXPERT_RAM|FREETOKEN_MIRROR_HOST_ROWS|FREETOKEN_MODEL|FREETOKEN_MEMORY_RATIO|FREETOKEN_MIRROR_RESERVE_ROWS|FREETOKEN_MIRROR_TIEBREAK)=' \
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
  # Lever 4. Unset => the pool's own default (3 * num_experts), so an arm that
  # does not ask for a reserve is byte-identical to before this line existed.
  # Stripped from the host's file above like every other knob this sweep owns:
  # without that, a reserve left in serve.env would silently apply to every arm
  # and three "3E/2E/E" arms would all be the same reserve wearing three names.
  if [ -n "${FT_RESERVE:-}" ]; then
    echo "export FREETOKEN_MIRROR_RESERVE_ROWS=$FT_RESERVE"
  fi
  # Lever 2. FT_TIEBREAK=0 reproduces pre-lever-2 victim selection with the
  # pool still attached, which is the only honest A/B for what the tie-break
  # buys: detaching the pool instead would change three things at once.
  if [ -n "${FT_TIEBREAK:-}" ]; then
    echo "export FREETOKEN_MIRROR_TIEBREAK=$FT_TIEBREAK"
  fi
  # FT_ENVS: extra "NAME=value" words exported into THIS arm only (e.g. an
  # A/B that forces the pre-fix extend tile via FREETOKEN_EXTEND_*); recorded in
  # the arm's .env beside its record so the number stays attributable.
  for kv in ${FT_ENVS:-}; do echo "export $kv"; done
  if [ -n "${FT_VENV:-}" ]; then
    echo "export UV_PROJECT_ENVIRONMENT=$FT_VENV"
    echo "export UV_NO_SYNC=1"
    echo "export PYTHONPATH=$REPO/python"
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

echo "[$ARM] starting: model=$(basename "$MODEL") rows=$ROWS ratio=$RATIO port=$PORT code=$REPO@$(git -C "$REPO" rev-parse --short HEAD) venv=${FT_VENV:-uv-default}"
systemd-run --user --unit="$UNIT" --property=OOMScoreAdjust=1000 \
  --setenv=PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" \
  --setenv=FREETOKEN_HOST_ENV="$ARMENV" \
  --setenv=FREETOKEN_PORT="$PORT" \
  --setenv=FREETOKEN_EXTRA_ARGS="${FT_EXTRA:-} $KV_FLAGS" \
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

# Host RAM has to be sampled at a DEFINED point, or arms are not comparable:
# a sample taken after an 80K request includes whatever the KV arena grew into,
# and on WSL2 that can be host-backed. Two points are recorded -- here, with
# the model resident and nothing served yet (this is the model's residency,
# the number the RAM knob is supposed to move), and again after the probe.
avail_ready=$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo)
CG_EARLY=$(systemctl --user show -p ControlGroup --value "$UNIT")
# RSS at readiness, plus the /dev/zero slice of it. Both are needed: pinned
# memory obtained through torch maps from /dev/zero (which is also why the
# cgroup charges it to `file` and `free` calls it `shared`), while memory
# pinned with cudaHostRegister over an ordinary allocation stays anonymous.
# A metric that only counted /dev/zero would read a profile as free the moment
# it stopped using torch's allocator.
read -r rss_ready pinned_ready <<<"$(python3 - "/sys/fs/cgroup$CG_EARLY/cgroup.procs" <<'PYREADY'
import collections, sys
tot = collections.Counter(); path = None
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    try:
        smaps = open(f"/proc/{int(line)}/smaps").read().splitlines()
    except Exception:
        continue
    for ln in smaps:
        head = ln.split()
        if head and "-" in head[0] and ":" not in head[0]:
            path = head[5] if len(head) > 5 else "[anon]"
        elif ln.startswith("Rss:"):
            kb = int(ln.split()[1])
            if kb:
                tot[path or "[anon]"] += kb
print(sum(tot.values()) * 1024, tot.get("/dev/zero", 0) * 1024)
PYREADY
)"

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
     PROBE_GEN_TOKENS=128 PROBE_PASSES=2 \
     "$REPO/scripts/probe_decode.py" $SIZES > "$OUT/$ARM-probe.jsonl"; then
  echo "[$ARM] probe failed; recording the arm anyway" >&2
fi
cat "$OUT/$ARM-probe.jsonl"

# Correctness is the model's OUTPUT, not the fault counters: a stale free-row
# publish once served the WRONG experts with coverage_faults at 0 (plan.md).
# FT_POST runs while the server is still up -- the only moment recall.py or
# needles.py can reach it, since this script stops the unit on the way out.
# The command sees FREETOKEN_URL and FREETOKEN_MODEL_NAME already pointing at
# this arm, and its output is kept beside the arm's other artefacts. A failure
# here is recorded, never silent, but does not discard the arm: the numbers
# above were still measured, and a recall failure is itself a finding.
if [ -n "${FT_POST:-}" ]; then
  echo "[$ARM] post: $FT_POST"
  if ! FREETOKEN_URL="http://127.0.0.1:$PORT" FREETOKEN_MODEL_NAME="$NAME" \
       timeout "${FT_POST_TIMEOUT:-3600}" bash -c "$FT_POST" \
       > "$OUT/$ARM-post.txt" 2>&1; then
    echo "[$ARM] POST FAILED (exit $?) -- see $ARM-post.txt" >&2
  fi
  tail -20 "$OUT/$ARM-post.txt" || true
fi

curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/stats" > "$OUT/$ARM-stats.json" 2>/dev/null || true

# Neither cgroup number answers this sweep on its own, and both were tried:
#   memory.current counts page cache, which the arms do not populate the same
#   way (the baseline reads its banks normally, the pool reads O_DIRECT). Two
#   identical baselines read 23.77 and 20.93 GiB and bigger pools read SMALLER.
#   memory.stat's anon misses the banks entirely: CUDA pinned host memory maps
#   from /dev/zero, so the kernel charges it to file, not anon (a baseline
#   measured anon 3.02 GiB against file 19.02 GiB).
# The metric is the MemAvailable delta at readiness, taken above; these are kept
# only as attribution, so a surprising delta can be explained. See
# results/README.md for the full account of the three definitions.
sync
sleep 5
avail_after=$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo)

CG=$(systemctl --user show -p ControlGroup --value "$UNIT")
CGDIR="/sys/fs/cgroup$CG"
mem_now=$(awk '/^anon /{print $2}' "$CGDIR/memory.stat" 2>/dev/null || echo 0)
mem_file=$(awk '/^file /{print $2}' "$CGDIR/memory.stat" 2>/dev/null || echo 0)
mem_unevict=$(awk '/^unevictable /{print $2}' "$CGDIR/memory.stat" 2>/dev/null || echo 0)
mem_shmem=$(awk '/^shmem /{print $2}' "$CGDIR/memory.stat" 2>/dev/null || echo 0)
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

CODE_USED="$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- python || echo -dirty)" \
  VENV_USED="${FT_VENV:-uv-default}" RATIO_USED="$RATIO" KV_USED="${KV:-q8q8}" python3 - "$ARM" "$MODEL" "$ROWS" "$mem_now" "$mem_peak" "$gpu_used" \
  "$OUT/$ARM-probe.jsonl" "$OUT/$ARM-stats.json" "$OUT/$ARM-record.json" \
  "$mem_file" "$mem_unevict" "$mem_cur" "$avail_before" "$avail_after" \
  "$rss_kb" "$lck_kb" "$avail_ready" "$pinned_ready" "$rss_ready" "$mem_shmem" <<'PY'
import json, os, sys
arm, model, rows, anon, peak, gpu, probe, stats, out = sys.argv[1:10]
mfile, unevict, cur, av_before, av_after, rss_kb, lck_kb = sys.argv[10:17]
av_ready, pinned_ready, rss_ready, shmem = sys.argv[17:21]
gib = lambda b: round(int(b) / 2**30, 2)
rec = {"arm": arm, "model": os.path.basename(model), "rows": rows,
       # The code of record: a number not tied to a commit was "not measured in
       # this tree". "-dirty" marks uncommitted edits under python/.
       "commit": os.environ.get("CODE_USED", ""), "venv": os.environ.get("VENV_USED", ""),
       "ratio": os.environ.get("RATIO_USED", ""),
       # The KV lane this arm ran on. Without it a 256K Ornith row cannot be
       # told from the same arm on a cheaper lane.
       "kv": os.environ.get("KV_USED", ""),
       # THE metric: host RAM the MODEL holds, sampled with the server ready
       # and nothing served yet. This is what the mirror's capacity knob moves.
       "ram_gib": gib(int(av_before) - int(av_ready)),
       # The pinned region itself at that same point (CUDA pinned host memory
       # maps from /dev/zero on WSL2), which for the baseline is the expert
       # banks and for the mirror should be the pool.
       "rss_ready_gib": gib(rss_ready), "devzero_gib": gib(pinned_ready),
       # And after the probe, so KV growth is visible rather than folded in.
       "ram_after_80k_gib": gib(int(av_before) - int(av_after)),
       # Attribution, so a surprising ram_gib can be explained rather than
       # guessed at. rss/locked are summed over the arm's processes; anon/file
       # are the cgroup's split (the baseline's banks are file-backed mmap).
       "rss_gib": gib(int(rss_kb) * 1024), "locked_gib": gib(int(lck_kb) * 1024),
       "anon_gib": gib(anon), "file_gib": gib(mfile),
       "unevict_gib": gib(unevict), "shmem_gib": gib(shmem),
       "current_gib": gib(cur),
       "peak_current_gib": gib(peak), "gpu_mib": gpu}
try:
    # Two passes per size. Pass 1 pays the one-off growable-KV commit and the
    # decode-graph recapture that follows it; whether that stall lands before or
    # after the first token decides whether it is charged to TTFT or to decode,
    # which is how one arm read 127.7 tok/s and the next 46.4 for the same total
    # wall clock. Pass 2 needs no growth, so it is the steady number; pass 1 is
    # kept beside it as _p1 rather than discarded.
    for line in open(probe):
        d = json.loads(line)
        k = d["prompt_tokens"] // 1000
        sfx = "" if d.get("pass", 1) == 2 else "_p1"
        rec[f"decode_{k}k{sfx}"] = round(d.get("decode_tok_s") or 0, 1)
        rec[f"ttft_{k}k{sfx}"] = round(d.get("ttft_s") or 0, 2)
        rec[f"total_{k}k{sfx}"] = round(d.get("total_s") or 0, 1)
except Exception as exc:                      # a failed probe must not erase the arm
    rec["probe_error"] = str(exc)
try:
    m = (json.load(open(stats)).get("scheduler", {}).get("moe", {}) or {}).get("mirror")
    if m:
        rec["coverage_faults"] = m.get("coverage_faults")
        rec["starved"] = m.get("starved_writebacks")
        rec["swaps"] = m.get("swaps")
        rec["retained_rows"] = m.get("retained_rows")
        rec["free_evict_rate"] = round(m.get("free_eviction_rate", 0), 3)
        # Prefill-buffer invalidations, counted separately from decode swaps
        # since 2026-09-22: folding them into free_evict_rate put it above 1.0.
        rec["buffer_free_evictions"] = m.get("buffer_free_evictions", 0)
except Exception:
    pass
# One file per arm, and table.py builds the TSV from all of them. Appending
# straight to the TSV wrote the header from the first arm's keys and each row
# from its own, so baseline rows (no mirror counters) sat under mirror headers.
with open(out, "w") as f:
    json.dump(rec, f, indent=2)
print(json.dumps(rec, indent=2))
PY

python3 "$(dirname "${BASH_SOURCE[0]}")/table.py" "$OUT" || true

echo "[$ARM] stopping"
systemctl --user stop "$UNIT" || true
