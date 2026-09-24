#!/usr/bin/env bash
# checkpoint1.sh's four arms on a rented native-Linux box (Vast container: root, no systemd,
# memlock capped, CUDA 13.0 toolkit beside the image's 12.8). Same arms, same env, same probe,
# same needles/recall as checkpoint1.sh; only the owner-machine plumbing is replaced:
#   * server under measure.sh FT_LAUNCHER=nohup (setsid) -- its $ARM-server.log is the journal;
#   * GPU "idle" = <= 16 MiB (a Vast GPU never reads 0; headroom-box.sh precedent);
#   * host lock /root/gpu.lock instead of the owner's ~/.cache lock, no systemctl/embedder;
#   * R6(arm) reads the server's RLIMIT_MEMLOCK instead of the user manager's
#     DefaultLimitMEMLOCK: a Vast container caps memlock, so R6 is EXPECTED to fail here
#     (environment limit; banks are still cudaHostRegister-pinned).
# Each arm's artefacts are moved into results/<ck>-box/ as soon as the arm ends, so they can be
# mirrored off the box arm by arm. Run detached:
#   setsid nohup tasks/exclusive-expert-ram/checkpoint-box.sh ck4 > /root/ck4-run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO="$PWD"
HERE="$REPO/tasks/exclusive-expert-ram"
OUT="$HERE/results"                 # where measure.sh writes
CK="${1:-ck4}"
BOX="$OUT/$CK-box"                  # where this run's artefacts end up
mkdir -p "$BOX"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}" PATH="${CUDA_HOME:-/usr/local/cuda-13.0}/bin:$PATH"
export FT_LAUNCHER=nohup FT_VENV="${FT_VENV:-/root/venv}" FT_RATIO="${FT_RATIO:-1.00}" FT_PORT=1920
export FT_MODEL="${FT_MODEL:-/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
POST_TIMEOUT=10800
die() { echo "checkpoint-box: $*" >&2; exit 1; }

exec 9>/root/gpu.lock
flock -n 9 || { echo "waiting for /root/gpu.lock"; flock 9; }

preflight() {
  [ -z "$(git -C "$REPO" status --porcelain -- python)" ] || die "python/ has uncommitted edits"
  echo "code: $(git -C "$REPO" log --oneline -1)"
  pgrep -f 'ft serve' >/dev/null && die "an ft serve is running"
  pgrep -f '^[^ ]*python[0-9.]* -m pytest' >/dev/null && die "a pytest is running"
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" -le 16 ] || { nvidia-smi; die "GPU holds ${used} MiB, must be <= 16"; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || die "MemAvailable ${avail} GiB < 22"
  echo "preflight ok: GPU ${used} MiB, MemAvailable ${avail} GiB, no server, no pytest"
}

NEEDLES="NEEDLES_THINK_MAX_TOKENS=65536 NEEDLES_OUT=\"$OUT/\$ARM_NAME-needles.json\" python3 $HERE/needles.py 21000 120000; python3 $HERE/recall.py 21000 120000 240000"

collect() {  # the arm's files -> $BOX, plus the headroom/acceptance lines of its log
  local a=$1 j
  mv "$OUT/$a-server.log" "$BOX/$a-journal.txt" 2>/dev/null || true
  for f in "$OUT/$a"-*.* "$OUT/$a.env"; do
    [ -e "$f" ] || continue
    mv "$f" "$BOX/"
  done
  git -C "$REPO" checkout -q -- "tasks/exclusive-expert-ram/results/" 2>/dev/null || true  # tracked ck4-whole.env etc.
  j="$BOX/$a-journal.txt"
  { echo "captures=$(grep -c 'Start capturing CUDA graphs' "$j" || true)" \
         "kv_grows=$(grep -c 'KV grew' "$j" || true)" \
         "tracebacks=$(grep -c 'Traceback' "$j" || true)" \
         "ooms=$(grep -ci 'out of memory' "$j" || true)"
    grep -E "Prefill headroom|Expert arena parked|Growable-KV (ceiling|pre-commit|arena)|Committed growable|KV grew|capturing CUDA graphs|Free GPU memory|Free memory after|OutOfMemory|out of memory|Traceback|Prefill warmup|moe-cache-auto resolved" \
      "$j" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-400; } > "$BOX/$a-headroom.txt" || true
}

arm() {  # name, then env assignments
  local name=$1; shift
  preflight
  echo "=== arm $name $(date -u +%FT%TZ) ==="
  nvidia-smi dmon -s pucm -d 5 -o T > "$BOX/$name-dmon.txt" 2>&1 & local smi=$!
  env "$@" FT_POST_TIMEOUT=$POST_TIMEOUT "$HERE/measure.sh" "$name" || echo "$name measure exit $?"
  kill $smi 2>/dev/null || true
  collect "$name"
  echo "=== arm $name done $(date -u +%FT%TZ) ==="
  sleep 20
}

preflight
[ "${2:-}" = "preflight" ] && exit 0
echo "checkpoint label: $CK (box)"
{ nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
  uname -r; nproc; free -g; ulimit -l; "$FT_VENV/bin/python" -c 'import torch; print("torch", torch.__version__, torch.version.cuda)'
  "$CUDA_HOME/bin/nvcc" --version | tail -1; git -C "$REPO" log --oneline -1; } > "$BOX/$CK-box-env.txt" 2>&1

T0="FREETOKEN_PREFILL_TRANSIENT_MEASURE=0 FREETOKEN_PREFILL_TRANSIENT_MB=0"
for a in ${ARMS:-whole mirror-1m mirror whole-close}; do
  case "$a" in
    whole)       arm $CK-whole       FT_ROWS=0 FT_POST="${NEEDLES//\$ARM_NAME/$CK-whole}" ;;
    whole-1m)    arm $CK-whole-1m    FT_ROWS=0 FT_SIZES="8000 1000000" ;;  # same-box 1M reference
    mirror-1m)   arm $CK-mirror-1m   FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000 1000000" FT_POST="${NEEDLES//\$ARM_NAME/$CK-mirror-1m}" ;;
    mirror)      arm $CK-mirror      FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000 80000 713000" ;;
    whole-close) arm $CK-whole-close FT_ROWS=0 ;;
    mirror-nd)   arm $CK-mirror-nd   FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000 80000" FT_POST="${NEEDLES//\$ARM_NAME/$CK-mirror-nd}" ;;  # pool arm with needles/recall, no 713K
    # Prefill-transient A/B (-def = default, -t0 = pre-fix cushion-only headroom: no startup
    # measurement, transient 0, so the arena is neither parked nor held 0.65 GiB below the
    # ratio plan through decode). --moe-collect-stats on both sides for the decode hit rate.
    whole-def|whole-t0|mirror-def|mirror-t0|whole-8k-def|whole-8k-t0|mirror-8k-def|mirror-8k-t0|whole-st|mirror-st|whole-8k-st|mirror-8k-st)
      # -st: the static reservation (dynamic prefill headroom off) on a tree that has it
      envs=""; case "$a" in *-t0) envs="$T0" ;; *-st) envs="FREETOKEN_DYNAMIC_PREFILL_HEADROOM=0" ;; esac
      case "$a" in
        whole-8k-*)  arm $CK-$a FT_ROWS=0 FT_SIZES="8000" FT_EXTRA="--moe-collect-stats" FT_ENVS="$envs" ;;
        mirror-8k-*) arm $CK-$a FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000" FT_EXTRA="--moe-collect-stats" FT_ENVS="$envs" ;;
        whole-*)     arm $CK-$a FT_ROWS=0 FT_EXTRA="--moe-collect-stats" FT_ENVS="$envs" ;;
        mirror-*)    arm $CK-$a FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000 80000 713000" FT_EXTRA="--moe-collect-stats" FT_ENVS="$envs" ;;
      esac ;;
    *) die "unknown arm $a" ;;
  esac
done

# R3 as checkpoint1.sh. R6(arm): the journal half of acceptance.sh R6 plus the memlock limit
# the server actually ran with (recorded by measure.sh's process; here: this shell's, which
# the nohup'd server inherits).
r6_arm() {
  local j="$1" start lim
  start=$(grep -n "ServerArgs(model_path" "$j" | tail -1 | cut -d: -f1)
  lim=$(ulimit -l)
  echo "RLIMIT_MEMLOCK (server inherits): $lim"
  ! tail -n +"$start" "$j" | grep -qi "settled pageable" &&
    ! tail -n +"$start" "$j" | grep -qi "mlock.*fail" &&
    [ "$lim" = unlimited ] &&
    echo "R6(arm) ok: no pageable fallback, no mlock failure, memlock unlimited"
}
for x in ${ARMS:-whole mirror-1m mirror whole-close}; do
  a=$CK-$x
  [ -f "$BOX/$a-journal.txt" ] || { echo "$a: no journal"; continue; }
  printf '%s R3: ' "$a"
  FREETOKEN_LOG="$BOX/$a-journal.txt" bash "$REPO/benchmarks/switchyard_soak/checks/acceptance.sh" R3 \
    > "$BOX/$a-acceptance-R3.txt" 2>&1 && echo PASS || echo "FAIL (see $a-acceptance-R3.txt)"
  printf '%s R6(arm): ' "$a"
  r6_arm "$BOX/$a-journal.txt" > "$BOX/$a-acceptance-R6.txt" 2>&1 && echo PASS || echo "FAIL (see $a-acceptance-R6.txt; expected on Vast)"
done
# Needles/recall of the 1M pool arm against the whole-model reference of THIS run, and the
# pool arms' decode against the owner-machine record (compare_records.py reads results/).
python3 "$HERE/compare_needles.py" "$BOX" $CK-whole ${NEEDLES_ARM:-$CK-mirror-1m} > "$BOX/$CK-needles-compare.txt" 2>&1 || true
cp "$OUT"/nemotron-reserve-2e-record.json "$OUT"/nemotron-reserve-2e-1m-record.json "$BOX/" 2>/dev/null || true
python3 "$HERE/compare_records.py" "$BOX" $CK > "$BOX/$CK-records-compare.txt" 2>&1 || true
rm -f "$BOX"/nemotron-reserve-2e-record.json "$BOX"/nemotron-reserve-2e-1m-record.json
{ python3 "$HERE/compare_transient_ab.py" "$BOX" $CK; python3 "$HERE/compare_transient_ab.py" "$BOX" $CK st; } > "$BOX/$CK-transient-ab.txt" 2>&1 || true
python3 "$HERE/compare_box.py" "$BOX" $CK > "$BOX/$CK-box-compare.txt" 2>&1 || true
cat "$BOX/$CK-transient-ab.txt" "$BOX/$CK-needles-compare.txt" "$BOX/$CK-records-compare.txt" "$BOX/$CK-box-compare.txt"
echo "checkpoint $CK (box) arms done $(date -u +%FT%TZ)"
