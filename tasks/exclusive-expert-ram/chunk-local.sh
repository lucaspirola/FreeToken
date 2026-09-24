#!/usr/bin/env bash
# Prefill chunk size at long context, now that the dynamic headroom holds the transient
# free only while prefill runs: --max-prefill-length 8192 / 12288 / 16384 on Nemotron 3.5
# Lightning and Ornith NVFP4, whole model and RAM saver (mirror, rows -1), prompts
# 8K (decode reference) / 32K / 80K / 256K, two passes. Per arm the journal gives the
# measured transient ("Prefill headroom: transient X GiB measured on a N-token chunk") and
# the reserve/release moves; compare_chunks.py tabulates prefill, decode and the transient.
#   CH_OUT   results dir (default results/chunk-local)
#   CH_ARMS  ";"-separated "model kind chunk" triples (default: all 12)
# Waits for MemAvailable >= 23 GiB and 0 MiB on the GPU before each arm [agent practice].
# Run as a systemd transient user unit (--setenv=PATH), never from an agent shell.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="${CH_OUT:-$HERE/results/chunk-local}"
mkdir -p "$OUT"
quiet() {
  until [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -ge 23 ]; do sleep 60; done
  echo "host quiet at $(date -Is): $(grep MemAvailable /proc/meminfo)"
}
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00 FT_SIZES="8000 32000 80000 256000"
DEFAULT="nemotron whole 8192;nemotron whole 12288;nemotron whole 16384;nemotron saver 8192;nemotron saver 12288;nemotron saver 16384;ornith whole 8192;ornith whole 12288;ornith whole 16384;ornith saver 8192;ornith saver 12288;ornith saver 16384"
IFS=";" read -r -a ARMLIST <<< "${CH_ARMS:-$DEFAULT}"
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
for spec in "${ARMLIST[@]}"; do
  set -- $spec
  model=$1 kind=$2 chunk=$3
  name="ch-$model-$kind-$chunk"
  case "$model" in
    nemotron) unset FT_MODEL FT_NAME; extra="" ;;
    ornith)   export FT_MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4" FT_NAME=ornith
              extra="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith" ;;
  esac
  unset FT_RESERVE
  case "$kind" in whole) rows=0 ;; saver) rows=-1; [ "$model" = nemotron ] && export FT_RESERVE=256 ;; esac
  quiet
  flock 9
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = 0 ] || { echo "GPU holds $used MiB, skipping $name"; flock -u 9; continue; }
  FT_ROWS=$rows FT_EXTRA="$extra --max-prefill-length $chunk --moe-collect-stats" "$HERE/measure.sh" "$name" \
    || echo "$name measure exit $?"
  journalctl --user -u "ft-measure-$name" -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$name-journal.txt" || true
  mv "$HERE"/results/$name-* "$OUT/" 2>/dev/null || true
  flock -u 9
  sleep 30
done
python3 "$HERE/compare_chunks.py" "$OUT" > "$OUT/chunks-compare.txt" 2>&1 || true
cat "$OUT/chunks-compare.txt"
echo "chunk-local done"
