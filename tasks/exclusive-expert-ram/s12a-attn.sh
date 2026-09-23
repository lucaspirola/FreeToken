#!/usr/bin/env bash
# S12a re-measure with the extend-attention register-cap fix (016412c, branch reorg-attn):
# the same Ornith arm as s12a.sh (whole model in RAM, arena, q8_0 KV, 8K-250K, pass 2,
# empty GPU), without the roofline benches (unchanged by an attention-only patch).
# Run as a systemd transient unit (--setenv=PATH), never from an agent shell.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results/ornith-s12a-attn"
mkdir -p "$OUT"
PY=/home/lucas/ai/FreeToken/.venv/bin/python
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
export FT_MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4" FT_NAME=ornith
export FT_EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
empty_gpu() {
  [ -z "$(git status --porcelain -- python)" ] || { echo "python/ dirty"; exit 1; }
  systemctl is-active --quiet freetoken-serve && { echo "freetoken-serve is up"; exit 1; }
  systemctl --user list-units --state=active --no-legend 'ft-measure-*' | grep -q . && { echo "ft-measure unit up"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = 0 ] || { echo "GPU holds $used MiB"; exit 1; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || { echo "MemAvailable $avail GiB"; exit 1; }
  echo "preflight ok: GPU 0 MiB, MemAvailable $avail GiB, code $(git log --oneline -1)"
}
empty_gpu
PYTHONPATH=$PWD/python "$PY" -m freetoken.models.check_experts "$FT_MODEL" > "$OUT/check_experts.txt" 2>&1 || true
FT_ROWS=0 FT_SIZES="8000 32000 80000 128000 250000" "$HERE/measure.sh" ornith-s12a-attn
journalctl --user -u ft-measure-ornith-s12a-attn -o cat --no-pager | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/ornith-s12a-attn-journal.txt" || true
mv "$HERE"/results/ornith-s12a-attn-* "$OUT/" 2>/dev/null || true
echo "arm done"
