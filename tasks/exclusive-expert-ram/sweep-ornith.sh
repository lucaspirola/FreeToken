#!/usr/bin/env bash
# Host RAM x decode curve for Ornith-1.5-35B-A3B-NVFP4.
#
# A different model, not a different flag: qwen3_5_moe, 256 experts per MoE
# layer against Lightning's 128, GATED silu experts (separate gate|up) against
# Lightning's ungated relu2, I=512, hidden 2048, 256K context.
#
# Geometry read off the checkpoint itself (not from the config):
#   one expert row = gate 524288 + up 524288 + down 524288 B of packed NVFP4,
#   plus three e4m3 block-scale planes (65536 B each) and three f32 globals
#   = 1769484 B = 1.688 MiB, against Lightning's 5.36 MiB.
#   40 layers x 256 experts = 10240 rows -> 16.88 GiB for the whole model,
#   MORE than Lightning's 15.41 on a 28 GiB host. The baseline arm is
#   therefore the one at risk here; if it will not start, that is the
#   finding, not a failed run.
#   config.json has no decoder_sparse_step and no mlp_only_layers, so every
#   one of the 40 layers is MoE -- which is what the model's own spec assumes
#   (layer_to_bank = identity in models/qwen3_5_moe/weight.py).
#   That spec's regex was checked against this checkpoint's weight_map: it
#   matches 92160 tensors (40 x 256 x 3 projections x 3 kinds) and excludes
#   the 768 mtp.layers.* expert tensors of the MTP head, which is not served.
#
# Known caveat, measured not assumed: no host has a tuned NVFP4 prefill table
# for Ornith's GEMM shapes (only Lightning's two, only on the 5080), so its
# prefill runs the MiniMax-M2 default tiles. The server now says so in its log
# (fused_nvfp4._warn_untuned_prefill). That handicaps Ornith's TTFT in EVERY
# arm below equally, baseline included, so the RAM x decode comparison between
# arms stands; the absolute prefill numbers are not Ornith's ceiling.
#
# Capacities are resolved after the auto arm reports its geometry ("Mirror
# pool: N rows pinned ... complement of M GPU residents"), because the floor
# depends on how many expert slots this model's VRAM leaves.
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."

export FT_MODEL="$HOME/ai/models/Ornith-1.5-35B-A3B-NVFP4"
export FT_NAME=ornith
# The launcher hardcodes Lightning's 1M ceiling and served name; FT_EXTRA is
# appended last and the last flag wins.
export FT_EXTRA="--num-tokens 262144 --max-seq-len-override 262144 --served-model-name ornith"

# Three KV lanes, because the owner asked for the trade-off, not one number:
# q8/q8 is the reference, q8 K + q6 V and q6 K + q5 V are the two asymmetric
# pairs the server validates (K is kept more precise than V in both, which is
# the pair that survives long context). Each lane is a separate arm and the
# lane name lands in the record, so no row is ambiguous later.
run() {
  local arm="$1" rows="$2"
  echo "=============== $arm (rows=$rows) $(date +%H:%M:%S)"
  for _ in $(seq 10); do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 22 ] && break
    sleep 20
  done
  # The GPU must be ours alone: --memory-ratio is a fraction of FREE VRAM, so
  # an arm sharing the card sizes itself to a smaller GPU and its decode number
  # is void. This cost a whole sweep on 2026-09-21.
  for _ in $(seq 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-1}" -eq 0 ] && break
    echo "[$arm] waiting: GPU holds ${used} MiB"; sleep 10
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-1}" -ne 0 ]; then
    echo "[$arm] REFUSED: GPU still holds ${used} MiB" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader >&2
    return 1
  fi
  systemctl --user reset-failed "ft-measure-$arm" 2>/dev/null
  FT_ROWS="$rows" timeout 3600 tasks/exclusive-expert-ram/measure.sh "$arm" 2>&1 | tail -40
  journalctl --user -u "ft-measure-$arm" --no-pager -o cat 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*m//g' > "tasks/exclusive-expert-ram/results/$arm-journal.txt"
  systemctl --user reset-failed "ft-measure-$arm" 2>/dev/null
  sleep 15
}

# FT_KV selects the lane for this invocation; the arm name carries it so the
# three lanes never overwrite each other's records.
LANE="${FT_KV:-q8q8}"
export FT_KV="$LANE"

for arg in "$@"; do
  case "$arg" in
    baseline) run "ornith-baseline-$LANE" 0 ;;
    auto)     run "ornith-auto-$LANE"     -1 ;;
    lanes)
      # Every lane, both arms, baseline bracketing each lane. ~6 arms.
      for lane in q8q8 q8q6 q6q5; do
        FT_KV="$lane" run "ornith-baseline-$lane" 0
        FT_KV="$lane" run "ornith-auto-$lane"    -1
      done
      ;;
    *)        run "ornith-$arg-$LANE"   "$arg" ;;
  esac
done
echo "=============== ornith sweep done $(date +%H:%M:%S)"
