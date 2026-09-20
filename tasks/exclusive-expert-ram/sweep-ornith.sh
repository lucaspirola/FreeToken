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

run() {
  local arm="$1" rows="$2"
  echo "=============== $arm (rows=$rows) $(date +%H:%M:%S)"
  for _ in $(seq 10); do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 22 ] && break
    sleep 20
  done
  FT_ROWS="$rows" timeout 1800 tasks/exclusive-expert-ram/measure.sh "$arm" 2>&1 | tail -40
  systemctl --user reset-failed "ft-measure-$arm" 2>/dev/null
  sleep 15
}

for arg in "$@"; do
  case "$arg" in
    baseline) run ornith-baseline 0 ;;
    auto)     run ornith-auto     -1 ;;
    *)        run "ornith-$arg"   "$arg" ;;
  esac
done
echo "=============== ornith sweep done $(date +%H:%M:%S)"
