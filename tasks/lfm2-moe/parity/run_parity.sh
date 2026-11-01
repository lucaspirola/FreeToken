#!/usr/bin/env bash
# FreeToken vs exllamav3 v1.6.6 parity on a tiny lfm2_moe EXL3 model.
#
#   run_parity.sh <exl3_model_dir> <tag> [n_new]
#
# Steps: ft_run.py (FreeToken greedy run + per-layer record), exl3_run.py (exllamav3 on the same
# ids), ft_run.py --ref (each mixer / feed-forward also run on exllamav3's input), then
# score.py, routing_flips.py and iso_moe_ties.py. FT_PROMPT_REPEAT=n lengthens the prompt.
#
# Build the model first (FreeToken venv, then the exllamav3 venv):
#   TINY_GATE_STD=0.25 $FT_PY make_tiny.py <hf_dir_with_tokenizer> m/bf16_r
#   $EXL3_PY -m exllamav3.conversion.convert_model -i m/bf16_r -w work_r -o m/exl3_r -b 5.0 -hq -cr 64 -cc 512
#
# Environment: FT_PY (FreeToken venv python), EXL3_PY (venv with torch 2.11.0+cu128 and the
# exllamav3-1.6.6+cu128.torch2.11.0 wheel), PARITY_OUT (work dir; outputs go to out/ and logs/).
# Needs the GPU: refuses to run while freetoken-serve is active.
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
REPO="$(cd "$HERE/../../.." && pwd)"
FT_PY=${FT_PY:-$HOME/ai/FreeToken/.venv/bin/python}
EXL3_PY=${EXL3_PY:-$HOME/ai/FreeToken/.work/lfm25/exl3/venv/bin/python}
cd "${PARITY_OUT:-$HOME/ai/FreeToken/.work/lfm25/parity}"
M=$1 TAG=$2 N=${3:-24}
export TVM_FFI_CUDA_ARCH_LIST=${TVM_FFI_CUDA_ARCH_LIST:-12.0}
FLAGS=(--model "$M" --max-running-requests 1 --linear-state-slots 13 --num-tokens 32768
       --max-seq-len-override 32768 --kv-cache-dtype q8_0 --attention-backend triton
       --moe-backend offload --max-prefill-length 8192)
mkdir -p out logs
if [ "$(systemctl is-active freetoken-serve || true)" = active ]; then
  echo "freetoken-serve is active: stop here"
  exit 1
fi
PYTHONPATH=$REPO/python $FT_PY "$HERE/ft_run.py" "out/ft_$TAG.pt" "$N" -- "${FLAGS[@]}" > "logs/ft_$TAG.log" 2>&1
$EXL3_PY "$HERE/exl3_run.py" "$M" "out/ft_$TAG.pt" "out/ex_$TAG.pt" > "logs/ex_$TAG.log" 2>&1
PYTHONPATH=$REPO/python $FT_PY "$HERE/ft_run.py" "out/iso_$TAG.pt" "$N" --ref "out/ex_$TAG.pt" -- "${FLAGS[@]}" > "logs/iso_$TAG.log" 2>&1
same=$($FT_PY -c "import torch;print(torch.load('out/ft_$TAG.pt')['output_ids'] == torch.load('out/iso_$TAG.pt')['output_ids'])")
echo "isolated run kept the chained output ids: $same"
$FT_PY "$HERE/score.py" "out/iso_$TAG.pt" "out/ex_$TAG.pt"
$FT_PY "$HERE/routing_flips.py" "$M" "out/iso_$TAG.pt" "out/ex_$TAG.pt"
$FT_PY "$HERE/iso_moe_ties.py" "$M" "out/iso_$TAG.pt" "out/ex_$TAG.pt"
