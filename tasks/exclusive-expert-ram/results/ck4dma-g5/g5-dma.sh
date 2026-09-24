#!/bin/bash
# "dma-wb on g5": ck4 whole / mirror-1m / mirror / whole-1m on exp/mirror-dma-wb 773d9f8 code.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
cd /root/FreeToken-dma
echo "=== dma-wb on g5 $(date -u +%FT%TZ) GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
ARMS="whole mirror-1m mirror whole-1m" tasks/exclusive-expert-ram/checkpoint-box.sh ck4dma
echo G5-DMA-DONE
