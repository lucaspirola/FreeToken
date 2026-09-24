#!/bin/bash
# exp/dynamic-transient on ft-g5 (native Linux, gen5): does the dynamic prefill headroom start without OOM?
# checkpoint-box ARMS="whole mirror-nd", FT_SIZES="8000 80000", Nemotron, ratio 1.00. Slotted between gap1 and gap2.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
until grep -q G5-GAP-DONE /root/g5-gap.log; do sleep 3; done
cd /root/FreeToken-dyn
echo "=== dyn-g5 start $(date -u +%FT%TZ) $(git log --oneline -1)"
FT_SIZES="8000 80000" ARMS="whole mirror-nd" NEEDLES_ARM=dyn-mirror-nd tasks/exclusive-expert-ram/checkpoint-box.sh dyn
echo DYN-G5-DONE $(date -u +%FT%TZ)
