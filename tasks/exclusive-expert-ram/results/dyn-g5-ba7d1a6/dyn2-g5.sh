#!/bin/bash
# exp/dynamic-transient ba7d1a6 on ft-g5: rerun of dyn-g5 (whole + mirror-nd, needles/recall, 8K/80K, ratio 1.00).
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
cd /root/FreeToken-dyn2
echo "=== dyn2-g5 start $(date -u +%FT%TZ) $(git log --oneline -1)"
FT_SIZES="8000 80000" ARMS="whole mirror-nd" NEEDLES_ARM=dyn2-mirror-nd tasks/exclusive-expert-ram/checkpoint-box.sh dyn2
B=tasks/exclusive-expert-ram/results/dyn2-box
for a in whole mirror-nd; do echo "scan_noreserve dyn2-$a:"; python3 tasks/exclusive-expert-ram/scan_noreserve.py $B/dyn2-$a-journal.txt; done > $B/dyn2-scan-noreserve.txt 2>&1
cat $B/dyn2-scan-noreserve.txt
echo DYN2-G5-DONE $(date -u +%FT%TZ)
