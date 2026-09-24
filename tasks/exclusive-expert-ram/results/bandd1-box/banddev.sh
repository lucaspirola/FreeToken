#!/bin/bash
# ft-dev (gen4): duplicate-aware eviction A/B on exp/mirror-dma-wb e70849e, Nemotron pool mirror-np with
# --moe-collect-stats (hit rate), band off then on, then the plain (no stats) arm. Each run takes the lock itself.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
S=/root/banddev-status; rm -f $S
cd /root/FT-gap2 && git log --oneline -1 >> $S
FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_MIRROR_DUP_BAND=0" ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh bandd0 > /root/banddev0.log 2>&1
echo "bandd0 rc=$? $(date -u +%FT%TZ)" >> $S
FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_MIRROR_DUP_BAND=1" ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh bandd1 > /root/banddev1.log 2>&1
echo "bandd1 rc=$? $(date -u +%FT%TZ)" >> $S
ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh gapdev4 > /root/gapdev4.log 2>&1
echo "gapdev4 rc=$? $(date -u +%FT%TZ)" >> $S
echo ALLDONE >> $S
