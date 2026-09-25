#!/bin/bash
# exp/decode-headroom on ft-dev: probe what decode windows take (FREETOKEN_DECODE_MEM_PROBE=1),
# Nemotron mirror at 8K/80K (mirror-np) and 8K/1M (mirror-1m), 512 decode tokens.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
S=/root/dhdev-status; rm -f $S
until grep -q ALLDONE /root/lfuab-status 2>/dev/null; do sleep 60; done
W=/root/FT-dh; cd $W && git log --oneline -1 | cut -c1-70 >> $S
PYTHONPATH=$W/python flock /root/gpu.lock /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/engine > /root/dhdev-tests.txt 2>&1
rc=$?; echo "tests rc=$rc $(tail -1 /root/dhdev-tests.txt)" >> $S
[ $rc -ne 0 ] && { echo ALLDONE >> $S; exit 1; }
for a in mirror-np mirror-1m; do
  FT_GEN=512 FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_DECODE_MEM_PROBE=1" ARMS=$a NEEDLES_ARM=dh-mirror-1m \
    tasks/exclusive-expert-ram/checkpoint-box.sh dh > /root/dh-$a.log 2>&1
  echo "dh $a rc=$? $(date -u +%T)" >> $S
done
echo ALLDONE >> $S
