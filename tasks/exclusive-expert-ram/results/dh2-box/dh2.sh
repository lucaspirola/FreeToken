#!/bin/bash
# Decode-level target A/B on ft-dev (exp/decode-headroom a2f08c0): default 0.375 GiB vs 128 MiB
# (FREETOKEN_DECODE_FREE_TARGET_MB=128), Nemotron mirror-np alternated x2, then one mirror-1m at 128,
# probe on. The dh arms measured a decode window's need at <= 0.02 GiB (1M) and 0 KV-commit overhead.
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
S=/root/dh2-status; rm -f $S
cd /root/FT-dh && git log --oneline -1 | cut -c1-60 >> $S
for r in 1 2; do
  for t in def 128; do
    envs="FREETOKEN_DECODE_MEM_PROBE=1"; [ $t = 128 ] && envs="$envs FREETOKEN_DECODE_FREE_TARGET_MB=128"
    FT_GEN=512 FT_EXTRA="--moe-collect-stats" FT_ENVS="$envs" ARMS=mirror-np \
      tasks/exclusive-expert-ram/checkpoint-box.sh dt$t-$r > /root/dh2-$t-$r.log 2>&1
    echo "dt$t-$r rc=$? $(date -u +%T)" >> $S
  done
done
FT_GEN=512 FT_EXTRA="--moe-collect-stats" FT_ENVS="FREETOKEN_DECODE_MEM_PROBE=1 FREETOKEN_DECODE_FREE_TARGET_MB=128" \
  ARMS=mirror-1m tasks/exclusive-expert-ram/checkpoint-box.sh dt128m > /root/dh2-1m.log 2>&1
echo "dt128m rc=$? $(date -u +%T)" >> $S
echo ALLDONE >> $S
