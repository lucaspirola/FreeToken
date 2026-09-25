#!/bin/bash
# ft-dev: transient snapshots, continued. -ne = native caching allocator (expandable_segments:False, what
# WSL runs: engine._ensure_expandable_segments skips WSL); -ws = native + FREETOKEN_PREFILL_WORKSPACE_TEST
# (one cached segment of the allocator peak). Ornith via scripts/serve-ornith.sh (262144 tokens).
until grep -q ALLDONE /root/snapdev-status 2>/dev/null; do sleep 20; done
sed -e 's#^S=/root/snapdev-status; rm -f $S#S=/root/snapdev2-status; rm -f $S#' -e '/^run /d' -e '/^echo ALLDONE/d' \
    -e 's#\$R/scripts/serve-default.sh#$R/scripts/${SERVE:-serve-default.sh}#' /root/snapdev.sh > /root/snapdev2-body.sh
source /root/snapdev2-body.sh
NE=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
NEMO=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4; ORN=Ornith-1.5-35B-A3B-NVFP4
run nemo-mirror-ne $NEMO -1 $NE
SERVE=serve-ornith.sh run ornith $ORN 0
SERVE=serve-ornith.sh run ornith-ne $ORN 0 $NE
run nemo-mirror-ws $NEMO -1 $NE FREETOKEN_PREFILL_WORKSPACE_TEST=0
SERVE=serve-ornith.sh run ornith-ws $ORN 0 $NE FREETOKEN_PREFILL_WORKSPACE_TEST=0
echo ALLDONE >> $S
