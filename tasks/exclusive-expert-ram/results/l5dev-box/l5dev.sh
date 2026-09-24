#!/bin/bash
# ft-dev: e7e029f (fault check + snapshot after the launch): GPU tests, then mirror-np (vs gapdev4 = e70849e).
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
S=/root/l5dev-status; rm -f $S
cd /root/FT-gap2 && git log --oneline -1 >> $S
PYTHONPATH=$PWD/python flock /root/gpu.lock /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/scheduler > /root/l5dev-tests.txt 2>&1
echo "tests rc=$? $(tail -1 /root/l5dev-tests.txt)" >> $S
ARMS="mirror-np" tasks/exclusive-expert-ram/checkpoint-box.sh l5dev > /root/l5dev.log 2>&1
echo "l5dev rc=$? $(date -u +%FT%TZ)" >> $S
echo ALLDONE >> $S
