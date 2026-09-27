#!/bin/bash
# Task 3 (box): EXL3 decode split-K in-kernel reduction + fused epilogues. Tests + layer bench on exp/ornith-exl3,
# then an s5 probe on scratch/exl3-t3 (= scratch/exl3-dma b85402f + the change; control = ring3's TAG=rule on b85402f).
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TVM_FFI_CUDA_ARCH_LIST=12.0
S=/root/t3-status; rm -f $S
W=/root/FT-exl3-t3
[ -d $W ] || git -C /root/FT-exl3-dma worktree add -q --detach $W
cd $W && git fetch -q /root/ft-t3.bundle scratch/exl3-t3 && git reset -q --hard FETCH_HEAD && git log --oneline -1 >> $S
PYTHONPATH=$W/python flock /root/gpu.lock timeout 1800 /root/venv/bin/python -m pytest -q -p no:cacheprovider tests/kernels/test_exl3.py tests/layers/test_exl3_linear_layer.py tests/moe/test_exl3_banks.py > /root/t3-tests.txt 2>&1
rc=$?; echo "tests rc=$rc $(tail -1 /root/t3-tests.txt) $(date -u +%FT%TZ)" >> $S
[ $rc -ne 0 ] && { echo STOP >> $S; echo ALLDONE >> $S; exit 1; }
PYTHONPATH=$W/python flock /root/gpu.lock timeout 1200 /root/venv/bin/python tasks/ornith-exl3/perf/bench_decode_epilogue.py > /root/t3-bench.txt 2>&1
echo "bench rc=$? $(date -u +%FT%TZ)" >> $S
while ! grep -q "exl3 rule done" /root/ring3-status 2>/dev/null; do grep -q ALLDONE /root/ring3-status 2>/dev/null && break; sleep 30; done
WT=$W SIZES="8000 32000 80000 128000" TAG=t3 RATIO=1.00 flock /root/gpu.lock /root/s5wt.sh
echo "exl3 t3 done $(date -u +%FT%TZ)" >> $S
echo ALLDONE >> $S
