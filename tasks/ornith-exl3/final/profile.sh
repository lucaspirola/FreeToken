#!/usr/bin/env bash
# After the measurement arms: nsys of the 32K/80K prefill (saver, whole), then the kernel roofline
# bench on the idle GPU (tasks/ornith-exl3/perf/bench_roofline.py, M=8192), under the host GPU lock.
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); P=$F/profile; mkdir -p $P
while systemctl --user is-active --quiet ft-final-chain || systemctl --user is-active --quiet ft-final-control; do sleep 30; done
$F/nsys-prefill.sh $P "saver whole" > $P/nsys-prefill.log 2>&1
for f in $P/prefill-*.sqlite; do python3 $F/nsys_kernels.py $f 30 > ${f%.sqlite}-kernels.txt 2>&1; done
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -le 32 ]; do sleep 5; done
echo "roofline start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg) sm $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader)" > $P/roofline.log
cd $WT && PYTHONPATH=$WT/python /home/lucas/ai/FreeToken/.venv/bin/python tasks/ornith-exl3/perf/bench_roofline.py --json $P/roofline.jsonl >> $P/roofline.log 2>&1
echo "PROFILEDONE $(date -Is)" >> $P/roofline.log
