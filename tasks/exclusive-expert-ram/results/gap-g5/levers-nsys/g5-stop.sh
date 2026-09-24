#!/bin/bash
# ft-g5 ends 17:00 UTC: let gap5 run whole, mirror-1m, mirror; abort at whole-1m's start or 15:50 UTC,
# whichever comes first; then the 2-min launch bench; then mark G5-FINAL-STOP.
LOG=/root/g5-stop.log
while :; do
  if grep -q "\[gap5-whole-1m\] starting" /root/g5-gap5.log; then echo "whole-1m started: abort $(date -u +%T)" >> $LOG; break; fi
  if grep -q "G5-GAP5-DONE" /root/g5-gap5.log; then echo "queue done $(date -u +%T)" >> $LOG; break; fi
  if [ "$(date -u +%H%M)" -ge 1550 ]; then echo "15:50 deadline: abort $(date -u +%T)" >> $LOG; break; fi
  sleep 5
done
for pat in "[l]aunchbench.sh" "[g]5-gap5.sh" "[c]heckpoint-box.sh" "[m]easure.sh" "[n]eedles" "[r]ecall" "[p]robe_decode"; do pkill -f "$pat"; done
sleep 2; pkill -f "[f]t serve"; sleep 15; pkill -9 -f "[f]t serve"; sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $LOG
CUDART=$(ls /usr/local/cuda-13.0/lib64/libcudart.so* 2>/dev/null | head -1)
[ -z "$CUDART" ] && CUDART=$(/root/venv/bin/python -c "import glob,nvidia,os;print(glob.glob(os.path.dirname(nvidia.__file__)+'/cuda_runtime/lib/libcudart.so*')[0])")
echo "cudart $CUDART" >> /root/launchbench.txt
timeout 600 /root/venv/bin/python /root/bench_graph_launch2.py "$CUDART" >> /root/launchbench.txt 2>&1
echo "launchbench rc=$? $(date -u +%T)" >> $LOG
echo G5-FINAL-STOP >> $LOG
