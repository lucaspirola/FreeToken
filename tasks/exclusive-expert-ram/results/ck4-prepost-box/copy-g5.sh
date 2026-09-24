#!/bin/bash
set -x
G="ssh -p 16583 -o StrictHostKeyChecking=accept-new"
D=root@89.155.246.210
date -u
rsync -a -e "$G" /root/FreeToken-headroom $D:/root/ --exclude "tasks/exclusive-expert-ram/results/ck4-box" --exclude "tasks/exclusive-expert-ram/results/ck4pre*"
rsync -a -e "$G" /root/.cache/flashinfer /root/.cache/torch_extensions /root/.cache/tvm-ffi $D:/root/.cache/
rsync -a -e "$G" /usr/local/cuda-13.0 $D:/usr/local/
rsync -a -e "$G" /root/venv $D:/root/
rsync -a -e "$G" /root/pcie_bw.py /root/_*.so $D:/root/
rsync -a -e "$G" /root/models $D:/root/
date -u
echo COPY-DONE
