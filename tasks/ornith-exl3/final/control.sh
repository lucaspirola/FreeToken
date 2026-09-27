#!/usr/bin/env bash
# Control arm after chain.sh part 2: q8q8 at the 256K ceiling with the extend attention forced onto
# the triton kernel (FREETOKEN_EXTEND_BACKEND=triton), the kernel the q6_0/q5_0/q4_0 lanes fall back
# to (kernel/extend_flashinfer.py eligible(): formats 0/1 only). Separates kernel from KV format.
F=$(dirname "$(readlink -f "$0")")
while systemctl --user is-active --quiet ft-final-chain; do sleep 30; done
FT_KV=q8q8 FT_ENVS_EXTRA=FREETOKEN_EXTEND_BACKEND=triton $F/arm.sh kv-q8q8-256k-triton -1 "8000 256000" FT_GEN=512
echo "control done $(date -Is)" >> $F/results/status.txt
