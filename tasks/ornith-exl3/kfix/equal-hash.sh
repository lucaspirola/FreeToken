#!/usr/bin/env bash
# equal_hash.py on the base tree (b967140) and on this tree, back to back, under the host GPU lock.
export PATH=/usr/lib/wsl/lib:/usr/local/cuda/bin:$PATH
F=$(dirname "$(readlink -f "$0")"); WT=$(cd "$F/../../.." && pwd); R=$F/results; BASE=/home/lucas/ai/FreeToken-wt/kbase
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
for t in base:$BASE new:$WT; do n=${t%%:*}; d=${t#*:}
  PYTHONPATH=$d/python TVM_FFI_CUDA_ARCH_LIST=12.0 /home/lucas/ai/FreeToken/.venv/bin/python $F/equal_hash.py $R/hash-$n.txt > $R/hash-$n.log 2>&1
done
{ echo "base $(git -C $BASE log --oneline -1 | cut -c1-40) new $(git -C $WT log --oneline -1 | cut -c1-40) +$(git -C $WT diff --stat | tail -1)"; wc -l $R/hash-base.txt $R/hash-new.txt; diff $R/hash-base.txt $R/hash-new.txt && echo IDENTICAL; echo HASHDONE; } > $R/hash-diff.txt
