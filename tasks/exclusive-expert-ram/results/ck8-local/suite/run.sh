#!/bin/bash
# Full 8-directory suite (same command as the ft-dev r5fsuite.sh), local RTX 5080, model unloaded,
# under the host GPU lock. Output next to this script.
O=$(dirname "$(readlink -f "$0")"); T=/home/lucas/ai/FreeToken-wt/round5; S=$O/status
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock; flock 9
echo "start $(date -Is) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) $(uptime)" >> $S
cd $T; echo "tree $(git log --oneline -1 | cut -c1-120)" >> $S
rm -rf $O/ext
PYTHONPATH=$T/python TORCH_EXTENSIONS_DIR=$O/ext /home/lucas/ai/FreeToken/.venv/bin/python -m pytest -q -p no:cacheprovider tests/moe tests/engine tests/scheduler tests/kernels tests/kvcache tests/server tests/tokenizer tests/layers > $O/suite.txt 2>&1
rc=$?
echo "suite rc=$rc $(tail -1 $O/suite.txt) | illegal: $(grep -c 'illegal memory access' $O/suite.txt) | failed-lines: $(grep -c '^FAILED' $O/suite.txt)" >> $S
rm -rf $O/ext; echo ALLDONE >> $S
