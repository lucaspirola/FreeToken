# Sourced by the splitkv-decode box jobs (ft-dev). Whole-model Nemotron arm, ratio 1.00, port 1920.
R=${R:-/root/FT-splitkv}
K=/root/K; mkdir -p $K/results
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
PY=/root/venv/bin/python
MODEL=${MODEL:-/root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
mk_env() {  # $1 = env file
  { echo "export FREETOKEN_MODEL=$MODEL"
    echo "export FREETOKEN_MEMORY_RATIO=1.00"
    echo "export FREETOKEN_MIRROR_EXPERT_RAM=0"; echo "export FREETOKEN_MIRROR_HOST_ROWS=0"
    echo "export UV_PROJECT_ENVIRONMENT=/root/venv"; echo "export UV_NO_SYNC=1"
    echo "export PYTHONPATH=$R/python"; } > "$1"
}
gpu_idle() {  # wait until nothing of ours or anyone's is on the GPU
  local t0=$(date +%s)
  while pgrep -f "ft serve" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 16 ]; do
    [ $(( $(date +%s) - t0 )) -gt 300 ] && { echo "GPU not idle after 300s"; nvidia-smi; return 1; }; sleep 5; done
}
wait_ready() {  # $1 log, $2 timeout s
  local t0=$(date +%s)
  until grep -q "API server is ready" "$1"; do
    grep -q -E "Traceback|Error:" "$1" && ! pgrep -f "ft serve" >/dev/null && { echo "server died"; return 1; }
    [ $(( $(date +%s) - t0 )) -gt ${2:-900} ] && { echo "not ready after ${2:-900}s"; return 1; }; sleep 5; done
  echo "ready after $(( $(date +%s) - t0 ))s"
}
stop_server() { pkill -f "ft serve" || true; local t0=$(date +%s); while pgrep -f "ft serve" >/dev/null; do [ $(( $(date +%s) - t0 )) -gt 60 ] && pkill -9 -f "ft serve"; sleep 2; done; sleep 5; }
