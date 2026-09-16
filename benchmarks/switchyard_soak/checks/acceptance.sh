#!/usr/bin/env bash
# Acceptance checks for the single-lane default profile (campaign 2026-09-16/17).
# Usage: benchmarks/switchyard_soak/checks/acceptance.sh R2|R3|R4|R5|R6
# Each check reads the CURRENT server run (lines after the last ServerArgs in the log) or
# the archived soak results; exit 0 = met. R7 is plain pytest and R1 a file test (see the
# campaign record); they are not here.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.."
LOG=${FREETOKEN_LOG:-/home/lucas/.cache/freetoken/logs/ft_serve.log}
RUN=${SOAK_RUN:-benchmarks/switchyard_soak/runs/single-lane-df06962}

# Lines of the current server run, materialised once so `grep -q` under pipefail cannot
# SIGPIPE the producer and turn a met check into a false failure.
start=$(grep -n "ServerArgs(model_path" "$LOG" | tail -1 | cut -d: -f1)
RUN_LOG=$(tail -n +"$start" "$LOG")
current_run() { printf '%s\n' "$RUN_LOG"; }

case "${1:-}" in
  R2)
    bash -n scripts/serve-default.sh
    grep -q -- '--max-running-requests 1 ' scripts/serve-default.sh
    grep -q -- '--kv-grow-step-tokens 65536' scripts/serve-default.sh
    grep -q 'FREETOKEN_EXPERT_ARENA=.*:-1' scripts/serve-default.sh
    grep -q 'FREETOKEN_GROWABLE_OVERLAP=.*:-1' scripts/serve-default.sh
    grep -q '^ExecStart=/home/lucas/ai/FreeToken/scripts/serve-default.sh$' scripts/systemd/freetoken-serve.service
    cmp -s scripts/systemd/freetoken-serve.service /etc/systemd/system/freetoken-serve.service
    grep -q 'sudo systemctl {start,stop,restart,status} freetoken-serve' scripts/systemd/freetoken-serve.service
    systemctl is-active --quiet freetoken-serve
    grep -q "max_running_req=1," <<<"$RUN_LOG"
    echo "R2 ok: launcher + installed unit + live single-lane server"
    ;;
  R3)
    caps=$(grep -c "Start capturing CUDA graphs" <<<"$RUN_LOG" || true)
    grows=$(grep -c "KV grew" <<<"$RUN_LOG" || true)
    tb=$(grep -c "Traceback" <<<"$RUN_LOG" || true)
    echo "captures=$caps kv_grows=$grows tracebacks=$tb"
    [ "$caps" -eq 1 ] && [ "$grows" -ge 1 ] && [ "$tb" -eq 0 ]
    ;;
  R4)
    grep -q "Scheduler loop: overlap (kv_grow_step_tokens=65536, growable_overlap=True)" <<<"$RUN_LOG"
    echo "R4 ok: overlap loop hosting growable KV"
    ;;
  R5)
    python3 - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
for name in ("soakStage/results-switchyard_stage", "soakPass/results-switchyard_passthrough",
             "soakPass13/results-switchyard_passthrough"):
    d = json.load(open(f"{run}/{name}/summary.json"))
    assert d["passed"] and d["failures"] == 0 and d["completed_duration"], (name, d)
    print(name, d["requests"], "req", "p95", d["latency_p95_ms"])
s = json.load(open(f"{run}/stats_after_soakPass13.json"))
sp = s["scheduler"]["session_spill"]
assert sp["spills_failed"] == 0 and sp["restores_failed"] == 0 and sp["restores"] > 0, sp
assert s["requests"]["aborts"]["error"] == 0
assert s["scheduler"]["prefill"]["invariant"]["violations"] == 0
print("R5 ok: 0 errors, restores", sp["restores"], "/ 0 failed, 0 invariant violations")
PY
    ;;
  R6)
    ! grep -qi "settled pageable" <<<"$RUN_LOG"
    ! grep -qi "mlock.*fail" <<<"$RUN_LOG"
    grep -q '^LimitMEMLOCK=infinity' scripts/systemd/freetoken-serve.service
    pid=$(systemctl show -p MainPID --value freetoken-serve)
    grep -q "Max locked memory.*unlimited" "/proc/$pid/limits"
    test -f /etc/systemd/system/user@1000.service.d/memlock.conf
    echo "R6 ok: banks pinned, memlock unlimited (pid $pid), drop-in present"
    ;;
  *) echo "usage: $0 R2|R3|R4|R5|R6" >&2; exit 2 ;;
esac
