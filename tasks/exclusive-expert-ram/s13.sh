#!/usr/bin/env bash
# Plan S13 Verify on the GPU: two arms in the record config (auto pool, reserve 2E), each
# followed by s13_pins.py 112000 32000 as FT_POST.
#   s13-session  --pin-prefix-scope session --pin-prefix-max-tokens 262144 (plan S13's proposal)
#   s13-shared   control: --pin-prefix-scope shared, same session key -- separates the pin
#                from the session lease (a hit here would mean the lease, not the pin, kept it)
# Run as a systemd transient unit (needs --setenv=PATH for nvidia-smi), never an agent shell.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
HERE="$PWD/tasks/exclusive-expert-ram"
OUT="$HERE/results"
export FT_VENV=/home/lucas/ai/FreeToken/.venv FT_RATIO=1.00
exec 9>/home/lucas/.cache/freetoken/gpu-host.lock
flock 9
PFX="${S13_PREFIX:-s13}"; HANDOFF="${S13_HANDOFF:-32000}"
# Session keys are unique per arm: the disk spill tier is persistent across restarts, so a
# shared key let s13c-shared restore s13c-session's 961K session B from disk (TTFT 6.9 s).
for arm in $PFX-session $PFX-shared; do
  # checkpoint1.sh's preflight, inline (calling it would wait on the lock held above)
  [ -z "$(git status --porcelain -- python)" ] || { echo "python/ dirty"; exit 1; }
  systemctl is-active --quiet freetoken-serve && { echo "freetoken-serve is up"; exit 1; }
  systemctl --user list-units --state=active --no-legend 'ft-measure-*' | grep -q . && { echo "ft-measure unit up"; exit 1; }
  pgrep -f '^[^ ]*python[0-9.]* -m pytest' >/dev/null && { echo "pytest running"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" = 0 ] || { echo "GPU holds $used MiB"; exit 1; }
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail" -ge 22 ] || { echo "MemAvailable $avail GiB"; exit 1; }
  echo "[$arm] preflight ok: GPU 0 MiB, MemAvailable $avail GiB, code $(git log --oneline -1)"
  # serve-default.sh ships the default (shared) scope since 16442c8: the session arm sets it.
  extra="--pin-prefix-scope shared"; [ "$arm" = $PFX-session ] && extra="--pin-prefix-scope session --pin-prefix-max-tokens 262144"
  env FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000" FT_EXTRA="$extra" FT_POST_TIMEOUT=7200 \
    FT_POST="S13_KEY_A=$arm-a S13_KEY_B=$arm-b S13_OUT=$OUT/$arm-pins.json python3 $HERE/s13_pins.py 112000 $HANDOFF" \
    "$HERE/measure.sh" "$arm"
  journalctl --user -u "ft-measure-$arm" -o cat --no-pager > "$OUT/$arm-journal.txt" || true
done
