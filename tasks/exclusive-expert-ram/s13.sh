#!/usr/bin/env bash
# Plan S13 Verify on the GPU: two arms in the record config (auto pool, reserve 2E), each
# followed by s13_pins.py 112000 32000 as FT_POST.
#   s13-session  serve-default.sh as shipped (--pin-prefix-scope session)
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
for arm in s13-session s13-shared; do
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
  extra=""; [ "$arm" = s13-shared ] && extra="--pin-prefix-scope shared"
  env FT_ROWS=-1 FT_RESERVE=256 FT_SIZES="8000" FT_EXTRA="$extra" FT_POST_TIMEOUT=3600 \
    FT_POST="S13_OUT=$OUT/$arm-pins.json python3 $HERE/s13_pins.py 112000 32000" \
    "$HERE/measure.sh" "$arm"
  journalctl --user -u "ft-measure-$arm" -o cat --no-pager > "$OUT/$arm-journal.txt" || true
done
