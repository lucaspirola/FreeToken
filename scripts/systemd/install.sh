#!/usr/bin/env bash
# Install (or re-render) the freetoken-serve SYSTEM unit for this host, plus the
# user-manager memlock drop-in that lets the whole model be pinned in RAM.
#
#   sudo scripts/systemd/install.sh          install/update for the invoking (sudo) user
#   scripts/systemd/install.sh --print       render the unit to stdout only (no root)
#
# After install:  sudo systemctl reset-failed freetoken-serve; sudo systemctl start freetoken-serve
# The user@UID memlock drop-in takes effect for --user units after the user manager
# restarts (WSL: `wsl --shutdown`); the system unit does not depend on it.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(cd "$HERE/../.." && pwd)
USER_NAME=${SUDO_USER:-$USER}
HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
UID_NUM=$(id -u "$USER_NAME")

render() {
  sed -e "s|@USER@|$USER_NAME|g" -e "s|@HOME@|$HOME_DIR|g" \
      -e "s|@REPO@|$REPO|g" -e "s|@UID@|$UID_NUM|g" "$HERE/freetoken-serve.service.in"
}

if [ "${1:-}" = "--print" ]; then
  render
  exit 0
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo $0   (or $0 --print to preview)" >&2
  exit 1
fi

render > /etc/systemd/system/freetoken-serve.service
install -d "/etc/systemd/system/user@$UID_NUM.service.d"
printf '[Service]\nLimitMEMLOCK=infinity\n' > "/etc/systemd/system/user@$UID_NUM.service.d/memlock.conf"
install -d -o "$USER_NAME" -g "$USER_NAME" "$HOME_DIR/.cache/freetoken/logs"
systemctl daemon-reload
echo "installed /etc/systemd/system/freetoken-serve.service for $USER_NAME ($REPO)"
echo "installed /etc/systemd/system/user@$UID_NUM.service.d/memlock.conf"
