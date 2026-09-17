#!/usr/bin/env bash
# Install (or re-render) the freetoken-serve SYSTEM unit for this host and make the
# "whole model in RAM" capability (unlimited RLIMIT_MEMLOCK for this user) effective NOW and
# permanent across reboots.
#
#   sudo scripts/systemd/install.sh            install/update for the invoking (sudo) user
#   sudo scripts/systemd/install.sh --enable   ...and start the server at boot (grabs the GPU)
#   scripts/systemd/install.sh --print         render the unit to stdout only (no root)
#
# What it writes (all persistent, all idempotent):
#   /etc/systemd/system/freetoken-serve.service            the server (LimitMEMLOCK=infinity)
#   /etc/systemd/system/user@UID.service.d/memlock.conf    user manager: unlimited memlock
#   /etc/systemd/system.conf.d/freetoken-memlock.conf      DefaultLimitMEMLOCK=infinity (services)
#   /etc/systemd/user.conf.d/freetoken-memlock.conf        DefaultLimitMEMLOCK=infinity (user units)
#   /etc/security/limits.d/90-freetoken-memlock.conf       shells / ssh / su for this user (PAM)
# and raises the running user manager's memlock limit with prlimit so `systemd-run --user`
# and new user units pin immediately, without waiting for a re-login or `wsl --shutdown`.
#
# After install:  ft-up   (and ft-down to hand the GPU back) -- both land in ~/.local/bin
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
  echo "run as root: sudo $0 [--enable]   (or $0 --print to preview)" >&2
  exit 1
fi

# 1. The server unit itself.
render > /etc/systemd/system/freetoken-serve.service

# 2. Permanent memlock limits: user manager drop-in, manager defaults, PAM limits.
install -d "/etc/systemd/system/user@$UID_NUM.service.d" /etc/systemd/system.conf.d /etc/systemd/user.conf.d /etc/security/limits.d
printf '[Service]\nLimitMEMLOCK=infinity\n' > "/etc/systemd/system/user@$UID_NUM.service.d/memlock.conf"
printf '[Manager]\nDefaultLimitMEMLOCK=infinity\n' > /etc/systemd/system.conf.d/freetoken-memlock.conf
printf '[Manager]\nDefaultLimitMEMLOCK=infinity\n' > /etc/systemd/user.conf.d/freetoken-memlock.conf
printf '%s soft memlock unlimited\n%s hard memlock unlimited\n' "$USER_NAME" "$USER_NAME" > /etc/security/limits.d/90-freetoken-memlock.conf

install -d -o "$USER_NAME" -g "$USER_NAME" "$HOME_DIR/.cache/freetoken/logs"
systemctl daemon-reload

# 2b. The two commands anyone needs: ft-up / ft-down on the user's PATH (~/.local/bin).
install -d -o "$USER_NAME" -g "$USER_NAME" "$HOME_DIR/.local/bin"
for cmd in ft-up ft-down; do
  ln -sfn "$REPO/scripts/$cmd" "$HOME_DIR/.local/bin/$cmd"
done

# 3. Make it effective for the CURRENT session: raise the limit on the live user manager
#    (children started from now on inherit it). Best effort; the system unit never needs it.
mgr_pid=$(pgrep -u "$USER_NAME" -x systemd | head -1 || true)
if [ -n "$mgr_pid" ]; then
  prlimit --pid "$mgr_pid" --memlock=unlimited:unlimited && \
    echo "raised memlock on the running user manager (pid $mgr_pid)"
fi

if [ "${1:-}" = "--enable" ]; then
  systemctl enable freetoken-serve
  echo "enabled at boot (it shares the GPU with whatever else is running there)"
fi

echo "installed /etc/systemd/system/freetoken-serve.service for $USER_NAME ($REPO)"
echo "installed ft-up / ft-down in $HOME_DIR/.local/bin (bring the server up / hand the GPU back)"
echo "installed memlock=unlimited for $USER_NAME: user@$UID_NUM drop-in, system/user manager defaults, limits.d"
echo "NEXT (as $USER_NAME, after the first successful start): scripts/tune-memory-ratio.sh"
echo "  -> binary-searches the largest --memory-ratio THIS GPU + model can serve (tries 1.00 first,"
echo "     bisects down only on failure) and persists it in $HOME_DIR/.config/freetoken/serve.env."
echo "     Free VRAM is wasted expert slots; another host's ratio does not transfer."

# 4. WSL: the VM's RAM cap is set on the Windows side and must hold the pinned banks.
if grep -qi microsoft /proc/version; then
  total_gib=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
  echo "WSL detected: this VM has ${total_gib} GiB RAM. Pinning needs ~expert banks + 4 GiB (>= 20 GiB"
  echo "for Nemotron 3.5 Lightning) -> set [wsl2] memory=<N>GB in %USERPROFILE%\\.wslconfig on Windows"
  echo "if it is smaller, then 'wsl --shutdown'. /etc/wsl.conf must keep [boot] systemd=true."
fi
