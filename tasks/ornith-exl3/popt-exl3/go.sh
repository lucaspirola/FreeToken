#!/usr/bin/env bash
# go.sh NAME OUT cmd... : run gpu.sh as a transient user unit popt-exl3-NAME and wait for it
F=$(dirname "$(readlink -f "$0")")
systemd-run --user --quiet --wait --collect --unit "popt-exl3-$1" -E TREE=${TREE:-} -p WorkingDirectory=$F bash $F/gpu.sh "$F/$2" "${@:3}"
tail -3 "$F/$2"
