#!/usr/bin/env bash
# Decode round (ab3.sh) then the checkpoint gate ck9s + 8-dir suite (gate-chain.sh), back to back.
F=$(dirname "$(readlink -f "$0")")
$F/ab3.sh
$F/gate-chain.sh > $F/gate-chain.log 2>&1
