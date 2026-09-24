#!/bin/bash
# B: pre-fix cf5d2c8 vs post-fix 9dac3b5, mirror arm (8K/80K/713K), both at ratio 0.90.
date -u
cd /root/FreeToken-prefix && FT_RATIO=0.90 ARMS=mirror tasks/exclusive-expert-ram/checkpoint-box.sh ck4pre90
cd /root/FreeToken-headroom && FT_RATIO=0.90 ARMS=mirror tasks/exclusive-expert-ram/checkpoint-box.sh ck4post90
date -u
echo B-DONE
