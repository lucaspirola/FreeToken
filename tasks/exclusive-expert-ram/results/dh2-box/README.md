# dh2: decode-level free target A/B, default 0.375 GiB vs 128 MiB (ft-dev RTX 5080; box numbers)

exp/decode-headroom a2f08c0, Nemotron NVFP4 mirror, ratio 1.00, `FREETOKEN_DECODE_MEM_PROBE=1`,
`--moe-collect-stats`, FT_GEN=512, checkpoint-box.sh. Arms in order dtdef-1, dt128-1, dtdef-2, dt128-2
(mirror-np, 8K/80K; A/B/A/B), then dt128m (mirror-1m, 8K/1M, needles). Results in `../dt*-box/`,
table from `dh2tab.py` in `dh2tab.txt`, run log `dh2-status`.

Decode tok/s p1 / p2, MoE decode hit rate, slots after the prefill-headroom release:

| Arm | 8K | 80K | hit | slots after release |
|---|---|---|---|---|
| dtdef-1 | 162.0 / 166.6 | 153.7 / 156.1 | 0.9624 | 2144 |
| dt128-1 | 164.2 / 170.1 | 156.9 / 158.8 | 0.9657 | 2192 |
| dtdef-2 | 161.9 / 166.5 | 153.7 / 156.2 | 0.9624 | 2144 |
| dt128-2 | 164.3 / 170.0 | 156.8 / 158.8 | 0.9657 | 2192 |

Run-to-run spread is 0.1 tok/s; 128 MiB is +3.5 tok/s (+2.1%) at 8K p2 and +2.6 (+1.7%) at 80K p2,
from 48 more expert slots during decode. Every np arm: 5 decode windows, drop 0.00 GiB, 0 coverage
faults, 0 starved write-backs, one capture, 0 tracebacks, 0 OOM.

dt128m (1M): 8K 164.4 / 165.6, 1M 71.0 / 72.7 tok/s, hit 0.9972, RAM 12.2 GiB (band 12.26 +/- 0.6 ok),
0 faults, 0 starved, one capture. 27 windows; 25 at 0.00, the 2 windows at 1M context drop 0.02 GiB
(min free 0.12 GiB), the same 2 windows and the same 0.02 GiB drop as the default-target arm
dh-mirror-1m (`../dh-box`, same commit: min free 0.37 GiB; 1M 68.5 / 69.9, not bracketed with dt128m).
Needle answers (21K, 120K) identical to dh-mirror-1m, only timings differ. The compare_* tracebacks in
`dh2-1m.log` are the missing whole/np reference arms of a 1M-only run (fixed later in 8e1d5aa).

Decision: 128 MiB wins by the criterion (decode better beyond the spread, no window short of the target,
0 faults, 0 starved). The 0.02 GiB at 1M is a decode need both targets show and 128 MiB covers; the
owner's WSL host shows the same (`../dhw-wsl`).

Note: dt128m waited 09:15-09:30 on a gpu.lock self-deadlock (mirror-queue.sh held the lock around
checkpoint-box.sh, which locks itself); the exl3 worker killed the parent and finished the tail with
mirror-queue-tail.sh. The arm started only after that; its preflight saw GPU 2 MiB, no server, no pytest, so its numbers stand.
