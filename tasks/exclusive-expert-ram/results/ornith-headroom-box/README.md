# Prefill headroom on native Linux (rented RTX 5080, 2026-09-24)

Diagnostics for branch `reorg-headroom` (code 82207c8), not R-S12a records: the owner's
numbers come from the owner's machine.

Box: RTX 5080 16 GB, driver 595.58.03, native Linux in Docker (no WDDM, no systemd), venv
/root/venv, clone /root/FreeToken-headroom (extensions built in place). Runner:
`ARMS="whole saver nemotron" tasks/exclusive-expert-ram/headroom-box.sh` (FT_LAUNCHER=nohup,
/root/gpu.lock held per arm), ratio **1.00** for every arm. Ornith: `--num-tokens 262144`,
growable KV 65536-token steps. Nemotron: `scripts/serve-default.sh` as is (1M tokens).
`*-headroom.txt` = the acceptance lines grepped from each journal.

## What changed

Before (exp/reorg 0b2d133, results/ornith-s12a-grow-ab-box): after a grow that shrinks the
arena, the engine guaranteed only the 256 MiB VMM cushion + 128 MiB free; the startup plan
reserved nothing for activations at ratio 1.00. Ratio 1.00 OOMed at the LinearStatePool
before graphs; 0.99-0.94 in the warmup or the first 8K chunk; 0.91 in the chunk after the
65536 -> 131072 grow.

After: the engine measures one full `--max-prefill-length` chunk at startup (empty prefix,
and after a one-chunk prefix) and every growable-KV headroom prices
`max(VMM cushion, transient)` (+128 MiB margin as the shrink/fill target). The arena is
parked at its floor while pools, graphs and the measurement allocate, then filled back
to what leaves that headroom free.

## Measured transient (8192-token chunk)

| model | empty prefix | after 8K prefix | allocator peak | headroom |
|---|---|---|---|---|
| Ornith 1.5 35B-A3B NVFP4 | 0.98 GiB | 0.98 GiB | 0.97 GiB | 0.98 + 0.12 GiB |
| Nemotron 3.5 Lightning 30B-A3B NVFP4 | 0.65 GiB | 0.65 GiB | 0.59 GiB | 0.65 + 0.12 GiB |

(~125 and ~83 KiB per chunk token; the pre-measurement estimate is 128 KiB/token.)

## Ratio 1.00 acceptance

TTFT (monotonic) s / prefill tok/s / decode tok/s:

| arm | size | pass 1 | pass 2 |
|---|---|---|---|
| ornith-whole (FT_ROWS=0) | 8K | 0.19* / 41,339 / 158 | 0.74 / 10,881 / 151 |
| | 32K | 3.86 / 8,295 / 142 | 3.84 / 8,348 / 145 |
| | 80K | 15.38 / 5,203 / 132 | **15.37 / 5,205 / 130** |
| | 128K | 35.35 / 3,622 / 122 | 35.31 / 3,626 / 121 |
| ornith-saver (FT_ROWS=-1) | 8K | 0.18* / 45,369 / 98 | 0.74 / 10,895 / 140 |
| | 32K | 3.87 / 8,275 / 107 | 3.82 / 8,376 / 138 |
| | 80K | 15.90 / 5,034 / 88 | **15.50 / 5,163 / 91** |
| | 128K | 35.85 / 3,572 / 98 | 35.71 / 3,585 / 105 |
| nemotron-whole | 8K | 0.71 / 11,242 / 171 | 1.97 / 4,073 / 152 |
| | 80K | 12.38 / 6,466 / 151 | 12.35 / 6,480 / 154 |
| | 256K | 70.08 / 3,653 / 122 | 68.41 / 3,743 / 138 |
| | 713K | 415.88 / 1,715 / 100 | 414.59 / 1,720 / 103 |

*8K pass 1 hits the warm-up's prefix cache. Journals: captures=1, tracebacks=0, OOMs=0 in
all three (kv_grows 4 / 4 / 28). Saver: coverage_faults 0, ram_gib 14.71 (whole 20.82).
The Ornith 80K numbers match the ratio-0.85 A/B of ornith-s12a-grow-ab-box (15.36-15.43 s).

## Arena and coverage floor, before -> after (ratio 1.00)

| | Ornith whole | Ornith saver | Nemotron |
|---|---|---|---|
| ratio plan (`--moe-cache-auto`) | 6510 | 6510 | 2296 |
| usable at ready | 6510 (did not start) -> **5800** | **5808** | 2296 -> **2136** |
| free after startup | OOM -> 1.11 GiB | 1.11 GiB | 0.78 GiB |
| after the 131072 grow | - -> 5376 (post-commit free 1.13 GiB) | 5384 | 1752 at 720896 tok |
| ceiling plan (final arena) | 5152 -> **4672** | 4672 | 1584 at 1M |
| mirror coverage floor | - | 4600 (pool 6412 rows, 10.60 GiB) | - |

The Ornith "before" ceiling plan (5152) is from the 1.00 startup journal in
ornith-s12a-grow-ab-box. -480 slots = the 0.73 GiB of transient above the cushion plus the
live budget (graphs/workspaces the ratio arithmetic never saw). The saver pool is sized
from the 1.0 GiB estimate: with the old cushion-only estimate its floor would sit ~450
slots higher (~5050, 0.75 GiB / 1.69 MiB per slot) -- above the 4672 plan, so the load
would be refused. Here the floor clears the plan by 72 slots.

## Tests on the box

`PYTHONPATH=$PWD/python /root/venv/bin/python -m pytest -p no:cacheprovider --timeout 300 -v
tests/moe tests/engine tests/scheduler` under /root/gpu.lock, model unloaded: 971 passed,
7 skipped (marlin/b12x backends, the mm repeated-image test), no timeout (pytest-box.log).
