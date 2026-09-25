# reorg-headroom fix on Ornith EXL3 5.0bpw + RAM saver — BOX NUMBERS (ft-dev Vast RTX 5080), 2026-09-24

Code: scratch/exl3-headroom @ 58ee8da = exp/ornith-exl3 5131557 + reorg-headroom 82207c8..38e6790
cherry-picked (not merged into exp/ornith-exl3; waits for ck4 on ft-ck). Tests on the box:
EXL3 suites + tests/engine/test_prefill_headroom.py + tests/moe/test_mirror_headroom.py -> 303 passed.
Server: `step5.sh` TAG=hr RATIO=1.00 (same flags as step 3 + --cuda-memory-telemetry), probe
8K/32K/80K/128K x2, VRAM sampled every 1 s (`s5-hr-r1.00-vram.tsv`).

## Q1 — does the startup measurement cover the FLA/GDN prefill workspace? Yes.
`Prefill headroom: transient 1.00 GiB measured on a 8192-token chunk (empty prefix 1.00 GiB, after a
8192-token prefix 1.00 GiB, allocator peak 0.97 GiB); headroom 1.00 GiB ...; expert arena 4368 -> 5432
of 6076 slots, 1.13 GiB free`. The measurement is a real `_prefill_forward` of a full chunk through
every layer (GDN FLA kernels, triton attention, EXL3 dense GEMM, EXL3 MoE prefill), empty prefix and
extend. In serving: 4 growable-KV commits to 131072 tokens, each shrinking the arena 5408 -> 5056 to hold
the 1.67 GiB required; peak VRAM on the 1 s trace 15736 / 16303 MiB (>= ~0.55 GiB never touched),
zero OOM over 2 chats + 8 probe requests up to 128K. Before the fix the same shape died at 0.90/0.95
with 28-42 MiB free in FLA `wy_fast` / `chunk_delta_h` right after a grow that did not shrink the arena.

## Q2 — the ratio-1.00 LinearStatePool miss at startup: same commit, different mechanism.
It happened at engine init (LinearStatePool, 540 MiB asked, 526 MiB free), before any prefill and before
any measurement exists, so it is not the post-grow transient. 82207c8 cures it by *parking* the expert
arena at its mirror-coverage floor right after the cache is built ("Expert arena parked 6076 -> 4368 slots
(3.15 GiB) until the prefill transient is measured"): KV/state pools, page table, graph capture and the
measurement all allocate with 3 GiB spare, and only then does the arena refill to headroom. Without the
parking the ratio-1.00 plan hands the arena everything and init-time allocations the plan does not price
(14-32 MiB of them here) OOM.

## Probe (ratio 1.00, pass 2 of record; pass 1 in brackets)

| prompt | TTFT s | prefill tok/s | decode tok/s |
|---|---|---|---|
| 8K   | 10.33  | 775 (774) | 43.6 (24.1) |
| 32K  | 41.76  | 767 (766) | 43.3 (30.4) |
| 80K  | 108.82 | 735 (732) | 35.7 (30.0) |
| 128K | 183.23 | 699 (698) | 36.8 (33.3) |

vs ratio 0.85 without the fix (step4-probe): prefill identical (not expert-slot bound), decode +20% at
8K/32K from ~560 more resident experts. Chats identical to the 0.95 run (542 / 930 tokens, finish=stop).
