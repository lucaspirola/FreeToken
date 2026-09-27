# Step 5: DMA-writeback ring size rule on Ornith EXL3 (box ft-dev, RTX 5080, 2026-09-24)

All numbers are box numbers. Setup: s5wt.sh, ratio 1.00, q8_0 KV, two greedy chats, probe 8K/32K/80K/128K x2.

| arm (files s5-<arm>-r1.00-*) | ring | p1 decode tok/s | p2 decode tok/s | staged / all writebacks |
|---|---|---|---|---|
| r32lag1 | 32 (old cap), lag 1 | 73.9/97.4/94.1/92.1 | 118.0/115.2/103.0/99.2 | 83.0% |
| r256lag1 | 256 (reserve cap) | 79.0/95.9/92.7/90.4 | 118.0/114.6/102.4/99.0 | 96.0% |
| r256trace | 256 + FREETOKEN_MIRROR_WB_TRACE | 79.2/95.8/92.7/90.3 | 117.9/114.2/102.4/99.0 | trace: ring2-exl3-256.trace |
| **rule** | **66 (b85402f rule)** | **81.2/98.0/94.6/92.3** | **118.1/115.0/103.0/99.6** | **94.7% (99.0% of ring-eligible)** |

The two greedy chats are byte-identical across all arms. With the trace, per-step writebacks are
p50 15, p95 52, p99 75, max 187 (ring_replay.py). The rule and its derivation are in exp/mirror-dma-wb,
`tasks/exclusive-expert-ram/results/ringrule-box/README.md`.
