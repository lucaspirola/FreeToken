# DMA-writeback staging ring: size rule (box ft-dev, RTX 5080, 2026-09-24)

All numbers are box numbers (ft-dev, rented RTX 5080, ratio 1.00, one arm at a time, port 1920/30100).

## Rule (exp/mirror-dma-wb 4206a76, scratch/exl3-dma b85402f)

    rows = ceil(top_k * moe_layers * decode_batch * uncovered / 2),   4 <= rows <= reserve // 3
    uncovered = 1 - warm-start arena slots / (moe_layers * experts)

This replaces the fixed 96 MiB / 32-row cap. The rule has no per-model constants. The ring has to hold
about one step's writebacks. Under lag-1 servicing, each MoE layer's resolve re-reads the landed count,
so the previous step's DMAs free the ring early in the next step. `ring_replay.py` with
`REPLAY_PHI=0.2` reproduces the measured staged share of the 32-row EXL3 run and the 17-row Nemotron run.

The per-step writebacks were traced with `FREETOKEN_MIRROR_WB_TRACE` on runs whose ring never filled:

| model | ceiling top_k x L | uncovered | p50 | p95 | p99 | max | rule |
|---|---|---|---|---|---|---|---|
| Ornith EXL3 (256-row ring) | 320 | 0.41 | 15 | 52 | 75 | 187 | 66 |
| Nemotron NVFP4 (85-row ring) | 138 | 0.22 | 4 | 14 | 20 | 83 | 16 |

Each ring row costs one arena slot for the whole decode:
- Nemotron: 85 rows instead of 17 means 10% more swaps and 4% slower decode.
- EXL3: 256 rows instead of 32 means 11% more swaps.

## Ornith EXL3 5.0bpw + RAM saver (scratch/exl3-dma, s5 probe, decode tok/s)

| ring | p1 8K/32K/80K/128K | p2 8K/32K/80K/128K | staged / all writebacks | staged / ring-eligible | swaps |
|---|---|---|---|---|---|
| off (SM stores, control) | 40.5/58.8/55.8/58.7 | - / - /76.6/81.6 | 0 | 0 | |
| 32, lag 2 (773d9f8) | 63.6/96.0/91.5/88.8 | - / - /102.4/99.0 | 61% | | |
| 32, lag 1 | 73.9/97.4/94.1/92.1 | 118.0/115.2/103.0/99.2 | 83.0% | 86.7% | 63775 |
| 256, lag 1 | 79.0/95.9/92.7/90.4 | 118.0/114.6/102.4/99.0 | 96.0% | 100% | 70845 |
| **rule = 66, lag 1** | **81.2/98.0/94.6/92.3** | **118.1/115.0/103.0/99.6** | **94.7%** | **99.0%** | 64763 |

The 1536 prefill-buffer writebacks are SM stores that never go through the ring. The greedy chats were
byte-identical across the 32, 256 and rule runs.

## Nemotron-3.5 Lightning NVFP4 saver (exp/mirror-dma-wb, checkpoint-box mirror-np, decode tok/s)

| ring | p1 8K/80K | p2 8K/80K | staged / ring-eligible | swaps |
|---|---|---|---|---|
| 85 (reserve cap) | 141.8/127.9 | 150.4/139.0 | 100% | 5042 |
| 17 (old 96 MiB default) | 147.3/134.4 | 154.0/142.9 | 94.5% | 4591 |
| **rule = 16** | **146.6/134.3** | **153.1/143.0** | **94.2%** | 4619 |

For reference, whole-model decode is 171/159 (wb1-box). The prefill-buffer writebacks (768) are
outside the ring, as for EXL3.

Files: `ringrule-*` (rule arm), `../ring17-box`, `../ring85-box`, `../ring85trace-box`.
The traces are `ring85trace-nemotron.trace` and `ringrule-nemotron.trace`. The EXL3 runs and traces are
in scratch/exl3-dma `tasks/ornith-exl3/results/step5-ring-rule-2026-09-24/`.

## Answers (`../ringrulend-box`, rule = 16, arm mirror-nd, 4206a76)

The needles (21K/120K) and the recall (21K/120K/240K) match the whole model's (wb0-whole) field by
field: 0 differences (`ringrulend-needles-vs-wb0-whole.txt`). Decode on the same arm: p1
146.5/133.9, p2 153.2/142.9 tok/s at 8K/80K, the same as the mirror-np rule arm above. R3 PASS.
