# DMA writebacks for the bounded mirror (exp/mirror-dma-wb 773d9f8), ft-dev

Box numbers: ft-dev (Vast, RTX 5080, PCIe gen 4, EPYC 7402P, docker), Nemotron-3.5-Lightning
NVFP4, ratio 1.00, reserve 256 rows, one arm at a time on port 1920. Before = `../wb0-box/`
(dc043d6: reorg-headroom head + tooling, same box, same session). After = this directory.

## What changed

A victim with no pool duplicate was SM-stored into its mapped pinned pool row from inside the
decode graph. The resolve kernel now copies it into a 17-row VRAM ring (91 MiB), and the host
issues ring -> pool `cudaMemcpyAsync` between steps. Every decision (victims, rows, retention,
free stack) is unchanged. `FREETOKEN_MIRROR_WB_STAGE_MB=0` restores the old byte path.

## Copy primitives (`bench_mirror_copy.py`, 5.36 MiB rows, 4 banks)

| per writeback, n=1..8 rows | ft-dev | ft-ck (second host, coordinator's run) |
|---|---|---|
| SM store to pinned pool (before, on the decode stream) | 2.1-2.4 GB/s, 2610 us/row | 12.2-13.8 GB/s |
| copy-engine D2H (after, off the decode stream) | 28.1-28.8 GB/s | 28.2 GB/s |
| slot -> VRAM ring (after, what the decode stream pays) | 23 us/row (240-323 GB/s) | not measured |
| admission, SM loads (unchanged; kinds kernel = old kernel) | 25.6-26.2 GB/s | 24-26 GB/s |

## Nemotron decode, tok/s (probe_decode.py, 127 generated tokens)

| arm | 8K p1 | 80K p1 | 8K p2 | 80K p2 |
|---|---|---|---|---|
| whole (wb0) | 171.4 | 154.7 | 171.2 | 159.0 |
| saver before (wb0-mirror-nd) | 83.0 | 72.5 | 104.5 | 95.4 |
| **saver after (wb1-mirror-nd)** | **147.2** | **134.5** | **153.9** | **143.4** |
| whole bracket after (wb1-whole-close) | 171.5 | 154.9 | 171.3 | 159.2 |

The saver's pass-2 gap to whole goes from 39%/40% to 10%/10%. The whole-model path is unaffected
(bracket within 0.2 tok/s). Prefill is unchanged (8K p1 19.9K vs 20.5K tok/s, 80K 6644 vs 6657).

## Correctness

* Needles 21K/120K (7 kinds each) and recall 21K/120K/240K: 17 of 17 answers identical to whole
  (`wb1-needles-vs-wb0-whole.txt`; before: `../wb0-box/wb0-needles-compare.txt`, also 0 differences).
* Coverage faults 0, starved writebacks 0. 30,817 writebacks, of which 27,348 staged and DMA'd
  (the rest are ring-full fallbacks and prefill-buffer writebacks, both still SM stores). 2,122
  admissions read a still-pending ring slot.
* R3: one graph capture, 17 KV grows, 0 tracebacks.
* GPU tests on the box, `wb1-gpu-tests.txt`: 548 passed, including
  `tests/moe/test_mirror_dma_writeback.py` (8: byte-exact admissions and pool rows under
  per-step, lagging and absent service; a 4-row ring overflowing into the fallback; identical
  decisions, rows and bytes vs SM stores; graph replay) and the swap_smoke/graph_race_repro device
  scripts.

## Not in this run

The Ornith EXL3 end-to-end is `scratch/exl3-dma` (exp/ornith-exl3 + reorg-headroom picks + this
commit), measured separately.
