# Closing the checkpoint gap on gen5: the dma-wb levers (box ft-g5 and ft-dev, 2026-09-24)

All numbers here are box numbers:
- ft-g5 is a rented RTX 5080 with PCIe gen5. Pinned H2D is 56.6 GB/s and D2H 39.9 GB/s
  (`levers-nsys/pcie_bw-ftg5.txt`).
- ft-dev is a rented RTX 5080 with PCIe gen4.

Common to every run: Nemotron-3.5-Lightning NVFP4, ratio 1.00, port 1920, one arm at a time,
two probe passes. Pass 2 is the number of record.

The pool arm keeps the whole model in RAM as whole ∪ mirror, bounded (`FREETOKEN_MIRROR_HOST_ROWS=-1`,
reserve 256). The target was pool decode within 9% of whole.

## The levers (exp/mirror-dma-wb)

| lever | commit | what it changes |
|---|---|---|
| L1 | d1ed596 | No copy nodes in the decode graph. `begin_layer` is one fused Triton kernel (victim/prior fills + the layer's `prev_slot` slice). The stats and writeback-ring snapshots moved to a side stream that waits on the compute stream: the per-layer D2H and D2D memcpy nodes had queued behind the writeback DMA on the copy engine. |
| L2 | 40c154d | Writebacks are issued after the forward is launched. |
| L3 | 2392b93 | The mirror bookkeeping moved into the v2 LRU kernel (`MIRROR_BOOK`), one node per MoE layer fewer. |
| L4 | e70849e | Duplicate-aware eviction (`FREETOKEN_MIRROR_DUP_BAND`, default on). Among candidates whose LFU count is within max(1, √min) of the coldest (Poisson noise of the count), the coldest with a pool row is evicted, which makes the eviction free. |
| L5 | e7e029f | `fault_check` and the ring snapshot run after the launch, off the host critical path. |

Every lever keeps one capture, 0 coverage faults and byte-exact pools (tests). GPU tests:
- ft-g5: 768 passed at e70849e and at e7e029f.
- ft-dev: 768 passed at e7e029f.

## What each lever buys (ft-g5 nsys, 8K decode, pass 2, 121 steps)

| | step period | graph GPU | D2H writeback | host sync → launch | launch call → graph start |
|---|---|---|---|---|---|
| whole | 5375 us | 5096 us | – | ~150 us | ~65 us |
| L1 | 5827 us | 5320 us | 14.40 MB/step | ~254 us | ~177 us |
| L2 | 5821 us | 5315 us | 14.40 MB/step | 251.6 us | 184.1 us |
| L3 | 5806 us | 5303 us | 14.40 MB/step | 254.2 us | 178.2 us |
| L4 | 5786 us | 5285 us | **8.83 MB/step** | 253.8 us | 177.4 us |
| L5 | **5739 us** | 5282 us | – | **206.8 us** | 179.7 us |

Files: `gap-nsys/steps-graph-level.txt`, `levers-nsys/L*-steps.txt` and `L*-host-gap.txt`.

- **L1** removed the copy nodes. It took the pool from 87.5-92.4% of whole (773d9f8) to the
  gap1 numbers below.
- **L2 and L3** are each worth 0-20 us/step on gen5. On gen4 they were neutral too (gapdev2/3).
- **L4** cut writeback bytes by 39%.
- **L5** takes 47 us off the host gap.

**Remaining gap.** Graph GPU time is still +186 us over whole: the in-graph miss fetch, ~630 us/step,
serial. The other half is host time inside `cudaGraphLaunch`: 176 us for the pool graph vs 63 us
for whole. The launch call returns 2 us before the graph starts, so the GPU is not what waits.
- **Microbench.** A straight-line 892-kernel graph replays in 3.2 us (`bench_graph_launch.py`).
- **Ruled out** (`bench_graph_launch2.py`, `launchbench.txt`). None of these moves the 3.2 us:
  92 distinct Triton functions, kernels reading 4 GiB pinned or 8 GiB `cudaHostRegister`'d memory,
  kernels writing it, 60-argument kernels, 92 separate registrations.
- **Status.** The cause is still open. What differs in the real server is still untested: other host
  threads that make CUDA calls, and the driver lock.

## Duplicate-aware eviction (L4) A/B (ft-dev, e70849e, `--moe-collect-stats`, mirror-np)

| DUP_BAND | free-eviction rate | writebacks/token | hit rate | swaps/token | 8K p2 | 80K p2 |
|---|---|---|---|---|---|---|
| 0 (one-bucket tie-break) | 0.708 | 3.33 | 0.9479 | 7.27 | 161.5 | 151.8 |
| 1 (band) | **0.802** | **2.69** | 0.9466 | 7.45 | 162.3 | 151.5 |

- The hit rate drops by 0.13 points.
- The bands' unit test rate was 0.8850 (tie-break) vs 0.9809 (band).
- Files: `../bandd0-box`, `../bandd1-box`.

## Final: pool vs whole on ft-g5 (L5 e7e029f, `../gap5-box`)

| context | whole p1 / p2 | pool p1 / p2 | pool / whole p1 / p2 |
|---|---|---|---|
| 8K (mirror-1m arm) | 184.6 / 184.7 | 176.1 / 169.2 | 95.4% / **91.6%** |
| 8K (mirror arm) | 184.6 / 184.7 | 175.1 / 167.7 | 94.9% / **90.8%** |
| 80K (mirror arm) | 169.5 / 172.1 | 160.6 / 163.9 | 94.8% / **95.2%** |
| 1M (mirror-1m arm) | 88.5 / 91.7 † | 81.1 / 88.3 | 91.6% / **96.3%** |
| 713K (mirror arm) | – | 96.1 / 106.6 | |

† The 1M whole reference is gap1-whole-1m: same box, d1ed596. The whole path does not touch any
lever. The L5 whole-1m arm was aborted at its start because the ft-g5 contract ended at 17:00 UTC.

- **Correctness.** Needles/recall show 0 differences vs whole (`gap5-needles-compare.txt`, 17 SAME
  at 21K/120K/240K). captures=1 on all three arms (R3). coverage_faults 0.
- **Before the levers** (gap1 at L1 → L5, pass 2):
  - 8K 166.5 → 169.2 and 168.9 → 167.7;
  - 80K 165.4 → 163.9;
  - 1M 86.5 → 88.3.
- **Verdict.** The target is met at 80K and 1M. 8K pass 2 sits on the line: 91.6% and 90.8% in the
  two pool arms.
- **Queue cut short.** The lev2-5 mirror-np arms and the g5 band arms were cut by the contract end.
  The per-lever effects above come from the nsys traces; the band A/B ran on ft-dev.

## No regression on gen4 (ft-dev, mirror-np, pass 2)

| code | 8K | 80K |
|---|---|---|
| ring rule, before the levers | 153.1 | 143.0 |
| L1 (gapdev) / L3 (gapdev3) | 161.5 | 152.2 |
| L4 (gapdev4) | 162.6 | 151.9 |
| L5 (l5dev) | **164.3** | **153.5** |
| whole-close (gapdev3) | 171.4 | 159.0 |
