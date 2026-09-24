# dma-wb on g5: ck4 arms on exp/mirror-dma-wb 773d9f8, and an nsys trace of the 8K gap

Box: ft-g5 (Vast 52354198, RTX 5080, PCIe gen 5, native Linux, driver 580), the same box as `../ck4g5-box`.
Code: the python/ tree of 773d9f8 (git tree hash 27a1256, verified), applied on 9dac3b5 as box
commit 7e8b7dc in the worktree /root/FreeToken-dma. The applied patch is python/ scripts/ tests/
and the tasks/*.sh|*.py of `git diff 9dac3b5 773d9f8`.
Run: `checkpoint-box.sh ck4dma` with ARMS="whole mirror-1m mirror whole-1m", 07:20Z to 09:20Z,
driven by `g5-dma.sh`. The flags are the same as ck4g5: ratio 1.00, port 1920, reserve 256.
The log confirms the DMA writebacks are on: "mirror DMA writebacks: 17-row VRAM staging ring
(91.1 MiB)".
`ck4dma-attempt1/` holds a first launch that died at start: the new worktree lacked the
prebuilt `_pinned_tensor` extension, which was then copied in from /root/FreeToken-headroom.

## Result: the DMA writebacks do not close the gap on this box

| point | whole (same box) | pool, dma-wb | ratio | pool, SM-store (ck4g5) |
|---|---|---|---|---|
| 8K p1 / p2 (mirror-1m) | 184.6 / 184.8 | 162.1 / 161.7 | 87.8% / 87.5% | 162.1 / 162.4 |
| 8K p1 / p2 (mirror) | 184.6 / 184.8 | 162.2 / 162.7 | 87.9% / 88.0% | 162.1 / 162.6 |
| 80K p1 / p2 (mirror) | 169.4 / 172.0 | 149.6 / 159.0 | 88.3% / 92.4% | 149.1 / 159.2 |
| 1M p1 / p2 (mirror-1m vs whole-1m) | 88.1 / 91.6 | 74.5 / 82.8 | 84.6% / 90.4% | 74.7 / 81.2 |

(`ck4dma-box/ck4dma-box-compare.txt`; wall-clock decode. This box runs native Linux, whose
clock does not step, so wall and monotonic agree.)

Other gates:
* faults 0, starved 0;
* needles/recall 0 differences vs ck4dma-whole;
* captures 1 and R3 PASS on every arm;
* 1M RAM 12.18 GiB, PASS;
* R6 fails as an environment limit, since Vast caps memlock at 64 KiB.

Against the owner record, 9 of 10 points pass (`ck4dma-records-compare.txt`); 8K p1 is at 90.0%.
Decode is unchanged from the SM-store writebacks at every point: on gen 5, SM stores
already reach 37-50 GB/s. The writeback bandwidth was ft-dev's (gen 4) problem, not this box's.

## Transient-reservation A/B at 8K (`../ck4g5t-box`, code 9dac3b5)

| arm | arena slots | decode 8K p1 / p2 | decode expert hit rate |
|---|---|---|---|
| whole-8k-def | 2136 of 2298 | 183.9 / 183.8 | 0.9227 (4137 missing of 53552) |
| mirror-8k-def | 2136 of 2298 | 162.7 / 171.5 (88.5% / 93.3%) | 0.9430 (3051 missing), 8.0 swaps per token |
| whole-8k-t0, mirror-8k-t0 | did not start | | |

* `-t0` means FREETOKEN_PREFILL_TRANSIENT_MEASURE=0 plus FREETOKEN_PREFILL_TRANSIENT_MB=0,
  i.e. the pre-fix cushion-only headroom.
* The `-t0` arms OOM at startup: `linear_state_pool` asks for 598 MiB with 556 MiB free.
  Without the measurement the arena is never parked, so ratio 1.00 leaves no room for the pool.
  This is the same start OOM the pre-fix cf5d2c8 hit at 1.00 on ft-ck (`../ck4-prepost-box`).
* The default arms already answer the question for this box: the pool's decode hit rate is
  HIGHER than whole's, yet the pool decodes 12% slower. The 8K gap on ft-g5 is not a hit-rate or
  slot-count effect. (This mirror arm's p2 of 171.5 is the one pool 8K pass on this box above
  91%; the other five 8K pool passes here read 161.7-162.7.)

## nsys: where the 8K step time goes (`ck4dma-nsys/`)

One 8K request, two passes, 128 tokens each. The trace starts after three warmups, on the same
code and flags; the server runs under `nsys launch`, and `nsys start/stop` brackets the request.
* `whole.nsys-rep` and `mirror.nsys-rep`: `--cuda-graph-trace=graph`.
  `steps-graph-level.txt` comes from `nsys_steps.py`. A step is one graph launch, and its window
  runs to the next launch.
* `*-node.nsys-rep`: `--cuda-graph-trace=node`. `in-graph-node-level.txt` comes from
  `nsys_graph_nodes.py`.

Per decode step, pass 2 (graph-level trace):

| | whole | pool (mirror, dma-wb) | delta |
|---|---|---|---|
| step period | 5549 us (180 tok/s) | 6149 us (163 tok/s) | **+600 us** |
| decode graph on GPU | 5101 us | 5502 us | +401 us |
| between graphs: D2H memcpy (writebacks) | 0.3 us | 372 us, 14.7 MB (17.7 copies), streams 13 and 17 | +372 us |
| GPU busy (union) | 5134 us | 5908 us | +774 us |
| GPU idle | 415 us | 241 us | -174 us |
| cudaGraphLaunch (host) | 166 us | 255 us | +89 us |

Inside the graph (node-level trace), per step:
* The miss fetch runs on SM copies in both arms: whole 557 us (`fast_index_copy_multi`, 23/step),
  pool 615 us (`fast_index_copy_kinds`, 46/step). That is +58 us.
* The pool's bookkeeping adds 46 elementwise kernels, +32 us.
* The pool adds 23 D2H and 23 D2D memcpy nodes, 30 us of copy time.
* In-graph kernel time totals 5004 vs 5179 us (+175 us). The rest of the +401 us of graph
  duration (about 220 us) is bubbles between nodes; the 46 memcpy nodes per step are the new
  dependency points.

Reading:
1. About 370 us per step is the DMA writeback, 14.7 MB at about 40 GB/s D2H. It runs between
   graphs, so it sits on the critical path; the next step's graph does not overlap it. The
   GPU's idle time hides about 175 us of it.
2. About 400 us is the graph getting longer. Only about 90 us of that is extra kernel work.
   About 220 us is serialization inside the graph around the per-layer D2H/D2D copy nodes.
3. The host-side launch cost of the bigger graph (+89 us) is mostly hidden.

Fix directions (not implemented):
* Let the writeback DMA overlap the next step's graph on its own stream, and wait on it only
  when a staging row is reused.
* Replace the per-layer memcpy nodes in the graph with kernel writes, or batch them, so the
  graph has no copy-engine dependency points.
80K has the same shape (`steps-graph-level-80k.txt`, one 80K request, graph-level):

| pass | step period whole / pool | delta | graph delta | between-graph D2H (writebacks) |
|---|---|---|---|---|
| p1 | 5861 / 6703 us | +842 us | +464 us | 437 us, 17.2 MB |
| p2 | 5914 / 6342 us | +428 us | +401 us | 254 us, 9.9 MB |

The graph grows by a steady ~400-460 us per step. The writeback term follows the step's miss
traffic, and it is larger on p1, which matches the p1 < p2 pattern in the ck4 tables.
