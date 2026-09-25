# nsys at 1M context: whole-1m vs mirror-1m on exp/reorg-next (3d6f249 = ad93cc4 + tooling), 2026-09-25

Produced by `nsys1m-local.sh` on the owner's RTX 5080 (WSL).
* Each arm starts a fresh server, gets three 8K warm-ups, then one pass-1 1M request
  (probe_decode prompt, 1,000,032 tokens).
* `nsys_window.py` traces only decode: tokens 34 to about 233. It starts nsys from the token
  stream, so prefill is not in the report.
* The graph trace is at node level.
* Summaries:
  * `in-graph-node-level.txt` (`nsys_graph_nodes.py`);
  * `overlap.txt` (`nsys_overlap.py`): attention against everything else, with streams and
    overlap.
* The `.sqlite` exports can be regenerated from the `.nsys-rep` files and are not committed.

## Per decode step

| | whole-1m | mirror-1m | Δ |
|---|---|---|---|
| graph launch period, median (mean) | 9721 (9822) us | 10102 (10290) us | +381 us (+3.9%) |
| in-graph kernel time | 8905 us | 9500 us | +596 us |
| decode attention stage 1, grid (1,2,84) = 168 CTAs, 128 regs | 3581.0 us | 3581.4 us | 0 |
| decode attention stage 2 | 22.0 us | 21.8 us | 0 |
| expert copy kernels | fast_index_copy_multi 1537.8 us (23 launches) | fast_index_copy_kinds 2033.7 us (46 launches) | **+496 us** |
| _ensure_experts_sized_kernel_v2 | 79.9 us | 103.0 us | +23 us |
| mirror bookkeeping (_resolve_swaps 58.3, _publish_freed 16.1) | - | 74.4 us | +74 us |
| attention overlapped by other-stream kernels | 0 | 0.2 us | - |
| D2H write-back DMA, outside the graph | 0 | 48.9 copies, **43.0 MB**, 1104 us of DMA | - |
| arena slots during 1M decode | 1528 | 1512 | −16 |

Step period by quarter of the traced window (ms):
* whole: 9.81, 9.62, 9.55, 9.82;
* mirror: 10.25, 10.35, 9.90, 10.03.

## What this shows

1. **The coordinator's hypothesis is rejected.** Split-KV v2 is selected in both arms with the
   same geometry, and it takes the same 3581 us per step.
   * The expert copy kernels are in the graph, on the same stream as attention, and never run
     beside it: 0 us of overlap.
   * So the copies do not take SMs from v2's 2 CTAs per SM.
2. **The pool arm's extra cost at 1M is its expert traffic, not attention.**
   * Copy kernels that do real work (>5 us), per step:
     * whole: 11.4 fetches, mean 134 us;
     * mirror: 12.5 fetches (mostly 100-400 us) plus about 5.9 short staging copies (5-50 us)
       of write-back victims.
   * A mirror fetch takes about 40% longer than a whole fetch at similar arena size.
   * Beside the fetches runs 43 MB per step of D2H write-back DMA; at 8K it was 9.5 MB. At 1M
     the arena holds only 1512 of about 2944 experts, and every evicted expert that is not
     duplicated in the pool must be written back.
   * The likely reason the fetches are slower is that they share PCIe/host memory with that
     write-back stream. The discriminating run is not done yet: mirror-1m with
     `FREETOKEN_MIRROR_WB_STAGE_MB=0` (SM-store write-backs inside the graph instead of
     concurrent DMA), or with a larger duplicate budget.
3. **The traced window shows 3.9%, the probe window showed 9.8%.** Probe p1 was 90.2%, while
   this window (tokens 34-233) has mirror at 96.2% of whole by median step.
   * The probe's 127-token window starts at token 1, right after the 1M prefill.
   * In the first quarter here mirror is 4.5% slower, and 2-3% in the later quarters.
   * So the first decode steps after a 1M prefill likely cost the pool arm more (a cold arena,
     and many GPU-only victims to write back). That weighs on pass 1's short window.
   * p2 was already 93.5%. A trace starting at token 1 would confirm this.
4. The arena gained equally in both arms from expandable segments. At 1M decode, ck4m had
   1504 / 1488 slots (whole / mirror) and ck4n has 1528 / 1512. This does not explain why the
   probe gain differs.

No fix yet: nothing in this trace is a one-line change. The next step decides between write-back
contention and pool sizing.

## Step-1 traces (`*-s0`) and the write-back switch (`bins-s0.txt`)

These runs were made with `NS_START_AT=700`, which starts the trace in the last seconds of the
1M prefill, so decode steps 1-206 are all in the trace. The GPU host lock was held across three
arms, in this order:
* mirror-wb0 (`FREETOKEN_MIRROR_WB_STAGE_MB=0`: write-backs are SM stores inside the graph, with
  no DMA staging ring);
* whole;
* mirror.

Arena slots during 1M decode: whole 1528, mirror 1512, mirror-wb0 1528 (it has no staging ring).

Step time (ms) and write-back MB per step, in 16-step bins:

| steps | whole | mirror | mirror wb MB/step | mirror-wb0 |
|---|---|---|---|---|
| 1-16 | 10.39 | 12.29 | 28.1 | 12.75 |
| 17-32 | 9.39 | 10.83 | 70.6 | 11.98 |
| 33-48 | 9.78 | 10.51 | 70.3 | 11.66 |
| 49-64 | 9.57 | 10.01 | 55.9 | 11.10 |
| 65-80 | 9.95 | 11.07 | 66.4 | 11.77 |
| 81-96 | 9.65 | 10.12 | 56.2 | 11.19 |
| 97-112 | 10.25 | 11.47 | 74.8 | 12.23 |
| 113-128 | 9.70 | 10.45 | 66.8 | 11.64 |
| **1-128 mean** | **9.84** | **10.84 (90.7%)** | 61.1 | 11.79 (83.4%) |
| 129-206 | 9.69 | 10.12 (95.8%) | 33.6 | 10.55 (91.8%) |

The 1-128 ratio (90.7%) reproduces the probe's 1M p1 (90.2%). After step 128 the ratio is 95.8%.

Expert copy kernels per step, split into fetches (> 50 us) and the mirror's staging copies of
write-back victims (5-50 us):

| steps | whole fetches (mean) | mirror fetches (mean) | mirror staging | mirror bookkeeping |
|---|---|---|---|---|
| 1-16 | 9.9 (212 us) | 16.0 (235 us) | 3.5 | 205 us (whole 82) |
| 17-32 | 9.2 (122 us) | 12.7 (186 us) | 9.1 | 180 us |
| 33-64 | 11.2 (125 us) | 12.1 (150 us) | 9.0 | 175 us |
| 65-128 | 11.8 (135 us) | 13.4 (167 us) | 8.8 | 182 us |
| 129-206 | 10.8 (129 us) | 11.5 (144 us) | 5.0 | 174 us |

Where the pool arm's extra 1.0 ms per step over steps 1-128 comes from (steps 129+ in brackets):
* **Fetch slowdown under the write-back DMA: about 0.5 ms [+0.17].**
  * The pool arm fetches take 37 us longer on average (176 vs 139 us).
  * The slowdown tracks the write-back volume: +52% at 70 MB/step (steps 17-32), +12% at
    34 MB/step (steps 129+).
* **More misses: about 0.3 ms [+0.09].** The pool arm has 13.3 fetches per step against 11.1,
  and 16.0 against 9.9 in steps 1-16. That covers the cold start after the 1M prefill plus 16
  fewer arena slots.
* **Write-back burst: about 0.15 ms [+0.08].** It runs at 56-75 MB/step in steps 17-128, twice
  the later 34 MB/step, costing about 9 staging copies per step.
* **Mirror bookkeeping: about 0.1 ms [+0.1]**, constant (resolve_swaps, publish_freed and a
  longer ensure_experts).

The in-graph SM write-back path is not a fix. mirror-wb0 is slower everywhere: 83% over steps
1-128 and 92% after, because its in-graph write-back stores cost more than the fetch slowdown
the DMA causes. The DMA ring stays.

No fix is applied. Directions the data supports:
1. Issue the write-back DMA while decode attention runs. Stage 1 takes 3.6 ms per step at 1M and
   uses no PCIe, so write-backs would stop overlapping the fetch kernels. This would remove most
   of the ~0.5 ms fetch slowdown.
2. Cut the write-back volume at long context with a larger duplicate budget, so more victims are
   already in the pool. This costs host RAM.
