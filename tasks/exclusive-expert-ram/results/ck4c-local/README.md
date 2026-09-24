# ck4c: arena compaction A/B on the owner's RTX 5080 (WSL), commit 63c7da6. Pre-fix run

Run 2026-09-24 by `checkpoint1.sh ck4c` from the detached worktree `dt-measure` at 63c7da6
(dynamic headroom + compaction 5374b4e). Settings: ratio 1.00, port 1920, `--moe-collect-stats`,
GPU at 0 MiB before every arm. Mirror arms ran 8K/80K/713K and whole arms 8K/32K/80K, two
passes each.

| arm | profile |
|---|---|
| def | dynamic prefill headroom + compaction |
| dc  | dynamic, `FREETOKEN_ARENA_COMPACTION=0` (D2H write-back still on) |
| st  | static reservation, `FREETOKEN_DYNAMIC_PREFILL_HEADROOM=0` |
| t0  | no transient reserved: `FREETOKEN_PREFILL_TRANSIENT_MEASURE=0`, `_MB=0` |

**The def and dc arms ran under the dyn-g5 bug**, fixed in ba7d1a6. After a request's
teardown KV shrink, the next prefill started at the decode level with no reserve. The count
per journal is in `ck4c-noreserve-scan.txt`: 3 of 201 prefill batches on the mirror def/dc
arms, 1 of 33 on the whole def/dc arms, 0 on st and t0. On WSL, this host pages instead of
OOMing, so those prefills completed. The def and dc arms are re-run on the fix as ck4f,
together with the ck4 gate set. The gate arms started here (whole with needles) were stopped
on the bug report and are not results.

## Numbers (`ck4c-compaction.txt`)

| mirror | def | dc | st | t0 |
|---|---|---|---|---|
| decode hit rate | 0.9404 | 0.9350 | 0.9235 | 0.9416 |
| mirror swaps / token | 8.3 | 9.0 | 10.5 | 8.0 |
| checkpoint re-reads | 0 | 0 | 0 | 0 |
| 8K p1 / p2 | 185.8 / 166.2 | 171.0 / 165.6 | 166.7 / 165.8 | 190.6 / 173.3 |
| 80K p1 / p2 | 155.7 / 160.1 | 151.8 / 161.1 | 151.4 / 159.7 | 154.8 / 165.8 |
| 713K p1 / p2 | 96.7 / 104.2 | 95.3 / 104.0 | 87.0 / 96.6 | 89.2 / 100.5 |

| whole | def | dc | st | t0 |
|---|---|---|---|---|
| decode hit rate | 0.9423 | 0.9391 | 0.9309 | 0.9429 |
| 8K p1 / p2 | 186.9 / 184.2 | 177.2 / 174.0 | 177.2 / 177.0 | 195.3 / 193.5 |
| 32K p1 / p2 | 172.4 / 168.0 | 182.6 / 163.0 | 187.5 / 176.0 | 185.6 / 188.3 |
| 80K p1 / p2 | 164.5 / 174.5 | 154.2 / 167.3 | 160.1 / 163.2 | 176.0 / 179.0 |

Compaction on the reserve shrink (2088 -> 1944, 144 doomed slots):

* mirror: 45-122 experts moved per reserve, 240-650 MiB D2D in 1.7-2.8 ms. The first
  call is 64 ms because it compiles the copy kernel. 0-25 experts are left for write-back
  from the GPU. Before this, 95-144 experts per reserve were re-read from the checkpoint
  (ck4d: 656 rows).
* whole: 63 moved per call on average, 10.4 ms per call including the first call's
  compile.

Misses per decode window (release to next reserve): mirror 699 (def) vs 772 (dc); whole
710 vs 755.

## Reading

* Compaction closes most of the hit-rate gap that the reserve's blind eviction opened. Mirror
  def is 0.0012 below t0, where ck4d's dynamic-only arm was 0.0072 below ck4t's t0. Whole
  def is 0.0006 below t0.
* Mirror decode: def is within 4% of t0 at 8K/80K and ahead of both t0 and st at 713K.
* Whole decode still trails t0 by 3-10% at equal hit rate. Each request's release
  (empty_cache, then committing about 0.75 GiB of arena) runs after the first token, so it
  lands inside the timed decode. The ms timing added to the reserve/release log lines in
  ba7d1a6 is there to measure this in ck4f.
