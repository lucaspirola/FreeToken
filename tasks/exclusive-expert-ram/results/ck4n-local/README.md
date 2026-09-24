# ck4n: the ck4 on exp/reorg-next, owner's RTX 5080 (WSL), 2026-09-25

Code: exp/reorg-next. That is exp/reorg 1376216 plus four merges:
* transient-slack (expandable segments on WSL);
* splitkv-decode (split-KV v2);
* ratio-default (ratio 1.00 default, memory prediction);
* pinned-err.

Run from the detached worktree `next-measure` at f03c081 (4543acc plus measurement tooling),
with the extensions rebuilt. Settings: ratio 1.00, port 1920, 0 MiB before every arm, embedder
stopped. The journals confirm "Enabled expandable_segments", "Startup memory prediction" and
"Triton decode launch: kv_splits=84".

Parts:
* `nx-*`: `recheck-local.sh` with RC_SIZE="8000 80000": whole and mirror (reserve 256)
  alternated x3, 512 decode tokens, passes 1 and 2 (`nx-compare.txt`).
* `ck4n-*`: `checkpoint1.sh ck4n` with ARMS="whole-1m mirror-1m", WHOLE1M_NEEDLES=1 and
  NEEDLES_REF=ck4n-whole-1m.
  * Both arms carry needles/recall.
  * 1M decode uses the default 128-token window.
* `cgroup-samples.tsv` (+ `.procs.tsv`, `.maps/`): the cgroup/smaps sampler over all arms.
* `ck4n-host-before.txt`: MemAvailable 29.3 GiB. julia processes had started 6 s and 64 s
  earlier; nothing was stopped.

Tests on ft-dev at 4543acc: 1599 passed, 0 failed, 14 skipped (`../reorg-next-ftdev`).

## Gate

| item | result | |
|---|---|---|
| 8K, median of 3 alternated pairs, >= 91% | p1 94.2% (91.4/96.3/94.2), p2 96.6% (95.1/101.3/96.6) | PASS |
| 80K, median of 3 alternated pairs, >= 91% | p1 96.3% (92.8/104.4/96.3), p2 97.8% (92.2/98.8/97.8) | PASS |
| 1M mirror-1m vs whole-1m (single arms, 128 tokens) | p1 **90.2%** (89.6 vs 99.3), p2 93.5% (97.2 vs 104.0) | **p1 0.8 pt below** |
| 8K in the 1M arms | p1 93.8%, p2 95.7% | PASS |
| 1M vs owner record (76.9 / 72.9) | 116.5% / 133.3% | PASS |
| needles/recall, mirror-1m vs whole-1m | 0 differences (14 needle answers, recall 21K/120K/240K) | PASS |
| coverage faults / starved write-backs | 0 / 0 on all 4 mirror arms | PASS |
| one capture, no Traceback, no OOM | captures=1 on all 8 arms; R3 and R6 PASS | PASS |
| prefills at the decode level | 0 on all 8 arms | PASS |
| 1M RAM, ram_gib, 12.26 ± 0.6 | 11.71 | IN |
| per-server anon (cgroup) at ready / after 1M p1 / after 1M p2 | 12.70 / 13.19 / 12.99 GiB. 583afc8 ck4m had 12.54 / 13.46 / 13.18 with a pool 0.08 GiB larger (1866 vs 1850 rows) | same |

## 1M p1: split-KV speeds up the whole model, not the pool arm

Median gap between streamed tokens at 1M (ms per token):

| run | whole-1m p1 / p2 | mirror-1m p1 / p2 |
|---|---|---|
| ck4m (583afc8) | 10.7 / 10.0 | 10.7 / 10.5 |
| tsw (583afc8 + expandable segments) | - | 10.7 / 9.6 |
| ck4n (reorg-next, adds split-KV v2) | **9.5 / 9.1** | **10.6 / 9.7** |

* Split-KV v2 cuts the whole model's 1M step by 1.2 ms (+11% decode at p1, +9% at p2).
* The mirror arm's step is unchanged from 583afc8.
* Mirror 1M p1 decode has been flat at 88.5-89.6 across ck4m, tsw and ck4n. The ratio falls
  because the reference got faster, not because the pool arm got slower.
* The first gap (about 36 ms on both arms) is not the difference.
* Not yet explained: why split-KV's saving does not reach the pool arm. One candidate is that
  the pool path's side work (write-backs, fused copy kernels) now hides behind or contends with
  the 2-CTA-per-SM split-KV grid. A 1M nsys of both arms would settle it.
