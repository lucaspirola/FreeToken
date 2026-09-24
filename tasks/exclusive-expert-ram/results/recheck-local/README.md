# Re-check of the two ck4f gate items on ba7d1a6, owner's RTX 5080 (WSL)

Run 2026-09-24 by `recheck-local.sh` (label `rc`) from the detached worktree `dt-measure` at
fc96095. The engine code is identical to ba7d1a6: fc96095 changes only the probe and the
scripts. Settings: ratio 1.00, port 1920, GPU at 0 MiB before every arm.

This run is the bisect baseline for the exp/dt-dma 583afc8 gate.

## 1. 8K decode, 512 tokens, whole and mirror alternated three times (`rc-compare.txt`)

| pass | run 1 | run 2 | run 3 | median | gate >= 91% |
|---|---|---|---|---|---|
| p1 mirror / whole | 173.8 / 193.9 (89.6%) | 173.8 / 193.0 (90.0%) | 173.1 / 192.5 (89.9%) | 89.9% | **FAIL** |
| p2 mirror / whole | 176.2 / 192.5 (91.5%) | 175.9 / 193.8 (90.7%) | 176.8 / 192.2 (92.0%) | 91.5% | PASS |

With a 512-token window the numbers are stable: each side varies by 1% across the three
runs. ck4f's 10% spread came from its 127-token windows.

The probe now records the stream's inter-chunk gaps:

| | first gap (after token 1) | median gap | decode excluding the first gap |
|---|---|---|---|
| whole | 31-38 ms | 5.0-5.1 ms | 194-197 |
| mirror | 21-29 ms | 5.5-5.6 ms | 174-178 |

* The first gap holds the dynamic-headroom release (about 20 ms, logged) plus the first decode
  step. Both arms pay it.
* Excluding it leaves the ratio unchanged: p1 88.9%, p2 91.5%.
* **So the release is not what fails the 8K gate.** The gap is a steady +0.5 ms per token on
  the pool path.
* ck4 on this machine (reorg-headroom, static reservation, 127 tokens) measured the same
  90.4-91.7% at 8K. The pool's per-step cost predates dynamic headroom.
* On ft-g5, nsys attributed it to write-back D2H between graphs plus about 400 us of longer
  graph per step (`ck4dma-g5/README.md`). `nsys-local.sh` repeats that trace on this machine.

Other checks:
* `scan_noreserve.py`: 0 prefills started at the decode level in any arm.
* captures=1 on every arm.
* coverage faults 0 and starved write-backs 0 on every mirror arm.

## 2. The 1M RAM point (`rc-mirror-1m`, 8K + 1M)

| | rc-mirror-1m | ck4f mirror-1m | ck4 mirror-1m (reorg-headroom) |
|---|---|---|---|
| ram_gib (MemAvailable delta, the gate metric) | **13.06** | 13.54 | 12.65 |
| rss_ready_gib | 13.71 | 13.72 | 13.71 |
| rss_gib | 14.37 | 14.00 | 14.05 |
| anon_gib | 13.23 | 12.85 | 12.92 |
| file_gib | 0.11 | 0.42 | 1.50 |
| pool | 1830 rows, 9.58 GiB | same | same |

The gate is 12.26 ± 0.6. At 13.06 it is 0.2 GiB above the band: **OUT by the metric**.

At ready, the process's RSS is the same as ck4's: 13.71 GiB in all three runs, with the same
pool. Page-cache ownership differs: file 0.11 here against 1.50 in ck4. The host was not quiet
(`rc-host-before.txt`, MemAvailable 29.7 GiB at the snapshot):
* two julia processes had started 9 s and 15 min earlier;
* codex, four claude sessions, two piro-board-mcp and dockerd were running;
* the user units anthropic-proxy@account1/2 and switchyard-dev were active.

None was touched. ram_gib moves with the host: one pool size read 10.81-13.54 across ck4f
and this run.

Decode at 1M: p1 77.8, p2 90.1. At 8K: 174.5 / 171.9 (128 tokens).
