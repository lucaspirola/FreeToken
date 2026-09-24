# Merge gate for exp/dt-dma 583afc8, owner's RTX 5080 (WSL), 2026-09-24

583afc8 merges exp/dynamic-transient ba7d1a6 (dynamic prefill headroom + arena compaction +
the teardown-shrink fix) with exp/mirror-dma-wb e7e029f (DMA write-backs through a VRAM
staging ring).

All arms ran from the detached worktree `dt-measure` at 583afc8. Common settings:
* ratio 1.00, port 1920, 0 MiB on the GPU before every arm, embedder stopped;
* compaction on (default), dynamic headroom on (default).

Parts:
* `dd-*`: the re-check protocol. Whole and mirror (reserve 256) alternated three times at
  8K with 512 decode tokens, passes 1 and 2 (`recheck-local.sh`, `dd-compare.txt`).
* `ck4m-*`: the ck4 set (`checkpoint1.sh ck4m`), in this order:
  1. mirror-1m (8K + 1M, needles/recall);
  2. whole (8K/32K/80K, needles/recall);
  3. mirror (8K/80K/713K);
  4. whole-1m (8K + 1M);
  5. whole-close.
  Outputs: `ck4m-box-compare.txt`, `ck4m-records-compare.txt`, `ck4m-needles-compare.txt`.
* `ref2e-mirror-1m`: the owner-record commit (5474e3b; code d9b0070; the
  `nemotron-reserve-2e-1m` run behind 12.26 GiB), run in the same session.
  * Worktree `ref-2e`, with the record's venv (`exclusive-expert-ram/.venv`, python 3.11).
  * Record configuration: auto pool, reserve 256, 8K + 1M, two passes.
  * Today's `measure.sh` and `probe_decode.py` were copied into that tree (python/ untouched),
    so both sides are measured the same way.
* `cgroup-samples.tsv`: `cgroup-sampler.sh` sampled every measure unit every 10 s: its cgroup
  memory.current/anon/file and the largest process's RSS.
  * `cgroup-summary.txt` gives the values at ready and after each request.
  * `ram-cgroup-compare.txt` compares the candidate with the reference.
* Host snapshots: `ck4m-host-before.txt` and `ref2e-host-before.txt`, with MemAvailable 29.3
  and 28.9 GiB. Short-lived julia processes were starting on the host, codex and claude
  sessions were running, and nothing was stopped.

## Gate

| item | result | |
|---|---|---|
| 8K, median of 3 alternated pairs, >= 91% pass by pass | p1 **95.0%** (98.0/95.0/94.1), p2 **97.1%** (92.5/98.3/97.1) | PASS |
| 80K, median of 3 alternated pairs (`d80-compare.txt`), >= 91% pass by pass | p1 **95.9%** (95.5/95.9/97.7), p2 **95.7%** (95.9/95.1/95.7) | PASS |
| 80K single arms, mirror vs ck4m-whole | p1 99.5%, p2 98.5% (superseded by the alternated run) | - |
| 1M mirror-1m vs whole-1m (same session) | p1 98.9% (88.5 vs 89.5), p2 96.5% (92.1 vs 95.4) | PASS |
| 1M vs owner record (76.9 / 72.9) | 88.5 / 92.1 (115% / 126%) | PASS |
| 713K vs owner record (91.7 / 107.6) | 106.4 / 109.7 (116% / 102%) | PASS |
| needles/recall identical to whole | 0 differences (14 needle answers, recall 21K/120K/240K) | PASS |
| coverage faults / starved write-backs | 0 / 0 on every mirror arm (dd x3, ck4m mirror-1m, ck4m mirror) | PASS |
| one capture, no Traceback | captures=1 and 0 Traceback on all 11 arms; R3 and R6 PASS on the 5 ck4m arms | PASS |
| prefills at the decode level | 0 in the dd arms (`scan_noreserve.py`) | PASS |
| 1M RAM, ram_gib (MemAvailable delta), 12.26 ± 0.6 | 12.54; the reference re-run in the same session read 12.74 | IN |
| 1M RAM, cgroup memory.current, candidate − reference within ± 0.6 | 1M p1 −0.21, 1M p2 **−0.86** (the candidate is lower) | see below |

### 8K: the pool's per-step cost is closed

| | whole median gap | mirror median gap | first gap (release + first step) |
|---|---|---|---|
| ba7d1a6 (`recheck-local`) | 5.0-5.1 ms | 5.5-5.6 ms | whole 31-38, mirror 21-29 ms |
| 583afc8 (`dd`) | 4.9-5.2 ms | 5.3-5.5 ms | whole 30-37, mirror 21-30 ms |

ba7d1a6 failed p1 at 89.9% under the same protocol. The DMA write-backs remove most of the
extra pool-path cost per step.

### 80K: the two whole arms disagree

| arm | 8K p1 / p2 | 80K p1 / p2 |
|---|---|---|
| ck4m whole | 176.1 / 176.5 | 163.9 / 166.1 |
| ck4m whole-close | 200.7 / 190.1 | 175.8 / 185.8 |
| ck4f whole (ba7d1a6, earlier today) | 188.5 / 186.3 | 173.2 / 175.1 |
| dd whole x3 | 189-194 / 188-198 | - |
| **ck4m mirror** | 176.8 / 178.4 | **163.1 / 163.6** |

* The ck4m whole arm ran slow. Its 8K is 7% below the six dd whole runs, and whole-close was
  7-14% above it.
* Against whole-close, 80K is 92.8% (p1) and 88.1% (p2).
* Against the median of the four whole 80K draws (175.05), it is 93.2% / 93.5%.
* The single arms' bracket spread (12% at 80K p2) was larger than the 8.7% noise floor, so 80K
  was re-run alternated x3 like `dd` (`d80-*`, 512 decode tokens):
  * whole 179-187 tok/s, mirror 171-179 tok/s;
  * medians 95.9% / 95.7%, PASS;
  * 0 faults/starved, captures=1, 0 prefills at the decode level.

### 1M RAM by cgroup (`ram-cgroup-compare.txt`, `cgroup-summary.txt`)

| point | memory.current cand / ref / Δ | anon cand / ref / Δ | Δ anon without the pool difference | file cand / ref |
|---|---|---|---|---|
| at ready | 12.68 / 13.48 / −0.80 | 12.54 / 12.11 / +0.43 | −0.10 | 0.10 / 1.31 |
| after 1M p1 | 13.60 / 13.81 / −0.21 | 13.46 / 12.40 / +1.06 | +0.53 | 0.11 / 1.35 |
| after 1M p2 | 13.34 / 14.20 / −0.86 | 13.18 / 12.78 / +0.41 | −0.12 | 0.12 / 1.36 |

* **The raw memory.current rule fails at 1M p2 (−0.86), and the candidate is the side that is
  lower.** The cause is page cache charged to the cgroup: the reference has 1.3 GiB `file`,
  the candidate 0.1. The reference's python 3.11 venv and its libraries were read cold into
  its own cgroup, while the candidate's venv had been cached by earlier arms. This is the
  reason measure.sh recorded memory.current as unusable in the first place.
* On anon (the server's own memory, including the pinned pool):
  * raw Δ is +0.43 at ready, +1.06 after 1M p1 and +0.41 after 1M p2;
  * with the pool-size difference removed, every point is inside ± 0.6 (−0.10, +0.53, −0.12).
* **The pool difference is not a free-VRAM effect.** Both runs have 2173 GPU residents and
  771 complement rows, but the candidate budgets 839 duplicates against 738, giving 1866 rows
  / 9.77 GiB against 1765 / 9.24 (+0.53 GiB). In exchange its coverage floor is 1336 slots
  instead of 1440, leaving the KV 104 more arena slots (0.54 GiB of VRAM).
  * ba7d1a6 had 1830 rows / 9.58 GiB, so the extra duplicate budget grew again in the merge
    with mirror-dma-wb.
* During 1M p1 the candidate's anon grows 0.92 GiB above ready, against 0.29 for the
  reference; about 0.3 of it is released by the end of p2. That growth is not yet attributed.
  The largest process's RSS stays flat at 12.12-12.13, so it sits in the other two server
  processes (about 1.2 GiB RSS each).

Decode on the reference re-run: 8K 174.8 / 155.2, 1M 70.2 / 80.5. TTFT at 1M: 1198 / 1433 s,
against the candidate's 794 / 795 s.
