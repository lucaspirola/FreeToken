# ck4 on the gen 5 box ft-g5 (RTX 5080, native Linux), commit 9dac3b5

Vast instance 52354198: Core Ultra 9 285K, 62 GB RAM, driver 580.126.09, **PCIe gen 5 x16**.
Run 2026-09-24 04:53Z to 06:55Z by `checkpoint-box.sh ck4g5` with ARMS="whole whole-1m
mirror-1m mirror whole-close". Settings: ratio 1.00, port 1920, /root/venv, the same code
and launcher as ck4 on ft-ck. The venv, CUDA 13.0, weights and repo were copied from ft-ck.
`whole-1m` is new: the whole-model arm at 8K + 1M, which gives 1M decode a same-box
reference (`compare_box.py`).

Copy bandwidth, measured before and after the arms on an idle GPU:

| | H2D | D2H |
|---|---|---|
| DMA, `ck4g5-pcie-bw.txt` (pinned, 256 MiB) | 56.3 GB/s | 40.0 GB/s |
| mirror SM copy, `ck4g5-bench-mirror-copy.txt` (n=8) | 50.4 GB/s | 37-40 GB/s |

For comparison, ft-ck (gen 4): DMA 28.3/28.2, SM copy 25.7 H2D / 12.2 D2H.

## Verdict against the owner's gate

| gate | result | |
|---|---|---|
| 1M arm RAM 12.26 ± 0.6 GiB | 12.07 GiB | PASS |
| coverage faults 0, starved write-backs 0 | 0 / 0 on both pool arms | PASS |
| one graph capture | captures=1 on all 5 arms, 0 tracebacks, 0 OOMs (R3 PASS) | PASS |
| needles/recall identical to ck4g5-whole | 0 differences (14 needle answers, recall at 21K/120K/240K) | PASS |
| decode within 9% of the same box's whole model, pass by pass | 2 of 8 comparable points inside | **FAIL** |
| R6 memlock | RLIMIT_MEMLOCK 64 KiB in this container | environment limit, not judged |

Box stability: whole-close is within 0.6% of whole at every point.

| point | whole (same box) | pool arm | ratio |
|---|---|---|---|
| 8K p1 / p2 (mirror-1m) | 184.7 / 184.9 | 162.1 / 162.4 | 87.8% / 87.8% |
| 8K p1 / p2 (mirror) | 184.7 / 184.9 | 162.1 / 162.6 | 87.8% / 87.9% |
| 80K p1 / p2 (mirror) | 168.6 / 171.2 | 149.1 / 159.2 | 88.4% / **93.0% ok** |
| 1M p1 / p2 (mirror-1m vs whole-1m) | 88.3 / 91.7 | 74.7 / 81.2 | 84.6% / 88.5% |
| 713K p1 / p2 (mirror) | none | 87.7 / 99.6 | |

Against the owner record (`ck4g5-records-compare.txt`), 9 of 10 points pass: 1M 74.7/81.2
vs 76.9/72.9 (97%/111%), 713K 95.6%/92.6%, 80K 96%/103%, and 8K 91.6/95.1% on mirror-1m and
89.9%/94.5% on the mirror arm (8K p1 just below).

## What the gen 5 box tells us about the gap

Extra ms per token that the pool arm pays over the same host's whole-model decode:

| point | ft-g5 (gen 5) | owner (ck3, gen 5, WSL) | ft-ck (gen 4) |
|---|---|---|---|
| 8K p2 | 0.74 | 0.40 | 1.05 |
| 8K p1 | 0.75 | 0.14 | 1.83 |
| 1M p1 (base: 8K whole) | 7.97 | 7.84 | 15.28 |
| 1M p2 (base: 8K whole) | 6.91 | 5.86 | 10.52 |

* **Long context tracks the link.** At 1M, ft-g5 pays what the owner's machine pays (7.97
  vs 7.84 ms at p1), and ft-ck pays twice that. That fits the ck4 analysis: the
  miss/write-back traffic that dominates at 1M is bandwidth-bound, and ft-ck's gen 4 halves
  it.
* **Short context does not.** At 8K, ft-g5 has faster copies than ft-ck in both directions
  but still pays 0.74 ms/token, about 2x the owner's 0.40. The first-pass excess seen on
  ft-ck (24%) is gone here: p1 equals p2. So the 8K gap on ft-g5 is a per-step cost of the
  pool path that the owner's WSL machine does not show, and it is not link bandwidth.
  Owner-side comparison: the owner's pool arm decodes 8K at 185/177 while ft-g5's decodes at
  162/163. At the same time ft-g5's whole model (185) is slightly slower than the owner's
  (190). Candidate causes, none tested yet:
  * the native-Linux driver (580) vs WSL's paravirtualised one on the pool's mapped-memory
    stores and event syncs;
  * the 64 KiB memlock cap in the Vast container, although no pageable fallback is logged;
  * the per-step host work of the mirror (victim selection, bookkeeping) on this CPU.
  An nsys trace of one 8K decode step, pool vs whole on the same box, would separate them.

On this evidence, the same-box decode gate fails on native Linux for a reason the owner's
machine does not show. The pool is correct (faults 0, starved 0, needles identical) and
within the owner-record band.
