# ck4 on the owner's machine (RTX 5080, WSL2), python/ of 9dac3b5 (checkpoint1.sh at 8f90e0b)

Run on 2026-09-24 from 11:09 to 13:15 local time with `checkpoint1.sh ck4`, as the transient user unit
ft-ck4-local. Arms: whole, whole-1m, mirror-1m, mirror, whole-close. Port 1920, ratio 1.00.
Each arm ran in its own transient unit (measure.sh, FT_LAUNCHER=systemd).

Preconditions:
* The preflight stopped the piro-board embedder, and nothing restarted it. qwen3-embedding-4b was not touched.
* freetoken-serve stayed inactive.
* The GPU read 0 MiB before the first arm, and MemAvailable was 29 GiB.
* No torch pytest ran locally during the run.
* A Windows-side GPU user (overlay, desktop) would not show from WSL, so one can't be ruled out.

The arm files sit loose in `results/` as `ck4-*`, the way ck3's do. The compare files are
`ck4-box-compare.txt` (same-commit whole, the owner's gate), `ck4-records-compare.txt` (the owner
record nemotron-reserve-2e*) and `ck4-needles-compare.txt`.

## Clock correction (one point)

`probe_decode.py` computed decode tok/s from the wall clock, and WSL2 steps its wall clock.
At mirror-1m 1M pass 2, the wall-minus-monotonic gap grew by 15.8 s during the prefill and by
0.69 s inside the 2 s decode window. The result was 56.7 tok/s on the wall clock against 82.1 on the monotonic clock.
Every other point differs by at most 1-2 tok/s. The compare files keep the raw wall numbers.
The table below uses monotonic numbers: (gen_tokens - 1) / (total_mono_s - ttft_mono_s).
`scripts/probe_decode.py` now reports decode on the monotonic clock
(`decode_tok_s_wall` keeps the old one).

## Judgement 1: against the same-commit whole arm (the owner's gate, >= 91%)

| point | whole (same commit) | pool arm | ratio | |
|---|---|---|---|---|
| 8K p1 / p2 (mirror-1m) | 174.5 / 181.8 | 159.5 / 164.7 | 91.4% / 90.6% | ok / **below** |
| 8K p1 / p2 (mirror) | 174.5 / 181.8 | 160.1 / 164.3 | 91.7% / 90.4% | ok / **below** |
| 80K p1 / p2 (mirror) | 165.8 / 167.1 | 151.1 / 158.9 | 91.1% / 95.1% | ok / ok |
| 1M p1 / p2 (mirror-1m vs whole-1m) | 87.8 / 92.3 | 72.1 / 82.1 | 82.1% / 88.9% | **below / below** |
| 713K p1 / p2 (mirror) | none | 84.1 / 97.9 | | |

| gate | result | |
|---|---|---|
| 1M arm RAM 12.26 ± 0.6 GiB | 12.65 GiB | PASS |
| coverage faults 0 / starved write-backs 0 | 0 / 0 on both pool arms | PASS |
| one graph capture, no tracebacks | captures=1 and 0 tracebacks on all 5 arms (R3 kv_grows=22 on mirror) | PASS |
| needles/recall identical to ck4-whole | 0 differences (14 needle answers, recall at 21K/120K/240K) | PASS |
| R6 memlock | ok on all 5 arms (no pageable fallback, DefaultLimitMEMLOCK=infinity) | PASS |
| decode >= 91% of same-commit whole | 4 of 10 points below (8K p2 twice, 1M p1 and p2) | **FAIL** |

Box drift (whole-close vs whole) is within 4% at 8K and 32K. At 80K, whole-close decoded
98.8 (p1) and 125.0 (p2) against 165.8 / 167.1. The log shows the same KV growth
(65536 -> 131072, arena 1944 -> 1904) and the same events as the whole arm, with no warning,
fault or capture. Its per-40-token throughput was 90-94 tok/s on p1, where whole showed 165-176.
I found no cause in the log.
* Its probes began 1:44 after the expert-bank load started, against 2:40 for the whole arm.
  CLAUDE.md warns that the first minutes after a start are slow. The load itself had finished
  (13:12:48, 18.3 GB at about 230 MB/s).
* Windows-side GPU use cannot be seen from WSL.
So the 80K point of the run is bracketed by a slow close. The pool's 80K numbers pass against
the opening whole arm either way.

## Judgement 2: against the owner record (nemotron-reserve-2e / -2e-1m, >= 91%)

| point | record | pool arm | ratio | |
|---|---|---|---|---|
| 8K p1 / p2 (mirror-1m) | 177.0 / 170.7 | 159.5 / 164.7 | 90.1% / 96.5% | **below** / ok |
| 8K p1 / p2 (mirror) | 180.3 / 172.1 | 160.1 / 164.3 | 88.8% / 95.5% | **below** / ok |
| 80K p1 / p2 | 154.8 / 154.1 | 151.1 / 158.9 | 97.6% / 103.1% | ok |
| 713K p1 / p2 | 91.7 / 107.6 | 84.1 / 97.9 | 91.7% / 91.0% | ok / ok (borderline) |
| 1M p1 / p2 | 76.9 / 72.9 | 72.1 / 82.1 | 93.8% / 112.6% | ok |

8 of 10 pass. The wall-clock file shows 713K p2 at 90.8% and 1M p2 at 77.8%, and the monotonic
clock moves both. The 8K p1 misses are what ck4 on the boxes showed too.

## Reading

* Correctness and the memory gates hold on the owner's machine, R6 included:
  faults 0, starved 0, needles identical, one capture, RAM 12.65 GiB at 1M.
* Decode: the gap at 8K is 9-10% on this machine, matching ft-g5's 12%, and just outside the
  band. The 1M gap is 11-18%. Per token at 1M p1, the pool pays 2.5 ms more than whole-1m
  (13.87 vs 11.39 ms).
* The whole model decodes slower here today than in ck3: 8K whole was 181.4 now against 190.1 in
  ck3, a 5% drop with no known cause.
