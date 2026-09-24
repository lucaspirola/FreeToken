# ck4t: prefill-transient reservation A/B on the owner's machine (python/ of 9dac3b5)

Run on 2026-09-24 from 13:23 to 14:5x local time with `checkpoint1.sh ck4t` (8d964fe), as the transient unit
ft-ck4t-local. Arms, in order: whole-def, whole-t0, mirror-t0, mirror-def. The preconditions
are those of ck4: the embedder stopped and not restarted, 0 MiB on the GPU, MemAvailable 29 GiB,
port 1920, ratio 1.00. Both sides ran with --moe-collect-stats.
* `-def` holds the measured transient (1.00 GiB here) plus the 0.12 GiB margin free below the
  ratio plan through decode.
* `-t0` is the pre-fix cushion-only headroom: FREETOKEN_PREFILL_TRANSIENT_MEASURE=0 and
  FREETOKEN_PREFILL_TRANSIENT_MB=0.

Files: `../ck4t-*`, and `../ck4t-transient-ab.txt` (compare_transient_ab.py). Decode is on the
monotonic clock. R3 and R6 PASS on all four arms.

| | arena at start (min over run) | decode hit rate | mirror swaps/token |
|---|---|---|---|
| mirror-def | 1944 of 2173, 1.15 GiB free (1560) | 0.9234 | 10.5 |
| mirror-t0 | 2173 of 2173, 0.00 GiB free (1704) | 0.9407 | 8.2 |
| whole-def | 1944 of 2173 (1904) | 0.9306 | |
| whole-t0 | 2173 of 2173 (2048) | 0.9425 | |

| decode | mirror def | mirror t0 | t0/def | whole def | whole t0 | pool/whole def | pool/whole t0 |
|---|---|---|---|---|---|---|---|
| 8K p1 | 160.1 | 176.7 | 110.4% | 164.9 | 181.0 | 97.1% | 97.6% |
| 8K p2 | 163.0 | 168.7 | 103.5% | 182.1 | 185.8 | 89.5% | 90.8% |
| 80K p1 | 133.5 | 150.9 | 113.1% | 165.6 | 170.7 | 80.6% | 88.4% |
| 80K p2 | 158.5 | 166.9 | 105.3% | 184.8 | 184.2 | 85.8% | 90.6% |
| 713K p1 | 82.2 | 90.0 | 109.4% | | | | |
| 713K p2 | 98.4 | 102.2 | 103.8% | | | | |

What the reservation costs, and why t0 cannot simply be the default:
* The ~229 slots it holds back lower the hit rate. The pool gains 3.5-13% of decode at every
  point, and swaps per token fall 22%.
* Whole gains much less (a +10% 8K p1 outlier; otherwise -6% to +3%), because its misses are
  cheaper.
* The pool-vs-whole ratio improves most at 80K: p1 80.6% -> 88.4%, p2 85.8% -> 90.6%.
* It explains a material part of the pool gap here, but not all of it. At t0 the pool still sits
  at 88-91% of whole at 8K p2 and 80K. The rest is the per-swap mechanics seen in the ft-g5 nsys
  trace (`../ck4dma-g5/README.md`).
* t0 is not viable as it stands. With 0.00 GiB free, the first long prefill ran from paged memory
  on WSL: 32K/80K p1 prefill at 787-935 tok/s instead of 7000-9500, which is the problem the
  reservation fixed. On native Linux the t0 arms do not even start (linear-state pool OOM, ft-g5
  `../ck4g5t-box`).
