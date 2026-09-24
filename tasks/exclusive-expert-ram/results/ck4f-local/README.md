# ck4f: ck4 gate on dynamic headroom + compaction + the dyn-g5 fix, owner's RTX 5080 (WSL)

Run 2026-09-24 by `checkpoint1.sh ck4f` from the detached worktree `dt-measure` at ba7d1a6.
That commit is exp/dynamic-transient: dynamic prefill headroom, arena compaction (5374b4e) and
the teardown-shrink fix. Settings: ratio 1.00, port 1920, GPU at 0 MiB before every arm.

Arms:
* compaction A/B: mirror-def and mirror-dc (8K/80K/713K); whole-def and whole-dc (8K/32K/80K).
  All four run with `--moe-collect-stats`.
* ck4 gate set: whole (8K/32K/80K plus needles/recall), mirror-1m (8K and 1M plus
  needles/recall), mirror (8K/80K/713K), whole-close.

The files are copied here from `dt-measure`. Comparison outputs: `ck4f-compaction.txt`,
`ck4f-box-compare.txt`, `ck4f-records-compare.txt` and `ck4f-needles-compare.txt`.

## The fix holds

`scan_noreserve.py` finds 0 prefill batches started at the decode level in every ck4f
journal: 418 on mirror-1m, 201 on each mirror arm. ck4c had 1-3 per dynamic arm.
Every prefill after a teardown shrink now logs its reserve. When the shrink already left
enough free VRAM, the reserve is a no-op ("1944 -> 1944, 0.8 ms").

## Gate (owner's criteria)

| gate | result | |
|---|---|---|
| coverage faults 0, starved write-backs 0 | 0 / 0 on mirror-1m and mirror | PASS |
| one graph capture, no tracebacks (R3) | captures=1 on all 8 arms, R3 PASS | PASS |
| R6 memlock (arm) | PASS on all 8 arms | PASS |
| needles/recall identical to ck4f-whole | 0 differences (14 needle answers, recall 21K/120K/240K) | PASS |
| decode >= 91% of same-commit whole, pass by pass | 6 of 8 comparable points inside; the mirror arm's 8K p1/p2 are below | **FAIL (2 points)** |
| 1M arm RAM 12.26 ± 0.6 GiB (`ram_gib`) | 13.54 GiB | **OUT by the metric; see attribution** |

| point | whole | pool arm | ratio |
|---|---|---|---|
| 8K p1 / p2 (mirror-1m) | 188.5 / 186.3 | 177.1 / 179.3 | 94.0% / 96.2% |
| 8K p1 / p2 (mirror) | 188.5 / 186.3 | 166.2 / 161.1 | **88.2% / 86.5%** |
| 80K p1 / p2 (mirror) | 173.2 / 175.1 | 166.0 / 166.7 | 95.8% / 95.2% |
| 1M p1 / p2 (mirror-1m vs record) | 76.9 / 72.9 | 80.1 / 84.7 | 104% / 116% |
| 713K p1 / p2 (mirror vs record) | 91.7 / 107.6 | 90.1 / 102.8 | 98% / 96% |

ck4 on this machine (reorg-headroom, static reservation) had 4 of 10 points below, at
90.4-91.7% for 8K and 82-89% for 1M. This run moves 1M above the record and mirror-1m's 8K to
94-96%. The mirror arm's 8K sits at 86-88%. That arm has the same configuration as mirror-1m
at 8K, which decoded 177-179, so the two 8K draws differ by 10% between identical
configurations. 8K decode is 127 tokens, about 0.7 s, and each request's release (about 20 ms,
now logged) falls inside it. This run adds no whole-1m arm, so 1M is judged against the owner
record only.

Box drift: whole-close is within 0.4% of whole at every point except 32K p2 (+3.5%).

### RAM attribution

`ram_gib` is MemAvailable before start minus MemAvailable at ready, a host-wide number. The
server's own memory matches ck4:

| arm | pool | ram_gib | rss_ready | rss | anon | file |
|---|---|---|---|---|---|---|
| ck4 mirror-1m (reorg-headroom) | 1830 rows, 9.58 GiB | 12.65 | 13.71 | 14.05 | 12.92 | 1.50 |
| ck4f mirror-1m | 1830 rows, 9.58 GiB | 13.54 | 13.72 | 14.00 | 12.85 | 0.42 |
| ck4f mirror | 1830 rows, 9.58 GiB | 10.81 | 13.73 | 13.94 | 12.92 | 0.12 |

Two ck4f arms have the same pool and a process RSS within 0.1 GiB of each other, yet read
13.54 and 10.81. The spread comes from host-wide changes outside the server; julia, codex and
several claude processes were live on the host during the run. The server's footprint did not
move.

## Compaction A/B on the fix (`ck4f-compaction.txt`)

| | mirror def | mirror dc | whole def | whole dc |
|---|---|---|---|---|
| decode hit rate | 0.9396 | 0.9339 | 0.9419 | 0.9386 |
| mirror swaps / token | 7.2 | 7.9 | - | - |
| misses per decode window (mean) | 832 | 918 | 721 | 772 |
| experts moved per compaction / ms | 31.1 / 4.07 | - | 65.2 / 8.27 | - |
| write-back rows (all from the GPU) | 255 | 657 | - | - |

The hit rates repeat ck4c (mirror 0.9404 / 0.9350, whole 0.9423 / 0.9391) now that every
prefill reserves. Per-point decode between def and dc stays within ±5%, the same noise as
whole-close vs whole. Only 713K p1 differs more: mirror def (compaction on) 99.1 vs dc (compaction off) 92.1, in favour of compaction.

Reserve and release now carry timings. A release (0.75 GiB committed) takes about 20 ms. A
reserve takes 15-20 ms, except the first one at 136 ms, when the compaction kernel compiles.
