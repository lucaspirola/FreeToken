# B: is the headroom fix behind the first-pass gap? (ft-ck, 2026-09-24)

ck4 on ft-ck showed pool pass 1 about 24% below the whole model at 8K and 80K (`../ck4-box/`).
To test whether the measured-prefill-headroom fix (82207c8, in 9dac3b5) causes it, the ck4
mirror arm ran on the pre-fix commit `exp/reorg` cf5d2c8 and on the post-fix commit 9dac3b5.
The pre-fix tree was a git worktree at /root/FreeToken-prefix. Both runs used the same
`checkpoint-box.sh` and measure.sh, serve-default.sh, probe and sizes (8K/80K/713K),
FT_ROWS=-1 and FT_RESERVE=256, the same venv and the same extension builds. The runs were
sequential, with the GPU otherwise idle. Driver scripts: `b-chain.sh`, `b-chain*.log`.

## Pre-fix does not run at ratio 1.00 on native Linux, which is evidence for the fix

| run | ratio | outcome |
|---|---|---|
| ck4pre (`ck4pre-r100-oom/`) | 1.00 | OOM at startup: LinearStatePool asks for 598 MiB with 563 MiB free, before graph capture |
| ck4pre95 | 0.95 | starts, then OOMs in the first 80K prefill (mamba2 `_chunk_state_fwd`, 29 MiB free) |
| ck4pre90 | 0.90 | serves 8K and 80K, then OOMs in the 713K prefill (5 MiB free) |
| ck4post95, ck4post90 | 0.95, 0.90 | full arm, 0 OOM, 0 tracebacks, 1 capture, 0 faults, 0 starved |

Pre-fix reserves nothing for the prefill transient (0.65 GiB per 8192-token chunk on this
model), so it cannot finish the ck4 arm at any ratio tried. Post-fix measures that transient
and parks the arena until it has done so, then serves every size.

## Pass-1 decode, pre-fix against post-fix at equal ratio (tok/s)

| ratio | point | pre cf5d2c8 | post 9dac3b5 | post/pre |
|---|---|---|---|---|
| 0.95 | 8K p1 | 138.6 | 137.8 | 99.4% |
| 0.90 | 8K p1 | 118.4 | 124.8 | 105.4% |
| 0.90 | 80K p1 | 116.6 | 123.2 | 105.7% |

Post-fix full arms, p1 / p2: 0.95 gives 8K 137.8/145.2, 80K 125.9/144.1, 713K 59.7/81.7.
0.90 gives 8K 124.8/132.9, 80K 123.2/135.2, 713K 60.0/81.9.

**Verdict: the headroom fix is not the cause.** At equal ratio, post-fix pass 1 matches or
beats pre-fix, and pre-fix already shows the gap: its 8K p1 at 0.95 is 138.6, which is 80% of
the box's whole-model 173.3.

## Arena slots and coverage restores during the first requests (journals)

| run | resolved arena | at ready | first 80K request |
|---|---|---|---|
| pre95 | 2162 | 2162 GPU residents, 660 duplicates (30.5% free evictions) | KV 65536->131072, MoE slots stay 2162; OOM |
| post95 | 2162 | parked 2162 -> 1360, warm start 1360 residents, 1 duplicate (0.1%) | arena back at 2144, shrinks 2144 -> 2096 for the KV commit; restores 5 experts, later 32 |
| pre90 | 2017 | 2017 residents, 660 duplicates (32.7%) | slots stay 2017 through 393K; OOM at 713K |
| post90 | 2017 | parked 2017 -> 1216, warm start 1216 residents, 2 duplicates | slots stay 2017 through 262K; restores 13 experts at the 713K request |

Post-fix starts pass 1 with a parked arena, 800 fewer warm residents and almost no duplicates,
yet it is no slower. So the park/refill and the coverage restores (5-32 experts, a few
hundred MB of copies once) do not account for a 20% pass-1 drop.

What remains box-specific and hits the pool arm only is write-back speed. On ft-ck, SM stores
into mapped host memory run at 12-14 GB/s against 24-26 GB/s for admissions
(`ftck-bench-mirror-copy.txt`), on top of the gen 4 link. Pass 1 has the most evictions
because the pool is still sorting out residency. The ck4g5 run on the gen 5 box (ft-g5,
`../ck4g5-box/`) tests this directly.
