#!/usr/bin/env python3
"""Decode steps from step 1 in 16-step bins (nsys --cuda-graph-trace=node exports whose trace
starts before the first token, NS_START_AT).

  nsys_bins.py <export.sqlite> ...

A decode step = one cudaGraphLaunch; its GPU span starts at the first kernel carrying that
launch's correlationId and ends where the next step starts. Per bin (steps 1-16, 17-32, ... 113-128,
then everything after): mean step time, write-back bytes (D2H memcpy, in or outside the graph,
started inside the step), real expert-copy kernels per step (fast_index_copy* > 5 us; the
mirror's staging copies of write-back victims are the short ones, 5-50 us, fetches are longer),
mean duration of the copies > 50 us (fetches), and total copy-kernel time. The last three
columns (exp/wb-schedule) split the write-back DMA's busy time: total us/step, and the share of
it that runs beside decode attention (stage 1/2 kernels) and beside fetch kernels (> 50 us).
"""
import sqlite3, sys, bisect, statistics
from collections import defaultdict

for p in sys.argv[1:]:
    c = sqlite3.connect(p)
    names = dict(c.execute("select id,value from StringIds"))
    launches = [r[0] for r in c.execute(
        "select r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id "
        "where s.value like 'cudaGraphLaunch%' order by r.start")]
    lset = set(launches)
    first = {}
    kern = []
    for cid, st, en, sn in c.execute("select correlationId,start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        if cid in lset:
            first[cid] = min(first.get(cid, st), st)
        kern.append((st, en, names.get(sn, "")))
    starts = [first[cid] for cid in launches if cid in first]
    if len(starts) < 3:
        print(p, "too few steps"); continue
    n = len(starts) - 1
    step_t = [starts[i + 1] - starts[i] for i in range(n)]
    wb = [0] * n; cp_n = [0] * n; cp_t = [0] * n; fetch = defaultdict(list)
    d2h = []
    for st, en, b in c.execute("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2"):
        i = bisect.bisect_right(starts, st) - 1
        if 0 <= i < n:
            wb[i] += b
            if b >= 65536:                 # the write-back rows, not the tiny snapshots
                d2h.append((st, en))

    def merged(iv):
        out = []
        for a, b in sorted(iv):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out

    def overlap_by_step(xs, ys):
        """Overlap of merged intervals xs with ys, credited to the step xs start in."""
        res = [0] * n
        j = 0
        for a, b in xs:
            i = bisect.bisect_right(starts, a) - 1
            while j < len(ys) and ys[j][1] <= a:
                j += 1
            k = j
            while k < len(ys) and ys[k][0] < b:
                if 0 <= i < n:
                    res[i] += max(0, min(b, ys[k][1]) - max(a, ys[k][0]))
                k += 1
        return res
    dma_iv = merged(d2h)
    attn_iv = merged([(st, en) for st, en, nm in kern if "decode" in nm and ("stage1" in nm or "stage2" in nm)])
    fetch_iv = merged([(st, en) for st, en, nm in kern if "index_copy" in nm and en - st > 50000])
    dma_t = [0] * n
    for a, b in dma_iv:
        i = bisect.bisect_right(starts, a) - 1
        if 0 <= i < n:
            dma_t[i] += b - a
    dma_attn = overlap_by_step(dma_iv, attn_iv)
    dma_fetch = overlap_by_step(dma_iv, fetch_iv)
    for st, en, name in kern:
        if "index_copy" not in name:
            continue
        i = bisect.bisect_right(starts, st) - 1
        if 0 <= i < n and en - st > 5000:
            cp_n[i] += 1; cp_t[i] += en - st
            if en - st > 50000:
                fetch[i].append(en - st)
    print(f"===== {p}: {n} decode steps traced from step 1")
    print(f"   {'steps':>9} {'step ms':>8} {'wb MB/step':>10} {'copies/step':>11} {'copy us/step':>12} {'fetch mean us':>13} {'wb DMA us':>9} {'% in attn':>9} {'% in fetch':>10}")
    bins = [(i, min(i + 16, n)) for i in range(0, min(128, n), 16)] + ([(128, n)] if n > 128 else [])
    for a, b in bins:
        k = b - a
        f = [x for i in range(a, b) for x in fetch[i]]
        print(f"   {a + 1:>4}-{b:<4} {sum(step_t[a:b]) / k / 1e6:8.2f} {sum(wb[a:b]) / k / 1e6:10.2f} "
              f"{sum(cp_n[a:b]) / k:11.1f} {sum(cp_t[a:b]) / k / 1e3:12.0f} {statistics.mean(f) / 1e3 if f else 0:13.0f} "
              f"{sum(dma_t[a:b]) / k / 1e3:9.0f} {100 * sum(dma_attn[a:b]) / max(1, sum(dma_t[a:b])):9.0f} "
              f"{100 * sum(dma_fetch[a:b]) / max(1, sum(dma_t[a:b])):10.0f}")
