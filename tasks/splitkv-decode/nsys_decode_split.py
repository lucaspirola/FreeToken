#!/usr/bin/env python3
"""Decode-step split from an nsys sqlite export (server launched with --cuda-graph-trace=node).

A step = one cudaGraphLaunch. Per step: the period between launches, all GPU kernel
time (in-graph and not), grouped into decode attention (stage 1 / stage 2), the rest
of the graph, and non-graph kernels; memcpys; GPU idle = period - union of busy spans.

    nsys_decode_split.py FILE.sqlite [...]
"""
import sqlite3
import statistics as S
import sys
from collections import defaultdict


def union(iv, lo, hi):
    iv.sort(); u = 0; ce = lo
    for a, b in iv:
        a = max(a, ce); b = min(b, hi)
        if b > a:
            u += b - a; ce = b
    return u


for path in sys.argv[1:]:
    c = sqlite3.connect(path)
    names = dict(c.execute("select id,value from StringIds"))
    launches = [r[0] for r in c.execute(
        "select r.start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id "
        "where s.value like 'cudaGraphLaunch%' order by r.start")]
    ker = c.execute("select start,end,shortName,coalesce(graphNodeId,0) from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    try:
        mem = c.execute("select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
    except sqlite3.OperationalError:
        mem = []
    if len(launches) < 8:
        print(f"== {path}: only {len(launches)} graph launches"); continue
    # Kernels of a step start after its launch API call; window = [launch_i, launch_{i+1}).
    wins = list(zip(launches[2:-2], launches[3:-1]))
    n = len(wins)
    per = [b - a for a, b in wins]
    lo, hi = wins[0][0], wins[-1][1]
    agg = defaultdict(float); cnt = defaultdict(int)
    busy = []
    for s, e, sn, node in ker:
        if s < lo or s >= hi:
            continue
        nm = names.get(sn, str(sn))
        key = ("G " if node else "N ") + nm
        agg[key] += e - s; cnt[key] += 1; busy.append((s, e))
    mt = defaultdict(float); mb = defaultdict(float)
    for s, e, k, b in mem:
        if lo <= s < hi:
            mt[k] += e - s; mb[k] += b; busy.append((s, e))
    tot_busy = union(busy, lo, hi)
    st1 = sum(v for k, v in agg.items() if "decode_grouped_stage1" in k)
    st2 = sum(v for k, v in agg.items() if "decode_stage2" in k)
    graph = sum(v for k, v in agg.items() if k.startswith("G "))
    nong = sum(v for k, v in agg.items() if k.startswith("N "))
    us = lambda x: x / n / 1e3
    period = S.mean(per)
    print(f"== {path}: {n} steps, period mean {period/1e3:.1f} us (median {S.median(per)/1e3:.1f}) "
          f"= {1e9/period:.1f} tok/s under trace")
    n_s1 = sum(v for k, v in cnt.items() if "decode_grouped_stage1" in k) / n
    print(f"   attention stage1 {us(st1):8.1f} us/step ({n_s1:.1f} launches/step)")
    print(f"   attention stage2 {us(st2):8.1f} us/step")
    print(f"   rest of graph    {us(graph - st1 - st2):8.1f} us/step")
    print(f"   non-graph kerns  {us(nong):8.1f} us/step")
    print(f"   memcpy           " + ", ".join(f"kind{k}: {us(v):.1f} us {mb[k]/n/1e6:.2f} MB" for k, v in mt.items()))
    print(f"   GPU busy {us(tot_busy):.1f} us, idle {period/1e3 - us(tot_busy):.1f} us per step; "
          f"attention share of period {100*(st1+st2)/n/period:.1f}%")
    for k, v in sorted(agg.items(), key=lambda x: -x[1])[:16]:
        print(f"     {k[:78]:78} {cnt[k]/n:6.1f}/step {us(v):8.1f} us")
