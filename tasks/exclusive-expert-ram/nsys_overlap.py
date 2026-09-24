#!/usr/bin/env python3
"""Per decode step at long context: decode attention vs everything that runs beside it.

  nsys_overlap.py <export.sqlite> ...     (nsys --cuda-graph-trace=node exports)

Steps = cudaGraphLaunch calls in the capture. For each kernel group it prints count and GPU
time per step, and how much of that time overlaps decode attention (stage 1/2 kernels) on
another stream -- work that competes with the split-KV grid for SMs. Also: memcpy per step by
kind (in-graph vs outside the graph), the step period (median gap between consecutive graph
launches' first kernels), and the attention kernel launch geometry (grid, block, registers).
"""
import sqlite3, statistics, sys
from collections import defaultdict


def merge(iv):
    iv = sorted(iv); out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a, b):  # total intersection of two merged interval lists
    i = j = t = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            t += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def group(name):
    n = name.lower()
    if "decode" in n and ("stage1" in n or "stage2" in n or "grouped" in n):
        return "ATTN " + name
    if "index_copy" in n or "copy_kinds" in n or "copy_multi" in n:
        return "COPY " + name
    return name


for p in sys.argv[1:]:
    c = sqlite3.connect(p)
    names = dict(c.execute("select id,value from StringIds"))
    steps = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id "
                      "where s.value like 'cudaGraphLaunch%'").fetchone()[0] or 1
    ks = [(names.get(sn, str(sn)), st, en, stream, g, gx, gy, gz, bx, reg)
          for sn, st, en, stream, g, gx, gy, gz, bx, reg in c.execute(
              "select shortName,start,end,streamId,graphNodeId,gridX,gridY,gridZ,blockX,registersPerThread "
              "from CUPTI_ACTIVITY_KIND_KERNEL")]
    attn = [(s, e, st) for n, s, e, st, *_ in ks if group(n).startswith("ATTN")]
    attn_iv = merge([[s, e] for s, e, _ in attn])
    attn_streams = {st for _, _, st in attn}
    print(f"===== {p}: {steps} graph launches")
    geo = defaultdict(int)
    for n, s, e, st, g, gx, gy, gz, bx, reg in ks:
        if group(n).startswith("ATTN"):
            geo[(n, gx, gy, gz, bx, reg)] += 1
    for (n, gx, gy, gz, bx, reg), k in sorted(geo.items(), key=lambda x: -x[1])[:6]:
        print(f"   attention launch: {n} grid ({gx},{gy},{gz}) block {bx} regs {reg}  x{k / steps:.1f}/step")
    agg = defaultdict(lambda: [0, 0, 0, set(), 0])
    for n, s, e, st, g, *_ in ks:
        a = agg[group(n)]; a[0] += 1; a[1] += e - s; a[3].add(st)
        a[4] += 1 if (g is None or g == 0) else 0
    other = defaultdict(list)
    for n, s, e, st, g, *_ in ks:
        if st not in attn_streams:
            other[group(n)].append([s, e])
    for k in other:
        agg[k][2] = overlap(merge(other[k]), attn_iv)
    tot_attn = sum(e - s for s, e in attn_iv)
    print(f"   decode attention busy {tot_attn / steps / 1e3:.1f} us/step")
    print(f"   {'kernel':62} {'n/step':>7} {'us/step':>9} {'ovl attn us/step':>17} {'outside graph':>13} streams")
    for k, (n, t, ov, sts, og) in sorted(agg.items(), key=lambda x: -x[1][1])[:22]:
        print(f"   {k[:62]:62} {n / steps:7.1f} {t / steps / 1e3:9.1f} {ov / steps / 1e3:17.1f} {og / max(n, 1):13.0%} {sorted(sts)}")
    for kind, ing, n, t, b in c.execute(
            "select copyKind, (graphNodeId is not null and graphNodeId != 0), count(*), sum(end-start), sum(bytes) "
            "from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind, 2"):
        print(f"   memcpy kind {kind} {'in-graph' if ing else 'outside '}: {n / steps:.1f}/step {t / steps / 1e3:.1f} us/step {b / steps / 1e6:.2f} MB/step")
    # step period: first in-graph kernel start of each launch, via correlation of launches
    launches = [r[0] for r in c.execute(
        "select r.start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id "
        "where s.value like 'cudaGraphLaunch%' order by r.start")]
    if len(launches) > 2:
        d = [b - a for a, b in zip(launches, launches[1:])]
        print(f"   graph launch period median {statistics.median(d) / 1e3:.1f} us, mean {statistics.mean(d) / 1e3:.1f} us")
