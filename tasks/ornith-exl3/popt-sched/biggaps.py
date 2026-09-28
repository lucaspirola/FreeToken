#!/usr/bin/env python3
"""Largest idle gaps on the compute stream with the CUDA API calls that ran during each.
    biggaps.py FILE.sqlite [N (default 12)]"""
import sqlite3, sys
from collections import defaultdict
db = sqlite3.connect(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
names = dict(db.execute("select id, value from StringIds"))
k = db.execute("select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
m = db.execute("select start, end, streamId, bytes from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
r = db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
per = defaultdict(float)
for s, e, st, _ in k: per[st] += e - s
main = max(per, key=per.get)
allev = sorted([(s, e, st, names[n]) for s, e, st, n in k] + [(s, e, st, f"memcpy{b}") for s, e, st, b in m])
t0 = allev[0][0]
ev = [x for x in allev if x[2] == main]
gaps = sorted(((s1 - e0, e0, s1, n0, n1) for (s0, e0, _, n0), (s1, e1, _, n1) in zip(ev, ev[1:])), reverse=True)[:N]
for gap, a, b, n0, n1 in sorted(gaps, key=lambda x: x[1]):
    api = defaultdict(float)
    for s, e, n in r:
        if e > a and s < b:
            api[names[n]] += min(e, b) - max(s, a)
    other = [(round((s - t0) / 1e6, 1), round((e - s) / 1e6, 2), st, nm[:30]) for s, e, st, nm in allev if st != main and e > a and s < b and e - s > 5e5][:4]
    top = ", ".join(f"{n[:28]} {v/1e6:.1f}" for n, v in sorted(api.items(), key=lambda x: -x[1])[:4])
    print(f"@{(a - t0)/1e6:8.1f} ms gap {gap/1e6:6.1f} ms  {n0[:26]} -> {n1[:26]} | api: {top} | other streams: {other}")
