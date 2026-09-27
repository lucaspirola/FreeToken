#!/usr/bin/env python3
"""Idle gaps on the compute stream of an nsys sqlite, grouped by (kernel before, kernel after).
    gaps.py FILE.sqlite [MIN_US (default 50)] [TOP]"""
import sqlite3, sys
from collections import defaultdict
db = sqlite3.connect(sys.argv[1]); mn = float(sys.argv[2]) if len(sys.argv) > 2 else 50; top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
names = dict(db.execute("select id, value from StringIds"))
k = db.execute("select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
m = db.execute("select start, end, streamId, 0 from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
per = defaultdict(float)
for s, e, st, _ in k: per[st] += e - s
main = max(per, key=per.get)
ev = sorted([(s, e, names[n]) for s, e, st, n in k if st == main] + [(s, e, "memcpy") for s, e, st, _ in m if st == main])
g = defaultdict(lambda: [0.0, 0]); tot = 0.0
# also GPU-wide idle: union of everything
for (s0, e0, n0), (s1, e1, n1) in zip(ev, ev[1:]):
    gap = s1 - e0
    if gap > mn * 1e3:
        x = g[(n0[:38], n1[:38])]; x[0] += gap; x[1] += 1; tot += gap
print(f"compute stream {main}: gaps > {mn:.0f} us total {tot/1e6:.1f} ms")
for (a, b), (t, c) in sorted(g.items(), key=lambda x: -x[1][0])[:top]:
    print(f"  {t/1e6:8.1f} ms {c:5d}x  {a} -> {b}")
