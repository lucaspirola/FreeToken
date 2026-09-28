#!/usr/bin/env python3
"""Attribute generic kernels by their neighbours on the same stream (nsys sqlite).

    neighbours.py FILE.sqlite [PATTERN (default: elementwise)] [TOP]

Groups every kernel whose short name contains PATTERN by (previous kernel, next kernel) on its
stream, with total time, count, mean grid size and mean duration."""
import sqlite3, sys
from collections import defaultdict
db = sqlite3.connect(sys.argv[1]); pat = sys.argv[2] if len(sys.argv) > 2 else "elementwise"
top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start, end, streamId, shortName, gridX*gridY*gridZ, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by streamId, start").fetchall()
g = defaultdict(lambda: [0, 0, 0, set()])
for i, (s, e, st, sn, grid, dn) in enumerate(rows):
    n = names.get(sn, str(sn))
    if pat not in n:
        continue
    prev = names.get(rows[i-1][3], "?") if i and rows[i-1][2] == st else "-"
    nxt = names.get(rows[i+1][3], "?") if i + 1 < len(rows) and rows[i+1][2] == st else "-"
    key = (st, prev[:40], n[:28], nxt[:40])
    x = g[key]; x[0] += e - s; x[1] += 1; x[2] += grid; x[3].add(names.get(dn, "")[:160])
tot = sum(v[0] for v in g.values())
print(f"{pat}: {tot/1e6:.1f} ms total")
for k, (t, c, gr, dn) in sorted(g.items(), key=lambda x: -x[1][0])[:top]:
    print(f"{t/1e6:8.1f} ms {c:6d}x grid~{gr//c:7d} {t/c/1e3:8.1f} us  st{k[0]} {k[1]} -> [{k[2]}] -> {k[3]}")
    for d in list(dn)[:1]:
        print(f"           {d}")
