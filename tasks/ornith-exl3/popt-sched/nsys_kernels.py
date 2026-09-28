#!/usr/bin/env python3
"""Kernel time of one traced prefill request (nsys sqlite from nsys-prefill.sh).

    nsys_kernels.py FILE.sqlite [TOP]

Prints the span (first kernel start -> last kernel end), GPU busy time (union of kernel and memcpy
intervals), idle, and per-kernel totals (ms, calls, share of busy), plus memcpy by kind."""
import sqlite3, sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1]); top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
names = dict(db.execute("select id, value from StringIds"))
k = db.execute("select start, end, coalesce(demangledName, shortName), shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
try:
    m = db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
except sqlite3.OperationalError:
    m = []
iv = sorted([(s, e) for s, e, *_ in k] + [(s, e) for s, e, *_ in m])
busy = 0; cur_s = cur_e = None
for s, e in iv:
    if cur_e is None or s > cur_e:
        if cur_e is not None: busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
if cur_e is not None: busy += cur_e - cur_s
span = (iv[-1][1] - iv[0][0]) if iv else 0
print(f"span {span/1e6:.1f} ms, GPU busy {busy/1e6:.1f} ms, idle {(span-busy)/1e6:.1f} ms, kernels {len(k)}, memcpys {len(m)}")
tot = defaultdict(lambda: [0, 0])
for s, e, dn, sn in k:
    n = names.get(sn, str(sn))
    tot[n][0] += e - s; tot[n][1] += 1
ksum = sum(v[0] for v in tot.values())
print(f"kernel time {ksum/1e6:.1f} ms")
for n, (t, c) in sorted(tot.items(), key=lambda x: -x[1][0])[:top]:
    print(f"  {t/1e6:9.1f} ms {100*t/ksum:5.1f}%  {c:6d}x  {n[:90]}")
kinds = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
mt = defaultdict(lambda: [0, 0, 0])
for s, e, kind, b in m:
    x = mt[kinds.get(kind, str(kind))]; x[0] += e - s; x[1] += 1; x[2] += b
for kd, (t, c, b) in mt.items():
    print(f"  memcpy {kd}: {t/1e6:.1f} ms, {c}x, {b/2**20:.0f} MiB")
