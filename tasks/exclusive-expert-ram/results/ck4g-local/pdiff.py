#!/usr/bin/env python3
"""pdiff.py A.tsv B.tsv : files whose cached bytes grew from A to B, grouped by top dir."""
import sys, collections
def load(p):
    return {l.split("\t")[0]: int(l.split("\t")[2]) for l in open(p)}
a, b = load(sys.argv[1]), load(sys.argv[2])
grow = {p: b[p] - a.get(p, 0) for p in b if b[p] - a.get(p, 0) > 0}
shrink = sum(a[p] - b.get(p, 0) for p in a if a[p] - b.get(p, 0) > 0)
G = 2**30
print(f"grew {sum(grow.values())/G:.3f} GiB in {len(grow)} files; shrank {shrink/G:.3f} GiB")
grp = collections.Counter()
for p, v in grow.items():
    parts = p.split("/")
    key = "/".join(parts[:7]) if "site-packages" not in p else "/".join(parts[:parts.index("site-packages")+2])
    grp[key] += v
for k, v in grp.most_common(25): print(f"{v/G:8.3f} GiB  {k}")
print("--- top files")
for p, v in sorted(grow.items(), key=lambda x: -x[1])[:25]: print(f"{v/2**20:9.1f} MiB  {p}")
