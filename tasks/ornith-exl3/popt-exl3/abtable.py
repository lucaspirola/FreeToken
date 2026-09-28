#!/usr/bin/env python3
"""Summarise an abtrees.sh TSV (lines: tree <tab> key=ms:sha ...): median ms per tree, change, hash equality.
    abtable.py results/ab-dense.tsv  -> markdown"""
import statistics as st, sys
from collections import defaultdict
t = defaultdict(lambda: defaultdict(list)); h = defaultdict(lambda: defaultdict(set)); keys = []
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if f[0] not in ("base", "new"):
        continue
    for kv in f[1:]:
        k, v = kv.split("="); ms, sha = v.split(":")
        if k not in keys: keys.append(k)
        t[k][f[0]].append(float(ms)); h[k][f[0]].add(sha)
print("| shape | 757caee ms | new ms | change | n base/new | output sha1 |\n|---|---:|---:|---:|---:|---|")
for k in keys:
    b, n = st.median(t[k]["base"]), st.median(t[k]["new"])
    same = "identical" if h[k]["base"] == h[k]["new"] and len(h[k]["base"]) == 1 else f"DIFFER {h[k]}"
    print(f"| {k} | {b:.3f} | {n:.3f} | {100 * (n / b - 1):+.1f}% | {len(t[k]['base'])}/{len(t[k]['new'])} | {same} |")
