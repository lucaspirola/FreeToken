#!/usr/bin/env python3
"""Pool arms against the SAME box's whole-model arms, pass by pass (the ck4 decode gate).

    compare_box.py <box_dir> <ck>      # e.g. results/ck4-box ck4

The reference for each decode_<size>k[_p1] key comes from <ck>-whole, or from <ck>-whole-1m
for keys whole does not have (1M). Pass: pool decode >= 91% of the reference. Also prints
whole-close against whole (box drift) and the pool arms' faults/starved/RAM.
"""
import json
import os
import re
import sys

d, ck = sys.argv[1], sys.argv[2]


def load(n):
    p = os.path.join(d, f"{ck}-{n}-record.json")
    return json.load(open(p)) if os.path.exists(p) else None


ref = {}
for n in ("whole-1m", "whole"):  # whole wins where both have the key
    r = load(n)
    if r:
        ref.update({k: (v, n) for k, v in r.items() if re.fullmatch(r"decode_\d+k(_p1)?", k)})
bad = 0
for arm in ("mirror-1m", "mirror"):
    a = load(arm)
    if not a:
        continue
    print(f"== {ck}-{arm} ({a.get('commit')}) vs same-box whole")
    for k in sorted(k for k in a if re.fullmatch(r"decode_\d+k(_p1)?", k)):
        if k not in ref:
            print(f"  {k:18} {a[k]:7.1f}  (no same-box reference)")
            continue
        v, n = ref[k]
        ratio = a[k] / v
        ok = ratio >= 0.91
        bad += not ok
        print(f"  {k:18} {a[k]:7.1f} vs {v:7.1f} ({n:8}) {ratio:6.1%}  {'ok' if ok else 'BELOW'}")
    for k in ("coverage_faults", "starved"):
        ok = a.get(k) == 0
        bad += not ok
        print(f"  {k:18} {a.get(k)}  {'ok' if ok else 'NONZERO'}")
    print(f"  ram_gib            {a.get('ram_gib')}  free_evict_rate {a.get('free_evict_rate')}")
    if arm == "mirror-1m":
        ok = abs(a["ram_gib"] - 12.26) <= 0.6
        bad += not ok
        print(f"  ram 12.26 +/- 0.6  {'ok' if ok else 'OUT'}")
w, c = load("whole"), load("whole-close")
if w and c:
    print("== whole-close vs whole (box drift)")
    for k in sorted(k for k in w if k in c and re.fullmatch(r"decode_\d+k(_p1)?", k)):
        print(f"  {k:18} {c[k]:7.1f} vs {w[k]:7.1f}  {c[k] / w[k]:6.1%}")
print(f"{bad} point(s) outside the band")
sys.exit(1 if bad else 0)
