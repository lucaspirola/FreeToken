#!/usr/bin/env python3
"""Natural-text arms: decode tok/s (sum completion tokens / sum seconds) and md5 per task vs REF.
    natcmp.py DIR REF ARM..."""
import json, sys, os
d, ref, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
def nat(a):
    t = json.load(open(f"{d}/{a}-natural.json"))["tasks"]
    return sum(x["completion_tokens"] for x in t) / sum(x["s"] for x in t), {x["task"]: x["md5"] for x in t}
rt, rm = nat(ref)
for a in [ref] + arms:
    if not os.path.exists(f"{d}/{a}-natural.json"): print(a, "missing"); continue
    tps, m = nat(a)
    same = sum(m.get(k) == v for k, v in rm.items())
    print(f"{a:8s} {tps:7.1f} tok/s  {100 * tps / rt:6.1f}% of {ref}  md5 {same}/{len(rm)} vs {ref}")
