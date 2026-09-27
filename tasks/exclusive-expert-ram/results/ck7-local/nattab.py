#!/usr/bin/env python3
"""Natural-text arms: aggregate decode tok/s (sum completion tokens / sum seconds), % of the
first arm (the whole-model reference), and whether every task's output md5 matches it.
  nattab.py REF.json ARM.json ..."""
import json, sys

def load(p):
    t = json.load(open(p))["tasks"]
    return sum(x["completion_tokens"] for x in t) / sum(x["s"] for x in t), {x["task"]: x["md5"] for x in t}

ref, rmd5 = load(sys.argv[1])
for p in sys.argv[1:]:
    tps, md5 = load(p)
    same = sum(md5[k] == rmd5.get(k) for k in md5)
    print(f"{p:45} {tps:7.1f} tok/s {100 * tps / ref:6.1f}%  md5 same as ref {same}/{len(md5)}")
