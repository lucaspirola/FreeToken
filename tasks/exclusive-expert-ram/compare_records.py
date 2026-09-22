#!/usr/bin/env python3
"""Checkpoint acceptance numbers: a checkpoint arm against the record, pass by pass.

    compare_records.py <results_dir> <ck>      # e.g. results ck2

Pairs (the /goal's record, frozen exp/exclusive-expert-ram):
  <ck>-mirror-1m  vs nemotron-reserve-2e-1m   (1M: 76.9 pass 1 / 72.9 pass 2)
  <ck>-mirror     vs nemotron-reserve-2e      (8K/80K/713K)
Criteria: decode >= 91% of the record at every size and pass (faster is fine), RAM of the
1M arm in 12.26 +/- 0.6 GiB, coverage faults 0, starved write-backs 0. A single point
failing only with the ~2.5 s delivery-stall signature is re-measured, not accepted as a
regression (plan 7.2) -- this script only reports it.
"""
import json
import os
import re
import sys

d, ck = sys.argv[1], sys.argv[2]
load = lambda n: json.load(open(os.path.join(d, f"{n}-record.json")))
bad = 0
for arm, ref in ((f"{ck}-mirror-1m", "nemotron-reserve-2e-1m"), (f"{ck}-mirror", "nemotron-reserve-2e")):
    a, r = load(arm), load(ref)
    print(f"== {arm} ({a.get('commit')}) vs {ref} ({r.get('commit')})")
    for k in sorted(k for k in r if re.fullmatch(r"decode_\d+k(_p1)?", k)):
        if k not in a:
            continue
        ratio = a[k] / r[k] if r[k] else 0
        ok = ratio >= 0.91
        bad += not ok
        tot = k.replace("decode", "total")
        print(f"  {k:18} {a[k]:7.1f} vs {r[k]:7.1f}  {ratio:6.1%}  {'ok' if ok else 'BELOW'}"
              f"   total {a.get(tot)} s, ttft {a.get(k.replace('decode', 'ttft'))} s")
    for k in ("coverage_faults", "starved"):
        ok = a.get(k) == 0
        bad += not ok
        print(f"  {k:18} {a.get(k)}  {'ok' if ok else 'NONZERO'}")
    print(f"  ram_gib            {a.get('ram_gib')} (record {r.get('ram_gib')})"
          f"  free_evict_rate {a.get('free_evict_rate')} (record {r.get('free_evict_rate')})")
    if arm.endswith("-1m"):
        ok = abs(a["ram_gib"] - 12.26) <= 0.6
        bad += not ok
        print(f"  ram 12.26 +/- 0.6  {'ok' if ok else 'OUT'}")
print(f"{bad} point(s) outside the band")
sys.exit(1 if bad else 0)
