#!/usr/bin/env python3
"""Table of the policy A/B probe arms: decode tok/s, % of the mean of the whole arms, swaps,
writebacks and decode hit rate (1 - missing/active, from --moe-collect-stats) per point.
  poltab.py DIR [GLOB]  (GLOB default pol*-[0-9]*-probe.jsonl; the idle-seed A/B uses "sd-[0-9]*-probe.jsonl")"""
import glob, json, os, sys
from collections import defaultdict

D = sys.argv[1]
rows = defaultdict(list)
for f in sorted(glob.glob(os.path.join(D, sys.argv[2] if len(sys.argv) > 2 else "pol*-[0-9]*-probe.jsonl"))):
    arm = os.path.basename(f)[:-len("-probe.jsonl")]
    pol = arm.split("-", 2)[2]
    for line in open(f):
        r = json.loads(line)
        rows[(r["target"], r["pass"])].append((arm, pol, r))
for key in sorted(rows):
    whole = [r["decode_tok_s"] for _, p, r in rows[key] if p == "whole"]
    ref = sum(whole) / len(whole) if whole else None
    print(f"== {key[0]//1000}K pass {key[1]} (whole mean {ref:.1f})")
    for arm, pol, r in rows[key]:
        m, d = r.get("mirror_delta", {}), r.get("decode_delta", {})
        act, miss = d.get("active") or d.get("active_experts"), d.get("missing") or d.get("missing_experts")
        hit = f"{1 - miss / act:.4f}" if act and miss is not None else "-"
        print(f"  {arm:14} {r['decode_tok_s']:6.1f} {100 * r['decode_tok_s'] / ref:6.1f}%  gap {r['median_gap_ms']} ms  "
              f"swaps {m.get('swaps', '-'):>5} wb {m.get('writebacks', '-'):>5}  hit {hit}  "
              f"decode keys {sorted(d)[:4] if not act else ''}")
