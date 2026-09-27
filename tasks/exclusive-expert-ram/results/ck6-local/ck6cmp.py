#!/usr/bin/env python3
"""Pass-by-pass decode of arm B against arm A (probe.jsonl), plus each arm's record counters.

  ck6cmp.py DIR A B [BAR%]     (BAR default 91: the gate is B >= 91% of A, i.e. within 9%)
"""
import json, os, sys

D, A, B = sys.argv[1:4]
BAR = float(sys.argv[4]) if len(sys.argv) > 4 else 91.0


def probes(arm):
    out = {}
    p = os.path.join(D, f"{arm}-probe.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "decode_tok_s" in r:
                out[(r["target"], r["pass"])] = r
    return out


def record(arm):
    p = os.path.join(D, f"{arm}-record.json")
    return json.load(open(p)) if os.path.exists(p) else {}


pa, pb = probes(A), probes(B)
bad = 0
print(f"== {B} vs {A} (bar {BAR:.0f}%)")
for k in sorted(set(pa) | set(pb)):
    a, b = pa.get(k), pb.get(k)
    if not a or not b:
        print(f"  {k[0]//1000}K p{k[1]}: missing in {'A' if not a else 'B'}"); bad += 1; continue
    ratio = 100.0 * b["decode_tok_s"] / a["decode_tok_s"]
    ok = ratio >= BAR
    bad += not ok
    print(f"  {k[0]//1000:>4}K p{k[1]}: {b['decode_tok_s']:7.1f} vs {a['decode_tok_s']:7.1f}  {ratio:6.1f}%  "
          f"gap {b.get('median_gap_ms')} vs {a.get('median_gap_ms')} ms  ttft {b.get('ttft_s')} vs {a.get('ttft_s')} s"
          f"  {'ok' if ok else 'BELOW'}")
for arm in (A, B):
    r = record(arm)
    print(f"  {arm}: " + ", ".join(f"{k}={r.get(k)}" for k in
                                   ("ram_gib", "rss_ready_gib", "anon_gib", "coverage_faults", "starved", "code")))
print(f"{bad} point(s) outside the band")
