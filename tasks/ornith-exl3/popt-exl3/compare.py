#!/usr/bin/env python3
"""Per-arm table of probe (pass 2) and extend (both passes) results, with identity checks.
    compare.py DIR ARM[,ARM...] ...   (arms joined by ',' are replicates, averaged)"""
import json, sys, os, statistics as st

def load(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            try: out.append(json.loads(l))
            except Exception: pass
    return out

d = sys.argv[1]; groups = [g.split(",") for g in sys.argv[2:]]
def m(xs): xs = [x for x in xs if x is not None]; return st.mean(xs) if xs else float("nan")
rows = {}
sha = {}
for g in groups:
    key = "+".join(g)
    for a in g:
        for r in load(f"{d}/{a}-probe.jsonl"):
            if r.get("pass") != 2: continue
            k = ("probe", r["target"])
            rows.setdefault(k, {}).setdefault(key, []).append((r["ttft_mono_s"], r["prefill_tok_s_mono"], r["decode_tok_s"], r.get("gap1_ms")))
            sha.setdefault(k, {}).setdefault(a, r["out_sha1"])
        for r in load(f"{d}/{a}-extend.jsonl"):
            k = ("ext" if r.get("add") else "ctx", r["depth"], r.get("add") or 0, r["pass"])
            rows.setdefault(k, {}).setdefault(key, []).append((r["ttft_mono_s"], r["extend_tok_s"], r.get("decode_tok_s"), None))
            sha.setdefault(k, {}).setdefault(a, r["out_sha1"])
keys = ["+".join(g) for g in groups]
print("shape".ljust(26) + "".join(f"{k[:22]:>24s}" for k in keys) + "  identical")
for k in sorted(rows, key=lambda k: (k[0] != "probe", k)):
    cells = []
    for g in keys:
        v = rows[k].get(g)
        if not v: cells.append(" " * 24); continue
        t = m([x[0] for x in v]); tp = m([x[1] for x in v]); dc = m([x[2] for x in v])
        cells.append(f"{t:8.3f}s {tp:6.0f} {dc:6.1f}".rjust(24))
    ident = len(set(sha[k].values())) == 1
    print(" ".join(map(str, k)).ljust(26) + "".join(cells) + ("  yes" if ident else f"  NO {sorted(set(v[:8] for v in sha[k].values()))}"))
print("cells: TTFT s, prefill/extend tok/s, decode tok/s")
