#!/usr/bin/env python3
"""256K control: per arm 8K/256K decode tok/s by pass, and saver/whole ratios per tree (mean of arms)."""
import json, sys, os, statistics as st
d = sys.argv[1]; arms = {}
for f in sorted(os.listdir(d)):
    if f.endswith("-probe.jsonl"):
        a = f[:-12]; arms[a] = {}
        for l in open(f"{d}/{f}"):
            r = json.loads(l); arms[a][(r["target"], r["pass"])] = r["decode_tok_s"]
        print(a, " ".join(f"{t//1000}K p{p}: {v}" for (t, p), v in sorted(arms[a].items())))
for tree in ("cn", "cb"):
    for k in ((8000, 1), (8000, 2), (256000, 1), (256000, 2)):
        s = [arms[a][k] for a in arms if a.startswith(tree + "-s") and k in arms[a]]
        w = [arms[a][k] for a in arms if a.startswith(tree + "-w") and k in arms[a]]
        if s and w:
            print(f"{tree} {k[0]//1000}K p{k[1]}: saver {st.mean(s):.1f} whole {st.mean(w):.1f} = {100*st.mean(s)/st.mean(w):.1f}%  (saver arms {s}, whole arms {w})")
