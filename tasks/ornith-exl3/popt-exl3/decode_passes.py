#!/usr/bin/env python3
"""Probe decode tok/s, both passes, every arm of ab1 and ab3 (8K and 80K), and the mean per tree and
mode: the no-regression check with every sample rather than pass 2 of two arms."""
import json, statistics as st
from collections import defaultdict
m = defaultdict(list)
for d, arms in (("ab1", "b-w1 n-w1 n-w2 b-w2 b-s1 n-s1 n-s2 b-s2"), ("ab3", "b-wd1 n-wd1 n-wd2 b-wd2 b-sd1 n-sd1 n-sd2 b-sd2")):
    for a in arms.split():
        for l in open(f"results/{d}/{a}-probe.jsonl"):
            x = json.loads(l)
            if x["target"] in (8000, 80000):
                mode = "whole" if "w" in a.split("-")[1] else "saver"
                m[(mode, x["target"], a[0])].append(x["decode_tok_s"])
                print(d, a, x["target"], x["pass"], x["decode_tok_s"])
for mode in ("saver", "whole"):
    for t in (8000, 80000):
        b, n = m[(mode, t, "b")], m[(mode, t, "n")]
        print(f"{mode} {t}: 757caee {st.mean(b):.1f} (n={len(b)}, sd {st.stdev(b):.1f})  new {st.mean(n):.1f} (n={len(n)}, sd {st.stdev(n):.1f})  {100 * (st.mean(n) / st.mean(b) - 1):+.1f}%")
