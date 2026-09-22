"""Checkpoint 7.2: compare a pool arm's needle/recall answers with the whole-model
reference recorded on the same commit, field for field. Exit 1 on any difference.

usage: compare_needles.py <results_dir> <reference_arm> <arm>
"""
import json, os, re, sys

D, REF, ARM = sys.argv[1:4]
FIELDS = ("answer", "completion_tokens", "reasoning_chars", "finish_reason", "correct")
bad = 0

def needles(arm):
    d = json.load(open(os.path.join(D, f"{arm}-needles.json")))
    return {(s["target"], q): v for s in d["sizes"] for q, v in s["questions"].items()}

def recall(arm):
    out = {}
    for line in open(os.path.join(D, f"{arm}-post.txt")):
        line = line.strip()
        if line.startswith("{") and '"planted"' in line:
            r = json.loads(line); out[r["target"]] = r
    return out

a, b = needles(REF), needles(ARM)
for k in sorted(set(a) | set(b)):
    if k not in a or k not in b:
        print(f"MISSING {k}"); bad += 1; continue
    diff = [f for f in FIELDS if a[k].get(f) != b[k].get(f)]
    print(f"{'SAME' if not diff else 'DIFF'} needles {k[0]:>7} {k[1]:<13}" + (f" {diff}" if diff else ""))
    bad += bool(diff)
ra, rb = recall(REF), recall(ARM)
for t in sorted(set(ra) | set(rb)):
    if t not in ra or t not in rb:
        print(f"MISSING recall {t}"); bad += 1; continue
    diff = [f for f in ("answer", "completion_tokens", "finish_reason", "all_three") if ra[t].get(f) != rb[t].get(f)]
    print(f"{'SAME' if not diff else 'DIFF'} recall  {t:>7}" + (f" {diff}" if diff else ""))
    bad += bool(diff)
print(f"{bad} difference(s)")
sys.exit(1 if bad else 0)
