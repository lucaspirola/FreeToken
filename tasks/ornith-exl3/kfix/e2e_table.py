#!/usr/bin/env python3
"""A/B table of the kfix e2e arms (e2e/kf-*): per probe size and pass, TTFT / prefill tok/s /
decode tok/s per arm, out_sha1 agreement with the first base arm of the same mode, and the natural
text md5s + decode tok/s. Usage: e2e_table.py [E2E_DIR]"""
import glob, json, os, sys
from collections import defaultdict

E = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e")


def probe(arm):
    try:
        return {(d["target"], d["pass"]): d for d in map(json.loads, open(f"{E}/{arm}-probe.jsonl"))}
    except FileNotFoundError:
        return {}


def med(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2 if v else None


for mode in ("saver", "whole"):
    arms = sorted(os.path.basename(p)[:-12] for p in glob.glob(f"{E}/kf*-probe-*-{mode}-*-probe.jsonl"))
    if not arms:
        continue
    P = {a: probe(a) for a in arms}
    ref = next((a for a in arms if "-base-" in a), arms[0])
    keys = sorted({k for a in arms for k in P[a]}, key=lambda k: (k[1], k[0]))
    print(f"\n### {mode}: probe (arms {', '.join(arms)}; sha vs {ref})\n")
    print("| size | pass | base TTFT s | new TTFT s | base prefill tok/s | new prefill tok/s | base decode | new decode | new/base decode | out_sha1 same |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for k in keys:
        row = {}
        for t in ("base", "new"):
            ds = [P[a][k] for a in arms if f"-{t}-" in a and k in P[a]]
            row[t] = (med([d["ttft_mono_s"] for d in ds]), med([d["prefill_tok_s_mono"] for d in ds]), med([d["decode_tok_s"] for d in ds]))
        same = sum(P[a][k]["out_sha1"] == P[ref][k]["out_sha1"] for a in arms if k in P[a])
        n = sum(k in P[a] for a in arms)
        r = row["new"][2] / row["base"][2] if row["new"][2] and row["base"][2] else None
        f = lambda v, p=2: "–" if v is None else f"{v:.{p}f}"
        print(f"| {k[0]} | {k[1]} | {f(row['base'][0], 3)} | {f(row['new'][0], 3)} | {f(row['base'][1], 0)} | {f(row['new'][1], 0)} | "
              f"{f(row['base'][2], 1)} | {f(row['new'][2], 1)} | {f(r, 3)} | {same}/{n} |")
    cov = {a: sum((d.get("mirror_delta") or {}).get("coverage_faults", 0) + (d.get("mirror_delta") or {}).get("starved_writebacks", 0) for d in P[a].values()) for a in arms}
    print(f"\ncoverage faults + starved writebacks per arm: {cov}")
for p in sorted(glob.glob(f"{E}/kf*-acceptance-R3.txt")):
    print(f"- {os.path.basename(p)[:-19]}: {open(p).read().strip().splitlines()[-1]}")
nat = sorted(glob.glob(f"{E}/kf-nat-*-natural.json"))
if nat:
    print("\n### natural text\n\n| arm | decode tok/s | md5 same as first base arm of the mode |\n|---|---:|---|")
    for mode in ("saver", "whole"):
        ns = [p for p in nat if f"-{mode}-" in p]
        rb = next((p for p in ns if "-base-" in p), None)
        rm = {x["task"]: x["md5"] for x in json.load(open(rb))["tasks"]} if rb else {}
        for p in ns:
            t = json.load(open(p))["tasks"]
            tps = sum(x["completion_tokens"] for x in t) / sum(x["s"] for x in t)
            print(f"| {os.path.basename(p)[:-13]} | {tps:.1f} | {sum(x['md5'] == rm.get(x['task']) for x in t)}/{len(t)} |")
