#!/usr/bin/env python3
"""Copied from kfix (tasks/ornith-exl3/kfix/gate_compare.py) with the label as a variable.
ck9k natural-text and /clear-replay comparisons, in the format of ck8o's
(tasks/exclusive-expert-ram/results/ck8-local/gate/ck8o-{nat,replay}-compare.txt):
natural decode tok/s vs the first whole arm and md5 per task vs it (and vs ck8o's whole arm, a
cross-commit check); replay prompt / cached token counts per request vs ck8o."""
import json, os

F = os.path.dirname(os.path.abspath(__file__)); G = f"{F}/gate"; L = os.environ.get("CK_LABEL", "ck9e")
CK8 = os.path.join(F, "../../exclusive-expert-ram/results/ck8-local/gate")


def nat(p):
    t = json.load(open(p))["tasks"]
    return sum(x["completion_tokens"] for x in t) / sum(x["s"] for x in t), {x["task"]: x["md5"] for x in t}


ref_tps, ref = nat(f"{G}/{L}-nat-whole-1-natural.json")
_, ck8 = nat(f"{CK8}/ck8o-nat-whole-1-natural.json")
with open(f"{G}/{L}-nat-compare.txt", "w") as out:
    for a in ("whole-1", "mirror-1", "whole-2", "mirror-2"):
        tps, md5 = nat(f"{G}/{L}-nat-{a}-natural.json")
        out.write(f"{L}-nat-{a}-natural.json  {tps:6.1f} tok/s  {100 * tps / ref_tps:5.1f}%  md5 same as ref {sum(md5[k] == ref.get(k) for k in md5)}/{len(md5)}"
                  f"  same as ck8o-nat-whole-1 {sum(md5[k] == ck8.get(k) for k in md5)}/{len(md5)}\n")
a, b = json.load(open(f"{G}/{L}-replay.json")), json.load(open(f"{CK8}/ck8o-replay.json"))
with open(f"{G}/{L}-replay-compare.txt", "w") as out:
    n = diff = 0
    for agent, steps in a.items():
        for step, r in steps.items():
            t = r.get("tokens") or {}
            o = (b.get(agent, {}).get(step, {}) or {}).get("tokens") or {}
            same = (t.get("prompt"), t.get("cached")) == (o.get("prompt"), o.get("cached"))
            n += 1; diff += not same
            out.write(f"{agent:12s} {step:7s} {L} prompt {t.get('prompt')} cached {t.get('cached')} | ck8o prompt {o.get('prompt')} "
                      f"cached {o.get('cached')} {'SAME' if same else 'DIFF'}\n")
    out.write(f"{n} requests, {diff} differ ({L} vs ck8o; Codex reports no usage)\n")
print(open(f"{G}/{L}-nat-compare.txt").read() + open(f"{G}/{L}-replay-compare.txt").read())
