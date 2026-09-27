#!/usr/bin/env python3
"""Per-decode-step GPU time by kernel/memcpy from a --cuda-graph-trace=node sqlite export.

  kernels.py A.sqlite [B.sqlite]

The decode window of each pass is found from the kernels themselves: a pass is a run of
kernel activity with no gap > 50 ms; inside a run, decode steps are counted by the launches
of the kernel that runs once per step (the least frequent kernel present in every step is not
knowable a priori, so the step count is the number of `_resolve`-free anchor launches: the
first kernel name that appears exactly once per 40-ish layer group is not needed -- we use the
step count from the API: cudaGraphLaunch calls inside the run). Prefill kernels are excluded by
taking only the time after the last kernel longer than 2 ms (prefill GEMMs) in the run.
"""
import sqlite3, sys
from collections import defaultdict

def load(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id,value from StringIds"))
    ker = [(s, e, names.get(n, n)) for s, e, n in
           c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    kinds = {1: "memcpy H2D", 2: "memcpy D2H", 8: "memcpy D2D", 10: "memcpy P2P"}
    mem = [(s, e, kinds.get(k, str(k))) for s, e, k in
           c.execute("select start,end,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
    api = [(s, names.get(n, n)) for s, n in
           c.execute("select start,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
    return ker, mem, api

def passes(ker):
    runs, cur = [], [ker[0]]
    for a, b in zip(ker, ker[1:]):
        if b[0] - cur[-1][1] > 50e6:
            runs.append(cur); cur = []
        cur.append(b)
    runs.append(cur)
    return [r for r in runs if len(r) > 10000]

def main(path):
    ker, mem, api = load(path)
    out = {}
    for pi, run in enumerate(passes(ker), 1):
        big = [k for k in run if k[1] - k[0] > 2e6]
        lo = (big[-1][1] if big else run[0][0]) + 50e6   # skip prefill and the first decode steps
        hi = run[-1][1] - 20e6
        steps = sum(1 for s, n in api if lo <= s < hi and n.startswith("cudaGraphLaunch"))
        agg = defaultdict(lambda: [0, 0])
        for s, e, n in ker:
            if lo <= s < hi:
                agg[n][0] += 1; agg[n][1] += e - s
        for s, e, n in mem:
            if lo <= s < hi:
                agg[n][0] += 1; agg[n][1] += e - s
        tot = sum(v[1] for v in agg.values())
        print(f"-- {path} pass {pi}: {steps} steps in {(hi-lo)/1e6:.0f} ms, "
              f"{(hi-lo)/max(steps,1)/1e3:.1f} us/step wall, summed GPU {tot/max(steps,1)/1e3:.1f} us/step")
        out[pi] = (steps, agg)
        for n, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:30]:
            print(f"   {n[:70]:70} {c/max(steps,1):7.2f}/step {t/max(steps,1)/1e3:8.1f} us/step")
    return out

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
