#!/usr/bin/env python3
"""Per-decode-step breakdown from an nsys sqlite export (--cuda-graph-trace=graph).

A decode step = one CUDA graph launch (CUPTI_ACTIVITY_KIND_GRAPH_TRACE). For each step i the
window is [graph_i.start, graph_{i+1}.start). Reports per pass (launch runs separated by
>50 ms): step period, graph GPU time, non-graph kernels and memcpys in the window (by name /
kind, time and bytes, stream), GPU busy union and idle, and the CPU API calls in the window.
"""
import sqlite3, sys, statistics as S
from collections import defaultdict

def main(path):
    c = sqlite3.connect(path)
    g = c.execute("select start,end,streamId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start").fetchall()
    runs, cur = [], [g[0]]
    for a, b in zip(g, g[1:]):
        if b[0] - a[0] > 50e6:
            runs.append(cur); cur = []
        cur.append(b)
    runs.append(cur)
    names = dict(c.execute("select id,value from StringIds"))
    ker = c.execute("select start,end,streamId,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    mem = c.execute("select start,end,streamId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
    api = c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
    kinds = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
    for pi, run in enumerate(runs, 1):
        if len(run) < 20:
            continue
        steps = run[5:-1]  # skip the first steps (post-prefill) and the last (no next start)
        nxt = {s[0]: run[run.index(s) + 1][0] for s in steps}
        per = [nxt[s[0]] - s[0] for s in steps]
        gdur = [s[1] - s[0] for s in steps]
        kagg = defaultdict(lambda: [0, 0, set()]); magg = defaultdict(lambda: [0, 0, 0, set()])
        aagg = defaultdict(lambda: [0, 0]); busy_tot = 0
        lo, hi = steps[0][0], nxt[steps[-1][0]]
        for s in steps:
            w0, w1 = s[0], nxt[s[0]]
            iv = [(s[0], s[1])]
            for k in ker:
                if k[0] >= w0 and k[0] < w1:
                    a = kagg[names.get(k[3], k[3])]; a[0] += 1; a[1] += k[1] - k[0]; a[2].add(k[2]); iv.append((k[0], k[1]))
            for m in mem:
                if m[0] >= w0 and m[0] < w1:
                    a = magg[kinds.get(m[3], m[3])]; a[0] += 1; a[1] += m[1] - m[0]; a[2] += m[4]; a[3].add(m[2]); iv.append((m[0], m[1]))
            iv.sort(); u = 0; ce = w0
            for a0, a1 in iv:
                a0 = max(a0, ce); a1 = min(a1, w1)
                if a1 > a0: u += a1 - a0; ce = a1
            busy_tot += u
        for a in api:
            if lo <= a[0] < hi:
                x = aagg[names.get(a[2], a[2])]; x[0] += 1; x[1] += a[1] - a[0]
        n = len(steps)
        print(f"-- pass {pi}: {n} steps, period mean {S.mean(per)/1e3:.1f} us median {S.median(per)/1e3:.1f} us "
              f"(={1e9/S.mean(per):.1f} tok/s); graph GPU {S.mean(gdur)/1e3:.1f} us; GPU busy {busy_tot/n/1e3:.1f} us, idle {(S.mean(per)-busy_tot/n)/1e3:.1f} us per step")
        for k, (cnt, t, st) in sorted(kagg.items(), key=lambda x: -x[1][1])[:8]:
            print(f"   kernel {k[:60]:60} {cnt/n:5.2f}/step {t/n/1e3:8.1f} us/step streams {sorted(st)}")
        for k, (cnt, t, b, st) in sorted(magg.items(), key=lambda x: -x[1][1]):
            print(f"   memcpy {k:5} {cnt/n:5.2f}/step {t/n/1e3:8.1f} us/step {b/n/1e6:8.2f} MB/step streams {sorted(st)}")
        for k, (cnt, t) in sorted(aagg.items(), key=lambda x: -x[1][1])[:8]:
            print(f"   api    {k[:50]:50} {cnt/n:5.2f}/step {t/n/1e3:8.1f} us/step")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=====", p); main(p)
