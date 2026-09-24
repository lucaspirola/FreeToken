#!/usr/bin/env python3
"""Host timeline of the inter-graph gap (nsys sqlite, --cuda-graph-trace=graph).
For each decode step: graph_i end -> graph_{i+1} start (GPU idle between graphs), where the
host was when graph_i ended (inside which API call), and the host time from the end of the
sync that waited for graph_i to the cudaGraphLaunch of graph_{i+1}, split by API name and by
the python time between API calls."""
import sqlite3, sys, statistics as S
from collections import defaultdict
for p in sys.argv[1:]:
    c = sqlite3.connect(p)
    names = dict(c.execute("select id,value from StringIds"))
    g = c.execute("select start,end,correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start").fetchall()
    api = c.execute("select start,end,nameId,correlationId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
    launch_of = {a[3]: a for a in api}
    runs, cur = [], [g[0]]
    for a, b in zip(g, g[1:]):
        if b[0] - a[0] > 50e6:
            runs.append(cur); cur = []
        cur.append(b)
    runs.append(cur)
    print(f"===== {p}")
    for pi, run in enumerate(runs, 1):
        if len(run) < 20:
            continue
        idle, post, byname, pyt, launch_lat = [], [], defaultdict(float), [], []
        steps = run[5:-1]
        for i, s in enumerate(steps):
            nxt = run[run.index(s) + 1]
            idle.append(nxt[0] - s[1])
            la = launch_of.get(nxt[2])
            if la is None:
                continue
            launch_lat.append(nxt[0] - la[0])   # launch call start -> graph start on GPU
            tid = la[4]
            # the last cudaStreamSynchronize/EventSynchronize on that thread ending after graph_i started
            syncs = [a for a in api if a[4] == tid and s[0] <= a[1] <= la[0]
                     and "ynchronize" in names[a[2]]]
            t0 = syncs[-1][1] if syncs else s[1]
            post.append(la[0] - t0)
            inwin = [a for a in api if a[4] == tid and t0 <= a[0] < la[0]]
            busy = 0
            for a in inwin:
                byname[names[a[2]].split("_v")[0]] += a[1] - a[0]; busy += a[1] - a[0]
            pyt.append(la[0] - t0 - busy)
        n = len(post)
        print(f"-- pass {pi}: idle between graphs mean {S.mean(idle)/1e3:.1f} us; host sync-end -> next launch "
              f"mean {S.mean(post)/1e3:.1f} us (API {sum(byname.values())/n/1e3:.1f}, rest {S.mean(pyt)/1e3:.1f}); "
              f"launch call -> graph start {S.mean(launch_lat)/1e3:.1f} us")
        for k, v in sorted(byname.items(), key=lambda x: -x[1])[:10]:
            print(f"     {k:40} {v/n/1e3:8.1f} us/step")
