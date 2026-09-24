#!/usr/bin/env python3
"""In-graph kernel time per decode step (nsys --cuda-graph-trace=node export): kernels and
memcpys with a graphNodeId, summed by name and divided by the decode step count (the
number of graph-launch API calls in the capture)."""
import sqlite3, sys
from collections import defaultdict
for p in sys.argv[1:]:
    c = sqlite3.connect(p)
    names = dict(c.execute("select id,value from StringIds"))
    steps = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like 'cudaGraphLaunch%'").fetchone()[0]
    agg = defaultdict(lambda: [0, 0])
    for sn, st, en in c.execute("select shortName,start,end from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null and graphNodeId != 0"):
        a = agg[names.get(sn, sn)]; a[0] += 1; a[1] += en - st
    tot = sum(v[1] for v in agg.values())
    print(f"===== {p}: {steps} graph launches, in-graph kernel time {tot/steps/1e3:.1f} us/step")
    for k, (n, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:14]:
        print(f"   {k[:60]:60} {n/steps:6.1f}/step {t/steps/1e3:8.1f} us/step")
    for kind, n, t, b in c.execute("select copyKind,count(*),sum(end-start),sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY where graphNodeId is not null and graphNodeId != 0 group by copyKind"):
        print(f"   memcpy kind {kind}: {n/steps:.1f}/step {t/steps/1e3:.1f} us/step {b/steps/1e6:.2f} MB/step")
