#!/usr/bin/env python3
"""Parse a FreeToken server.log (or a per-phase slice of one) from the Switchyard soak
into the numbers the results file wants: decode/prefill throughput, KV+mamba occupancy,
lanes per prefill batch, the §R7-ticket-1 starvation signature, back-pressure markers,
and the pressure episodes that used to be fatal.

Usage: analyze.py <log> [<log> ...]
"""
from __future__ import annotations

import re
import statistics as st
import sys

DEC = re.compile(
    r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), "
    r"#mamba-slot: (\d+)/(\d+), mamba usage: ([\d.]+), gen throughput \(token/s\): ([\d.]+), "
    r"#queue-req: (\d+)"
)
PRE = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"token usage: ([\d.]+), #mamba-slot: (\d+)/(\d+), mamba usage: ([\d.]+), "
    r"#running-req: (\d+), #queue-req: (\d+), input throughput \(token/s\): ([\d.]+) instant"
)
MARKERS = (
    ("invariant_violated", "finishability invariant violated"),
    ("committed_pages", "committed_pages_required"),
    ("linear_exhausted", "LinearStatePool exhausted"),
    ("eviction_did_not_free", "Eviction did not free enough space"),
    ("oversize_skip", "can never be admitted"),
    ("traceback", "Traceback (most recent call last)"),
    ("released_admission", "KV protection (admission pressure)"),
    ("released_gdn", "KV protection (GDN state-slot pressure)"),
    ("released_grace", "KV protection (grace expired)"),
    ("kv_grew", "KV grew"),
    ("kv_shrank", "KV shrank"),
    ("session_expired", "expired after idle timeout"),
    ("restored_cold", "Restored cold session"),
    ("discarded_cold", "client tokens diverge"),
    ("scheduler_idle", "Scheduler is idle"),
)


def q(vals, name):
    if not vals:
        return f"{name}: n=0"
    return (f"{name}: n={len(vals)} median={st.median(vals):.2f} "
            f"mean={st.mean(vals):.2f} max={max(vals):.2f}")


def analyze(path: str) -> None:
    decodes, prefills, markers = [], [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = DEC.search(line)
            if m:
                decodes.append(tuple(float(x) for x in m.groups()))
                continue
            m = PRE.search(line)
            if m:
                prefills.append(tuple(float(x) for x in m.groups()))
                continue
            for key, pat in MARKERS:
                if pat in line:
                    markers[key] = markers.get(key, 0) + 1

    print(f"=== {path}")
    print(f"decode batches: {len(decodes)}   prefill batches: {len(prefills)}")
    if decodes:
        run = [d[0] for d in decodes]
        print(q([d[6] for d in decodes], "decode aggregate tok/s (all)"))
        print(q([d[6] / d[0] for d in decodes if d[0] > 0], "decode per-stream tok/s (all)"))
        print(f"mean #running-req = {st.mean(run):.2f}")
        for sel, label in (([d for d in decodes if d[0] >= 12], ">= 12"),
                           ([d for d in decodes if d[0] == 16], "== 16")):
            if sel:
                print(f"  #running-req {label}: n={len(sel)} "
                      f"agg median={st.median([d[6] for d in sel]):.1f} "
                      f"mean={st.mean([d[6] for d in sel]):.1f} "
                      f"max={max(d[6] for d in sel):.1f} "
                      f"per-stream median={st.median([d[6] / d[0] for d in sel]):.2f}")
        print(q([d[2] for d in decodes], "decode token usage"))
        print(q([d[3] for d in decodes], "decode mamba slots used"))
        print(f"decode batches at mamba usage 1.00: {sum(1 for d in decodes if d[5] >= 1.0)}")
    if prefills:
        lanes = [p[0] for p in prefills]
        starv = [p for p in prefills if p[0] == 1 and p[1] <= 512 and p[8] >= 8]
        fresh = [p for p in prefills if p[2] > 0]
        print(q([p[9] for p in prefills], "prefill instant tok/s"))
        print(q([p[3] for p in prefills], "prefill token usage"))
        print(q([p[1] for p in prefills], "prefill #new-token"))
        print(f"lanes per prefill batch: mean={st.mean(lanes):.2f} "
              f"median={st.median(lanes):.0f} max={max(lanes):.0f}")
        print(f"STARVATION SIGNATURE (#new-seq==1, #new-token<=512, #queue-req>=8): "
              f"{len(starv)}/{len(prefills)} = {100 * len(starv) / len(prefills):.1f}%")
        print(f"passes with #cached-token>0 (proves a FRESH admit, so chunked_inflight<cap): "
              f"{len(fresh)}")
        print(f"prefill batches at mamba usage 1.00: {sum(1 for p in prefills if p[6] >= 1.0)}")
        print("prefill batches at 96/96 mamba slots: "
              f"{sum(1 for p in prefills if p[4] == p[5] and p[5] >= 96)}")

    hot_d = [d for d in decodes if d[2] >= 0.98 and d[7] > 0]
    hot_p = [p for p in prefills if p[3] >= 0.98 and p[8] > 0]
    print(f"\npressure episodes (token usage >= 0.98 AND #queue-req > 0): "
          f"decode {len(hot_d)}, prefill {len(hot_p)}")
    print(f"  token usage >= 0.99: decode {sum(1 for d in decodes if d[2] >= 0.99)}, "
          f"prefill {sum(1 for p in prefills if p[3] >= 0.99)}")
    capped = [p for p in prefills
              if p[3] >= 0.95 and p[8] > 0 and 0 < p[1] < 8192 and p[0] == 1]
    print(f"  single-seq prefill chunks < 8192 tokens at usage >= 0.95 with queue: {len(capped)}")
    print("\nmarkers: " + (", ".join(f"{k}={v}" for k, v in sorted(markers.items())) or "none"))
    print()


for arg in sys.argv[1:]:
    analyze(arg)
