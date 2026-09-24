#!/usr/bin/env python3
"""The 1M RAM gate by cgroup: candidate vs reference, same host and session.

  compare_ram_cgroup.py <samples.tsv> <results dir> <candidate arm> <reference arm>

At ready, after 1M p1 (request 2) and after 1M p2 (request 4) it prints cgroup
memory.current and anon for both arms, candidate minus reference, and that delta with the
pool-size difference removed. Pool sizes come from each arm's journal
("mirror expert pool attached: N host rows, X GiB"). The rule is the raw delta within
+/- 0.6 GiB [agent practice, the coordinator's gate]. The record's ram_gib (MemAvailable
delta) is printed next to it for continuity.
"""
import csv
import json
import re
import sys
from pathlib import Path

G = 1 << 30


def points(rows, arm):
    rs = [r for r in rows if r["unit"] == arm]
    out = {}
    for r in rs:
        if r["ready"] == "1" and "ready" not in out:
            out["ready"] = r
        k = int(r["probes_done"] or 0)
        # first sample at or after the request finished (8K p2 is 2 s long and can
        # finish inside the same 10 s sampling interval as 1M p1)
        for need, name in ((2, "1M p1"), (4, "1M p2")):
            if r["ready"] == "1" and k >= need and name not in out:
                out[name] = r
    return out


def pool(d, arm):
    p = d / f"{arm}-journal.txt"
    m = re.findall(r"mirror expert pool attached: (\d+) host rows, ([0-9.]+) GiB",
                   p.read_text(errors="replace")) if p.exists() else []
    return (int(m[-1][0]), float(m[-1][1])) if m else (None, None)


def main():
    tsv, d, cand, ref = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    pc, pr = points(rows, cand), points(rows, ref)
    (rc, gc), (rr, gr) = pool(d, cand), pool(d, ref)
    dpool = (gc - gr) if gc is not None and gr is not None else 0.0
    print(f"pool: {cand} {rc} rows / {gc} GiB, {ref} {rr} rows / {gr} GiB, difference {dpool:+.2f} GiB")
    verdict = []
    for pt in ("ready", "1M p1", "1M p2"):
        a, b = pc.get(pt), pr.get(pt)
        if not (a and b):
            print(f"{pt}: missing ({cand} {'ok' if a else '-'}, {ref} {'ok' if b else '-'})")
            continue
        for k, lab in (("cg_current", "memory.current"), ("cg_anon", "anon")):
            x, y = int(a[k]) / G, int(b[k]) / G
            dlt = x - y
            print(f"{pt:6} {lab:15} cand {x:6.2f}  ref {y:6.2f}  delta {dlt:+5.2f}  "
                  f"delta without the pool difference {dlt - dpool:+5.2f}")
            if pt != "ready" and k == "cg_current":
                verdict.append(abs(dlt) <= 0.6)
        x, y = int(a["vm_rss"] or 0) / G, int(b["vm_rss"] or 0) / G
        print(f"{pt:6} {'largest proc RSS':15} cand {x:6.2f}  ref {y:6.2f}  delta {x - y:+5.2f}")
    for arm in (cand, ref):
        p = d / f"{arm}-record.json"
        if p.exists():
            r = json.loads(p.read_text())
            print(f"{arm}: ram_gib {r.get('ram_gib')} (MemAvailable delta), end-of-arm current "
                  f"{r.get('current_gib')}, peak {r.get('peak_current_gib')}, rss_ready {r.get('rss_ready_gib')}")
    if verdict:
        print("gate (raw memory.current delta at 1M p1 and p2 within +/- 0.6 GiB):",
              "PASS" if all(verdict) else "FAIL")


if __name__ == "__main__":
    main()
