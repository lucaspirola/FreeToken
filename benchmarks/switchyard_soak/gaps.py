#!/usr/bin/env python3
"""Wall-clock gaps between consecutive batch log lines + LEADING/TRAILING silence.

Lesson from the 81ab30e soak: a scheduler policy is graded on the wall clock it spends
NOT scheduling, which shows up as gaps between batch-log lines.

Lesson from the ea7ed7c soak (2026-09-05): a *deadlock* produces ZERO such gaps, because
the silence starts after the LAST batch line and never ends -- there is no second line to
measure against. So this also reports, against the phase window taken from driver.log:
  leading silence  = phase start -> first batch line
  trailing silence = last batch line -> phase end   <-- the deadlock signature
  % of the phase spent scheduling
Trailing silence >= 120 s is a WARN; the passing tree records 1 s (0.1 %).

Usage: gaps.py <phase.log> [thresh_s] [phase_start_epoch phase_end_epoch]
"""
import datetime
import re
import sys

TS = re.compile(r"\[(\d{4}-\d{2}-\d{2})\|(\d{2}:\d{2}:\d{2})\|")
BATCH = re.compile(r"(Decode|Prefill) batch,")
USAGE = re.compile(r"token usage: ([\d.]+)")
QUEUE = re.compile(r"#queue-req: (\d+)")
RUN = re.compile(r"#running-req: (\d+)")

MARKERS = {
    "finishability invariant violated": "invariant_violated",
    "committed_pages_required": "committed_pages_required",
    "LinearStatePool exhausted": "linear_state_exhausted",
    "Eviction did not free enough space": "eviction_did_not_free",
    "can never be admitted": "oversize_skip",
    "Traceback (most recent call last)": "traceback",
    "expired after idle timeout": "session_idle_expired",
    "Restored cold session": "cold_restore",
}

path = sys.argv[1]
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
p_start = float(sys.argv[3]) if len(sys.argv) > 4 else None
p_end = float(sys.argv[4]) if len(sys.argv) > 4 else None

rows = []          # (epoch, line)
marker_hits = {}
with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        for pat, key in MARKERS.items():
            if pat in line:
                marker_hits[key] = marker_hits.get(key, 0) + 1
        m = TS.search(line)
        if not m or not BATCH.search(line):
            continue
        t = datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
        rows.append((t.timestamp(), line.rstrip()))

if not rows:
    print(f"{path}: NO BATCH LINES AT ALL -- the scheduler never ran")
    sys.exit(0)

gaps = []
for (t0, l0), (t1, l1) in zip(rows, rows[1:]):
    d = t1 - t0
    if d >= thresh:
        gaps.append((d, t0, l0, l1))

print(f"=== {path}")
print(f"batch lines: {len(rows)}   span "
      f"{datetime.datetime.fromtimestamp(rows[0][0]):%H:%M:%S} -> "
      f"{datetime.datetime.fromtimestamp(rows[-1][0]):%H:%M:%S} "
      f"({rows[-1][0] - rows[0][0]:.0f} s)")
print(f"gaps >= {thresh:.0f} s: {len(gaps)}")
for d, t0, l0, l1 in sorted(gaps, reverse=True)[:10]:
    u = USAGE.search(l0)
    q = QUEUE.search(l0)
    r = RUN.search(l0)
    print(f"  {d:7.0f} s starting {datetime.datetime.fromtimestamp(t0):%H:%M:%S}  "
          f"usage={u.group(1) if u else '?'} queue={q.group(1) if q else '?'} "
          f"running={r.group(1) if r else '?'}")

if p_start and p_end:
    lead = rows[0][0] - p_start
    trail = p_end - rows[-1][0]
    window = p_end - p_start
    sched = rows[-1][0] - rows[0][0]
    print(f"phase window {window:.0f} s")
    print(f"  leading  silence {lead:7.1f} s ({100 * lead / window:.1f} %)")
    print(f"  TRAILING silence {trail:7.1f} s ({100 * trail / window:.1f} %)"
          + ("   <-- WARN: deadlock signature" if trail >= 120 else ""))
    print(f"  scheduling wall clock {sched:.0f} s = {100 * sched / window:.1f} % of the phase")

print("markers: " + (", ".join(f"{k}={v}" for k, v in sorted(marker_hits.items())) or "none"))
