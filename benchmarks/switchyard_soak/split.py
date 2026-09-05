#!/usr/bin/env python3
"""Split a soak run's server.log into per-phase logs using the driver's phase timestamps.

Usage: split.py <run-dir>     # reads <run-dir>/{driver.log,server.log}
                              # writes <run-dir>/phase_{stage,pass}.log
"""
import datetime
import re
import sys

SP = sys.argv[1].rstrip("/")
marks = {}
for line in open(f"{SP}/driver.log", errors="replace"):
    m = re.match(r"(STAGE|PASS)_(START|END)_TS=(\d+)", line.strip())
    if m:
        marks[f"{m.group(1)}_{m.group(2)}"] = int(m.group(3))
print(marks)

ts_re = re.compile(r"\[(\d{4}-\d{2}-\d{2})\|(\d{2}:\d{2}:\d{2})\|")
out = {k: open(f"{SP}/phase_{k}.log", "w") for k in ("stage", "pass")}
cur = None
for line in open(f"{SP}/server.log", errors="replace"):
    m = ts_re.search(line)
    if m:
        t = datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
        e = t.timestamp()
        if marks.get("STAGE_START", 0) <= e <= marks.get("STAGE_END", -1):
            cur = "stage"
        elif marks.get("PASS_START", 0) <= e <= marks.get("PASS_END", -1):
            cur = "pass"
        else:
            cur = None
    if cur:
        out[cur].write(line)
for f in out.values():
    f.close()
print("wrote", ", ".join(f"{SP}/phase_{k}.log" for k in out))
