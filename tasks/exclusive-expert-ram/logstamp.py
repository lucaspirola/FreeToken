#!/usr/bin/env python3
"""Follow a server log and prefix every line with seconds since this watcher started.

    logstamp.py LOG OUT      (runs until killed; waits for LOG to appear)

The server's own timestamps are whole seconds and tqdm progress bars rewrite one
line with carriage returns, so neither says when a startup phase ended. Lines are
split on \\r as well as \\n and stamped on time.monotonic() as they are read (0.2 s
poll), which is what the startup-phase table is built from.
"""
import os
import re
import sys
import time

log, out = sys.argv[1], sys.argv[2]
t0 = time.monotonic()
wall0 = time.time()
while not os.path.exists(log):
    time.sleep(0.2)
ansi = re.compile(rb"\x1b\[[0-9;]*m")
with open(log, "rb") as f, open(out, "w", buffering=1) as o:
    o.write(f"# t0 wall={wall0:.3f}\n")
    buf = b""
    while True:
        chunk = f.read()
        if not chunk:
            time.sleep(0.2)
            continue
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        now = time.monotonic() - t0
        for p in parts:
            p = ansi.sub(b"", p).strip()
            if p:
                o.write(f"{now:9.2f} {p.decode(errors='replace')}\n")
