#!/usr/bin/env python3
"""Summarize cgroup-sampler.sh rows per arm, in GiB.

  summarize_cgroup.py <samples.tsv> [arm ...]

For each arm it prints one row at ready (the first sample after "API server is ready"),
one after each finished probe request (the first sample where probes_done reached k), and
the highest cg_current seen. Columns: cgroup memory.current / anon / file / unevictable,
and the server process's VmRSS / RssAnon / RssFile.
"""
import csv
import sys

G = 1 << 30
COLS = ("cg_current", "cg_anon", "cg_file", "cg_unevict", "vm_rss", "rss_anon", "rss_file")


def g(v):
    try:
        return f"{int(v) / G:6.2f}"
    except (TypeError, ValueError):
        return "     -"


def main():
    rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    arms = sys.argv[2:] or list(dict.fromkeys(r["unit"] for r in rows))
    print("arm                 point        " + " ".join(f"{c:>10}" for c in COLS))
    for arm in arms:
        rs = [r for r in rows if r["unit"] == arm]
        if not rs:
            continue
        marks, seen = [], set()
        for r in rs:
            if r["ready"] == "1" and "ready" not in seen:
                seen.add("ready"); marks.append(("at ready", r))
            k = int(r["probes_done"] or 0)
            if k and f"p{k}" not in seen and r["ready"] == "1":
                seen.add(f"p{k}"); marks.append((f"after req {k}", r))
        top = max(rs, key=lambda r: int(r["cg_current"] or 0))
        marks.append(("max current", top))
        for name, r in marks:
            print(f"{arm:19} {name:12} " + " ".join(f"{g(r[c]):>10}" for c in COLS))


if __name__ == "__main__":
    main()
