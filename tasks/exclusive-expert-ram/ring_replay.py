"""Replay a DMA-writeback ring trace at other ring sizes.

    python ring_replay.py TRACE [rows ...]

TRACE is the file FREETOKEN_MIRROR_WB_TRACE wrote on a run whose ring never
filled (ring_full_fallbacks == 0): the ring head at every step boundary, and
"D <head>" where a drain landed everything. Head deltas are then the rows each
step wanted to stage. Under lag-1 servicing a step's resolve kernel sees the
previous step's entries still pending, so a ring of R rows stages
min(demand, R - staged_previous) and falls back to SM stores for the rest.
Prints, per R: staged %, fallbacks, and the per-step demand distribution.
"""
from __future__ import annotations

import sys


def load(path):
    steps, prev = [], None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        drain = line.startswith("D")
        head = int(line.split()[-1])
        if prev is not None:
            steps.append(head - prev)
        if drain:
            steps.append(None)       # everything pending has landed
        prev = head
    return steps


def replay(steps, rows):
    staged = fallback = prev = 0
    for d in steps:
        if d is None:
            prev = 0
            continue
        s = min(d, max(rows - prev, 0))
        staged += s
        fallback += d - s
        prev = s
    return staged, fallback


def main():
    steps = load(sys.argv[1])
    demand = sorted(d for d in steps if d)
    total = sum(demand)
    n = len(demand)
    print(f"{n} steps that staged ({len(steps)} boundaries), {total} rows; per staging step "
          + " ".join(f"p{q}={demand[min(n - 1, int(q / 100 * n))]}" for q in (50, 90, 95, 99))
          + f" max={demand[-1]}")
    sizes = [int(x) for x in sys.argv[2:]] or [8, 16, 17, 24, 32, 48, 64, 85, 96, 128, 192, 256]
    for r in sizes:
        st, fb = replay(steps, r)
        print(f"  ring {r:4d}: staged {100 * st / max(total, 1):5.1f}%  fallbacks {fb}")


if __name__ == "__main__":
    main()
