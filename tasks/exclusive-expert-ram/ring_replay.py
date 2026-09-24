"""Replay a DMA-writeback ring trace at other ring sizes.

    python ring_replay.py TRACE [rows ...]

TRACE is the file FREETOKEN_MIRROR_WB_TRACE wrote on a run whose ring never
filled (ring_full_fallbacks == 0): the ring head at every step boundary, and
"D <head>" where a drain landed everything. Head deltas are then the rows each
step wanted to stage. Under lag-1 servicing a step's resolve kernel sees the
previous step's entries still pending, so a ring of R rows stages
min(demand, R - staged_previous) and falls back to SM stores for the rest.
Prints, per R: staged %, fallbacks, and the per-step demand distribution.
REPLAY_PHI="1 0.5 0.25" also replays with the previous step freed part-way
through the current one (see replay).
"""
from __future__ import annotations

import os
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


def replay(steps, rows, phi=1.0):
    """``phi``: fraction of a step's layers that run before the previous step's
    entries are freed (1.0 = the whole step, the pessimistic bound). The resolve
    kernel runs per MoE layer and reads the completed count each time, so the
    previous step's DMAs landing early in the step free the ring for its later
    layers; a step's demand is taken as spread evenly over its layers."""
    staged = fallback = prev = 0
    for d in steps:
        if d is None:
            prev = 0
            continue
        early = d * phi
        s1 = min(early, max(rows - prev, 0))
        s2 = min(d - early, max(rows - s1, 0))
        staged += s1 + s2
        fallback += d - s1 - s2
        prev = s1 + s2
    return round(staged), round(fallback)


def main():
    steps = load(sys.argv[1])
    demand = sorted(d for d in steps if d)
    total = sum(demand)
    n = len(demand)
    print(f"{n} steps that staged ({len(steps)} boundaries), {total} rows; per staging step "
          + " ".join(f"p{q}={demand[min(n - 1, int(q / 100 * n))]}" for q in (50, 90, 95, 99))
          + f" max={demand[-1]}")
    phis = [float(x) for x in os.environ.get("REPLAY_PHI", "1.0").split()]
    sizes = [int(x) for x in sys.argv[2:]] or [8, 16, 17, 24, 32, 48, 64, 85, 96, 128, 192, 256]
    for r in sizes:
        cells = []
        for phi in phis:
            st, fb = replay(steps, r, phi)
            cells.append(f"phi {phi:.2f}: staged {100 * st / max(total, 1):5.1f}% fallbacks {fb:6d}")
        print(f"  ring {r:4d}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
