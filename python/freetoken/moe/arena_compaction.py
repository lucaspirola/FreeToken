"""Arena compaction before a shrink: keep the hot set when the top slots are unmapped.

Shrinking the expert arena from ``current`` to ``n`` slots used to drop every
resident of ``[n, current)`` whatever its rank, so a request that moved the arena
down for its prefill came back to decode with ~100-150 of its hottest experts
missing (refill misses), and under the bounded mirror each non-duplicated one also
cost a disk re-read to keep coverage. Compaction moves the doomed residents worth
keeping into the least valuable slots below ``n`` first (device to device), using
the admission policy's own ranking, so the shrink drops the coldest experts
instead of the highest-numbered ones.

The plan is residency-generic. The only residency input is ``host_copy[id]``: the
expert also has a host copy, so dropping its GPU copy is free (no write-back):

* whole-model residency: every expert has one, so the plan is plain LFU
  compaction (the hottest doomed experts replace the coldest survivors);
* bounded mirror: a GPU-only expert must not be dropped without a write-back, so

  1. doomed duplicates (host copy present) are simply dropped unless they are
     hotter than a free target;
  2. doomed GPU-only experts move first, hottest first, into empty slots, then
     into slots whose occupant is a duplicate (overwriting those is free); a
     GPU-only occupant is never a target, since displacing it would need the
     write-back the move was meant to avoid;
  3. whatever GPU-only expert is still in ``[n, current)`` is written back by the
     residency's own ``before_shrink``.

Slot indices are data, not code: every decode kernel reads ``slot_for_id`` /
``id_of_slot`` from device memory on each launch (``_ensure_experts_sized_kernel_v2``
loads ``slot_for_id`` and rewrites the routed ids with it), so remapping an expert
to another slot needs no graph recapture.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompactionPlan:
    """``moves`` are ``(src_slot, dst_slot)`` pairs; ``src >= n > dst`` for all."""

    moves: list[tuple[int, int]] = field(default_factory=list)
    # Ids that lose their GPU copy because a move overwrote their slot (all have a
    # host copy by construction).
    displaced: list[int] = field(default_factory=list)
    doomed: int = 0              # residents of [n, current) before the plan
    dropped_with_copy: int = 0   # doomed, not moved, host copy present (free drop)
    left_uncovered: int = 0      # doomed, not moved, GPU-only (residency writes back)


def plan_compaction(
    id_of_slot: list[int],
    rank: list[tuple],
    host_copy,
    *,
    lo: int,
    n: int,
    current: int,
) -> CompactionPlan:
    """Choose the moves for shrinking ``[0, current)`` to ``[0, n)``.

    ``id_of_slot[s]`` is the expert in slot ``s`` (-1: empty); ``rank[s]`` orders
    residents by value to the policy (larger is hotter; compared only between
    occupied slots); ``host_copy(id)`` says whether the expert has a host copy.
    Targets are taken from ``[lo, n)`` only (``lo`` excludes the prefill double
    buffer, which the next prefill overwrites anyway).
    """
    plan = CompactionPlan()
    doomed = [s for s in range(n, current) if id_of_slot[s] >= 0]
    plan.doomed = len(doomed)
    if not doomed:
        return plan
    gpu_only = sorted((s for s in doomed if not host_copy(id_of_slot[s])),
                      key=lambda s: rank[s], reverse=True)
    covered = sorted((s for s in doomed if host_copy(id_of_slot[s])),
                     key=lambda s: rank[s], reverse=True)
    empty = [s for s in range(lo, n) if id_of_slot[s] < 0]
    # Occupied targets, coldest first: only experts whose GPU copy can go for free.
    free_to_overwrite = sorted(
        (s for s in range(lo, n) if id_of_slot[s] >= 0 and host_copy(id_of_slot[s])),
        key=lambda s: rank[s],
    )
    targets = empty + free_to_overwrite
    t = 0
    moved = set()
    for src in gpu_only + covered:
        if t >= len(targets):
            break
        dst = targets[t]
        occupant = id_of_slot[dst]
        # An occupied target is taken only by a strictly hotter mover. A GPU-only
        # mover colder than the coldest duplicate left is written back instead:
        # overwriting the duplicate would cost a GPU hit to save one row of PCIe.
        # The target stays available for the next (differently ordered) mover.
        if occupant >= 0 and not rank[src] > rank[dst]:
            continue
        plan.moves.append((src, dst))
        if occupant >= 0:
            plan.displaced.append(occupant)
        moved.add(src)
        t += 1
    for s in doomed:
        if s in moved:
            continue
        if host_copy(id_of_slot[s]):
            plan.dropped_with_copy += 1
        else:
            plan.left_uncovered += 1
    return plan
