"""Named schema for the mirror swap-kernel counters.

``_resolve_swaps_kernel`` and ``_writeback_buffer_kernel`` (``mirror_kernels.py``)
share one ``int64[7]`` counter tensor, written positionally inside Triton
kernels where there is no named-tuple equivalent. Everywhere else -- the two
Python readers, ``mirror_stats()`` and ``mirror_fault_check()`` in
``offload_cache.py`` -- used to unpack that tensor positionally too, with the
layout documented only in a kernel comment. That was the direct cause of two
real defects: growing the tensor ``[6] -> [7]`` (the buffer-eviction counter)
left a second positional unpack site that only a test caught, and
``free_eviction_rate`` once exceeded 1.0 (1.121, arm ``nemotron-lever2``,
2026-09-22) because one kernel bumped the numerator slot and not the
denominator's.

``MirrorStat`` is the single source of truth for the layout. The kernels take
each member's value as a ``tl.constexpr`` offset (still numeric inside the
kernel -- Triton has no enum support -- but sourced from here, not
re-typed), and every host-side reader unpacks ``by name`` through this enum
instead of position. The numeric values themselves are UNCHANGED from the
kernel comment's original layout; this step is behaviour-identical.
"""
from __future__ import annotations

from enum import IntEnum


class MirrorStat(IntEnum):
    """Index into the mirror's shared ``stats`` / ``stats_host`` int64[7]."""

    # Bumped by _resolve_swaps_kernel (decode admissions / prefill materialize).
    SWAPS = 0
    FREE_EVICTIONS = 1
    WRITEBACKS = 2
    VIOLATIONS = 3
    STARVED = 4
    RETAINED = 5
    # Bumped by _writeback_buffer_kernel (prefill buffer invalidation) only.
    # Kept separate from FREE_EVICTIONS: those evictions are not decode
    # admissions, so folding them into slot 1 (whose denominator is SWAPS,
    # bumped only by _resolve_swaps_kernel) let free_eviction_rate exceed 1.0.
    BUFFER_FREE_EVICTIONS = 6


#: Number of counters in the shared ``stats`` tensor. Kept in step with
#: ``MirrorStat`` by construction (``len(MirrorStat)``), so allocation sites
#: never need their own literal.
MIRROR_STAT_COUNT = len(MirrorStat)


def mirror_stats_from_vector(vec) -> dict:
    """Unpack a 7-element stats vector (list, tuple or tensor) BY NAME.

    ``vec`` is whatever ``.tolist()`` on the ``stats`` / ``stats_host`` tensor
    produces: a length-7 sequence in ``MirrorStat`` order. Returns the same
    keys ``OffloadMoeCache.mirror_stats()`` has always returned, computed from
    named lookups instead of positional unpacking.
    """
    values = list(vec)
    if len(values) != MIRROR_STAT_COUNT:
        raise ValueError(
            f"mirror stats vector has {len(values)} entries, expected "
            f"{MIRROR_STAT_COUNT} ({', '.join(m.name for m in MirrorStat)})"
        )
    swaps = values[MirrorStat.SWAPS]
    free_evictions = values[MirrorStat.FREE_EVICTIONS]
    buffer_free_evictions = values[MirrorStat.BUFFER_FREE_EVICTIONS]
    return {
        "swaps": swaps,
        "free_evictions": free_evictions,
        "writebacks": values[MirrorStat.WRITEBACKS],
        "coverage_faults": values[MirrorStat.VIOLATIONS],
        "starved_writebacks": values[MirrorStat.STARVED],
        # Decode swaps only, so this cannot exceed 1 -- see BUFFER_FREE_EVICTIONS
        # above for why buffer-invalidation evictions are excluded here.
        "free_eviction_rate": (free_evictions / swaps) if swaps else 0.0,
        "buffer_free_evictions": buffer_free_evictions,
        "retained_rows": values[MirrorStat.RETAINED],
    }


def mirror_fault_counts_from_vector(vec) -> tuple[int, int]:
    """Return ``(violations, starved)`` from a 7-element stats vector BY NAME."""
    values = list(vec)
    if len(values) != MIRROR_STAT_COUNT:
        raise ValueError(
            f"mirror stats vector has {len(values)} entries, expected "
            f"{MIRROR_STAT_COUNT} ({', '.join(m.name for m in MirrorStat)})"
        )
    return values[MirrorStat.VIOLATIONS], values[MirrorStat.STARVED]
