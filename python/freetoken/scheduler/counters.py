"""Cumulative scheduler counters, and the document they publish to ``/v1/stats``.

The soak (§U5/§U6) could see the scheduler's *rates* in the batch log but none of its
*decisions*: whether ``max_chunked_prefills`` ever bound, how the interleave share's divisor
was moving, whether the finishability invariant was ever violated on a tree that was not
running with ``FREETOKEN_SCHEDULER_INVARIANT`` set, why speculation declined, and whether a
session checkpoint failed. All of those were inferable at best, and several only from a
debug-level log line that a soak does not capture.

Everything here is a plain counter incremented on a path the scheduler already walks -- no
new work, no sampling, no allocation per pass. The counters live on the objects that make
the decisions (``PrefillManager``, ``SessionSpillStore``, ``SpecStats``); this module owns
their shape and the one function that renders them, so the wire document has a single
definition and can be unit-tested without a Scheduler (which needs a GPU to construct).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

# Lane-count buckets for the seatable-lanes divisor. Small counts are what matter -- 1 lane
# is the §R7 starvation shape and 2-4 is the healthy band the passing tree sits in -- so the
# resolution is per-lane there and geometric above it.
_LANE_BUCKETS = ((0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4"), (8, "5-8"), (16, "9-16"))
_LANE_OVERFLOW = "17+"


def lane_bucket(lanes: int) -> str:
    """Histogram key for a per-pass lane count."""
    for ceiling, name in _LANE_BUCKETS:
        if lanes <= ceiling:
            return name
    return _LANE_OVERFLOW


def _empty_lane_hist() -> Dict[str, int]:
    return {name: 0 for _ceiling, name in _LANE_BUCKETS} | {_LANE_OVERFLOW: 0}


@dataclass
class PrefillCounters:
    """What each prefill pass decided. All cumulative except the ``*_last``/``*_max`` gauges."""

    passes: int = 0
    # Fresh prompts the ``max_chunked_prefills`` cap skipped (prefill.py's ``continue``).
    # Non-zero means the belt-and-braces bound is binding and the reservation arithmetic
    # should be re-derived -- see the knob's own comment.
    fresh_admits_blocked_by_cap: int = 0
    # Lanes a pass admitted but could not finish: their remainder is deferred to a later
    # pass. One per lane per pass, so a 30-chunk prompt contributes 29.
    deferred_chunks: int = 0
    # Passes that stopped because a lane was refused (the admission loop's ``break``), i.e.
    # the queue tail went unserved for want of pool, table or budget rather than lanes.
    refusals: int = 0
    # Gauges: chunked prefills in flight at the top of the last pass, and the high-water mark.
    chunked_inflight: int = 0
    chunked_inflight_max: int = 0
    # The interleave share's divisor (``_seatable_lanes``). 0 means the share was not applied
    # this pass (interleaving off, or a single queued request).
    seatable_lanes_last: int = 0
    seatable_lanes: Dict[str, int] = field(default_factory=_empty_lane_hist)
    # Finishability invariant. ``checks`` counts every pass (the comparison is three
    # attribute reads next to the radix walk the same pass runs), so ``violations`` is
    # meaningful even with FREETOKEN_SCHEDULER_INVARIANT unset -- which is the whole point:
    # the soak that needed this number was not running with the env var on.
    invariant_checks: int = 0
    invariant_violations: int = 0
    invariant_worst_shortfall: int = 0

    def note_pass(self, *, seatable: int, chunked_inflight: int) -> None:
        self.passes += 1
        self.seatable_lanes_last = seatable
        self.seatable_lanes[lane_bucket(seatable)] += 1
        self.chunked_inflight = chunked_inflight
        self.chunked_inflight_max = max(self.chunked_inflight_max, chunked_inflight)

    def note_invariant(self, shortfall: int) -> None:
        """``shortfall = owed - budget``; <= 0 is the invariant holding."""
        self.invariant_checks += 1
        if shortfall > 0:
            self.invariant_violations += 1
            self.invariant_worst_shortfall = max(self.invariant_worst_shortfall, shortfall)

    def as_dict(self, max_chunked_prefills: int = 0) -> Dict[str, Any]:
        return {
            "passes": self.passes,
            "fresh_admits_blocked_by_cap": self.fresh_admits_blocked_by_cap,
            "deferred_chunks": self.deferred_chunks,
            "refusals": self.refusals,
            "chunked_inflight": self.chunked_inflight,
            "chunked_inflight_max": self.chunked_inflight_max,
            "max_chunked_prefills": max_chunked_prefills,
            "seatable_lanes_last": self.seatable_lanes_last,
            "seatable_lanes": dict(self.seatable_lanes),
            "invariant": {
                "checks": self.invariant_checks,
                "violations": self.invariant_violations,
                "worst_shortfall": self.invariant_worst_shortfall,
            },
        }


@dataclass
class SpillCounters:
    """Session checkpoint traffic. ``*_failed`` are the ones a soak cannot otherwise see:
    a spill that did not fit its budget is a warning, and a prefetch that never started is
    silent."""

    spills: int = 0
    spills_failed: int = 0
    restores: int = 0
    restores_failed: int = 0
    # A record whose stored state boundaries all sit after the client's divergence point:
    # nothing restorable, so the context is recomputed.
    restores_diverged: int = 0
    prefetches: int = 0
    prefetches_failed: int = 0
    prefetches_collected: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "spills": self.spills,
            "spills_failed": self.spills_failed,
            "restores": self.restores,
            "restores_failed": self.restores_failed,
            "restores_diverged": self.restores_diverged,
            "prefetches": self.prefetches,
            "prefetches_failed": self.prefetches_failed,
            "prefetches_collected": self.prefetches_collected,
        }


def build_scheduler_counters(
    prefill_manager: Any = None,
    spec: Any = None,
    spill_store: Any = None,
) -> Dict[str, Any]:
    """The ``/v1/stats["scheduler"]`` document.

    Every source is optional and duck-typed: speculation and session spill are off on most
    profiles, and the low-level loop tests drive the scheduler with stub managers. A source
    that is absent contributes ``None`` rather than a zero-filled block, so "off" and "on
    but idle" stay distinguishable on the wire -- the same ambiguity this ticket removes
    from ``cached_tokens``.
    """
    doc: Dict[str, Any] = {"prefill": None, "spec": None, "session_spill": None}
    counters = getattr(prefill_manager, "counters", None)
    if counters is not None:
        doc["prefill"] = counters.as_dict(
            getattr(prefill_manager, "max_chunked_prefills", 0)
        )
    stats = getattr(spec, "stats", None)
    if stats is not None:
        doc["spec"] = stats.as_dict()
    spill = getattr(spill_store, "counters", None)
    if spill is not None:
        doc["session_spill"] = spill.as_dict()
    return doc
