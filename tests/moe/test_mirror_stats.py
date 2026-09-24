"""CPU tests for the named mirror-counter schema (``moe/mirror_stats.py``).

Feeds a fake 10-vector straight to the unpacking helpers -- no device, no
``OffloadMoeCache`` -- so this runs in the CPU suite alongside the rest of
``tests/moe``. Guards the two defects a positional unpack caused: growing the
vector without updating every unpack site, and folding buffer-invalidation
evictions into the same counter as decode free-evictions (which pushed
``free_eviction_rate`` above 1.0).
"""
import pytest

from freetoken.moe.mirror_stats import (
    MIRROR_STAT_COUNT,
    MirrorStat,
    mirror_fault_counts_from_vector,
    mirror_stats_from_vector,
)


def test_mirror_stat_has_ten_named_slots():
    assert MIRROR_STAT_COUNT == 10
    assert len(MirrorStat) == 10
    assert [m.value for m in MirrorStat] == list(range(10))


def test_mirror_stats_from_vector_unpacks_by_name():
    # swaps, free_evict, d2h, violations, starved, retained, buffer_free_evict,
    # staged_writebacks, stage_redirects
    vec = [100, 40, 12, 0, 0, 55, 9, 10, 4, 2]
    stats = mirror_stats_from_vector(vec)
    assert stats["swaps"] == 100
    assert stats["free_evictions"] == 40
    assert stats["writebacks"] == 12
    assert stats["coverage_faults"] == 0
    assert stats["starved_writebacks"] == 0
    assert stats["retained_rows"] == 55
    assert stats["buffer_free_evictions"] == 9
    assert stats["staged_writebacks"] == 10
    assert stats["stage_redirects"] == 4
    assert stats["ring_full_fallbacks"] == 2


def test_free_eviction_rate_uses_only_the_decode_free_evictions():
    """free_evictions / swaps -- buffer_free_evictions must NOT be folded in.

    Regression for the defect that once produced free_eviction_rate == 1.121
    (arm nemotron-lever2, 2026-09-22): a large buffer_free_evictions count
    must not push the rate above what swaps alone justify.
    """
    swaps = 100
    free_evictions = 40
    buffer_free_evictions = 9_000  # deliberately huge; must be irrelevant here
    vec = [swaps, free_evictions, 0, 0, 0, 0, buffer_free_evictions, 0, 0, 0]
    stats = mirror_stats_from_vector(vec)
    assert stats["free_eviction_rate"] == pytest.approx(free_evictions / swaps)
    assert stats["free_eviction_rate"] <= 1.0
    assert stats["buffer_free_evictions"] == buffer_free_evictions


def test_free_eviction_rate_zero_swaps_is_zero_not_a_zero_division():
    vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stats = mirror_stats_from_vector(vec)
    assert stats["free_eviction_rate"] == 0.0


def test_buffer_free_evictions_reported_separately_from_free_evictions():
    vec = [10, 3, 0, 0, 0, 0, 7, 0, 0, 0]
    stats = mirror_stats_from_vector(vec)
    assert stats["buffer_free_evictions"] != stats["free_evictions"]
    assert stats["buffer_free_evictions"] == 7
    assert stats["free_evictions"] == 3


def test_mirror_stats_from_vector_accepts_a_tensor_like_tolist_result():
    # Mirrors what OffloadMoeCache.mirror_stats() actually passes: the result
    # of `.tolist()` on an int64 tensor, i.e. a plain list of Python ints.
    vec = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    stats = mirror_stats_from_vector(vec)
    assert isinstance(stats["swaps"], int)


def test_mirror_stats_from_vector_rejects_wrong_length():
    with pytest.raises(ValueError):
        mirror_stats_from_vector([1, 2, 3])
    with pytest.raises(ValueError):
        mirror_stats_from_vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


def test_mirror_fault_counts_from_vector_unpacks_violations_and_starved():
    vec = [100, 40, 12, 3, 5, 55, 9, 0, 0, 0]
    violations, starved = mirror_fault_counts_from_vector(vec)
    assert violations == 3
    assert starved == 5


def test_mirror_fault_counts_from_vector_rejects_wrong_length():
    with pytest.raises(ValueError):
        mirror_fault_counts_from_vector([1, 2, 3])
