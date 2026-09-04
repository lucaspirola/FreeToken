"""LinearStatePool exhaustion must degrade prefix reuse, never kill the scheduler.

Regression for the Switchyard soak crash (`RuntimeError: LinearStatePool exhausted: need 1,
have 0` at the prefill-chunk commit in ``_cache_req_hybrid``): the donation of a frozen GDN
snapshot into the radix tree is an optimization, so when the pool cannot back a replacement
ping-pong slot the commit is skipped and the request keeps its own working set.

CPU, real LinearStatePool + page_table, hand-built Reqs -- same shape as
``test_hybrid_cache_manager.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def _admit(cm, pool, page_table, table_idx, ids, pages, track_len):
    """Admit a chunked request holding live + ping-pong slots, with a pending track freeze."""
    mr = cm.match_req(_pend(ids))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[table_idx, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx,
              cached_len=track_len, output_len=1, uid=table_idx,
              sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1              # frozen = pp[0]
    req.mamba_last_track_seqlen = track_len
    cm.lock(mr.cuda_handle)
    return req


def _drain(pool):
    """Empty the free-list, as a full 16-way batch plus a lease-pinned snapshot cache does."""
    if pool.num_free_slots:
        pool.alloc(pool.num_free_slots)


def test_chunk_commit_skips_donation_when_pool_is_exhausted():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    req = _admit(cm, pool, page_table, 0, [1, 2, 3, 4, 5], [100, 101, 102, 103], 4)
    _drain(pool)
    assert pool.num_free_slots == 0
    assert cm.mamba_available_size == 0        # nothing evictable either: the crash precondition

    pp_before = req.mamba_ping_pong
    cm.cache_req(req, finished=False)          # used to raise LinearStatePool exhausted

    # Degraded, not fatal: the request keeps both ping-pong slots and its live slot, the
    # pending freeze is cleared (it is re-taken at the next boundary), and nothing was donated.
    assert req.mamba_ping_pong == pp_before
    assert req.mamba_last_track_seqlen is None
    assert pool.num_free_slots == 0
    assert cm.match_req(_pend([1, 2, 3, 4, 9])).cuda_handle.cached_len == 0
    assert cm._mamba_donation_skips == 1


def test_finish_commit_survives_an_exhausted_pool():
    """The finish path donates the live slot and frees the pair -- it must never alloc."""
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    req = _admit(cm, pool, page_table, 1, [7, 8, 9, 10, 11], [200, 201, 202, 203, 204], 4)
    _drain(pool)

    cm.cache_req(req, finished=True)
    assert pool.num_free_slots > 0             # the request's slots came back
    hit = cm.match_req(_pend([7, 8, 9, 10, 11, 12]))
    assert hit.cuda_handle.cached_len == 4 and hit.mamba_value is not None


def test_acquire_mamba_slot_prefers_eviction_over_the_reclaim_hook():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    # One unlocked donated snapshot in the tree, then an empty free-list.
    donor = _admit(cm, pool, page_table, 0, [1, 2, 3, 4, 5], [100, 101, 102, 103], 4)
    cm.cache_req(donor, finished=False)
    cm.unlock(donor.cache_handle)              # the donated node is now evictable cache
    _drain(pool)
    assert pool.num_free_slots == 0 and cm.mamba_available_size == 1

    calls = []
    cm.mamba_reclaim_hook = lambda n: calls.append(n) or False
    assert cm.acquire_mamba_slot() is not None
    assert calls == []                         # LRU eviction was enough; no lease was touched


def test_acquire_mamba_slot_falls_back_to_the_session_reclaim_hook():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    # A "lease": a donated snapshot whose node stays locked, so evict_mamba skips it.
    donor = _admit(cm, pool, page_table, 0, [1, 2, 3, 4, 5], [100, 101, 102, 103], 4)
    cm.cache_req(donor, finished=False)
    leased = donor.cache_handle                # still locked == the retain_prefix lease shape
    _drain(pool)
    assert pool.num_free_slots == 0 and cm.mamba_available_size == 0

    assert cm.acquire_mamba_slot() is None     # no hook installed: honest failure, no raise

    def release(n):
        cm.unlock(leased)
        return True

    cm.mamba_reclaim_hook = release
    slot = cm.acquire_mamba_slot()
    assert slot is not None                    # the lease released, eviction then found a slot


def test_donation_succeeds_once_the_reclaim_hook_frees_a_lease():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    donor = _admit(cm, pool, page_table, 0, [1, 2, 3, 4, 5], [100, 101, 102, 103], 4)
    cm.cache_req(donor, finished=False)
    leased = donor.cache_handle
    req = _admit(cm, pool, page_table, 1, [20, 21, 22, 23, 24], [300, 301, 302, 303], 4)
    _drain(pool)
    assert cm.mamba_available_size == 0

    cm.mamba_reclaim_hook = lambda n: (cm.unlock(leased), True)[1]
    frozen = req.mamba_ping_pong[0]
    cm.cache_req(req, finished=False)

    assert req.mamba_ping_pong[0] != frozen                       # replaced, so the tree took it
    assert cm.match_req(_pend([20, 21, 22, 23, 99])).mamba_value == frozen
    assert cm._mamba_donation_skips == 0


def test_one_running_request_spills_the_idle_lease_instead_of_dropping_the_donation():
    """The 1M profile: ``--max-running-requests 1 --linear-state-slots 5``.

    Five slots = padding + live + 2 ping-pong + exactly one idle session lease, so session B's
    first turn has nothing left when its chunk commit needs a replacement ping-pong slot. It
    crashed with the same ``LinearStatePool exhausted: need 1, have 0``. The residency policy
    says the idle conversation is the thing that gives: spill A's lease, keep B's donation.
    """
    pool = _pool(5)
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    # Session A finishes a turn and its prefix is retained as a lease (locked snapshot node).
    a = _admit(cm, pool, page_table, 0, [1, 2, 3, 4, 5], [100, 101, 102, 103], 4)
    cm.cache_req(a, finished=True)
    lease = cm.retain_prefix(a.input_ids, 4)
    assert pool.num_free_slots == 3                    # padding + A's leased snapshot are out

    # Session B is admitted (live + ping-pong) -> zero free, and the lease is not evictable.
    b = _admit(cm, pool, page_table, 1, [20, 21, 22, 23, 24], [300, 301, 302, 303], 4)
    assert pool.num_free_slots == 0 and cm.mamba_available_size == 0

    spills = []

    def _spill(n):
        spills.append(n)
        cm.unlock(lease)                              # == _release_soft_session_handle
        return True

    cm.mamba_reclaim_hook = _spill
    frozen = b.mamba_ping_pong[0]
    cm.cache_req(b, finished=False)                   # used to raise

    assert spills == [1]                              # the lease was spilled on demand
    assert b.mamba_ping_pong[0] != frozen             # and B's donation went through
    assert cm.match_req(_pend([20, 21, 22, 23, 99])).mamba_value == frozen
    assert cm._mamba_donation_skips == 0


def test_admission_reserves_state_slots_through_the_lease_spill():
    """Same five-slot pool, but the pressure lands at admission: three slots, one lease."""
    from freetoken.scheduler.prefill import PrefillAdder

    pool = _pool(5)
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    a = _admit(cm, pool, page_table, 0, [1, 2, 3, 4, 5], [100, 101, 102, 103], 4)
    cm.cache_req(a, finished=True)
    lease = cm.retain_prefix(a.input_ids, 4)
    pool.alloc(1)                                     # a live request holds one more slot
    # Two free, and the third is the lease's -- pinned, so nothing is evictable.
    assert pool.num_free_slots == 2
    assert cm.prefix_cache.mamba_evictable_size == 0

    assert cm.reserve_mamba_slots(3) is False         # no hook: honest refusal, no raise
    cm.mamba_reclaim_hook = lambda n: (cm.unlock(lease), True)[1]
    assert cm.reserve_mamba_slots(3) is True          # lease spilled -> the seat opens

    # And the adder consults exactly that path (rather than ensure_mamba_slots alone).
    import inspect

    assert "reserve_mamba_slots(3)" in inspect.getsource(PrefillAdder._try_allocate_one)


def test_scheduler_state_slot_reclaim_releases_the_lru_idle_lease():
    """The scheduler-side hook: LRU idle automatic leases only, one slot at a time."""
    from freetoken.scheduler.scheduler import Scheduler

    released = []

    class _Lease:
        def __init__(self, last_used):
            self.last_used_at = last_used
            self.reclaimable = True
            self.active_uid = None
            self.handle = object()

    sched = Scheduler.__new__(Scheduler)
    sched.cache_manager = SimpleNamespace(is_hybrid=True, mamba_available_size=0)
    sched._sessions = {"old": _Lease(1.0), "new": _Lease(2.0)}

    def _release(sid, reason):
        released.append(sid)
        sched.cache_manager.mamba_available_size = 1   # one slot is enough
        return True

    sched._release_soft_session_handle = _release
    assert sched._reclaim_soft_sessions_for_state_slot() is True
    assert released == ["old"]                         # LRU first, and it stopped at one


def test_scheduler_state_slot_reclaim_is_a_noop_without_sessions():
    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.cache_manager = SimpleNamespace(is_hybrid=True, mamba_available_size=0)
    sched._sessions = {}
    assert sched._reclaim_soft_sessions_for_state_slot() is False

    sched.cache_manager = SimpleNamespace(is_hybrid=False, mamba_available_size=0)
    assert sched._reclaim_soft_sessions_for_state_slot() is False


@pytest.mark.parametrize("num_slots", [8, 16])
def test_repeated_exhausted_commits_never_leak_or_raise(num_slots):
    """A long stall under a full pool must not drift the free-list or the request's slots."""
    pool = _pool(num_slots)
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    req = _admit(cm, pool, page_table, 0, [1, 2, 3, 4, 5], [100, 101, 102, 103], 4)
    _drain(pool)
    occupied = set(pool.occupied_slots)
    for _ in range(10):
        req.mamba_last_track_seqlen = 4
        cm.cache_req(req, finished=False)
    assert set(pool.occupied_slots) == occupied
    assert cm._mamba_donation_skips == 10
