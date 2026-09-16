"""Mechanism 2 of the staged-context-medium-v1 diagnosis (see
``tasks/kv-rollout/staged-context-medium-v1-diagnosis/diagnosis.md``): a soft/automatic
session's KV protection can be released by ``Scheduler._release_soft_session_handle``
whether or not a checkpoint was actually produced, unless the caller passes
``require_checkpoint=True``. Two call sites still defaulted to ``False``:

* ``_reclaim_soft_sessions_for_pending`` ("admission pressure")
* ``_reclaim_soft_sessions_for_state_slot`` ("GDN state-slot pressure"), reached from
  ``CacheManager.reserve_mamba_slots`` -> ``mamba_reclaim_hook`` whenever a GDN state slot
  is needed (prefill admission, a chunk-commit donation, a cold-session restore).

A lease released this way still has its radix node unlocked (``CacheManager.unlock`` ->
``dec_lock``), so the node becomes eligible for ``HybridRadixCache.evict_mamba`` even
though nothing durable exists to reconstruct the conversation from. This is the
correctness half of the staged-128K crash; the diagnosis explicitly could not attribute
the CUDA illegal-address fault to it, only rule it in as still open.

Everything here runs on the real ``CacheManager`` / hybrid radix tree / ``LinearStatePool``
on CPU, exercised through a scheduler-shaped stub in the manner of
``test_reclaim_and_match_memo.py``'s ``_SchedulerStub``: only the collaborators the
release path does not own (the lease map, the checkpoint call) are substituted.
"""

from __future__ import annotations

import torch

from .test_session_spill import _pools


class _SchedulerStub:
    """A ``Scheduler`` with only its collaborators replaced.

    ``__getattr__`` binds every other attribute straight off the real ``Scheduler``, so
    ``_release_soft_session_handle``, ``_reclaim_soft_sessions_for_pending`` and
    ``_reclaim_soft_sessions_for_state_slot`` are the genuine methods under test, not a
    reimplementation of them.
    """

    def __init__(self, cache_manager, *, checkpoint_succeeds: bool):
        self.cache_manager = cache_manager
        self._sessions: dict[str, object] = {}
        self._checkpoint_succeeds = checkpoint_succeeds
        self.checkpoint_attempts: list[str] = []

    def __getattr__(self, name: str):
        from freetoken.scheduler.scheduler import Scheduler

        return getattr(Scheduler, name).__get__(self, type(self))

    def _spill_soft_session(self, session_id, session) -> bool:
        # The checkpoint store is not under test; only whether a FAILED checkpoint can
        # still let the caller through.
        self.checkpoint_attempts.append(session_id)
        return self._checkpoint_succeeds


def _make_lease(handle, last_used_at: float):
    from freetoken.scheduler.scheduler import SessionLease

    return SessionLease(
        handle=handle,
        ttl_seconds=300.0,
        reclaimable=True,
        active_uid=None,
        last_used_at=last_used_at,
    )


def _two_locked_hybrid_sessions():
    """Two resident automatic sessions, A and B, each with a locked GDN snapshot node."""
    kv, linear, manager = _pools()
    tokens_a = torch.tensor([1, 2, 3], dtype=torch.int32)
    tokens_b = torch.tensor([11, 12, 13], dtype=torch.int32)
    pages_a = manager._page_to_token(manager._allocate(len(tokens_a)))
    pages_b = manager._page_to_token(manager._allocate(len(tokens_b)))
    slot_a, slot_b = linear.alloc(2)
    manager.prefix_cache.insert(tokens_a, pages_a, slot_a)
    manager.prefix_cache.insert(tokens_b, pages_b, slot_b)
    handle_a = manager.retain_prefix(tokens_a, len(tokens_a))
    handle_b = manager.retain_prefix(tokens_b, len(tokens_b))

    # Exercise the exact hole from the diagnosis: compaction runs with only A represented
    # (a live ``reqs`` entry would carry A's own handle; B is idle, so a caller that forgot
    # to pass it as a retained handle -- the state ``_elastic_retained_session_handles``
    # closed at the scheduler layer -- leaves B's cache_handle.kv_indices alias out of the
    # remap). The tree's own node.value is unconditionally walked and stays correct; the
    # separate ``kv_indices`` tensor cached on the lease's handle is the one at risk.
    target = manager.compact_active_pages([], manager.committed_pages, kv.copy_pages,
                                           retained_handles=[handle_a])
    assert target <= manager.committed_pages

    # Drain every remaining GDN state slot so the pool is genuinely under pressure: nothing
    # free, nothing evictable (A and B are both locked).
    if linear.num_free_slots:
        linear.alloc(linear.num_free_slots)
    assert manager.mamba_available_size == 0

    return manager, handle_a, handle_b


def test_gdn_state_slot_pressure_refuses_release_without_a_checkpoint():
    """The still-open half of mechanism 2: ``_reclaim_soft_sessions_for_state_slot`` must
    not unlock an idle session's GDN snapshot when its checkpoint failed.

    On unmodified ``main`` this releases B (or A) anyway -- ``_release_soft_session_handle``
    is called with the GDN-pressure site's default ``require_checkpoint=False`` -- which
    hands a locked node back to ``evict_mamba`` with nothing durable behind it.
    """
    manager, handle_a, handle_b = _two_locked_hybrid_sessions()
    stub = _SchedulerStub(manager, checkpoint_succeeds=False)
    stub._sessions["A"] = _make_lease(handle_a, last_used_at=1.0)
    stub._sessions["B"] = _make_lease(handle_b, last_used_at=2.0)

    released = stub._reclaim_soft_sessions_for_state_slot(1)

    assert stub.checkpoint_attempts, "the release path must have tried to checkpoint someone"
    assert released is False, (
        "a session was released for GDN-state-slot pressure despite every checkpoint "
        "attempt failing -- its radix node is now unlocked with nothing durable behind it"
    )
    assert stub._sessions["A"].handle is handle_a
    assert stub._sessions["B"].handle is handle_b


def test_admission_pressure_refuses_release_without_a_checkpoint():
    """Same hole, the other call site: ``_reclaim_soft_sessions_for_pending``'s
    "admission pressure" release also defaults ``require_checkpoint=False``."""
    from freetoken.scheduler.utils import PendingReq
    from freetoken.core import SamplingParams

    manager, handle_a, handle_b = _two_locked_hybrid_sessions()
    stub = _SchedulerStub(manager, checkpoint_succeeds=False)
    stub._sessions["A"] = _make_lease(handle_a, last_used_at=1.0)
    stub._sessions["B"] = _make_lease(handle_b, last_used_at=2.0)

    # A queued request big enough that admission is short even after the lock-delta
    # correction (_reclaim_and_match_memo's ``pressured()`` closure), so the reclaim must
    # actually try to release someone.
    huge_ids = torch.arange(200_000, 200_000 + manager.num_pages * 4, dtype=torch.int32)
    pending = PendingReq(uid=999, input_ids=huge_ids, sampling_params=SamplingParams(max_tokens=4))

    released = stub._reclaim_soft_sessions_for_pending(pending, session_id=None)

    assert stub.checkpoint_attempts, "the release path must have tried to checkpoint someone"
    assert released is False, (
        "a session was released for admission pressure despite every checkpoint attempt "
        "failing -- its radix node is now unlocked with nothing durable behind it"
    )
    assert stub._sessions["A"].handle is handle_a
    assert stub._sessions["B"].handle is handle_b


def test_a_successful_checkpoint_still_releases_under_gdn_pressure():
    """The guard must not become a hard lock: a REAL checkpoint still frees the slot."""
    manager, handle_a, handle_b = _two_locked_hybrid_sessions()
    stub = _SchedulerStub(manager, checkpoint_succeeds=True)
    stub._sessions["A"] = _make_lease(handle_a, last_used_at=1.0)
    stub._sessions["B"] = _make_lease(handle_b, last_used_at=2.0)

    released = stub._reclaim_soft_sessions_for_state_slot(1)

    assert released is True
    still_locked = [sid for sid, lease in stub._sessions.items() if lease.handle is not None]
    assert len(still_locked) == 1, "exactly one lease should have been released"


def test_refused_gdn_reclaim_degrades_reserve_mamba_slots_to_a_false_return():
    """The exact code path a refusal must reach: ``CacheManager.reserve_mamba_slots`` ->
    ``mamba_reclaim_hook`` -> refusal -> ``reserve_mamba_slots`` returns False ->
    ``PrefillAdder._try_allocate_one`` (``python/freetoken/scheduler/prefill.py``, the
    ``if not self.cache_manager.reserve_mamba_slots(3): return self.cache_manager.unlock(handle)``
    line) unlocks its own speculative match and returns ``None`` -- a deferred admission,
    not a deadlock or an exception. This test verifies the hook side of that chain; the
    admission side is exercised directly by reading the cited line.
    """
    manager, handle_a, handle_b = _two_locked_hybrid_sessions()
    stub = _SchedulerStub(manager, checkpoint_succeeds=False)
    stub._sessions["A"] = _make_lease(handle_a, last_used_at=1.0)
    stub._sessions["B"] = _make_lease(handle_b, last_used_at=2.0)
    manager.mamba_reclaim_hook = stub._reclaim_soft_sessions_for_state_slot

    assert manager.acquire_mamba_slot() is None, (
        "with every checkpoint refused, the reclaim hook must fail closed and "
        "acquire_mamba_slot must return None (never raise, never spin)"
    )
    assert stub._sessions["A"].handle is handle_a
    assert stub._sessions["B"].handle is handle_b
