"""Regression for the S13 pin/admission deadlock (measured 2026-09-23, s13b-session).

Under ``--pin-prefix-scope session``, a session's own pin (``CacheManager.pin_prefix`` /
``_pin_session_prefix``) locks its prefix directly (``inc_lock``) -- a stronger hold than a
soft session lease's protection. ``Scheduler._reclaim_soft_sessions_for_pending`` used to
release only the lease; a pin outliving it (or a shared-scope pin with no lease behind it
at all, as reproduced here) left the tree's evictable pool short no matter how much soft
protection was released, so a later fresh request that needed those pages was deferred
forever (``fresh_admits_deferred`` reached 106,632 in 20 minutes on the GPU with a
104,934-token pin blocking a 961,390-token admit into a 1,048,576-token pool -- see
``tasks/exclusive-expert-ram/results/s13b-session-stuck.txt``).

The fix: once the candidate soft leases are exhausted and the request is still KV-short,
``CacheManager.release_pins_for_admission`` releases pins least-recently-matched first,
only as many as it takes, counted in ``pin_admission_releases`` -- same call site, same
spirit as the soft-session release it sits beside.

These tests drive ``Scheduler._reclaim_soft_sessions_for_pending`` unbound against a
scheduler-shaped stub (``tests/scheduler/test_reclaim_and_match_memo.py``'s pattern) over a
real hybrid-radix ``CacheManager`` with a real ``LinearStatePool`` on CPU (page_size 1),
the way ``tests/kvcache/radix/test_hybrid_radix_pins.py`` builds one.
"""

from __future__ import annotations

import torch

from freetoken.core import Context, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.utils import PendingReq


def _setup_context() -> None:
    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _pool(num_slots: int = 16) -> LinearStatePool:
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _cm(pool: LinearStatePool, num_pages: int, min_tokens: int = 2,
        scope: str = "shared", max_pin_slots: int = -1) -> CacheManager:
    page_table = torch.zeros(4, num_pages, dtype=torch.int32)
    return CacheManager(num_pages, 1, page_table, "hybrid_radix", linear_state_pool=pool,
                        pin_prefix_min_tokens=min_tokens, pin_prefix_scope=scope,
                        pin_prefix_max_slots=max_pin_slots)


def _pend(uid: int, ids: list[int], max_tokens: int = 0) -> PendingReq:
    t = torch.tensor(ids, dtype=torch.int32)
    return PendingReq(uid=uid, input_ids=t,
                      sampling_params=SamplingParams(max_tokens=max_tokens))


def _insert(cm: CacheManager, pool: LinearStatePool, ids: list[int]):
    """Donate ``ids`` with a fresh snapshot, as a finished request would."""
    n = len(ids)
    pages, cm.free_slots = cm.free_slots[:n].clone(), cm.free_slots[n:]
    slot = pool.alloc(1)[0]
    matched, exists = cm.prefix_cache.insert(torch.tensor(ids, dtype=torch.int32), pages, slot)
    assert not exists
    cm.free_slots = torch.cat([cm.free_slots, pages[:matched]])
    return slot


def _insert_kv_only(cm: CacheManager, ids: list[int]):
    pages, cm.free_slots = cm.free_slots[:len(ids)].clone(), cm.free_slots[len(ids):]
    matched, exists = cm.prefix_cache.insert(torch.tensor(ids, dtype=torch.int32), pages, None)
    assert not exists
    cm.free_slots = torch.cat([cm.free_slots, pages[:matched]])


def _admit(cm: CacheManager, ids: list[int], session: str) -> None:
    """What the admission loop reports at the ``PromptAdmittedMsg`` point."""
    mr = cm.match_req(_pend(0, ids + [999]))
    cm.lock(mr.cuda_handle)
    cm.note_prompt_admitted(mr.cuda_handle, len(ids) + 1, session_key=session)
    cm.unlock(mr.cuda_handle)


class _SchedulerStub:
    """A ``Scheduler`` with only its collaborators replaced; see
    ``test_reclaim_and_match_memo.py`` for the rationale (unbound real methods, no
    prefill manager or session leases needed for this call)."""

    def __init__(self, cm: CacheManager):
        self.cache_manager = cm
        self.prefill_manager = None
        self._sessions: dict = {}

    def __getattr__(self, name: str):
        from freetoken.scheduler.scheduler import Scheduler

        return getattr(Scheduler, name).__get__(self, type(self))


def _wedged_pool(num_pages: int = 40):
    """A's 10-token prefix, pinned once two sessions match through it (shared scope, no
    session lease standing at all -- the old release has nothing to free)."""
    _setup_context()
    pool = _pool()
    cm = _cm(pool, num_pages=num_pages)
    a_ids = list(range(1, 11))
    _insert(cm, pool, a_ids)
    _admit(cm, a_ids, session="a")
    _admit(cm, a_ids, session="b")
    assert cm.prefix_counters.pinned_tokens == 10
    return cm


def test_a_pin_that_outlives_its_session_no_longer_starves_a_later_fresh_request():
    """The deadlock shape: B only fits once A's pin is released."""
    cm = _wedged_pool()
    stub = _SchedulerStub(cm)
    b = _pend(1, list(range(100, 135)))  # 35 fresh tokens, no session key
    assert b.input_len + b.output_len > cm.available_size, (
        "B must be refused today: this is the deadlock's starting condition"
    )
    assert b.input_len + b.output_len <= cm.available_size + cm.prefix_counters.pinned_tokens, (
        "and it must fit once the pin (and only the pin) is gone"
    )

    released = stub._reclaim_soft_sessions_for_pending(b, session_id=None)

    assert released is True
    assert cm.prefix_counters.pinned_tokens == 0, "the pin is gone"
    assert cm.prefix_counters.pin_admission_releases == 1, "counted, and only once"
    assert b.input_len + b.output_len <= cm.available_size, "B now fits"
    cm.check_integrity()


def test_a_request_that_fits_without_release_leaves_the_pin_alone():
    cm = _wedged_pool()
    stub = _SchedulerStub(cm)
    small = _pend(2, list(range(200, 205)))  # 5 tokens, fits the free pages alone
    assert small.input_len + small.output_len <= cm.available_size

    released = stub._reclaim_soft_sessions_for_pending(small, session_id=None)

    assert released is False
    assert cm.prefix_counters.pinned_tokens == 10, "untouched"
    assert cm.prefix_counters.pin_admission_releases == 0


def test_a_request_too_big_even_with_every_pin_released_is_unchanged():
    """If releasing every pin still cannot make it fit, today's refusal/deferral path is
    left to run: this function does not invent a new outcome for that case."""
    cm = _wedged_pool()
    stub = _SchedulerStub(cm)
    huge = _pend(3, list(range(300, 345)))  # 45 tokens > the whole 40-page pool
    assert huge.input_len > cm.num_pages * cm.page_size

    released = stub._reclaim_soft_sessions_for_pending(huge, session_id=None)

    assert released is True, "it tried: every pin it had is now gone"
    assert cm.prefix_counters.pinned_tokens == 0
    assert huge.input_len + huge.output_len > cm.available_size, (
        "still short -- the existing refusal/deferral path applies unchanged"
    )
    cm.check_integrity()


def test_mamba_lock_delta_predicts_the_state_slots_removed_by_match_lock():
    _setup_context()
    pool = _pool(8)
    cm = _cm(pool, num_pages=64)
    ids = [1, 2, 3, 4]
    _insert(cm, pool, ids)
    handle = cm.match_req(_pend(0, ids + [99])).cuda_handle
    assert handle.cached_len == len(ids)

    predicted = cm.mamba_lock_delta(handle)
    before = cm.mamba_available_size
    cm.lock(handle)
    after = cm.mamba_available_size
    cm.unlock(handle)

    assert predicted == before - after == 1
    assert cm.mamba_available_size == before


def test_reclaim_uses_post_match_gdn_budget_to_release_an_idle_lease():
    """Three slots look available before locking; the fourth slot comes from the lease."""
    from freetoken.scheduler.scheduler import SessionLease

    _setup_context()
    pool = _pool(6)
    cm = _cm(pool, num_pages=128, scope="shared", max_pin_slots=8)
    match_ids = [1, 2, 3, 4]
    _insert(cm, pool, match_ids)
    lease_ids = [20, 21, 22, 23]
    _insert(cm, pool, lease_ids)
    lease = cm.retain_prefix(torch.tensor(lease_ids, dtype=torch.int32), len(lease_ids))
    pool.alloc(1)  # The free slot plus the match snapshot leave exactly 3 available.
    assert cm.mamba_available_size == 3

    stub = _SchedulerStub(cm)
    stub._spill_soft_session = lambda *_a, **_k: True
    stub._sessions["lease"] = SessionLease(lease, 300.0, reclaimable=True)
    request = _pend(7, match_ids + [99])
    matched = cm.match_req(request).cuda_handle
    assert cm.mamba_lock_delta(matched) == 1
    assert cm.mamba_available_size - cm.mamba_lock_delta(matched) == 2

    assert stub._reclaim_soft_sessions_for_pending(request, session_id=None)

    assert stub._sessions["lease"].handle is None
    assert cm.mamba_available_size == 4
    cm.check_integrity()


def test_state_only_pressure_releases_lru_state_pins_until_three_slots_exist():
    """KV-only pins are skipped; state pins leave once their slots cover the admission."""
    _setup_context()
    pool = _pool(8)
    cm = _cm(pool, num_pages=128, scope="shared", max_pin_slots=8)
    for start in (1, 20, 40, 60):
        ids = list(range(start, start + 4))
        slot = _insert(cm, pool, ids)
        node = cm.match_req(_pend(0, ids + [999])).cuda_handle.node
        assert node.mamba_value == slot
        assert cm.pin_prefix(node)
    _insert_kv_only(cm, list(range(80, 84)))
    kv_node, _ = cm.prefix_cache._walk(torch.tensor([80, 81, 82, 83], dtype=torch.int32))
    assert cm.pin_prefix(kv_node)
    assert len(cm._pins) == 5
    cm._pins[kv_node].last_match = -1.0
    state_pins = [pin for pin in cm._pins.values() if pin.node.mamba_value is not None]
    for i, pin in enumerate(state_pins):
        pin.last_match = float(i)
    # Other live requests own the remaining slots. Only pin release can make state evictable.
    pool.alloc(pool.num_free_slots)
    assert cm.mamba_available_size == 1

    stub = _SchedulerStub(cm)
    request = _pend(7, list(range(100, 104)))
    assert request.input_len + request.output_len <= cm.available_size

    assert stub._reclaim_soft_sessions_for_pending(request, session_id=None)

    assert len(cm._pins) == 3
    assert kv_node in cm._pins
    remaining = {pin.node for pin in cm._pins.values() if pin.node.mamba_value is not None}
    assert remaining == {pin.node for pin in state_pins[-2:]}
    assert cm.mamba_available_size == 3
    assert cm.prefix_counters.pin_admission_releases == 2
    cm.check_integrity()
