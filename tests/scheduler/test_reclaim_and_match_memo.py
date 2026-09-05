"""The two §Y leftovers: a reclaim that measures the budget admission will actually see,
and a per-pass memo for the radix walk that measurement costs.

Soak §Y5b ended as a queue-discipline bug (fixed in ``125da19``), but two of its supporting
facts were left standing:

1. **Nothing was ever reclaimed.** The 576 s window contains **zero**
   ``Released soft session ... KV protection (admission pressure)`` lines, even though idle
   leases were resident and admission was failing on every pass.
   ``_reclaim_soft_sessions_for_pending`` compares what a request needs against
   ``cache_manager.available_size`` -- the budget BEFORE ``PrefillAdder._try_allocate_one``
   locks the request's own matched prefix out of the evictable pool. A turn that reuses a
   large evictable prefix therefore looks comfortable to the reclaim at the exact moment the
   gate is refusing it. ``CacheManager.lock_delta`` is the missing term.
   The same function only ever looked at ``pending_list[0]``; since ``125da19`` a pass admits
   PAST a prompt it cannot seat, so the request a release would unblock is often behind the
   head.

2. **Every refused pass re-walked the tree.** ``_seatable_lanes``, the admission loop and the
   reclaim each call ``CacheManager.match_req`` on the same prompts inside one scheduler
   iteration, and each call is O(prompt) -- 118 K-token prompts, sixteen deep. The memo makes
   it one walk per prompt per pass, and is *strict*: it is dropped by anything that can move
   the tree, so no gate ever reads a ``cached_len`` the allocation would not re-derive. (The
   cross-pass version is the one ``ea7ed7c`` shipped and the one tasks/lessons.md warns
   about; it is not this.)

Everything here runs on the real ``CacheManager`` / ``PrefillManager`` on CPU. The scheduler
methods are exercised unbound against a scheduler-shaped stub, the way
``test_admission_livelock`` exercises ``_only_idle_sessions``.
"""

from __future__ import annotations

import torch

WIDTH = 4_096
MAX_RUNNING = 8


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build(num_pages: int):
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    cm = CacheManager(num_pages=num_pages, page_size=1, page_table=pt, type="radix")
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    pm.interleave_chunks = True  # the soaked profile's setting
    return cm, tm, dm, pm


def _pending(uid: int, ids: torch.Tensor, max_tokens: int):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(
        uid=uid, input_ids=ids, sampling_params=SamplingParams(max_tokens=max_tokens)
    )


def _ids(first: int, length: int) -> torch.Tensor:
    return torch.arange(first, first + length, dtype=torch.int32)


def _drain(cm, tm, dm, pm, budget: int = 4_096, limit: int = 2_000) -> set[int]:
    served: set[int] = set()
    for _ in range(limit):
        batch = pm.schedule_next_batch(budget)
        if batch is not None:
            served.update(req.uid for req in batch.reqs)
            cm.allocate_paged(batch.reqs)
            for req in batch.reqs:
                req.complete_one()
            dm.filter_reqs(batch.reqs)
            continue
        batch = dm.schedule_next_batch()
        if batch is None:
            return served
        cm.allocate_paged(batch.reqs)
        for req in batch.reqs:
            req.append_host(torch.tensor([7], dtype=torch.int32))
            req.complete_one()
        for req in [r for r in batch.reqs if r.remain_len <= 0]:
            dm.remove_req(req)
            cm.cache_req(req, finished=True)
            tm.free(req.table_idx)
    raise AssertionError("the drain did not converge")


class _Lease:
    """``Scheduler.SessionLease`` cut to what the reclaim reads."""

    def __init__(self, handle, last_used_at: float):
        self.handle = handle
        self.last_used_at = last_used_at
        self.reclaimable = True
        self.active_uid = None
        self.protected_until = None
        self.expires_at = None
        self.ttl_seconds = 300.0


class _SchedulerStub:
    """A ``Scheduler`` with only its collaborators replaced.

    Every method the code under test reaches is the REAL one -- ``__getattr__`` binds it
    straight off ``Scheduler`` -- so these tests cannot pass by re-implementing the behaviour
    they claim to check, and a new private helper on the reclaim path (there is one already:
    ``_invalidate_match_memo``) does not need a new line here. Substituted, and only these:
    the two managers, the lease map, the disk checkpoint and the cold-restore, none of which
    is what the reclaim is being tested for.
    """

    def __init__(self, cm, pm):
        self.cache_manager = cm
        self.prefill_manager = pm
        self._sessions: dict[str, _Lease] = {}
        self.released: list[str] = []
        self.restored: list[str] = []
        self.prefetched = 0
        self._growable_shrink_pending = False

    def __getattr__(self, name: str):
        # Only reached when the attribute is NOT one of the substitutes above. An unbound
        # Scheduler method, bound to this stub; a genuine typo still raises AttributeError.
        from freetoken.scheduler.scheduler import Scheduler

        return getattr(Scheduler, name).__get__(self, type(self))

    def _spill_soft_session(self, session_id, session) -> None:
        self.released.append(session_id)  # the checkpoint itself is not under test

    def _prefetch_queued_session(self):
        self.prefetched += 1
        return None

    def _restore_cold_session(self, session_id, input_ids) -> bool:
        self.restored.append(session_id)
        return False


def _wedge_with_evictable_prefix():
    """The §Y5b state: a big EVICTABLE prefix, an idle lease, and a turn that reuses it.

    Returns the pool plus the head request. Admission will lock the head's own 700-token
    prefix out of ``available_size`` and then refuse it against what is left -- which is the
    whole point: pre-lock the pool looks roomy, post-lock it does not.
    """
    cm, tm, dm, pm = _build(num_pages=1_024)
    convo = _ids(0, 700)
    pm.pending_list = [_pending(1, convo, 4)]
    _drain(cm, tm, dm, pm)  # conversation A completes -> 700 tokens of evictable prefix

    lease_ids = _ids(500_000, 200)
    pm.pending_list = [_pending(2, lease_ids, 4)]
    _drain(cm, tm, dm, pm)
    # A second, resident conversation. Retained exactly once: locking the same prefix twice
    # would leave a reference standing after the release and free nothing.
    lease_handle = cm.retain_prefix(lease_ids, 200)

    head = _pending(10, torch.cat([convo, _ids(900_000, 5)]), 250)
    return cm, tm, dm, pm, head, lease_handle


# --------------------------------------------------------------------------- #
# (1) the pressure test uses the budget admission will actually see
# --------------------------------------------------------------------------- #
def test_lock_delta_is_exactly_what_locking_costs_available_size():
    """The primitive, checked against the mutation it predicts."""
    cm, tm, dm, pm = _build(num_pages=1_024)
    convo = _ids(0, 700)
    pm.pending_list = [_pending(1, convo, 4)]
    _drain(cm, tm, dm, pm)

    probe = _pending(10, torch.cat([convo, _ids(900_000, 5)]), 8)
    handle = cm.match_req(probe).cuda_handle
    assert handle.cached_len > 0, "the probe must actually hit the cached prefix"

    predicted = cm.lock_delta(handle)
    before = cm.available_size
    cm.lock(handle)
    after = cm.available_size
    cm.unlock(handle)

    assert predicted == before - after
    assert cm.available_size == before, "the prediction must not have moved anything"
    assert predicted > 0, "the probe's prefix was evictable, so locking it must cost"


def test_lock_delta_is_zero_for_an_already_locked_prefix():
    """A lease's prefix is protected, not evictable: locking it again costs nothing."""
    cm, tm, dm, pm = _build(num_pages=1_024)
    lease_ids = _ids(0, 300)
    pm.pending_list = [_pending(1, lease_ids, 4)]
    _drain(cm, tm, dm, pm)
    cm.retain_prefix(lease_ids, 300)

    probe = _pending(10, torch.cat([lease_ids, _ids(900_000, 5)]), 8)
    handle = cm.match_req(probe).cuda_handle
    assert cm.lock_delta(handle) == 0


def test_the_reclaim_releases_a_lease_the_pre_lock_budget_said_was_not_needed():
    """§Y5b's silence: zero release lines while admission failed on every pass."""
    cm, tm, dm, pm, head, lease_handle = _wedge_with_evictable_prefix()
    stub = _SchedulerStub(cm, pm)
    stub._sessions["idle"] = _Lease(lease_handle, 1.0)

    handle = cm.match_req(head).cuda_handle
    needed = head.input_len - handle.cached_len + head.output_len
    lock_delta = cm.lock_delta(handle)
    # The exact mismatch: comfortable before the lock, short after it.
    assert needed <= cm.available_size, "the OLD test would have seen no pressure"
    assert needed > cm.available_size - lock_delta, "the gate will refuse it"

    before = cm.available_size
    assert stub._reclaim_soft_sessions_for_pending(head, session_id=None) is True
    assert stub.released == ["idle"]
    assert stub._sessions["idle"].handle is None
    assert cm.available_size > before, "the released prefix is evictable again"


def test_the_reclaim_still_leaves_idle_leases_alone_when_the_request_fits():
    """The other direction: a stricter test must not turn into a spill storm."""
    cm, tm, dm, pm, _head, lease_handle = _wedge_with_evictable_prefix()
    stub = _SchedulerStub(cm, pm)
    stub._sessions["idle"] = _Lease(lease_handle, 1.0)

    small = _pending(30, _ids(700_000, 20), 4)
    assert small.input_len + small.output_len < cm.available_size

    assert stub._reclaim_soft_sessions_for_pending(small, session_id=None) is False
    assert stub.released == []


def test_a_continuation_is_charged_no_lock_delta():
    """A chunked lane locked its prefix in the pass that admitted it; do not charge twice
    (and do not pay for a match it already knows the answer to)."""
    cm, tm, dm, pm, _head, lease_handle = _wedge_with_evictable_prefix()
    stub = _SchedulerStub(cm, pm)
    stub._sessions["idle"] = _Lease(lease_handle, 1.0)

    calls: list[int] = []
    real_match = cm.match_req
    cm.match_req = lambda req: (calls.append(req.uid), real_match(req))[1]

    lane = _pending(40, _ids(800_000, 40), 4)
    stub._reclaim_soft_sessions_for_pending(lane, session_id=None, cached_len=8)
    assert calls == [], "a continuation must not be re-matched"


def test_the_blocked_prefill_reclaim_looks_past_the_head():
    """Since 125da19 the pass admits past a prompt it cannot seat, so the request a release
    unblocks is often not ``pending_list[0]``."""
    from freetoken.scheduler.scheduler import _RECLAIM_SCAN_DEPTH

    cm, tm, dm, pm, head, lease_handle = _wedge_with_evictable_prefix()
    stub = _SchedulerStub(cm, pm)
    stub._sessions["idle"] = _Lease(lease_handle, 1.0)

    # A tiny prompt at the head that needs nothing; the wedged turn sits behind it.
    tiny = _pending(20, _ids(700_000, 8), 2)
    pm.pending_list = [tiny, head]
    assert _RECLAIM_SCAN_DEPTH >= 2

    assert stub._reclaim_for_blocked_prefill() is True
    assert stub.released == ["idle"], (
        "the head fits, so the scan had to reach the request that does not"
    )
    assert stub.prefetched == 1


def test_the_blocked_prefill_reclaim_is_bounded():
    """The scan costs a radix walk per fresh entry; it must not walk an arbitrary queue."""
    from freetoken.scheduler.scheduler import _RECLAIM_SCAN_DEPTH

    cm, tm, dm, pm = _build(num_pages=4_096)
    stub = _SchedulerStub(cm, pm)
    stub._sessions["idle"] = _Lease(cm.retain_prefix(_ids(0, 1), 1), 1.0)
    pm.pending_list = [
        _pending(uid, _ids(uid * 10_000, 16), 4) for uid in range(1, _RECLAIM_SCAN_DEPTH + 6)
    ]
    seen: list[int] = []
    real_match = cm.match_req
    cm.match_req = lambda req: (seen.append(req.uid), real_match(req))[1]

    assert stub._reclaim_for_blocked_prefill() is False
    assert len(seen) <= _RECLAIM_SCAN_DEPTH


# --------------------------------------------------------------------------- #
# (2) the per-pass match memo
# --------------------------------------------------------------------------- #
def _count_matches(cm) -> list[int]:
    seen: list[int] = []
    real_match = cm.match_req
    cm.match_req = lambda req: (seen.append(req.uid), real_match(req))[1]
    return seen


def test_a_pass_walks_each_queued_prompt_at_most_once():
    """``_seatable_lanes`` and the admission loop ask the same question about the same
    prompts; before the memo the overlap was a second O(prompt) walk each."""
    cm, tm, dm, pm = _build(num_pages=4_096)
    pm.pending_list = [_pending(uid, _ids(uid * 10_000, 64), 8) for uid in range(1, 6)]
    seen = _count_matches(cm)

    pm.schedule_next_batch(256)

    assert len(seen) == len(set(seen)), f"a prompt was matched twice in one pass: {seen}"
    assert pm.counters.match_memo_hits > 0, "the scan/loop overlap must be a memo hit"
    assert pm.counters.match_calls == len(seen)
    assert pm.counters.match_tokens == 64 * len(seen)


def test_the_memo_does_not_survive_the_pass():
    """Strictly per-pass: the previous forward allocated, evicted and committed prefixes."""
    cm, tm, dm, pm = _build(num_pages=4_096)
    pm.pending_list = [_pending(uid, _ids(uid * 10_000, 64), 8) for uid in range(1, 4)]
    pm.schedule_next_batch(256)
    assert pm.match_memo, "the pass populated it"

    pm.pending_list = [_pending(uid, _ids(uid * 10_000, 64), 8) for uid in range(1, 4)]
    seen = _count_matches(cm)
    pm.schedule_next_batch(256)
    assert seen, "the next pass must re-walk, not reuse the previous pass's answers"


def test_an_admit_alone_does_not_drop_the_memo():
    """Locking a prefix and taking slots cannot falsify another prompt's match: no node
    leaves the tree. Invalidating there would throw the seat scan's walks away on every
    healthy pass, which is most of what the memo is for."""
    cm, tm, dm, pm = _build(num_pages=4_096)
    pm.pending_list = [_pending(uid, _ids(uid * 10_000, 64), 8) for uid in range(1, 6)]
    seen = _count_matches(cm)

    batch = pm.schedule_next_batch(4_096)
    assert batch is not None and len(batch.reqs) >= 2, "several lanes must be admitted"
    assert len(seen) == len(set(seen)), (
        f"an admit invalidated the memo and the tail was re-walked: {seen}"
    )


def test_the_memo_is_dropped_before_a_state_slot_reserve_can_evict():
    """The one in-pass mutation that deletes radix nodes. Checked at the adder, because
    reaching ``evict_mamba`` needs a hybrid pool this CPU fixture does not build."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, tm, dm, pm = _build(num_pages=4_096)
    adder = PrefillAdder(
        token_budget=256,
        reserved_size=0,
        cache_manager=cm,
        table_manager=tm,
        match_memo=pm.match_memo,
        counters=pm.counters,
    )
    req = _pending(1, _ids(0, 64), 8)
    adder._match(req)
    assert pm.match_memo, "the walk was memoized"

    adder._invalidate_match_memo()
    assert pm.match_memo == {}, "the guard must drop every entry, not just this one"

    # And it is the manager's dict that was cleared, not a private copy.
    assert adder.match_memo is pm.match_memo


def test_the_memo_is_dropped_when_a_lease_is_released():
    """A release hands a whole conversation's prefix back to the evictable pool."""
    cm, tm, dm, pm, head, lease_handle = _wedge_with_evictable_prefix()
    stub = _SchedulerStub(cm, pm)
    stub._sessions["idle"] = _Lease(lease_handle, 1.0)
    pm.match_memo[head.uid] = cm.match_req(head)

    assert stub._reclaim_soft_sessions_for_pending(head, session_id=None) is True
    assert pm.match_memo == {}


def test_the_memo_changes_no_admission_decision():
    """The safety property. Same queue, same pool, memo on and memo off: same batch."""
    def _first_batch(disable_memo: bool):
        cm, tm, dm, pm = _build(num_pages=2_048)
        if disable_memo:
            # An adder built with ``match_memo=None`` walks the tree every time, which is
            # the pre-memo behaviour.
            original = pm.match_memo

            class _NoMemo(dict):
                def get(self, *_a, **_k):
                    return None

                def __setitem__(self, *_a, **_k):
                    return None

            pm.match_memo = _NoMemo(original)
        pm.pending_list = [
            _pending(uid, _ids(uid * 10_000, 96 + (uid % 4) * 32), 16) for uid in range(1, 9)
        ]
        batch = pm.schedule_next_batch(512)
        return None if batch is None else [(r.uid, r.extend_len) for r in batch.reqs]

    assert _first_batch(disable_memo=False) == _first_batch(disable_memo=True)


def test_the_match_cost_is_published():
    """A soak could not previously say whether the scheduler's CPU went into scheduling or
    into prefix matching; the replay has reported it since §R7."""
    from freetoken.scheduler.counters import PrefillCounters

    counters = PrefillCounters()
    counters.note_pass(seatable=2, chunked_inflight=0)
    counters.note_pass(seatable=2, chunked_inflight=0)
    counters.note_match(1_000)
    counters.note_match(3_000)
    counters.match_memo_hits += 5

    match = counters.as_dict()["match"]
    assert match == {
        "calls": 2,
        "tokens": 4_000,
        "memo_hits": 5,
        "tokens_per_pass": 2_000,
    }
