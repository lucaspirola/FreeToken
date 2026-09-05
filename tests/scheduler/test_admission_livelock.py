"""One queued prompt that does not fit must not stop the ones that do.

Soak report §Y5b (`38617a7`, Switchyard stage route). At 21:31:43 the last decode batch
ran and the scheduler then emitted **no batch at all for 576 seconds**, with 1 running
request draining to 0, 15 requests queued, ``token usage 0.59``, ``#mamba-slot 26/96`` and
108 K tokens of the KV pool free. The head of the prefill queue was a long-context turn
whose remaining footprint did not fit what the pool's protected/live pages had left; the
admission loop refused it and ``break``ed, so the fifteen requests behind it -- which did
fit -- were never even examined. Nothing was running, so no page came back, so the next
pass took the identical decision. It ran 1,867,771 refused passes at ~3,240/s on one core
at 102 % CPU for 9m36s, and ended only when the clients' own 600 s timeouts closed their
sockets.

Three properties come out of that, and all three are pinned here:

1. a fresh prompt the pools cannot seat *right now* is SKIPPED, not stopped on, whenever
   the pass still has room to sell to something behind it;
2. skipping must not re-sell that room -- the finishability invariant
   ``owed(admitted set) <= available_size`` still holds on every pass (this is what
   ``ea7ed7c``'s continue-past-refusals got wrong, and why the standing reservation had to
   exist before the skip could be safe);
3. the seat scan (``_seatable_lanes``, the interleave share's divisor) has to make the same
   decision as the loop. When it did not, §Y5b showed up on ``/v1/stats`` as a
   seatable-lane histogram pinned at bucket 2 for 1.7 M consecutive passes that scheduled
   nothing whatsoever.

Everything here runs on the real ``CacheManager`` / ``PrefillManager`` on CPU.
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

    return PendingReq(uid=uid, input_ids=ids, sampling_params=SamplingParams(max_tokens=max_tokens))


def _ids(first: int, length: int) -> torch.Tensor:
    return torch.arange(first, first + length, dtype=torch.int32)


def _owed(pm) -> int:
    total = pm.decode_manager.inflight_tokens
    for pending in pm.pending_list:
        chunked = pending.chunked_req
        if chunked is not None:
            total += max(0, pending.input_len - chunked.cached_len) + pending.output_len
    return total


def _drain(cm, tm, dm, pm, budget: int = 4_096, limit: int = 2_000) -> set[int]:
    """Run the scheduler to a standstill, asserting the invariant on every pass."""
    served: set[int] = set()
    for _ in range(limit):
        assert _owed(pm) <= cm.available_size, "the admitted set must stay finishable"
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


def _wedge(num_pages: int, lease_len: int):
    """A pool whose free space is smaller than the request sitting at the head of the queue.

    Reproduces §Y5b's state: a chunk of the pool is held by a *protected* session prefix
    (``retain_prefix`` locks the radix node, exactly as a resident conversation's KV lease
    does), nothing is running, and the queue head needs more than what is left.
    """
    cm, tm, dm, pm = _build(num_pages=num_pages)
    lease_ids = _ids(0, lease_len)
    pm.pending_list = [_pending(1, lease_ids, 4)]
    _drain(cm, tm, dm, pm)
    lease = cm.retain_prefix(lease_ids, lease_len)
    return cm, tm, dm, pm, lease


def test_an_unadmittable_head_does_not_pin_the_queue_behind_it():
    """§Y5b itself: the head does not fit, four requests behind it do."""
    cm, tm, dm, pm, _lease = _wedge(num_pages=1_024, lease_len=600)
    room = cm.available_size
    head = _pending(10, _ids(100_000, 500), 64)      # 500 + 64 > room
    behind = [_pending(20 + i, _ids(200_000 + i * 1_000, 20), 4) for i in range(4)]
    pm.pending_list = [head] + behind
    assert head.input_len + head.output_len > room, "the head must not fit"
    assert all(p.input_len + p.output_len < room for p in behind), "the rest must fit"

    served = _drain(cm, tm, dm, pm)

    assert served >= {20, 21, 22, 23}, (
        "every request that fits the pool must be served while the head waits; "
        "this set was empty for 576 s in soak §Y5b"
    )
    assert pm.counters.fresh_admits_deferred > 0, "the head must be recorded as deferred"
    assert [p.uid for p in pm.pending_list] == [10], "the head keeps its place in the queue"


def test_the_deferred_head_is_admitted_once_the_pool_frees_the_room():
    """Deferral is not rejection: the head must run as soon as the room comes back."""
    cm, tm, dm, pm, lease = _wedge(num_pages=1_024, lease_len=600)
    head = _pending(10, _ids(100_000, 500), 64)
    pm.pending_list = [head, _pending(20, _ids(200_000, 20), 4)]
    served = _drain(cm, tm, dm, pm)
    assert 10 not in served and 20 in served

    cm.unlock(lease)  # the conversation's lease is checkpointed / released
    served = _drain(cm, tm, dm, pm)
    assert 10 in served, "a deferred request must not be starved forever"
    assert not pm.pending_list


def test_the_seat_scan_and_the_admission_loop_agree():
    """``_seatable_lanes`` > 0 with no batch is the §Y5b signature on ``/v1/stats``.

    The histogram sat at bucket 2 for 1,701,163 of 1,868,157 passes while not one batch
    was scheduled. The divergence is ``lock()``: the scan asks ``would_seat``, which never
    locks, while the real admit locks the matched prefix -- moving it out of ``evictable``
    and therefore out of ``available_size`` -- and re-checks the same gate against the
    smaller budget. A turn that reuses a large EVICTABLE prefix passes the scan and fails
    the admit, so the pass counts seats it will not fill. Whatever the two disagree about,
    a positive seat count must still mean a batch.
    """
    cm, tm, dm, pm = _build(num_pages=1_024)
    # Conversation A runs to completion: 700 tokens of *evictable* prefix in the tree.
    convo = _ids(0, 700)
    pm.pending_list = [_pending(1, convo, 4)]
    _drain(cm, tm, dm, pm)
    # A second, unrelated conversation stays resident, so the pool is tight.
    lease_ids = _ids(500_000, 200)
    pm.pending_list = [_pending(2, lease_ids, 4)]
    _drain(cm, tm, dm, pm)
    cm.retain_prefix(lease_ids, 200)

    head = _pending(10, torch.cat([convo, _ids(900_000, 5)]), 250)
    behind = [_pending(20 + i, _ids(200_000 + i * 1_000, 20), 4) for i in range(4)]
    pm.pending_list = [head] + behind
    # The pre-lock budget covers the head; the post-lock one does not. That gap is the bug.
    assert head.input_len - 700 + head.output_len <= cm.available_size
    assert head.input_len - 700 + head.output_len > cm.available_size - 700

    seen = []
    for _ in range(20):
        queued = len(pm.pending_list)
        batch = pm.schedule_next_batch(4_096)
        uids = None if batch is None else [r.uid for r in batch.reqs]
        seen.append(uids)
        if queued > 1:  # the divisor is only computed for a queue of more than one
            seatable = pm.counters.seatable_lanes_last
            assert (seatable > 0) == (batch is not None), (
                f"seat scan said {seatable} lanes and the pass produced {uids}"
            )
        if batch is None:
            break
        cm.allocate_paged(batch.reqs)
        for req in batch.reqs:
            req.complete_one()
        dm.filter_reqs(batch.reqs)
    assert any(uids and set(uids) & {20, 21, 22, 23} for uids in seen), (
        "the requests behind the unadmittable head must be served"
    )


def test_skipping_never_re_sells_the_pool():
    """The ``ea7ed7c`` trap: continuing past a refusal must not hand the same reclaimable
    tokens to the next lane. Sixteen prompts, a pool that fits about two, invariant checked
    on every pass of a full drain."""
    cm, tm, dm, pm = _build(num_pages=1_400)
    pm.pending_list = [
        _pending(uid, _ids(uid * 10_000, 64 + (uid % 5) * 90), 32) for uid in range(1, 17)
    ]
    served = _drain(cm, tm, dm, pm, budget=128)
    assert served == set(range(1, 17)), "every prompt must finish"
    assert not pm.pending_list
    assert pm.counters.invariant_violations == 0


def test_a_pass_that_schedules_nothing_lets_the_loop_rest():
    """The other half of §Y5b: 1.87 M passes in 576 s, one core at 102 %.

    ``normal_loop``/``overlap_loop`` take their 10 ms nap only when
    ``_only_idle_sessions`` says nothing can change, and that test used to require an
    EMPTY prefill queue -- which a queue nobody can admit never is. A pass that scheduled
    nothing has just proved another attempt will not either.
    """
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    stub = SimpleNamespace(
        prefill_manager=SimpleNamespace(runnable=True),
        decode_manager=SimpleNamespace(runnable=False),
        _pending_rebuild=None,
        _growable_shrink_pending=False,
        _admission_stalled=False,
    )
    assert Scheduler._only_idle_sessions(stub, None) is False, "a fresh queue is not idle"

    stub._admission_stalled = True
    assert Scheduler._only_idle_sessions(stub, None) is True, (
        "a refused pass must be allowed to rest instead of spinning"
    )

    # Anything that can still change the state keeps the loop hot.
    stub.decode_manager = SimpleNamespace(runnable=True)
    assert Scheduler._only_idle_sessions(stub, None) is False
    stub.decode_manager = SimpleNamespace(runnable=False)
    assert Scheduler._only_idle_sessions(stub, object()) is False, "an undrained batch"
