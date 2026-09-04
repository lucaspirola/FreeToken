"""Regression tests for the KV-page fatal in chunked-prefill continuations.

The 16-way Switchyard soak (2026-09-04) died at t=611 s with::

    RuntimeError: batch needs 6061 pages but only 3605 are physically allocatable
                  and 0 logical pages remain

raised from ``CacheManager.committed_pages_required`` inside ``_prepare_batch``.

``PrefillAdder.try_add_one`` had two paths and only one of them checked KV availability:
a *fresh* admit went through ``_try_allocate_one`` (which charges the whole remaining
prompt plus ``output_len`` against ``available_size``), while a *continuation* of an
already-chunked prompt was admitted unconditionally on the premise that "a continuation
already owns its resources". It owns its table slot, its GDN state slots and its
already-forwarded pages -- but NOT the pages for its next chunk, which are allocated
later in ``allocate_paged``. Continuations are placed at the head of ``pending_list``, so
the ungated path ran first; when the pool was fully committed the batch was unbackable
and the growth planner raised, killing the scheduler process. The shortage was transient
(sixteen sessions expired in the same two seconds), so back-pressure would have cleared
it.

The fix caps a chunk by the pages the pool can still back this pass and defers the
request when nothing fits, so the shortage becomes back-pressure instead of a fatal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

CHUNK = 8
WIDTH = 64
MAX_RUNNING = 4
POOL_PAGES = 32


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build_managers(num_pages=POOL_PAGES):
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
    return cm, tm, dm, pm


def _pending(uid: int, first_token: int, length: int, max_tokens: int = 4):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(
        uid=uid,
        input_ids=torch.arange(first_token, first_token + length, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def _forward(cm, batch):
    """Stand in for ``_prepare_batch`` + the forward: allocate the chunk, advance cached_len."""
    assert cm.committed_pages_required(batch.reqs) <= cm.num_pages
    cm.allocate_paged(batch.reqs)
    for req in batch.reqs:
        req.complete_one()
    return batch.reqs


def _decode_steps(cm, req, n: int) -> None:
    """Drive ``n`` decode steps, the growth that eats a continuation's unreserved pages."""
    for step in range(n):
        cm.allocate_paged([req])
        req.append_host(torch.tensor([900 + step], dtype=torch.int32))
        req.complete_one()


def _start_chunked_prompt(cm, pm):
    """Admit a 3-chunk prompt and forward its first chunk; return its pending entry."""
    prompt = _pending(uid=1, first_token=0, length=3 * CHUNK)
    pm.pending_list = [prompt]
    batch = pm.schedule_next_batch(CHUNK)
    assert batch is not None and len(batch.reqs) == 1
    _forward(cm, batch)
    assert prompt.chunked_req is not None and prompt.chunked_req.cached_len == CHUNK
    return prompt


def _fill_pool_behind_the_continuation(cm, pm, prompt, leave_free: int):
    """Admit a neighbour while the chunked prompt waits, and decode it down to
    ``leave_free`` free pages. This is the soak's shape: between two chunks of the same
    prompt the pages that prompt still needs are invisible to ``available_size``, so
    later admits and decode growth spend them."""
    held = pm.pending_list
    neighbour = _pending(uid=2, first_token=100, length=20)
    pm.pending_list = [neighbour]  # the chunked prompt waited out a decode burst
    batch = pm.schedule_next_batch(WIDTH)
    assert batch is not None
    (req,) = _forward(cm, batch)
    _decode_steps(cm, req, len(cm.free_slots) - leave_free)
    assert len(cm.free_slots) == leave_free
    pm.pending_list = held
    assert pm.pending_list[0] is prompt
    return req


def _finish(cm, tm, req) -> None:
    cm.cache_req(req, finished=True)
    tm.free(req.table_idx)


def _drain_prompt(cm, tm, pm, budget=CHUNK):
    """Run the remaining chunks of the head prompt to completion."""
    from freetoken.scheduler.prefill import ChunkedReq

    final = None
    while pm.runnable:
        batch = pm.schedule_next_batch(budget)
        assert batch is not None, "the continuation never made progress"
        for req in _forward(cm, batch):
            if not isinstance(req, ChunkedReq):
                final = req
    assert final is not None
    _finish(cm, tm, final)
    return final


def test_continuation_is_deferred_when_the_pool_cannot_back_its_next_chunk():
    """The soak's fatal, reproduced: a fully committed pool and a continuation whose next
    chunk needs pages nobody can allocate. It must be deferred, not admitted."""
    cm, tm, _dm, pm = _build_managers()
    prompt = _start_chunked_prompt(cm, pm)
    neighbour = _fill_pool_behind_the_continuation(cm, pm, prompt, leave_free=0)

    # Genuine exhaustion: nothing free, nothing evictable, no logical pages left to commit.
    assert cm.available_size == 0
    cached_len = prompt.chunked_req.cached_len
    # This is the fatal. Had the continuation been admitted for a full chunk, the batch it
    # landed in could not have been backed, and the growth planner would have killed the
    # scheduler process instead of applying back-pressure.
    with pytest.raises(RuntimeError, match="physically allocatable"):
        cm.committed_pages_required(
            [SimpleNamespace(device_len=cached_len + CHUNK, cached_len=cached_len)]
        )

    assert pm.schedule_next_batch(CHUNK) is None  # deferred
    assert pm.pending_list[0] is prompt  # and it keeps the head of the queue
    assert prompt.chunked_req.cached_len == cached_len  # nothing was forwarded

    # Back-pressure clears the way the soak's would have: the neighbour finishes and its
    # pages become reclaimable. The deferred continuation then runs to completion.
    _finish(cm, tm, neighbour)
    _drain_prompt(cm, tm, pm)

    cm.check_integrity()
    assert not pm.runnable


def test_continuation_chunk_is_capped_to_what_the_pool_can_back():
    """Short of a whole chunk but not of everything: forward what fits, do not raise."""
    cm, tm, _dm, pm = _build_managers()
    prompt = _start_chunked_prompt(cm, pm)
    neighbour = _fill_pool_behind_the_continuation(cm, pm, prompt, leave_free=3)

    batch = pm.schedule_next_batch(CHUNK)
    assert batch is not None
    (req,) = batch.reqs
    assert req.extend_len == 3  # capped by the pool, not by the token budget
    # The invariant the fatal violated: an admitted batch is always physically backable.
    _forward(cm, batch)
    assert len(cm.free_slots) == 0

    _finish(cm, tm, neighbour)
    _drain_prompt(cm, tm, pm)
    cm.check_integrity()


def test_a_deferred_continuation_is_not_starved_by_a_fresh_admit():
    """A queued fresh prompt must not overtake the blocked continuation and spend the
    pages it is waiting for."""
    cm, tm, _dm, pm = _build_managers()
    prompt = _start_chunked_prompt(cm, pm)
    neighbour = _fill_pool_behind_the_continuation(cm, pm, prompt, leave_free=0)

    fresh = _pending(uid=3, first_token=200, length=CHUNK)
    pm.pending_list.append(fresh)
    assert pm.schedule_next_batch(WIDTH) is None  # nobody runs, the continuation is first
    assert fresh.chunked_req is None

    _finish(cm, tm, neighbour)  # 24 pages come back, enough for exactly one of the two
    batch = pm.schedule_next_batch(WIDTH)
    assert batch is not None
    assert [req.uid for req in batch.reqs] == [prompt.uid]  # continuation served first
    assert pm.pending_list[-1] is fresh  # the fresh prompt is still queued behind it
    assert fresh.chunked_req is None


# --------------------------------------------------------- scheduler-side back-pressure


class _BlockedCache:
    """KV pool that stays exhausted until a session lease is released."""

    is_hybrid = False

    def __init__(self) -> None:
        self.free = 0
        self.unlocked: list = []

    @property
    def available_size(self) -> int:
        return self.free

    def match_req(self, _req):  # only reached for a fresh admit
        return SimpleNamespace(cuda_handle=SimpleNamespace(cached_len=0))

    def unlock(self, handle) -> None:
        self.unlocked.append(handle)
        self.free = 4096


def _blocked_scheduler(pending_list):
    from freetoken.scheduler.scheduler import Scheduler, SessionLease

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = _BlockedCache()
    scheduler.prefill_manager = SimpleNamespace(pending_list=pending_list)
    scheduler.config = SimpleNamespace(kv_grow_step_tokens=65_536)
    scheduler._sessions = {
        "idle": SessionLease("idle-handle", 300.0, reclaimable=True, last_used_at=1.0)
    }
    scheduler._session_spill_store = None
    scheduler._growable_shrink_pending = False
    scheduler.restored = []
    scheduler._restore_cold_session = lambda sid, ids: scheduler.restored.append(sid)
    return scheduler


def test_reclaim_for_blocked_prefill_spills_a_lease_for_a_blocked_continuation():
    """``_reclaim_for_blocked_prefill`` used to ``continue`` past every continuation, so a
    deferred one at the head of the queue got no reclaim at all and the queue stalled."""
    pending = _pending(uid=1, first_token=0, length=3 * CHUNK)
    pending.session_id = "talker"
    pending.chunked_req = SimpleNamespace(cached_len=CHUNK)
    scheduler = _blocked_scheduler([pending])

    assert scheduler._reclaim_for_blocked_prefill() is True
    assert scheduler.cache_manager.unlocked == ["idle-handle"]
    assert scheduler._sessions["idle"].handle is None
    # A continuation is mid-prefill with live state: no checkpoint may be restored over it.
    assert scheduler.restored == []


def test_reclaim_still_restores_a_checkpoint_for_a_fresh_admit():
    """The continuation branch must not change the fresh-admit behaviour."""
    pending = _pending(uid=1, first_token=0, length=CHUNK)
    pending.session_id = "talker"
    scheduler = _blocked_scheduler([pending])

    assert scheduler._reclaim_for_blocked_prefill() is True
    assert scheduler.restored == ["talker"]
