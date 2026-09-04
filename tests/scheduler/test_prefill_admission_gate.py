"""Regression tests for the fresh-admit gate, the admission scan and the match memo.

Companion to two measurement write-ups:

* ``benchmarks/results/nemotron35_lightning_5080_scheduler_bisect_2026-09-04.md`` — **R-2**,
  ``schedule_next_batch`` ended the pass at the first refusal, abandoning a median of 13
  queued requests of which 11 were seatable (23,628 lane slots over one 20,000-forward run);
  and **R-7**, a prompt bigger than the whole pool could never satisfy the gate and there was
  no rejection path, so the scheduler returned no batch forever with work outstanding.

* ``benchmarks/results/nemotron35_lightning_5080_switchyard_soak_2026-09-04.md`` §S — the
  live 16-way soak against ``81ab30e``, whose gate charged a fresh admit against
  ``cache_manager.max_size``. That ceiling still counts the KV held by *decoding* requests
  and by locked/retained *session* prefixes, neither of which admission can spend, so
  admissions kept arriving until ``token usage: 1.00`` and then no lane could buy its next
  chunk and nothing could complete to free one — 52% of the wall clock with no batch at all.
  The gate charges ``admissible_size`` instead: what is free now, plus what demand reclaim
  can still free (idle session leases), and nothing else.
"""

from __future__ import annotations

import torch

WIDTH = 512
MAX_RUNNING = 8
CHUNK = 8


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build_managers(num_pages: int):
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
    pm.interleave_chunks = True  # the growable multi-agent mode the soak runs
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
    """Stand in for ``_prepare_batch`` + the forward: the batch must be backable."""
    assert cm.committed_pages_required(batch.reqs) <= cm.num_pages
    cm.allocate_paged(batch.reqs)
    for req in batch.reqs:
        req.complete_one()
    return batch.reqs


def _seat_a_decoding_neighbour(cm, dm, pm, length: int, max_tokens: int = 4):
    """Run one short prompt to the end of its prefill and leave it decoding.

    Its pages are the occupancy that makes ``available_size`` small while the pool as a
    whole is still large -- the state the ``max_size`` gate could not tell apart from an
    empty pool.
    """
    held = pm.pending_list
    pm.pending_list = [
        _pending(uid=99, first_token=500_000, length=length, max_tokens=max_tokens)
    ]
    (req,) = _forward(cm, pm.schedule_next_batch(WIDTH))
    dm.running_reqs.add(req)
    pm.pending_list = held
    return req


def _retain_a_session_prefix(cm, dm, pm, length: int):
    """Finish one request and keep its prefix as a session lease: LOCKED, not evictable.

    Mirrors ``Scheduler._free_req_resources(retain_session=True)`` -- ``cache_req`` donates
    the KV to the radix tree, then ``retain_prefix`` locks it, which moves it out of
    ``evictable_size`` and therefore out of ``available_size``.
    """
    held = pm.pending_list
    pm.pending_list = [
        _pending(uid=98, first_token=700_000, length=length, max_tokens=1)
    ]
    (req,) = _forward(cm, pm.schedule_next_batch(WIDTH))
    pm.pending_list = held
    cm.cache_req(req, finished=True)
    handle = cm.retain_prefix(req.input_ids, req.cached_len)
    pm.table_manager.free(req.table_idx)
    return handle


# ---------------------------------------------------------------------------
# The soak's stall: what the finishability budget may and may not count
# ---------------------------------------------------------------------------


def test_pages_held_by_a_decoding_request_are_not_admission_budget():
    """§S: a fresh admit may not be charged against KV another request is decoding into.

    The prompt fits the pool's *maximum* several times over, so a gate against ``max_size``
    admits it -- and that is exactly how the soak reached ``token usage: 1.00`` with nothing
    able to advance. What it does not fit is what the pool can actually hand out.
    """
    cm, _tm, dm, pm = _build_managers(num_pages=160)
    _seat_a_decoding_neighbour(cm, dm, pm, length=120)

    prompt = _pending(uid=1, first_token=0, length=100)
    pm.pending_list = [prompt]

    # The discriminating condition: room under the ceiling, none in the pool.
    assert prompt.input_len + prompt.output_len <= cm.max_size
    assert prompt.input_len + prompt.output_len > cm.admissible_size

    assert pm.schedule_next_batch(CHUNK) is None, (
        "a fresh admit was charged against KV held by a decoding request"
    )
    assert prompt.chunked_req is None, "an unfinishable prompt was admitted into a lane"


def test_a_locked_session_prefix_is_not_admission_budget_until_reclaim_can_buy_it():
    """§S, the other half: retained leases are locked, so eviction cannot reach them.

    With no reclaim hook installed (unit tests, non-session serving) a locked prefix is
    simply unavailable and the prompt is refused. Once the scheduler advertises the lease as
    idle-and-reclaimable, the same prompt is admissible: ``_reclaim_soft_sessions_for_pending``
    will release it on demand, so those tokens really are obtainable.
    """
    cm, _tm, dm, pm = _build_managers(num_pages=160)
    lease = _retain_a_session_prefix(cm, dm, pm, length=120)
    retained = int(lease.cached_len)
    assert retained > 0

    prompt = _pending(uid=1, first_token=0, length=100)
    pm.pending_list = [prompt]

    need = prompt.input_len + prompt.output_len
    assert need <= cm.max_size
    assert need > cm.admissible_size, "the locked lease should not be free capacity"
    assert pm.schedule_next_batch(CHUNK) is None

    # Now the lease is advertised as idle and reclaimable, as Scheduler does.
    cm.reclaimable_tokens_hook = lambda: retained
    assert cm.admissible_size == cm.available_size + retained
    assert need <= cm.admissible_size

    batch = pm.schedule_next_batch(CHUNK)
    assert batch is not None, "reclaimable lease capacity was not counted as obtainable"
    assert [req.uid for req in batch.reqs] == [1]


def test_an_active_session_lease_is_never_counted_as_obtainable():
    """The hook reports only IDLE leases; an active one belongs to a live request.

    This is the distinction that makes the budget safe: counting a lease whose session has
    a request in flight is counting KV that reclaim will refuse to release.
    """
    cm, _tm, dm, pm = _build_managers(num_pages=160)
    _retain_a_session_prefix(cm, dm, pm, length=120)
    cm.reclaimable_tokens_hook = lambda: 0  # the session is active

    prompt = _pending(uid=1, first_token=0, length=100)
    pm.pending_list = [prompt]
    assert prompt.input_len + prompt.output_len > cm.admissible_size
    assert pm.schedule_next_batch(CHUNK) is None


# ---------------------------------------------------------------------------
# R-7 / R-2: the admission scan
# ---------------------------------------------------------------------------


def test_a_prompt_larger_than_the_whole_pool_is_refused_not_livelocked():
    """R-7: unsatisfiable at every pool state, so it must neither be admitted nor block.

    Admitting it would strand a chunked lane that pins its forwarded pages and can never
    reach a last chunk; leaving it at the head under the aging rule would eventually stall
    every fresh admit behind it. It is skipped instead, and the pass keeps making progress.
    """
    cm, _tm, _dm, pm = _build_managers(num_pages=64)

    huge = _pending(uid=1, first_token=0, length=64)
    small = _pending(uid=2, first_token=1_000, length=8, max_tokens=2)
    pm.pending_list = [huge, small]
    assert huge.input_len + huge.output_len > cm.max_size

    batch = pm.schedule_next_batch(WIDTH)
    assert batch is not None, "an unservable head request livelocked the scheduler"
    assert [req.uid for req in batch.reqs] == [2]
    assert huge.chunked_req is None, "an unfinishable prompt was admitted into a lane"

    _forward(cm, batch)
    # It is still queued and still refused, on an empty pool as much as on a full one.
    assert pm.pending_list[0] is huge
    assert pm.schedule_next_batch(WIDTH) is None
    assert huge.chunked_req is None


def test_a_refused_fresh_admit_does_not_block_a_continuation_behind_it():
    """R-2: continue past a refusal. A continuation is a different class from a fresh
    admit -- it was admitted in an earlier pass, so it is already ahead in FIFO order."""
    cm, _tm, dm, pm = _build_managers(num_pages=200)

    chunked = _pending(uid=1, first_token=0, length=6 * CHUNK)
    pm.pending_list = [chunked]
    _forward(cm, pm.schedule_next_batch(CHUNK))
    assert chunked.chunked_req is not None

    _seat_a_decoding_neighbour(cm, dm, pm, length=150)
    fresh = _pending(uid=2, first_token=1_000, length=100)
    pm.pending_list = [fresh, chunked]  # the refused fresh admit sits AHEAD of the lane

    # ``fresh`` cannot be seated: its whole remaining footprint does not fit what the pool
    # can hand out. It CAN be seated later, so this is a transient refusal and it keeps its
    # place -- not the permanent one the test above covers.
    assert fresh.input_len + fresh.output_len > cm.admissible_size
    assert fresh.input_len + fresh.output_len <= cm.max_size

    batch = pm.schedule_next_batch(CHUNK)
    assert batch is not None, "the pass stopped at the refusal instead of skipping it"
    assert [req.uid for req in batch.reqs] == [1], "the continuation behind it was dropped"
    assert fresh.chunked_req is None
    assert pm.pending_list[-1] is fresh  # refused, requeued, order preserved


def test_an_aged_refusal_reserves_the_queue_against_later_fresh_admits():
    """R-2's fairness half, with aging.

    Strict FIFO among fresh admits protects the long prompts -- they never win a size
    comparison against a short one, so unconditional overtaking means "every short prompt,
    forever". Applied unconditionally it also converts one temporarily unaffordable prompt
    into a dead scheduler, so the rule engages only once the refusal has been passed over
    ``admission_patience`` times.
    """
    cm, _tm, dm, pm = _build_managers(num_pages=200)
    _seat_a_decoding_neighbour(cm, dm, pm, length=150)

    big = _pending(uid=2, first_token=1_000, length=100)
    tiny = _pending(uid=3, first_token=2_000, length=2, max_tokens=1)
    pm.pending_list = [big, tiny]

    # ``tiny`` really is seatable; ``big`` is not. What decides is the aging rule alone.
    assert big.input_len + big.output_len > cm.admissible_size
    assert tiny.input_len + tiny.output_len + dm.inflight_tokens <= cm.admissible_size

    # While the refusal is young, the queue-mate goes first -- and the work it completes is
    # what eventually frees the KV ``big`` is waiting for.
    batch = pm.schedule_next_batch(CHUNK)
    assert batch is not None and [req.uid for req in batch.reqs] == [3]
    assert big.refused_passes == 1

    # Once it has been patient enough, it reserves the queue behind it.
    pm.pending_list = [big, tiny]
    big.refused_passes = pm.admission_patience
    tiny.chunked_req = None
    assert pm.schedule_next_batch(CHUNK) is None, (
        "a cheaper fresh admit overtook a refusal that had waited its turn"
    )
    assert [req.uid for req in pm.pending_list] == [2, 3]


# ---------------------------------------------------------------------------
# The match memo
# ---------------------------------------------------------------------------


def test_a_refused_prompt_is_not_re_matched_while_the_tree_is_unchanged():
    """§S: the whole CPU budget of a stall went into re-walking refused prompts.

    ``match_prefix`` is O(prompt), and a refused pass forwards nothing, so a queue of 118K
    -token prompts re-matched from scratch on every pass is the entire cost of a stalled
    scheduler. The memo makes the refusal arithmetic reuse the previous walk for as long as
    the tree that produced it is unchanged.
    """
    cm, _tm, dm, pm = _build_managers(num_pages=160)
    _seat_a_decoding_neighbour(cm, dm, pm, length=120)

    prompt = _pending(uid=1, first_token=0, length=100)
    pm.pending_list = [prompt]
    assert prompt.input_len + prompt.output_len > cm.admissible_size

    calls = []
    original = cm.match_req
    cm.match_req = lambda req: (calls.append(req.uid), original(req))[1]

    assert pm.schedule_next_batch(CHUNK) is None
    assert calls.count(1) == 1, "the first refusal has to walk the tree once"
    fingerprint = prompt.match_fp
    assert fingerprint is not None

    for _ in range(5):
        assert pm.schedule_next_batch(CHUNK) is None
    assert calls.count(1) == 1, "a refused prompt was re-matched against an unchanged tree"
    assert prompt.match_fp == fingerprint

    # A tree that has moved invalidates the memo: the walk must happen again, because the
    # match can only get deeper when nodes are added. Note it is the DONATION that moves it,
    # not the forward -- a request's own pages are allocated, not in the tree.
    other = _pending(uid=2, first_token=900_000, length=4, max_tokens=1)
    pm.pending_list = [other]
    (donor,) = _forward(cm, pm.schedule_next_batch(CHUNK))
    cm.cache_req(donor, finished=True)
    pm.table_manager.free(donor.table_idx)
    pm.pending_list = [prompt]
    assert cm.prefix_fingerprint() != fingerprint

    pm.schedule_next_batch(CHUNK)
    assert calls.count(1) == 2, "the memo outlived the tree state it was measured under"
