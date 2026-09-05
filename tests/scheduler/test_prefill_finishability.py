"""The admitted SET must stay finishable, not just each arrival at the instant it arrives.

Three admission-gate rewrites failed the live 16-way Switchyard soak in a row, each with
the same shape: a budget that is checked once, when a request is admitted, and never
re-validated against the requests already in flight.

* ``fad1fc4`` charged the wrong currency (whole prompt against the per-chunk page cap).
* ``81ab30e`` charged fresh admits against ``max_size``, i.e. the whole pool, so the KV
  held by decoders and by locked session prefixes was invisible; the pool reached
  ``token usage 1.00`` and no lane could buy its next chunk (soak report S5).
* ``ea7ed7c`` charged the right quantity against the right pool, but only at the moment of
  admission: an idle lease's reclaimable tokens counted as capacity for prompt A, A forwarded
  one 8K chunk and stayed chunked, and on the next pass the same lease tokens counted again
  for prompt B. Fourteen prefills got through that door one at a time; between them they had
  forwarded 237,819 tokens of a 262,144-token pool and still owed 222,538, with nothing
  decoding. Permanent deadlock (soak report T5).

The invariant these tests pin is the one none of them enforced::

    owed = SUM over in-flight chunked prefills of (input_len - forwarded) + output_len
         + DecodeManager.inflight_tokens
    owed <= CacheManager.available_size

``owed`` is what the pool has already PROMISED. Every token of it must be bought out of
free-or-evictable space before any of those requests can complete and hand a page back,
and the chunked-prefill scheduler advances every lane together, so nothing completes to
break the tie. ``PrefillManager._standing_reservation`` is what carries the promise across
passes; ``max_chunked_prefills`` is the belt-and-braces bound; and
``FREETOKEN_SCHEDULER_INVARIANT`` turns the same statement into a runtime assertion.
"""

from __future__ import annotations

import importlib

import pytest
import torch

WIDTH = 256
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
    return cm, tm, dm, PrefillManager(cm, tm, dm)


def _pending(uid: int, first_token: int, length: int, max_tokens: int):
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


def _owed(pm) -> int:
    """The left-hand side of the invariant, computed from outside the scheduler."""
    total = pm.decode_manager.inflight_tokens
    for pending in pm.pending_list:
        chunked = pending.chunked_req
        if chunked is not None:
            total += max(0, pending.input_len - chunked.cached_len) + pending.output_len
    return total


def test_a_second_prompt_is_refused_once_the_first_has_spoken_for_the_pool():
    """The regression: an in-flight prefill's unforwarded tail keeps costing admission.

    A 96-token prompt is admitted into a 160-token pool and forwards 32 tokens. It still
    owes 64 + 8, and ``available_size`` has dropped by exactly the 32 it forwarded -- so a
    second 96-token prompt does not fit, and would not have fitted at the moment the first
    one was admitted either. A gate that only looks at ``available_size`` sees 128 free,
    admits it, and the two of them then owe 232 against a 160-token pool.
    """
    cm, _tm, _dm, pm = _build(num_pages=160)
    first = _pending(uid=1, first_token=0, length=96, max_tokens=8)
    pm.pending_list = [first]
    batch = pm.schedule_next_batch(32)
    assert batch is not None and len(batch.reqs) == 1
    _forward(cm, batch)
    assert first.chunked_req is not None, "the prompt must still be mid-prefill"

    standing = pm._standing_reservation()
    assert standing == (96 - 32) + 8

    second = _pending(uid=2, first_token=10_000, length=96, max_tokens=8)
    pm.pending_list = pm.pending_list + [second]
    before = cm.available_size
    batch = pm.schedule_next_batch(32)

    # The pool has room for the second prompt's next CHUNK -- that is the trap -- but not
    # for its whole remaining footprint on top of the first prompt's.
    assert before > 96 + 8
    assert second.chunked_req is None, "the second prompt must not be admitted"
    assert batch is not None, "the first prompt must keep making progress"
    assert [r.uid for r in batch.reqs] == [1]
    _forward(cm, batch)
    assert _owed(pm) <= cm.available_size


def test_the_invariant_holds_over_a_full_drain():
    """Drive the queue to completion and assert the invariant on every single pass."""
    cm, _tm, dm, pm = _build(num_pages=384)
    pm.pending_list = [
        _pending(uid=i, first_token=i * 10_000, length=96, max_tokens=8)
        for i in range(1, 7)
    ]
    for _ in range(400):
        assert _owed(pm) <= cm.available_size
        batch = pm.schedule_next_batch(32)
        if batch is None:
            batch = dm.schedule_next_batch()
            if batch is None:
                break
            cm.allocate_paged(batch.reqs)
            for req in batch.reqs:
                req.append_host(torch.tensor([7], dtype=torch.int32))
                req.complete_one()
            for req in [r for r in batch.reqs if r.remain_len <= 0]:
                dm.remove_req(req)
                cm.cache_req(req, finished=True)
                pm.table_manager.free(req.table_idx)
            continue
        _forward(cm, batch)
        dm.filter_reqs(batch.reqs)
    assert not pm.pending_list, "every prompt must finish; a refusal must not be permanent"


def test_concurrent_chunked_prefills_are_capped():
    """Belt and braces: the cap bounds the lanes even if the reservation arithmetic slips."""
    cm, _tm, _dm, pm = _build(num_pages=4_096)
    pm.max_chunked_prefills = 3
    pm.pending_list = [
        _pending(uid=i, first_token=i * 10_000, length=96, max_tokens=8)
        for i in range(1, 8)
    ]
    for _ in range(6):
        batch = pm.schedule_next_batch(16)
        if batch is None:
            break
        _forward(cm, batch)
        chunked = sum(1 for p in pm.pending_list if p.chunked_req is not None)
        assert chunked <= pm.max_chunked_prefills


def test_the_cap_never_blocks_the_continuations_that_release_it():
    """A fresh admit refused by the cap must not stop the lanes behind it from advancing."""
    cm, _tm, _dm, pm = _build(num_pages=4_096)
    pm.max_chunked_prefills = 2
    pm.pending_list = [
        _pending(uid=i, first_token=i * 10_000, length=96, max_tokens=8)
        for i in range(1, 6)
    ]
    seen = set()
    for _ in range(200):
        batch = pm.schedule_next_batch(16)
        if batch is None:
            break
        seen.update(r.uid for r in batch.reqs)
        _forward(cm, batch)
        assert sum(1 for p in pm.pending_list if p.chunked_req is not None) <= 2
    assert seen == {1, 2, 3, 4, 5}, (
        "the cap must release as prefills finish, not wedge the queue"
    )
    assert not pm.pending_list


def test_the_debug_assertion_is_off_by_default(monkeypatch):
    from freetoken.scheduler.prefill import _invariant_mode

    monkeypatch.delenv("FREETOKEN_SCHEDULER_INVARIANT", raising=False)
    assert _invariant_mode() == ""
    for off in ("", "0", "off", "false", "no", "OFF"):
        monkeypatch.setenv("FREETOKEN_SCHEDULER_INVARIANT", off)
        assert _invariant_mode() == ""
    monkeypatch.setenv("FREETOKEN_SCHEDULER_INVARIANT", " Warn ")
    assert _invariant_mode() == "warn"


def _over_committed(pm):
    """The soak-report-T5 state: 3 in-flight prefills owing 300 against a 160-token pool.

    Built directly rather than by driving a broken gate: what is under test is the
    DETECTOR, and it has to fire on any tree -- including the two reverted ones, whose
    ``PrefillManager`` this file cannot import.
    """

    class _Chunked:
        cached_len = 0

    for uid in range(1, 4):
        pending = _pending(uid=uid, first_token=uid * 10_000, length=92, max_tokens=8)
        pending.chunked_req = _Chunked()
        pm.pending_list.append(pending)
    standing = pm._standing_reservation()
    assert standing == 3 * (92 + 8)
    return standing


def test_the_debug_assertion_raises_on_an_over_committed_admitted_set():
    cm, _tm, _dm, pm = _build(num_pages=160)
    standing = _over_committed(pm)
    assert standing > cm.available_size
    with pytest.raises(AssertionError, match="finishability invariant violated"):
        pm._check_finishability(standing, "raise")


def test_the_debug_assertion_warns_instead_when_asked_to(monkeypatch):
    """``warn`` is the mode a live soak runs in: report it, do not kill the server."""
    from freetoken.scheduler import prefill as prefill_mod

    said = []
    monkeypatch.setattr(
        prefill_mod.logger, "warning_rank0",
        lambda fmt, *a: said.append(fmt % a if a else fmt),
    )
    cm, _tm, _dm, pm = _build(num_pages=160)
    standing = _over_committed(pm)
    assert standing > cm.available_size
    pm._check_finishability(standing, "warn")
    assert len(said) == 1
    assert "finishability invariant violated" in said[0]
    assert "short by 140" in said[0]


def test_the_debug_assertion_stays_quiet_on_a_healthy_set(monkeypatch):
    """With ``raise`` armed, a whole drain must not trip it."""
    monkeypatch.setenv("FREETOKEN_SCHEDULER_INVARIANT", "raise")
    cm, _tm, _dm, pm = _build(num_pages=4_096)
    pm.pending_list = [
        _pending(uid=i, first_token=i * 10_000, length=96, max_tokens=8)
        for i in range(1, 4)
    ]
    for _ in range(200):
        batch = pm.schedule_next_batch(32)
        if batch is None:
            break
        _forward(cm, batch)
    assert not pm.pending_list

# --------------------------------------------------------------------------- #
# A cold session restore is the one thing that spends the pool between two passes
# --------------------------------------------------------------------------- #
# Soak §W6: nine ``finishability invariant`` warnings in the passthrough tail of the
# ca7e74b run, a CONSTANT 1,401-token shortfall over nine passes while both sides fell by
# one 8,192-token chunk each -- the pool was over-promised exactly once and then tracked
# the admitted set in lockstep. The two seconds before the first warning hold four
# ``Restored cold session`` lines, one of 79,104 tokens.
#
# The mechanism: ``Scheduler._restore_cold_session`` runs from the message path (before
# ``add_one_req``) and from ``_reclaim_for_blocked_prefill`` -- both BETWEEN two prefill
# passes, neither an admission gate. It ends with the restored prefix LOCKED, which takes
# those tokens out of ``available_size`` whether it allocated pages for them or merely
# re-protected an evictable prefix the tree still held. ``owed`` does not move, so the
# proof the gate made on the previous pass is quietly invalidated.


def _install_evictable_prefix(cm, ids):
    """An unlocked prefix in the radix tree: what a spilled session's KV looks like from
    the pool's side -- still resident, evictable, and therefore counted in
    ``available_size`` as capacity the admission gate is free to sell."""
    pages = cm._allocate(len(ids))
    cached_len, _handle = cm.prefix_cache.insert_prefix(ids, pages)
    cm._free(pages[:cached_len])


def test_finishability_reservation_is_the_invariant_it_is_named_after():
    """The exported figure has to equal the check's own left-hand side, or charging
    anything against it proves nothing."""
    cm, _tm, dm, pm = _build(num_pages=256)
    first = _pending(uid=1, first_token=0, length=96, max_tokens=8)
    pm.pending_list = [first]
    _forward(cm, pm.schedule_next_batch(32))
    assert first.chunked_req is not None
    assert pm.finishability_reservation() == (
        pm._standing_reservation() + dm.inflight_tokens
    )
    assert pm.finishability_reservation() == _owed(pm)


def test_session_restore_footprint_is_exactly_what_the_restore_costs_the_pool():
    """``available_size`` counts ``evictable_size``, and locking moves those tokens into
    ``protected_size`` -- so re-protecting a resident prefix costs the pool every bit as
    much as allocating one, which is how a restore can over-promise the pool without
    allocating a single new page."""
    cm, _tm, _dm, _pm = _build(num_pages=256)
    ids = torch.arange(500, 564, dtype=torch.int32)
    _install_evictable_prefix(cm, ids)

    predicted = cm.session_restore_footprint(ids)
    before = cm.available_size
    cm.retain_prefix(ids, len(ids))
    assert predicted == before - cm.available_size == len(ids)

    # Already protected: a second lock costs nothing, and the estimate has to say so.
    assert cm.session_restore_footprint(ids) == 0


def test_a_restore_the_pool_cannot_afford_is_what_breaks_the_invariant():
    """The §W6 sequence, end to end.

    A spilled session's prefix sits evictable in the tree, so the admission gate counts it
    and admits a chunked prefill against it. Restoring it then locks those same tokens --
    ``owed`` unchanged, ``available_size`` down by the whole prefix -- and the admitted set
    is no longer finishable. The charge that prevents it is the one the scheduler now
    makes: footprint against ``available_size - finishability_reservation()``.
    """
    cm, _tm, _dm, pm = _build(num_pages=256)
    spilled = torch.arange(500, 564, dtype=torch.int32)
    _install_evictable_prefix(cm, spilled)

    # 200 + 8 fits the 256-token pool at the instant it is admitted, because the spilled
    # prefix is evictable and ``available_size`` counts it. 200 + 8 + 64 does not.
    admitted = _pending(uid=1, first_token=0, length=200, max_tokens=8)
    pm.pending_list = [admitted]
    _forward(cm, pm.schedule_next_batch(32))
    assert admitted.chunked_req is not None, "the prompt must still be mid-prefill"
    assert _owed(pm) <= cm.available_size, "the gate proved this set finishable"

    reserved = pm.finishability_reservation()
    footprint = cm.session_restore_footprint(spilled)
    # This is the scheduler's refusal condition, evaluated on the state that produced §W6.
    assert footprint > cm.available_size - reserved

    # Do it anyway -- the pre-fix behaviour -- and the proof is gone, by exactly the amount
    # the refusal condition was short by.
    shortfall = footprint - (cm.available_size - reserved)
    cm.retain_prefix(spilled, len(spilled))
    assert _owed(pm) - cm.available_size == shortfall > 0


def test_a_restore_that_fits_under_the_reservation_keeps_the_invariant():
    """The other half: the charge must not refuse a restore the pool can actually back,
    or session reuse is paid for in recompute on every busy server."""
    cm, _tm, _dm, pm = _build(num_pages=1024)
    spilled = torch.arange(500, 564, dtype=torch.int32)
    _install_evictable_prefix(cm, spilled)

    admitted = _pending(uid=1, first_token=0, length=128, max_tokens=8)
    pm.pending_list = [admitted]
    _forward(cm, pm.schedule_next_batch(32))

    reserved = pm.finishability_reservation()
    footprint = cm.session_restore_footprint(spilled)
    assert footprint <= cm.available_size - reserved, "a roomy pool must not refuse"

    cm.retain_prefix(spilled, len(spilled))
    assert _owed(pm) <= cm.available_size
