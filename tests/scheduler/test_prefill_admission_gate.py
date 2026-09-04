"""Regression tests for the fresh-admit gate and the admission scan.

Companion to ``benchmarks/results/nemotron35_lightning_5080_scheduler_bisect_2026-09-04.md``,
which measured both faults on a CPU replay of the Switchyard stage route:

**R-1** — ``PrefillAdder._try_allocate_one`` charged a fresh admit its WHOLE remaining
prompt plus ``output_len`` against ``available_size``. 2,404 of 2,408 prefill passes
(99.8%) ended on that refusal, and in 98.3% of them the chunk the pass would actually have
forwarded fitted with room to spare: median headroom 23,155 pages against a median chunk
of 512 tokens. The refusal was driven entirely by the 115,189-token median whole prompt.
The gate now asks two separate questions -- "can this prompt ever finish?" against the
pool MAXIMUM, and "what will this pass write?" against ``available_size``.

**R-2** — ``schedule_next_batch`` ended the pass at the first refusal, abandoning a median
of 13 queued requests of which 11 were seatable by the pools' own accounting (23,628 lane
slots over one 20,000-forward run). Because R-1 put the refusal on a long prompt near the
head, the tail was skipped pass after pass. The scan now continues past a refusal, with
strict FIFO among fresh admits and a bound on the refusals it will walk.

**R-7** — a prompt bigger than the whole pool could never satisfy the gate and there was
no rejection path, so the scheduler returned no batch forever with work outstanding.
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
    whole is still large -- the state R-1's gate could not tell apart from a full pool.
    """
    held = pm.pending_list
    pm.pending_list = [_pending(uid=99, first_token=500_000, length=length, max_tokens=max_tokens)]
    (req,) = _forward(cm, pm.schedule_next_batch(WIDTH))
    dm.running_reqs.add(req)
    pm.pending_list = held
    return req


def test_a_fresh_long_prompt_is_admitted_when_only_its_chunk_has_to_fit():
    """R-1: gate the pass's chunk, not the whole remainder, against ``available_size``."""
    from freetoken.scheduler.prefill import ChunkedReq

    cm, _tm, dm, pm = _build_managers(num_pages=160)
    _seat_a_decoding_neighbour(cm, dm, pm, length=120)

    prompt = _pending(uid=1, first_token=0, length=100)
    pm.pending_list = [prompt]

    # The discriminating condition: the pool cannot hold this prompt's remainder right now,
    # but it can hold the chunk this pass would forward several times over -- and it can
    # hold the whole prompt once the neighbour finishes decoding.
    assert prompt.input_len + prompt.output_len > cm.available_size
    assert CHUNK + prompt.output_len < cm.available_size
    assert prompt.input_len + prompt.output_len <= cm.max_size

    batch = pm.schedule_next_batch(CHUNK)
    assert batch is not None, "the whole-prompt reservation refused a chunk that fits"
    (req,) = batch.reqs
    assert isinstance(req, ChunkedReq) and req.extend_len == CHUNK
    _forward(cm, batch)
    assert prompt.chunked_req is not None and prompt.chunked_req.cached_len == CHUNK


def test_a_prompt_larger_than_the_whole_pool_is_refused_not_livelocked():
    """R-7: unsatisfiable at every pool state, so it must neither be admitted nor block.

    Admitting it would strand a chunked lane that pins its forwarded pages and can never
    reach a last chunk; leaving it at the head under strict FIFO would stall every fresh
    admit behind it forever. It is skipped instead, and the pass keeps making progress.
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
    cm, _tm, dm, pm = _build_managers(num_pages=160)

    chunked = _pending(uid=1, first_token=0, length=6 * CHUNK)
    pm.pending_list = [chunked]
    _forward(cm, pm.schedule_next_batch(CHUNK))
    assert chunked.chunked_req is not None

    _seat_a_decoding_neighbour(cm, dm, pm, length=136)
    fresh = _pending(uid=2, first_token=1_000, length=40)
    pm.pending_list = [fresh, chunked]  # the refused fresh admit sits AHEAD of the lane

    # ``fresh`` cannot be seated this pass: its 9-token share of the budget plus its own
    # 4-token decode does not fit in the 16 pages the decoding neighbour leaves, alongside
    # the 7 already reserved (3 for the neighbour's decode, 4 for the lane's). It CAN be
    # seated later, so this is a transient refusal and it keeps its place -- not the
    # permanent one the test above covers.
    assert cm.available_size == 16 and dm.inflight_tokens == 3
    assert fresh.input_len + fresh.output_len <= cm.max_size

    batch = pm.schedule_next_batch(18)
    assert batch is not None, "the pass stopped at the refusal instead of skipping it"
    assert [req.uid for req in batch.reqs] == [1], "the continuation behind it was dropped"
    assert fresh.chunked_req is None
    assert pm.pending_list[-1] is fresh  # refused, requeued, order preserved
    _forward(cm, batch)


def test_a_later_fresh_admit_does_not_overtake_an_earlier_refused_one():
    """R-2's fairness half: strict FIFO among fresh admits, no overtaking.

    The permissive rule (overtake when strictly cheaper) was measured on the replay and
    rejected: it cost 25% of the prefill tokens on the stage profile and tripled
    wait-to-first-chunk p95, because the refusals land on the long prompts and "later and
    cheaper" then means "every short prompt, forever".
    """
    cm, _tm, dm, pm = _build_managers(num_pages=160)

    chunked = _pending(uid=1, first_token=0, length=6 * CHUNK)
    pm.pending_list = [chunked]
    _forward(cm, pm.schedule_next_batch(CHUNK))

    _seat_a_decoding_neighbour(cm, dm, pm, length=136)
    big = _pending(uid=2, first_token=1_000, length=40)
    tiny = _pending(uid=3, first_token=2_000, length=2, max_tokens=1)
    pm.pending_list = [big, tiny, chunked]

    # ``tiny`` really is seatable -- 3 tokens against 16 available and 7 reserved -- so what
    # keeps it out is the FIFO rule and nothing else. ``big`` is not: its 6-token share of
    # the budget plus its 4-token decode does not fit alongside the same reservation.
    assert cm.available_size == 16 and dm.inflight_tokens == 3
    reserved = dm.inflight_tokens + chunked.output_len
    assert tiny.input_len + tiny.output_len + reserved <= cm.available_size

    batch = pm.schedule_next_batch(18)

    assert batch is not None
    seated = [req.uid for req in batch.reqs]
    assert 3 not in seated, "a cheaper fresh admit overtook one the pools had refused"
    assert seated == [1], "the continuation behind the refusal was dropped"
    # Both fresh requests are still queued, in their original order.
    assert [req.uid for req in pm.pending_list] == [1, 2, 3]
