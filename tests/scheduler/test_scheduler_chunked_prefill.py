"""Regression test for the chunked-prefill page-accounting bug.

An earlier gemma4 perf change added ``cache_manager.cache_req(req, finished=False)`` to the
``ChunkedReq`` branch of ``Scheduler._process_last_data`` -- caching every *intermediate*
chunk. Under overlap scheduling the next chunk is created (snapshotting the prior chunk's
``cache_handle``) BEFORE that cache_req runs, so each continuation carries a stale handle
whose ``cached_len`` is behind reality and ``cache_req`` re-frees the prior chunk's pages.
Any multi-chunk prefill (codex's large prompts) then crashes the scheduler at the next idle
with ``CacheManager integrity check failed``.

The fix reverts to mini-sglang's behavior: do NOT cache intermediate chunks; the whole
prompt is inserted once when the final (non-chunked) chunk is processed.
"""

from __future__ import annotations

import torch

CHUNK = 8
WIDTH = 64
MAX_RUNNING = 4
UID = 7


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build_managers(num_pages):
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


def _drive_chunked_prefill(cm, tm, pm, n_chunks):
    """Drive one chunked-prefill request through ``n_chunks``, faithfully replicating overlap
    ordering: each iteration schedules+forwards the NEXT chunk (reading the prior chunk's
    cache_handle) BEFORE the PREVIOUS chunk is cached. Intermediate chunks are NOT cached
    (like the real scheduler's ``continue``); the whole prompt is cached once when the final
    non-chunked chunk is processed."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq
    from freetoken.scheduler.utils import PendingReq

    prompt_len = n_chunks * CHUNK
    pm.pending_list = [PendingReq(uid=UID, input_ids=torch.arange(prompt_len, dtype=torch.int32),
                                  sampling_params=SamplingParams(max_tokens=4))]
    last_batch = None
    final_req = None
    while pm.runnable or last_batch is not None:
        batch = pm.schedule_next_batch(CHUNK)          # step 2: schedule next chunk
        if batch is not None:
            cm.allocate_paged(batch.reqs)              # _prepare_batch allocates pages
            for r in batch.reqs:
                r.complete_one()                       # forward advances cached_len
        if last_batch is not None:                     # step 3: process the PREVIOUS batch
            for r in last_batch.reqs:
                if not isinstance(r, ChunkedReq):
                    cm.cache_req(r, finished=False)    # final chunk cached once
                    final_req = r
        last_batch = batch
    cm.cache_req(final_req, finished=True)             # request finishes -> release
    tm.free(final_req.table_idx)


def test_multichunk_overlap_no_double_free():
    """The fix: not caching intermediate chunks keeps page accounting consistent across a
    multi-chunk prefill, and still caches the whole prompt (reusable) exactly once."""
    cm, tm, _dm, pm = _build_managers(num_pages=64)
    _drive_chunked_prefill(cm, tm, pm, n_chunks=4)

    cm.check_integrity()  # free_slots + cache == num_pages; no chunk freed twice
    si = cm.prefix_cache.size_info
    assert si.protected_size == 0  # request released -> no leaked locks / ref_count drift
    assert si.evictable_size == 4 * CHUNK  # whole prompt retained in the prefix cache
    assert len(cm.free_slots) == cm.num_pages - 4 * CHUNK


def test_radix_hit_admission_reports_nonzero_cached_tokens():
    """A second request sharing a cached prefix admits with the prefix-cache hit recorded."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, tm, _dm, pm = _build_managers(num_pages=64)
    _drive_chunked_prefill(cm, tm, pm, n_chunks=4)  # whole 32-token prompt now in the radix cache

    cached_len = 4 * CHUNK
    prompt_len = cached_len + CHUNK
    pm.pending_list = [
        PendingReq(uid=UID + 1, input_ids=torch.arange(prompt_len, dtype=torch.int32),
                   sampling_params=SamplingParams(max_tokens=4))
    ]
    batch = pm.schedule_next_batch(prompt_len)
    assert batch is not None
    assert batch.prompt_admissions == [(UID + 1, prompt_len, cached_len)]
    assert batch.log_cached_tokens == cached_len


def test_chunked_prompt_admission_reports_complete_length_once():
    """The first prepared chunk carries the full prompt usage; continuations carry none."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, _tm, _dm, pm = _build_managers(num_pages=64)
    prompt_len = 3 * CHUNK
    pm.pending_list = [
        PendingReq(
            uid=UID,
            input_ids=torch.arange(prompt_len, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=4),
        )
    ]

    admissions = []
    while pm.runnable:
        batch = pm.schedule_next_batch(CHUNK)
        assert batch is not None
        admissions.append(list(batch.prompt_admissions))
        cm.allocate_paged(batch.reqs)
        for req in batch.reqs:
            req.complete_one()

    assert admissions[0] == [(UID, prompt_len, 0)]
    assert all(items == [] for items in admissions[1:])


def test_batched_prefill_carries_each_new_prompt_admission():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, _dm, pm = _build_managers(num_pages=64)
    pm.pending_list = [
        PendingReq(1, torch.arange(3, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(5, dtype=torch.int32), SamplingParams(max_tokens=2)),
    ]
    batch = pm.schedule_next_batch(16)
    assert batch is not None
    assert batch.prompt_admissions == [(1, 3, 0), (2, 5, 0)]


def test_interleaved_prefill_gives_each_waiting_agent_a_lane():
    """Growable multi-agent mode fills one aggregate batch with fair prompt chunks."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, _dm, pm = _build_managers(num_pages=64)
    pm.interleave_chunks = True
    pm.pending_list = [
        PendingReq(1, torch.arange(24, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(24, dtype=torch.int32) + 100, SamplingParams(max_tokens=2)),
    ]

    batch = pm.schedule_next_batch(16)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [1, 2]
    assert [req.extend_len for req in batch.reqs] == [8, 8]
    assert [req.uid for req in pm.pending_list] == [1, 2]


def test_single_lane_prefill_rotates_long_prompts_without_grouping_them():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, _dm, pm = _build_managers(num_pages=128)
    pm.interleave_chunks = True
    pm.max_batch_seqs = 1
    pm.pending_list = [
        PendingReq(1, torch.arange(24, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(24, dtype=torch.int32) + 100, SamplingParams(max_tokens=2)),
    ]

    first = pm.schedule_next_batch(16)
    second = pm.schedule_next_batch(16)

    assert first is not None and [(req.uid, req.extend_len) for req in first.reqs] == [(1, 16)]
    assert second is not None and [(req.uid, req.extend_len) for req in second.reqs] == [(2, 16)]
    assert [req.uid for req in pm.pending_list] == [1, 2]


def test_auto_single_lane_groups_fresh_small_prompts():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, _dm, pm = _build_managers(num_pages=64)
    pm.interleave_chunks = True
    pm.max_batch_seqs = 1
    pm.small_prompt_group_tokens = 8
    pm.pending_list = [
        PendingReq(1, torch.arange(8, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(8, dtype=torch.int32) + 100, SamplingParams(max_tokens=2)),
    ]

    batch = pm.schedule_next_batch(16)

    assert batch is not None
    assert [(req.uid, req.extend_len) for req in batch.reqs] == [(1, 8), (2, 8)]


def test_auto_small_prompt_group_must_fit_one_prefill_budget():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, _dm, pm = _build_managers(num_pages=64)
    pm.interleave_chunks = True
    pm.max_batch_seqs = 1
    pm.small_prompt_group_tokens = 8
    pm.pending_list = [
        PendingReq(1, torch.arange(8, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(8, dtype=torch.int32) + 100, SamplingParams(max_tokens=2)),
    ]

    batch = pm.schedule_next_batch(12)

    assert batch is not None
    assert [(req.uid, req.extend_len) for req in batch.reqs] == [(1, 8)]


def test_prefill_sequence_limit_auto_scope():
    cases = [
        (None, 65_536, 4, ((12, 12),), 1),
        (None, 0, 4, ((12, 12),), 0),
        (None, 65_536, 1, ((12, 12),), 0),
        (None, 65_536, 4, None, 0),
        (0, 65_536, 4, ((12, 12),), 0),
        (2, 65_536, 4, ((12, 12),), 2),
    ]
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import _resolve_max_prefill_seqs

    for explicit, grow_step, max_running, gguf_types, expected in cases:
        config = SimpleNamespace(
            max_prefill_seqs=explicit,
            kv_grow_step_tokens=grow_step,
            max_running_req=max_running,
            model_config=SimpleNamespace(gguf_expert_types=gguf_types),
        )
        assert _resolve_max_prefill_seqs(config) == expected


def test_auto_small_prompt_group_tokens_uses_measured_ada_crossover():
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import _auto_small_prompt_group_tokens

    auto = SimpleNamespace(max_prefill_seqs=None)
    explicit = SimpleNamespace(max_prefill_seqs=1)

    assert _auto_small_prompt_group_tokens(auto, 1, (8, 9)) == 1536
    assert _auto_small_prompt_group_tokens(auto, 1, (12, 0)) == 1280
    assert _auto_small_prompt_group_tokens(explicit, 1, (8, 9)) == 0
    assert _auto_small_prompt_group_tokens(auto, 0, (8, 9)) == 0


def test_interleaved_prefill_does_not_queue_blocked_agent_before_active_lane():
    """An agent that cannot reserve KV must not head-of-line block an admitted continuation."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, _dm, pm = _build_managers(num_pages=32)
    pm.interleave_chunks = True
    pm.pending_list = [
        PendingReq(1, torch.arange(20, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(20, dtype=torch.int32) + 100, SamplingParams(max_tokens=20)),
    ]

    first = pm.schedule_next_batch(16)
    second = pm.schedule_next_batch(16)

    assert first is not None and [req.uid for req in first.reqs] == [1]
    assert second is not None and second.reqs[0].uid == 1


# ---------------------------------------------------------------------------
# The interleave chunk share: token_budget / lanes the pass will SEAT, not / queue depth.
# Soak report §R7 ticket 1 / §U8; bisect §4(c) attributes the queue-depth divisor to f3c3ac4.
# ---------------------------------------------------------------------------


def _pending(uid, length, max_tokens):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(
        uid,
        torch.arange(uid * 1000, uid * 1000 + length, dtype=torch.int32),
        SamplingParams(max_tokens=max_tokens),
    )


def test_chunk_share_divides_by_seatable_lanes_not_queue_depth():
    """A deep queue whose pools will seat ONE lane must give that lane the WHOLE budget.

    The regression this pins: ``chunk_limit = token_budget // waiting`` charged the queue
    depth, so eight queued 40-token prompts in a pool that can only admit one of them cut
    the chunk to an eighth of the budget. Live, that is a 118 K-token prompt advancing 512
    tokens per 8,192-token pass -- 61 % of the soak's stage prefill passes.
    """
    _cm, _tm, _dm, pm = _build_managers(num_pages=64)
    pm.interleave_chunks = True
    # 40 + 8 = 48 owed each: the whole-footprint admission gate seats exactly one in a
    # 64-page pool, and refuses (and so ends the pass at) the second.
    pm.pending_list = [_pending(uid, 40, 8) for uid in range(1, 9)]

    batch = pm.schedule_next_batch(16)

    assert batch is not None
    assert [(req.uid, req.extend_len) for req in batch.reqs] == [(1, 16)]
    # The queue-depth divisor would have handed this lane 16 // 8 == 2 tokens.
    assert batch.reqs[0].extend_len == 16


def test_chunk_share_is_equal_across_equally_seatable_lanes():
    """Four lanes the pass can all seat split the budget four ways, in FIFO order."""
    _cm, _tm, _dm, pm = _build_managers(num_pages=256)
    pm.interleave_chunks = True
    pm.pending_list = [_pending(uid, 24, 2) for uid in range(1, 5)]

    batch = pm.schedule_next_batch(16)

    assert batch is not None
    assert [(req.uid, req.extend_len) for req in batch.reqs] == [
        (1, 4), (2, 4), (3, 4), (4, 4)
    ]


def test_chunk_share_redistributes_when_a_lane_takes_less_than_its_share():
    """A lane that cannot use its whole share hands the rest to the lanes behind it.

    The share is recomputed per iteration against the seats still to be filled, so the
    budget a short remainder leaves on the table is not lost.
    """
    _cm, _tm, _dm, pm = _build_managers(num_pages=256)
    pm.interleave_chunks = True
    pm.pending_list = [_pending(1, 3, 2), _pending(2, 40, 2), _pending(3, 40, 2)]

    batch = pm.schedule_next_batch(18)

    assert batch is not None
    # Shares are 6/6/6; lane 1 needs only 3, so lanes 2 and 3 split the remaining 15.
    assert [(req.uid, req.extend_len) for req in batch.reqs] == [(1, 3), (2, 7), (3, 8)]
    assert sum(req.extend_len for req in batch.reqs) == 18


class _DecodeStub:
    """Minimal stand-in for a running decode: ``DecodeManager.inflight_tokens`` reads
    ``remain_len`` and nothing else, and the set it lives in needs identity hashing."""

    def __init__(self, remain_len: int) -> None:
        self.remain_len = remain_len


def test_chunk_share_stays_safe_when_pages_run_out_mid_pass():
    """The seat scan charges one page per lane, so the pass can seat fewer than it counted.

    While the finishability invariant holds this gap cannot bite: ``owed <= available_size``
    bounds the unforwarded tails of every in-flight prefill PLUS the decode growth by the
    pool, so the chunks the seated lanes ask for always fit and the count is exact. The
    interesting case is a pool that has been over-promised -- the state
    ``_check_finishability`` exists to catch, and the one soak report T5 deadlocked in -- so
    that is what this builds, with the invariant deliberately off.

    The property: the pass spends the pages the pool can still back, down to the last one,
    and then DEFERS. Nothing lost (every backable page is used), nothing over-committed
    (``committed_pages_required`` stays inside the pool), and the lane the pages ran out on
    keeps its place in the queue instead of being admitted with a chunk that cannot be
    allocated.
    """
    _cm, _tm, dm, pm = _build_managers(num_pages=96)
    pm.interleave_chunks = True
    pm.pending_list = [_pending(uid, 30, 1) for uid in range(1, 4)]

    first = pm.schedule_next_batch(24)
    assert first is not None
    assert [(req.uid, req.extend_len) for req in first.reqs] == [(1, 8), (2, 8), (3, 8)]
    # Forward it for real, so the pool actually loses the pages.
    assert _cm.committed_pages_required(first.reqs) <= _cm.num_pages
    _cm.allocate_paged(first.reqs)
    for req in first.reqs:
        req.complete_one()
    free_before = _cm.available_size

    # Over-promise the pool: 60 tokens of decode growth on top of three continuations that
    # between them still owe 66 + 3, against 72 obtainable tokens.
    dm.running_reqs.add(_DecodeStub(60))
    backable = free_before - dm.inflight_tokens
    assert 0 < backable < free_before

    # A budget far larger than the pages left: the scan counts all three lanes (one page
    # each), the real pass runs out part-way through.
    second = pm.schedule_next_batch(4 * free_before)

    assert second is not None
    forwarded = sum(req.extend_len for req in second.reqs)
    # Nothing over-committed: the pass never promises pages the pool cannot allocate.
    assert _cm.committed_pages_required(second.reqs) <= _cm.num_pages
    # Nothing lost either: every page the pool could still back this pass was spent.
    assert forwarded == backable
    # The lane the pages ran out on is deferred with its place kept, not dropped.
    assert {req.uid for req in pm.pending_list} == {1, 2, 3}
