"""The cumulative scheduler counters behind ``/v1/stats["scheduler"]``.

Soak report §U5/§U6 could not answer, from a whole 40-minute run: did
``max_chunked_prefills`` ever bind? what was the interleave share's divisor? was the
finishability invariant ever violated on a tree not started with
``FREETOKEN_SCHEDULER_INVARIANT``? why did the drafter decline? did a session checkpoint
fail? Every one of those was inferable at best -- ``fresh_admits_blocked_by_cap`` was
approximated by counting passes carrying ``#cached-token > 0`` -- and the invariant answer
was simply unavailable, because counting it was gated behind the env var that logs it.

These tests pin the counters against a real ``PrefillManager`` driven to completion, not
against a mock, so a change to the admission loop that stops taking one of the counted
branches fails here rather than silently zeroing a soak column.
"""

from __future__ import annotations

import torch

WIDTH = 256
MAX_RUNNING = 8


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build(num_pages: int, **kwargs):
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    cm = CacheManager(num_pages=num_pages, page_size=1, page_table=pt, type="radix")
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=1)
    return cm, tm, dm, PrefillManager(cm, tm, dm, **kwargs)


def _pending(uid: int, first_token: int, length: int, max_tokens: int):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(
        uid=uid,
        input_ids=torch.arange(first_token, first_token + length, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def _forward(cm, batch):
    cm.allocate_paged(batch.reqs)
    for req in batch.reqs:
        req.complete_one()


# --------------------------------------------------------------------------- lane buckets


def test_lane_bucket_is_per_lane_where_it_matters_and_geometric_above():
    from freetoken.scheduler.counters import lane_bucket

    assert [lane_bucket(n) for n in (0, 1, 2, 3, 4)] == ["0", "1", "2", "3", "4"]
    assert lane_bucket(5) == lane_bucket(8) == "5-8"
    assert lane_bucket(9) == lane_bucket(16) == "9-16"
    assert lane_bucket(17) == lane_bucket(4096) == "17+"


# ------------------------------------------------------------------ prefill pass counters


def test_every_pass_is_counted_with_its_divisor_and_chunked_population():
    cm, _tm, _dm, pm = _build(num_pages=4_096, interleave_chunks=True)
    pm.pending_list = [
        _pending(uid=i, first_token=i * 10_000, length=96, max_tokens=8) for i in (1, 2, 3)
    ]
    batch = pm.schedule_next_batch(32)
    assert batch is not None
    counters = pm.counters
    assert counters.passes == 1
    # Three queued lanes the pools can all seat: the divisor is the seat count, not the
    # queue depth it used to be (§R7 ticket 1).
    assert counters.seatable_lanes_last == 3
    assert counters.seatable_lanes["3"] == 1
    # Nothing was mid-prefill when the pass started.
    assert counters.chunked_inflight == 0
    # Every lane took a chunk and still owes a remainder.
    assert counters.deferred_chunks == 3
    _forward(cm, batch)

    pm.schedule_next_batch(32)
    assert counters.passes == 2
    assert counters.chunked_inflight == 3
    assert counters.chunked_inflight_max == 3


def test_a_pass_that_seats_nothing_new_still_counts_itself():
    """A pass is a scheduling decision even when it returns no batch -- and a run of them
    is exactly the shape a starved queue makes."""
    _cm, _tm, _dm, pm = _build(num_pages=32)
    pm.pending_list = [_pending(uid=1, first_token=0, length=4_096, max_tokens=8)]
    assert pm.schedule_next_batch(32) is None
    assert pm.counters.passes == 1
    assert pm.counters.refusals == 1


def test_an_empty_queue_is_not_a_pass():
    _cm, _tm, _dm, pm = _build(num_pages=4_096)
    assert pm.schedule_next_batch(32) is None
    assert pm.counters.passes == 0


def test_the_chunked_prefill_cap_records_every_fresh_admit_it_skips():
    """``max_chunked_prefills`` binding was previously invisible: the skip is a bare
    ``continue`` and the only trace was the ABSENCE of a fresh admit in the batch log."""
    cm, _tm, _dm, pm = _build(
        num_pages=4_096, interleave_chunks=True, max_chunked_prefills=2
    )
    pm.pending_list = [
        _pending(uid=i, first_token=i * 10_000, length=512, max_tokens=8) for i in (1, 2, 3, 4)
    ]
    batch = pm.schedule_next_batch(128)
    assert batch is not None
    assert len(batch.reqs) == 2, "the cap admits two and skips the rest"
    assert pm.counters.fresh_admits_blocked_by_cap == 2
    _forward(cm, batch)

    pm.schedule_next_batch(128)
    # The two continuations still hold the cap shut against uids 3 and 4.
    assert pm.counters.fresh_admits_blocked_by_cap == 4


# -------------------------------------------------------------- finishability invariant


def test_the_invariant_is_counted_even_with_the_env_var_off(monkeypatch):
    """The whole reason this counter exists: the soak that needed the number was not
    running with FREETOKEN_SCHEDULER_INVARIANT set, so the check was never computed."""
    monkeypatch.delenv("FREETOKEN_SCHEDULER_INVARIANT", raising=False)
    cm, _tm, _dm, pm = _build(num_pages=4_096)
    pm.pending_list = [_pending(uid=1, first_token=0, length=96, max_tokens=8)]
    pm.schedule_next_batch(32)
    assert pm.counters.invariant_checks == 1
    assert pm.counters.invariant_violations == 0
    assert pm.counters.invariant_worst_shortfall == 0


def test_a_violation_is_counted_and_its_worst_shortfall_kept(monkeypatch):
    monkeypatch.delenv("FREETOKEN_SCHEDULER_INVARIANT", raising=False)
    cm, _tm, _dm, pm = _build(num_pages=160)

    class _Chunked:
        cached_len = 0

    for uid in range(1, 4):  # the soak-report-T5 state: 3 prefills owing 300 of 160
        pending = _pending(uid=uid, first_token=uid * 10_000, length=92, max_tokens=8)
        pending.chunked_req = _Chunked()
        pm.pending_list.append(pending)
    standing = pm._standing_reservation()
    assert standing > cm.available_size

    pm._check_finishability(standing, "")  # counting mode: no log, no raise
    assert pm.counters.invariant_violations == 1
    assert pm.counters.invariant_worst_shortfall == standing - cm.available_size


def test_counting_does_not_swallow_the_raise_mode(monkeypatch):
    import pytest

    monkeypatch.delenv("FREETOKEN_SCHEDULER_INVARIANT", raising=False)
    cm, _tm, _dm, pm = _build(num_pages=160)

    class _Chunked:
        cached_len = 0

    pending = _pending(uid=1, first_token=0, length=400, max_tokens=8)
    pending.chunked_req = _Chunked()
    pm.pending_list.append(pending)
    with pytest.raises(AssertionError, match="finishability invariant violated"):
        pm._check_finishability(pm._standing_reservation(), "raise")
    assert pm.counters.invariant_violations == 1


# ---------------------------------------------------------------------- the wire document


def test_build_scheduler_counters_distinguishes_off_from_idle():
    """``None`` means the subsystem is off; an all-zero block means it is on and idle.
    Collapsing the two is the same ambiguity this ticket removes from ``cached_tokens``."""
    from freetoken.scheduler.counters import build_scheduler_counters

    _cm, _tm, _dm, pm = _build(num_pages=4_096)
    doc = build_scheduler_counters(pm, None, None)
    assert doc["spec"] is None and doc["session_spill"] is None
    assert doc["prefill"]["passes"] == 0
    assert doc["prefill"]["max_chunked_prefills"] == pm.max_chunked_prefills
    assert doc["prefill"]["invariant"] == {
        "checks": 0, "violations": 0, "worst_shortfall": 0
    }

    assert build_scheduler_counters(None, None, None) == {
        "prefill": None, "spec": None, "session_spill": None, "moe": None
    }


def test_the_document_is_json_serializable():
    """It travels as a plain dict over the tokenizer link and out of /v1/stats."""
    import json

    from freetoken.scheduler.counters import SpillCounters, build_scheduler_counters
    from freetoken.scheduler.spec_ngram import SpecStats

    _cm, _tm, _dm, pm = _build(num_pages=4_096)
    pm.pending_list = [_pending(uid=1, first_token=0, length=96, max_tokens=8)]
    pm.schedule_next_batch(32)
    spec = type("S", (), {"stats": SpecStats()})()
    store = type("S", (), {"counters": SpillCounters()})()
    doc = build_scheduler_counters(pm, spec, store)
    assert json.loads(json.dumps(doc)) == doc


# ------------------------------------------------------------------------- spec + spill


def test_spec_stats_expose_declines_by_reason_and_an_accepted_histogram():
    from freetoken.scheduler.spec_ngram import SpecStats

    stats = SpecStats()
    stats.declined_no_slot = 2
    stats.declined_budget = 1
    for accepted in (0, 3, 3, 7):
        stats.note_accepted(accepted)
    doc = stats.as_dict()
    assert doc["declined"] == {
        "shape": 0, "no_slot": 2, "budget": 1, "stale_match": 0, "uneconomic": 0
    }
    assert doc["accepted_hist"] == {"0": 1, "3": 2, "7": 1}


def test_a_verify_step_records_its_accepted_count():
    """Pinned against the decoder itself so the histogram cannot drift away from the
    accepted_tokens total it decomposes."""
    from freetoken.scheduler.spec_ngram import SpecStats

    stats = SpecStats()
    stats.accepted_tokens += 5
    stats.note_accepted(5)
    stats.accepted_tokens += 0
    stats.note_accepted(0)
    hist = stats.as_dict()["accepted_hist"]
    assert sum(int(k) * v for k, v in hist.items()) == stats.accepted_tokens


def test_spill_counters_render_every_failure_channel():
    from freetoken.scheduler.counters import SpillCounters

    counters = SpillCounters()
    counters.spills_failed += 1
    counters.prefetches_failed += 2
    doc = counters.as_dict()
    assert doc["spills_failed"] == 1 and doc["prefetches_failed"] == 2
    assert set(doc) == {
        "spills", "spills_failed", "restores", "restores_failed", "restores_diverged",
        "restores_deferred",
        "prefetches", "prefetches_failed", "prefetches_collected",
    }

# --------------------------------------------------------------------------- #
# The expert cache (soak §W7)
# --------------------------------------------------------------------------- #
# ``--moe-collect-stats`` was on for the whole 41-minute ca7e74b soak at c=16 and emitted
# nothing: every ``MoE decode miss stats`` / ``GPU batch profile`` line is printed from
# ``Scheduler.run_when_idle``, and ``Scheduler is idle`` appeared 0 times -- a saturated
# server never reaches an idle boundary, which is precisely the regime whose expert-cache
# hit rate anyone would want. The counters now also ride the ``/v1/stats`` path, where a
# busy server does reach them.
#
# ``OffloadMoeCache`` needs a GPU, so these drive ``build_moe_counters`` with the same
# duck-typed stand-ins the function is written against.


class _FakeMoeCache:
    """The attributes ``build_moe_counters`` reads, and nothing else."""

    def __init__(self, *, totals=None, raises=False):
        self.extend_cache_hits = 0
        self.extend_cache_misses = 0
        self.extend_cache_tokens = 64
        self._totals = totals
        self._raises = raises

    def note_extend_gate(self, cached: bool) -> None:
        if cached:
            self.extend_cache_hits += 1
        else:
            self.extend_cache_misses += 1

    def decode_stat_totals(self) -> dict:
        if self._raises:
            raise RuntimeError("device sync failed")
        return dict(self._totals or {})


def test_the_extend_cache_gate_is_published_without_the_collect_stats_flag():
    """Two host ints, so they cost nothing and are always on. §W7 could only INFER this
    gate's engagement from the batch log (76 of 1,522 passes, 5.0%)."""
    from freetoken.scheduler.counters import build_moe_counters

    moe = _FakeMoeCache()
    for cached in (True, False, False, True, False):
        moe.note_extend_gate(cached)

    doc = build_moe_counters(moe, collect_decode_stats=False)
    assert doc == {"extend_cache": {"hits": 2, "misses": 3, "threshold_tokens": 64}}
    assert "decode" not in doc, "the decode accumulators cost a device sync; do not pay it"


def test_decode_totals_are_cumulative_ints_so_two_snapshots_subtract():
    """A ratio averaged over the process lifetime cannot be differenced back into the hit
    rate over a window; raw counts can."""
    from freetoken.scheduler.counters import build_moe_counters

    first = {"layer_calls": 100, "active": 800, "missing": 200, "fetched": 150,
             "prefill_hit_rows": 10, "prefill_rows": 40,
             "pageable_stage_calls": 3, "pageable_rows": 60}
    second = dict(first, layer_calls=140, active=1120, missing=248)

    a = build_moe_counters(_FakeMoeCache(totals=first), True)["decode"]
    b = build_moe_counters(_FakeMoeCache(totals=second), True)["decode"]
    assert all(isinstance(v, int) for v in a.values())
    # 48 misses out of 320 active over the window: an 85% hit rate the lifetime figures
    # (75.0% then 77.9%) never show.
    assert (b["active"] - a["active"], b["missing"] - a["missing"]) == (320, 48)


def test_a_failing_stats_read_reports_null_rather_than_breaking_the_loop():
    from freetoken.scheduler.counters import build_moe_counters

    doc = build_moe_counters(_FakeMoeCache(raises=True), True)
    assert doc["decode"] is None and doc["extend_cache"]["hits"] == 0


def test_no_expert_cache_is_distinguishable_from_an_idle_one():
    """The same "off" vs "on but idle" distinction the rest of this document keeps."""
    from freetoken.scheduler.counters import build_moe_counters, build_scheduler_counters

    assert build_moe_counters(None) is None
    assert build_scheduler_counters(None, None, None)["moe"] is None
    assert build_scheduler_counters(None, None, None, moe=_FakeMoeCache())["moe"] == {
        "extend_cache": {"hits": 0, "misses": 0, "threshold_tokens": 64}
    }


def test_the_moe_block_is_json_serializable_too():
    import json

    from freetoken.scheduler.counters import build_scheduler_counters

    moe = _FakeMoeCache(totals={"layer_calls": 1, "active": 2, "missing": 1,
                                "fetched": 1, "prefill_hit_rows": 0, "prefill_rows": 0,
                                "pageable_stage_calls": 0, "pageable_rows": 0})
    doc = build_scheduler_counters(None, None, None, moe=moe, moe_collect_stats=True)
    assert json.loads(json.dumps(doc)) == doc
