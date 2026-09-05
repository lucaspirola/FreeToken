from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.status import SchedulerStatusReporter, _usage_ratio


def _reporter(interval=40):
    logs: list[str] = []
    clock = {"t": 0.0}
    rep = SchedulerStatusReporter(
        log=logs.append,
        clock=lambda: clock["t"],
        decode_log_interval=interval,
    )
    return rep, logs, clock


def _req(extend, cached):
    return SimpleNamespace(extend_len=extend, cached_len=cached)


def _prefill_batch(new_tokens, cached_tokens, n_seqs):
    # The reporter must read the schedule-time snapshot (log_new_tokens/log_cached_tokens),
    # NOT the live reqs: by report time forward's complete_one() has advanced them to
    # decode state. Live reqs here carry deliberately-wrong values to prove that.
    reqs = [_req(extend=1, cached=10_000) for _ in range(n_seqs)]
    return SimpleNamespace(
        is_prefill=True, is_decode=False, reqs=reqs,
        log_new_tokens=new_tokens, log_cached_tokens=cached_tokens,
    )


def _decode_batch(n):
    return SimpleNamespace(is_prefill=False, is_decode=True, reqs=[_req(1, 0) for _ in range(n)])


def test_prefill_line_reports_tokens_and_throughput():
    rep, logs, clock = _reporter()
    clock["t"] = 0.5  # 30 new tokens over 0.5s -> 60 tok/s
    rep.report_batch(
        _prefill_batch(new_tokens=30, cached_tokens=12, n_seqs=2),
        running_reqs=2, queue_reqs=1, kv_used_pages=50, kv_total_pages=200, page_size=16,
    )
    assert len(logs) == 1
    line = logs[0]
    assert "#new-seq: 2" in line
    assert "#new-token: 30" in line  # snapshot, not the live reqs' extend_len (1 each)
    assert "#cached-token: 12" in line  # snapshot, not the live reqs' cached_len (10000 each)
    assert "token usage: 0.25" in line
    assert "#running-req: 2" in line
    assert "#queue-req: 1" in line
    assert "input throughput (token/s): 60.00" in line
    assert "60.00 average" in line


def test_prefill_line_carries_the_lane_fields_when_the_scheduler_passes_them():
    """``#seatable-lane`` (the interleave share's divisor) and ``#chunked-inflight`` were
    previously only inferable from this line -- the §R7 starvation signature had to be
    reconstructed from `#new-seq`/`#new-token`/`#queue-req`.

    The `benchmarks/switchyard_soak/analyze.py` PRE regex matches them optionally, so this
    also pins the field ORDER and spelling it parses.
    """
    rep, logs, clock = _reporter()
    clock["t"] = 0.5
    rep.report_batch(
        _prefill_batch(new_tokens=30, cached_tokens=12, n_seqs=2),
        running_reqs=2, queue_reqs=5, kv_used_pages=50, kv_total_pages=200, page_size=16,
        seatable_lanes=3, chunked_inflight=2,
    )
    assert "#queue-req: 5, #seatable-lane: 3, #chunked-inflight: 2, input throughput" in logs[0]


def test_prefill_line_is_unchanged_when_the_lane_fields_are_not_supplied():
    """The low-level loop tests and any caller that predates the fields must keep the old
    line byte-for-byte, so an older analyze.py still reads a new log."""
    rep, logs, clock = _reporter()
    clock["t"] = 0.5
    rep.report_batch(
        _prefill_batch(new_tokens=30, cached_tokens=12, n_seqs=2),
        running_reqs=2, queue_reqs=5, kv_used_pages=50, kv_total_pages=200, page_size=16,
    )
    assert "#queue-req: 5, input throughput" in logs[0]
    assert "#seatable-lane" not in logs[0]


def test_prefill_line_keeps_cumulative_average_for_same_request():
    rep, logs, clock = _reporter()
    batch = _prefill_batch(new_tokens=30, cached_tokens=0, n_seqs=1)
    batch.reqs[0].uid = 7
    clock["t"] = 0.5
    rep.report_batch(
        batch, running_reqs=0, queue_reqs=1,
        kv_used_pages=30, kv_total_pages=100, page_size=1,
    )
    continuation = _prefill_batch(new_tokens=30, cached_tokens=0, n_seqs=1)
    continuation.reqs[0].uid = 7
    clock["t"] = 2.0  # instant 20; cumulative 60 / 2 = 30 tok/s
    rep.report_batch(
        continuation, running_reqs=0, queue_reqs=1,
        kv_used_pages=60, kv_total_pages=100, page_size=1,
    )
    assert "20.00 instant, 30.00 average" in logs[-1]


def test_mamba_slots_reported_only_when_provided():
    rep, logs, clock = _reporter()
    clock["t"] = 1.0
    # non-hybrid (mamba_slots=None): no mamba field
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
    )
    assert "mamba" not in logs[-1]
    # hybrid: #mamba-slot: used/total and usage ratio
    clock["t"] = 2.0
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        mamba_slots=(37, 256),
    )
    assert "#mamba-slot: 37/256" in logs[-1]
    assert "mamba usage: 0.14" in logs[-1]


def test_swa_tokens_reported_only_when_provided():
    rep, logs, clock = _reporter(interval=1)
    clock["t"] = 1.0
    # non-SWA (swa_tokens=None): no swa field
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
    )
    assert "swa" not in logs[-1]
    # SWA: #swa-token: used/total and usage ratio, on both prefill and decode lines
    clock["t"] = 2.0
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        swa_tokens=(8448, 76800),
    )
    assert "#swa-token: 8448/76800" in logs[-1]
    assert "swa usage: 0.11" in logs[-1]
    clock["t"] = 3.0
    rep.report_batch(_decode_batch(1), running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1,
                     swa_tokens=(8448, 76800))
    assert "#swa-token: 8448/76800" in logs[-1]
    assert "swa usage: 0.11" in logs[-1]


def test_decode_lines_are_throttled_to_every_nth_forward():
    rep, logs, clock = _reporter(interval=3)
    for i, t in enumerate((1.0, 1.5), start=1):
        clock["t"] = t
        rep.report_batch(_decode_batch(2), running_reqs=2, queue_reqs=0,
                         kv_used_pages=60, kv_total_pages=200, page_size=16)
        assert logs == [], f"should not log before the interval (forward {i})"
    clock["t"] = 2.0  # 3rd forward -> log; 6 tokens over 2.0s gap -> 3 tok/s
    rep.report_batch(_decode_batch(2), running_reqs=2, queue_reqs=4,
                     kv_used_pages=62, kv_total_pages=200, page_size=16)
    assert len(logs) == 1
    line = logs[0]
    assert "#running-req: 2" in line
    assert "#queue-req: 4" in line
    assert "#token: 992" in line  # 62 pages * 16
    assert "token usage: 0.31" in line
    assert "gen throughput (token/s): 3.00" in line


def test_decode_counter_resets_each_interval():
    rep, logs, clock = _reporter(interval=2)
    clock["t"] = 1.0
    rep.report_batch(_decode_batch(5), running_reqs=5, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    clock["t"] = 2.0  # first emission: 10 tokens over 2.0s -> 5 tok/s
    rep.report_batch(_decode_batch(5), running_reqs=5, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    assert "gen throughput (token/s): 5.00" in logs[-1]
    # next window is measured from the previous emission, with a reset token count
    clock["t"] = 3.0
    rep.report_batch(_decode_batch(3), running_reqs=3, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    clock["t"] = 4.0  # 6 tokens over (4.0-2.0)=2.0s -> 3 tok/s
    rep.report_batch(_decode_batch(3), running_reqs=3, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    assert "gen throughput (token/s): 3.00" in logs[-1]


def test_zero_gap_and_zero_total_are_guarded():
    rep, logs, clock = _reporter(interval=1)
    # gap == 0 (clock unchanged since construction) and total == 0 must not raise
    rep.report_batch(_decode_batch(4), running_reqs=4, queue_reqs=0,
                     kv_used_pages=0, kv_total_pages=0, page_size=1)
    line = logs[-1]
    assert "gen throughput (token/s): 0.00" in line
    assert "token usage: 0.00" in line
    assert "#token: 0" in line  # owned-KV (dsv4) reports 0/0 pages


def test_interval_is_clamped_to_at_least_one():
    rep, _, _ = _reporter(interval=0)
    assert rep.decode_log_interval == 1
    rep_neg, _, _ = _reporter(interval=-5)
    assert rep_neg.decode_log_interval == 1


def test_usage_ratio_guard():
    assert _usage_ratio(0, 0) == 0.0
    assert _usage_ratio(5, 0) == 0.0
    assert _usage_ratio(5, 10) == 0.5
