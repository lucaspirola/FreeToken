from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.scheduler import Scheduler


class _Manager:
    def __init__(self, label: str):
        self.label = label
        self.runnable = True
        self.calls = 0

    def schedule_next_batch(self, *_args):
        self.calls += 1
        return self.label


def test_growable_scheduler_bounds_decode_starvation_during_helper_prefill():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(kv_grow_step_tokens=131_072, adaptive_scheduler=False)
    scheduler.prefill_budget = 8192
    scheduler.prefill_manager = _Manager("prefill")
    scheduler.decode_manager = _Manager("decode")
    scheduler._growable_decode_burst = 4
    scheduler._growable_decode_steps = 0
    scheduler._prepare_batch = lambda batch: batch
    scheduler._report_prompt_admissions = lambda _batch: None

    selected = [scheduler._schedule_next_batch() for _ in range(10)]

    assert selected == [
        "decode", "decode", "decode", "decode", "prefill",
        "decode", "decode", "decode", "decode", "prefill",
    ]


def _adaptive_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        kv_grow_step_tokens=131_072, adaptive_scheduler=True, page_size=16
    )
    scheduler.prefill_budget = 8192
    scheduler.prefill_manager = _Manager("prefill")
    scheduler.prefill_manager.pending_list = [object(), object()]
    scheduler.decode_manager = _Manager("decode")
    scheduler._growable_decode_burst = 32
    scheduler._growable_decode_steps = 0
    scheduler._scheduler_prefill_tps_ewma = None
    scheduler._scheduler_prefill_key = None
    scheduler._scheduler_decode_seconds_ewma = None
    scheduler._scheduler_prefill_slice_seconds = 8.0
    scheduler._scheduler_decode_slice_seconds = 0.25
    scheduler._scheduler_min_prefill_tokens_per_lane = 2048
    scheduler._prepare_batch = lambda batch: batch
    scheduler._report_prompt_admissions = lambda _batch: None
    return scheduler


def test_adaptive_prefill_budget_targets_time_without_starving_lanes():
    scheduler = _adaptive_scheduler()
    scheduler._scheduler_prefill_tps_ewma = 600.0
    assert scheduler._adaptive_prefill_budget() == 4800

    scheduler._scheduler_prefill_tps_ewma = 100.0
    assert scheduler._adaptive_prefill_budget() == 4096

    scheduler.prefill_manager.pending_list.extend([object(), object()])
    assert scheduler._adaptive_prefill_budget() == 8192


def test_adaptive_decode_burst_targets_elapsed_time_and_is_bounded():
    scheduler = _adaptive_scheduler()
    scheduler._scheduler_decode_seconds_ewma = 0.01
    assert scheduler._adaptive_decode_burst() == 25
    scheduler._scheduler_decode_seconds_ewma = 0.001
    assert scheduler._adaptive_decode_burst() == 64
    scheduler._scheduler_decode_seconds_ewma = 0.1
    assert scheduler._adaptive_decode_burst() == 8


def test_forward_observations_update_phase_ewmas(monkeypatch):
    scheduler = _adaptive_scheduler()
    times = iter((12.0, 20.0, 20.01))
    monkeypatch.setattr("freetoken.scheduler.scheduler.time.perf_counter", lambda: next(times))

    prefill = SimpleNamespace(
        scheduler_started_at=10.0,
        is_prefill=True,
        is_decode=False,
        log_new_tokens=1000,
        reqs=[SimpleNamespace(uid=1)],
    )
    scheduler._observe_scheduler_batch(prefill)
    assert scheduler._scheduler_prefill_tps_ewma == 500.0

    # A different lane must not inherit a short prompt's much higher throughput.
    other = SimpleNamespace(
        scheduler_started_at=10.0,
        is_prefill=True,
        is_decode=False,
        log_new_tokens=500,
        reqs=[SimpleNamespace(uid=99)],
    )
    scheduler._scheduler_prefill_key = (1,)
    scheduler._observe_scheduler_batch(other)
    assert scheduler._scheduler_prefill_tps_ewma == 50.0

    decode = SimpleNamespace(
        scheduler_started_at=19.99,
        is_prefill=False,
        is_decode=True,
    )
    scheduler._observe_scheduler_batch(decode)
    assert abs(scheduler._scheduler_decode_seconds_ewma - 0.02) < 1e-9
