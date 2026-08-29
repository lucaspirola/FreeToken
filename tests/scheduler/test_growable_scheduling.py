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
    scheduler.config = SimpleNamespace(kv_grow_step_tokens=131_072)
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
