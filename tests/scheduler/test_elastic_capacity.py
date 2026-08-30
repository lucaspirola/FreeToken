from types import SimpleNamespace

import freetoken.scheduler.scheduler as scheduler_module
from freetoken.scheduler.scheduler import Scheduler, _elastic_target_capacity


def test_elastic_capacity_tracks_demand_between_bounds():
    assert _elastic_target_capacity(2, 4, 0) == 2
    assert _elastic_target_capacity(2, 4, 2) == 2
    assert _elastic_target_capacity(2, 4, 3) == 3
    assert _elastic_target_capacity(2, 4, 4) == 4


def test_elastic_capacity_clamps_excess_demand():
    assert _elastic_target_capacity(4, 8, 99) == 8


def test_intermediate_shrink_waits_for_stable_demand(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        elastic_initial_requests=2, max_running_req=4
    )
    scheduler._elastic_capacity = 4
    scheduler._elastic_resize_pending = False
    scheduler._elastic_shrink_candidate = None
    scheduler._elastic_demand = lambda: 3

    now = [10.0]
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: now[0])
    scheduler._maybe_resize_elastic_capacity()
    assert scheduler._elastic_shrink_candidate == (3, 12.0)
    assert scheduler._elastic_resize_pending is True

    now[0] = 11.9
    scheduler._maybe_resize_elastic_capacity()
    assert scheduler._elastic_shrink_candidate == (3, 12.0)
