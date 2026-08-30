from freetoken.scheduler.scheduler import _elastic_target_capacity


def test_elastic_capacity_tracks_demand_between_bounds():
    assert _elastic_target_capacity(2, 4, 0) == 2
    assert _elastic_target_capacity(2, 4, 2) == 2
    assert _elastic_target_capacity(2, 4, 3) == 3
    assert _elastic_target_capacity(2, 4, 4) == 4


def test_elastic_capacity_clamps_excess_demand():
    assert _elastic_target_capacity(4, 8, 99) == 8
