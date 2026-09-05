from freetoken.engine.engine import _DENSE_GRAPH_BS, _elastic_graph_batch_sizes


def test_three_agent_elastic_tier_has_an_exact_graph():
    assert _elastic_graph_batch_sizes(2) == [1, 2]
    assert _elastic_graph_batch_sizes(3) == [1, 2, 3]
    assert _elastic_graph_batch_sizes(4) == [1, 2, 3, 4]


def test_full_width_tier_is_captured_not_padded_off_the_graph():
    """The 16-lane Switchyard profile's own capacity must have a graph.

    ``GraphRunner.can_use_cuda_graph`` gates on ``max(cuda_graph_bs)``, so a size the set
    never reaches decodes eagerly. Before 2026-09-05 the set stopped at 8 and every
    9-16-lane decode batch on ``--max-running-requests 16 --elastic-initial-requests 4``
    ran off the graph (73.5 % of the 13af13d soak's decode batches, 421 of 427 of which
    were taken at elastic capacity 16).
    """
    assert _elastic_graph_batch_sizes(16) == list(range(1, 17))
    assert max(_elastic_graph_batch_sizes(32)) == 32


def test_every_tier_capacity_has_its_own_graph():
    """No capacity may exceed the largest captured size, for any tier in range."""
    for capacity in range(1, 129):
        sizes = _elastic_graph_batch_sizes(capacity)
        assert sizes, capacity
        assert max(sizes) == capacity, (capacity, sizes)
        assert sizes == sorted(set(sizes)), (capacity, sizes)


def test_no_batch_in_the_common_range_ever_pads():
    """Dense to _DENSE_GRAPH_BS: padding is not free on an offload-MoE model.

    A padded row carries a hidden state, so it routes its own top-k experts and adds rows
    to the expert GEMV. Measured on Nemotron 3.5 Lightning at 12 lanes in a 16-request
    pool: eager 82.2 ms/step, padded up to a bs-16 graph 88.0 ms/step (-6.7 %). A sparse
    ladder is worse than no graph at all for every size that has to pad.
    """
    for capacity in range(1, _DENSE_GRAPH_BS + 1):
        assert _elastic_graph_batch_sizes(capacity) == list(range(1, capacity + 1))
    sizes = _elastic_graph_batch_sizes(_DENSE_GRAPH_BS)
    for batch in range(1, _DENSE_GRAPH_BS + 1):
        assert next(bs for bs in sizes if bs >= batch) == batch, batch


def test_graph_set_stays_sparse_above_the_dense_range():
    """Graph memory must not grow linearly with a large ceiling (the original intent)."""
    assert _elastic_graph_batch_sizes(64) == list(range(1, 17)) + [24, 32, 48, 64]
    assert len(_elastic_graph_batch_sizes(256)) == 16 + 8
    # every gap above the dense range stays within 1.5x, so padding waste is bounded
    sizes = _elastic_graph_batch_sizes(256)
    for lo, hi in zip(sizes, sizes[1:]):
        if lo >= _DENSE_GRAPH_BS:
            assert hi <= lo * 1.5, (lo, hi)


def test_env_cap_reproduces_the_pre_fix_ceiling(monkeypatch):
    """The A/B knob: the graph set never exceeds the cap, in the same binary."""
    monkeypatch.setenv("FREETOKEN_ELASTIC_GRAPH_MAX_BS", "8")
    assert _elastic_graph_batch_sizes(16) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert _elastic_graph_batch_sizes(4) == [1, 2, 3, 4]


def test_env_cap_never_exceeds_the_tier_and_survives_garbage(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ELASTIC_GRAPH_MAX_BS", "64")
    assert _elastic_graph_batch_sizes(16) == list(range(1, 17))
    for bad in ("not-a-number", "", "   "):
        monkeypatch.setenv("FREETOKEN_ELASTIC_GRAPH_MAX_BS", bad)
        assert _elastic_graph_batch_sizes(16) == list(range(1, 17))
