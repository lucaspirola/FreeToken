from freetoken.engine.engine import _elastic_graph_batch_sizes


def test_three_agent_elastic_tier_has_an_exact_graph():
    assert _elastic_graph_batch_sizes(2) == [1, 2]
    assert _elastic_graph_batch_sizes(3) == [1, 2, 3]
    assert _elastic_graph_batch_sizes(4) == [1, 2, 3, 4]


def test_larger_elastic_tier_keeps_sparse_graph_set():
    assert _elastic_graph_batch_sizes(8) == [1, 2, 3, 4, 8]
