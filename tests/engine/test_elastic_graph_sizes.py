import pytest

from freetoken.engine import graph
from freetoken.engine.engine import _DENSE_GRAPH_BS, _elastic_graph_batch_sizes
from freetoken.engine.graph import _DENSE_GRAPH_BS as _NON_ELASTIC_DENSE_BS
from freetoken.engine.graph import _determine_cuda_graph_bs


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


# ------------------------------------------------------------------ non-elastic ladder
#
# ``_determine_cuda_graph_bs`` is the ladder the ordinary (non-elastic) server captures.
# It is shared by every model, so the dense small end is gated on ``offload_moe``.


def test_offload_moe_ladder_never_pads_in_the_common_range():
    """Dense 1..16 for offload-MoE, measured on the non-elastic server.

    RTX 5080, Nemotron 3.5 Lightning NVFP4, ``--max-running-requests 16``, 12 decode lanes,
    three alternating repeats of each arm out of one binary (benchmarks/decode16/phaseE2.sh):

        sparse [1,2,4,8,16] (12 pads to 16): 138.69 / 144.86 / 137.74 tok/s
        dense  [1..16]      (12 gets bs-12): 149.83 / 150.33 / 152.55 tok/s

    1.074x on the means, with every dense run above every sparse run. A padded row is not
    free on an offload-MoE model: it carries a hidden state, routes its own top-6 experts
    and adds rows to every expert GEMV.
    """
    sizes = _determine_cuda_graph_bs(None, 16, 0, offload_moe=True)
    for batch in range(1, 17):
        assert next(bs for bs in sizes if bs >= batch) == batch, batch
    assert sizes == sorted(set(sizes))
    assert max(sizes) <= 16

    # and the same holds when the ceiling is far above the dense range
    sizes = _determine_cuda_graph_bs(None, 160, 0, offload_moe=True)
    for batch in range(1, 17):
        assert next(bs for bs in sizes if bs >= batch) == batch, batch
    assert sizes == sorted(set(sizes))
    assert all(bs <= 160 for bs in sizes)


def test_dense_model_ladder_is_unchanged():
    """A padded row is nearly free on a dense model, so the historical ladder stands.

    Pinned exactly: a future change to this shared helper must not widen the blast radius
    from offload-MoE models to every model.
    """
    assert _determine_cuda_graph_bs(None, 16, 0) == [1, 2, 4, 8, 16]
    assert _determine_cuda_graph_bs(None, 32, 0) == [1, 2, 4, 8, 16, 24, 32]
    assert _determine_cuda_graph_bs(None, 8, 0, offload_moe=False) == [1, 2, 4, 8]
    assert _determine_cuda_graph_bs(None, 160, 0, offload_moe=False) == [1, 2, 4] + list(
        range(8, 161, 8)
    )


def test_ladder_stays_sparse_above_the_dense_range_in_both_modes():
    """Graph memory must not grow linearly with a large ceiling."""
    for offload_moe in (False, True):
        sizes = _determine_cuda_graph_bs(None, 160, 0, offload_moe=offload_moe)
        above = [bs for bs in sizes if bs >= _NON_ELASTIC_DENSE_BS]
        assert above == list(range(16, 161, 8)), (offload_moe, above)


def test_env_override_makes_the_ab_one_binary(monkeypatch):
    """FREETOKEN_GRAPH_DENSE_BS=0|1 forces the rule off/on regardless of the model."""
    monkeypatch.setenv("FREETOKEN_GRAPH_DENSE_BS", "0")
    assert _determine_cuda_graph_bs(None, 16, 0, offload_moe=True) == [1, 2, 4, 8, 16]
    monkeypatch.setenv("FREETOKEN_GRAPH_DENSE_BS", "1")
    assert _determine_cuda_graph_bs(None, 16, 0, offload_moe=False) == list(range(1, 17))


def test_env_override_survives_garbage(monkeypatch):
    """A bad value is ignored (logged), and the ``offload_moe`` argument decides."""
    for bad in ("not-a-number", "", "   "):
        monkeypatch.setenv("FREETOKEN_GRAPH_DENSE_BS", bad)
        assert _determine_cuda_graph_bs(None, 16, 0, offload_moe=True) == list(range(1, 17))
        assert _determine_cuda_graph_bs(None, 16, 0, offload_moe=False) == [1, 2, 4, 8, 16]


def test_degenerate_and_explicit_inputs_are_untouched():
    for offload_moe in (False, True):
        assert _determine_cuda_graph_bs(None, 0, 0, offload_moe=offload_moe) == []
        assert _determine_cuda_graph_bs(None, -1, 0, offload_moe=offload_moe) == []
        # an explicit list short-circuits: it is returned verbatim, unsorted and all
        explicit = [5, 3, 9]
        assert _determine_cuda_graph_bs(explicit, 16, 0, offload_moe=offload_moe) is explicit


def test_the_two_dense_ranges_agree():
    """The elastic and non-elastic ladders must densify over the same range."""
    assert _NON_ELASTIC_DENSE_BS == _DENSE_GRAPH_BS


def test_graph_runner_derives_offload_moe_from_the_cache(monkeypatch):
    """``GraphRunner`` gates the dense ladder on having an offload-MoE cache.

    Capturing graphs needs a GPU, so the ladder call is intercepted at the top of
    ``__init__`` -- the argument it is really given is what matters here.
    """
    seen = {}

    class _Stop(Exception):
        pass

    def _spy(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(graph, "_determine_cuda_graph_bs", _spy)

    for cache, expected in ((None, False), (object(), True)):
        seen.clear()
        with pytest.raises(_Stop):
            graph.GraphRunner(
                stream=None,
                device=None,
                model=None,
                attn_backend=None,
                cuda_graph_bs=None,
                cuda_graph_max_bs=16,
                free_memory=0,
                max_seq_len=1,
                vocab_size=1,
                dummy_req=None,
                moe_offload_cache=cache,
            )
        assert seen["offload_moe"] is expected
