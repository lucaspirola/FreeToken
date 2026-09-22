"""The decode CUDA-graph batch-size ladder (``engine/graph.py::_determine_cuda_graph_bs``).

Kept from ``test_elastic_graph_sizes.py`` when refactor step S11 retired
``--elastic-initial-requests`` and its own graph-size set; these cases never depended on it.
"""

import pytest

from freetoken.engine import graph
from freetoken.engine.graph import _DENSE_GRAPH_BS, _determine_cuda_graph_bs


# ------------------------------------------------------------------ ladder
#
# ``_determine_cuda_graph_bs`` is the ladder the server captures.
# It is shared by every model, so the dense small end is gated on ``offload_moe``.


def test_offload_moe_ladder_never_pads_in_the_common_range():
    """Dense 1..16 for offload-MoE, measured on a fixed 16-request server.

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
        above = [bs for bs in sizes if bs >= _DENSE_GRAPH_BS]
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
