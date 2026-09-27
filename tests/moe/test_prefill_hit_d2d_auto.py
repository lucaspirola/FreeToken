"""Short prefill chunks take the hit/miss split without the flag; long ones keep the full-layer
copy unless --moe-prefill-hit-d2d asks for the split. CPU-only (stubbed cache)."""

from types import SimpleNamespace

import pytest

from freetoken.moe.offload_cache import OffloadMoeCache


def _begin(num_tokens, *, flag=False, tokens=2048, usable=True):
    asked = []

    def usable_fn(requested=True):
        asked.append(requested)
        return usable

    stub = SimpleNamespace(
        prefill_overlap=True, prefill_copy_stream=None,
        residency=SimpleNamespace(begin_prefill=lambda: False),
        prefill_hit_d2d=flag, prefill_hit_d2d_tokens=tokens, _hit_d2d_usable=usable_fn,
    )
    if usable:
        # the active path snapshots slots on the copy stream; not what this test is about
        stub._prefill_slot_snapshot = None
        stub.slot_for_id = None
    try:
        OffloadMoeCache.begin_prefill(stub, num_tokens)
    except (AttributeError, TypeError, RuntimeError, AssertionError):
        pass  # reached the (CUDA-only) snapshot copy: the split was chosen
    return getattr(stub, "_prefill_hit_d2d_active", None), asked


@pytest.mark.parametrize("n", [1, 300, 1000, 2048])
def test_short_chunk_takes_the_split_without_the_flag(n):
    active, asked = _begin(n, usable=False)
    assert asked == [False]  # consulted as the auto split, fallback logged at info
    assert active is False  # unusable -> full-layer copy


@pytest.mark.parametrize("n", [2049, 8192, None])
def test_long_chunk_keeps_the_full_layer_copy(n):
    active, asked = _begin(n, usable=False)
    assert asked == [] and active is False


def test_flag_forces_the_split_on_long_chunks():
    _, asked = _begin(8192, flag=True, usable=False)
    assert asked == [True]


def test_zero_disables_the_auto_split():
    _, asked = _begin(300, tokens=0, usable=False)
    assert asked == []
