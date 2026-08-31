from __future__ import annotations

import torch
import pytest

from freetoken.moe.offload_cache import OffloadMoeCache


def _sources():
    # Two GGUF signatures: gate/up is common, down alternates compact/large.
    gate = [torch.full((2, 16), layer + 1, dtype=torch.uint8) for layer in range(4)]
    down = [
        torch.full((2, 8 if layer < 2 else 12), layer + 11, dtype=torch.uint8)
        for layer in range(4)
    ]
    return {"gate_up": gate, "down": down}


def test_mixed_gguf_borrows_largest_decode_class_for_prefill_buffers():
    cache = OffloadMoeCache(
        num_layers=4,
        num_experts=2,
        cache_size=12,
        device=torch.device("cpu"),
        quant_format="gguf",
        prefill_overlap=True,
    )
    sources = _sources()
    cache.set_bank_sources(sources)

    assert cache._size_class_enabled
    assert cache._class_ranges == [(0, 4), (4, 12)]
    assert cache._lru_size == 12
    small = cache.bank_views(layer_id=0)
    large = cache.bank_views(layer_id=2)
    assert [tuple(t.shape) for t in small] == [(4, 16), (4, 8)]
    assert [tuple(t.shape) for t in large] == [(8, 16), (8, 12)]
    assert [tuple(t.shape) for t in cache.prefill_bank_buffers] == [
        (2, 2, 16),
        (2, 2, 12),
    ]
    assert cache.prefill_bank_buffers[0].data_ptr() == large[0].data_ptr()
    assert cache.prefill_bank_buffers[1].data_ptr() == large[1].data_ptr()

    # Each class rewrites global ownership into local tensor rows.
    ids0 = torch.tensor([[0, 1]], dtype=torch.int32)
    cache.ensure_experts(0, ids0)
    assert ids0.tolist() == [[0, 1]]
    assert cache.slot_for_id[0, :2].tolist() == [0, 1]
    ids2 = torch.tensor([[0, 1]], dtype=torch.int32)
    cache.ensure_experts(2, ids2)
    assert ids2.tolist() == [[0, 1]]
    assert cache.slot_for_id[2, :2].tolist() == [4, 5]
    assert cache.evict_slots[:2].tolist() == [0, 1]

    # A prefill using buffer zero invalidates only the borrowed owners in the
    # largest class; compact-class rows remain resident.
    cache.prefetch_prefill_layer(0)
    assert cache.slot_for_id[0, :2].tolist() == [0, 1]
    assert cache.slot_for_id[2, :2].tolist() == [-1, -1]

    # Overlap buffers accept the compact layer and preserve its real prefix.
    views = cache.wait_prefill_layer(0)
    assert torch.equal(views[0], sources["gate_up"][0])
    assert torch.equal(views[1][:, :8], sources["down"][0])


def test_mixed_gguf_size_class_rebuild_floor():
    cache = OffloadMoeCache(
        num_layers=4,
        num_experts=2,
        cache_size=12,
        device=torch.device("cpu"),
        quant_format="gguf",
        prefill_overlap=True,
    )
    cache.set_bank_sources(_sources())
    # Two classes need 2E decode slots, and overlap needs another 2E rows.
    try:
        cache.validate_rebuild(7)
    except ValueError as exc:
        assert "at least 8 slots" in str(exc)
    else:
        raise AssertionError("undersized compact cache was accepted")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_gguf_cuda_lru_and_copy_use_class_local_rows():
    device = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=8,
        device=device,
        quant_format="gguf",
        prefill_overlap=False,
    )
    sources = {
        "gate_up": [
            torch.arange(4 * 16, dtype=torch.uint8, device=device).view(4, 16),
            torch.arange(4 * 16, dtype=torch.uint8, device=device).view(4, 16) + 3,
        ],
        "down": [
            torch.arange(4 * 16, dtype=torch.uint8, device=device).view(4, 16) + 7,
            torch.arange(4 * 32, dtype=torch.uint8, device=device).view(4, 32) + 11,
        ],
    }
    cache.set_bank_sources(sources)

    for layer_id in (0, 1):
        raw = torch.tensor([[3, 1]], dtype=torch.int32, device=device)
        cache.ensure_experts(layer_id, raw)
        cache.copy_missing()
        torch.cuda.synchronize()
        assert raw.cpu().tolist() == [[1, 0]]
        views = cache.bank_views(layer_id=layer_id)
        assert torch.equal(views[0][1], sources["gate_up"][layer_id][3])
        assert torch.equal(views[0][0], sources["gate_up"][layer_id][1])
        assert torch.equal(views[1][1], sources["down"][layer_id][3])
        assert torch.equal(views[1][0], sources["down"][layer_id][1])
