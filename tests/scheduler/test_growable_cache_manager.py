from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from freetoken.core import Req
from freetoken.scheduler.cache import CacheManager


class _Handle:
    cached_len = 0


def _req(*, table_idx: int, cached_len: int, device_len: int) -> Req:
    req = Req(
        input_ids=torch.arange(device_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=1,
        uid=table_idx,
        sampling_params=None,
        cache_handle=_Handle(),
    )
    req.cached_len = cached_len
    return req


def test_growable_manager_exposes_only_committed_page_ids():
    table = torch.zeros((2, 256), dtype=torch.int32)
    manager = CacheManager(
        num_pages=128,
        page_size=1,
        page_table=table,
        type="naive",
        committed_pages=64,
        page_index_offset=1,
    )

    assert manager.available_size == 128
    assert manager.free_slots.tolist() == list(range(1, 65))
    first = manager._allocate(64)
    assert first.tolist() == list(range(1, 65))

    manager.add_committed_pages(128)
    assert manager.free_slots.tolist() == list(range(65, 129))
    assert manager.committed_pages == 128


def test_growable_manager_uses_aggregate_batch_demand():
    table = torch.zeros((3, 256), dtype=torch.int32)
    manager = CacheManager(
        num_pages=256,
        page_size=1,
        page_table=table,
        type="naive",
        committed_pages=64,
        page_index_offset=1,
    )
    manager._allocate(60)

    # Neither request is individually longer than the 64-page committed pool, but their
    # combined 16-page forward needs 12 more physical pages than the four still free.
    reqs = [
        _req(table_idx=0, cached_len=24, device_len=32),
        _req(table_idx=1, cached_len=40, device_len=48),
    ]
    assert manager.committed_pages_required(reqs) == 76


def test_growable_manager_reclaims_prefix_before_growing():
    table = torch.zeros((2, 256), dtype=torch.int32)
    manager = CacheManager(
        num_pages=256,
        page_size=1,
        page_table=table,
        type="naive",
        committed_pages=64,
        page_index_offset=1,
    )
    manager._allocate(60)
    manager.prefix_cache = SimpleNamespace(
        size_info=SimpleNamespace(evictable_size=12)
    )

    req = _req(table_idx=0, cached_len=16, device_len=32)
    assert manager.committed_pages_required([req]) == 64


def test_growable_manager_compacts_private_tail_and_removes_free_suffix():
    table = torch.zeros((2, 256), dtype=torch.int32)
    manager = CacheManager(
        num_pages=256,
        page_size=1,
        page_table=table,
        type="naive",
        committed_pages=128,
        page_index_offset=1,
    )
    handle = _Handle()
    handle.cached_len = 60
    req = _req(table_idx=0, cached_len=60, device_len=62)
    req.cache_handle = handle
    req.cached_len = 62
    table[0, :60] = torch.arange(1, 61, dtype=torch.int32)
    table[0, 60:62] = torch.tensor([100, 101], dtype=torch.int32)
    occupied = set(range(1, 61)) | {100, 101}
    manager.free_slots = torch.tensor(
        [page for page in range(1, 129) if page not in occupied], dtype=torch.int32
    )
    copied = []

    def copy_pages(src, dst):
        copied.append((src.tolist(), dst.tolist()))

    resulting = manager.compact_active_pages([req], 64, copy_pages)
    assert resulting == 64
    assert copied == [([100, 101], [61, 62])]
    assert table[0, 60:62].tolist() == [61, 62]

    manager.remove_committed_pages(64)
    assert manager.committed_pages == 64
    assert manager.free_slots.tolist() == [63, 64]


def test_growable_manager_compacts_hybrid_prefix_and_all_live_aliases():
    from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

    table = torch.zeros((2, 16), dtype=torch.int32)
    linear_pool = SimpleNamespace(track_chunk_size=1)
    manager = CacheManager(
        num_pages=16, page_size=1, page_table=table, type="hybrid_radix",
        linear_state_pool=linear_pool, committed_pages=8, page_index_offset=1,
    )
    tokens = torch.tensor([7, 8], dtype=torch.int32)
    old_prefix = torch.tensor([7, 8], dtype=torch.int32)
    manager.prefix_cache.insert(tokens, old_prefix, mamba_value=11,
                                pooled_sums=torch.tensor([[3.0]]))
    match = manager.prefix_cache.match_prefix(tokens, pooled=True)
    manager.prefix_cache.inc_lock(match.node)
    handle = HybridCacheHandle(match.cached_len, match.node, match.kv_indices)
    req = _req(table_idx=0, cached_len=3, device_len=4)
    req.cache_handle = handle
    table[0, :3] = torch.tensor([7, 8, 6], dtype=torch.int32)
    manager.free_slots = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)

    copied = []
    assert manager.compact_active_pages(
        [req], 5, lambda src, dst: copied.append((src.tolist(), dst.tolist()))
    ) == 5
    assert copied == [([6, 7, 8], [1, 2, 3])]
    assert table[0, :3].tolist() == [2, 3, 1]
    assert match.node.value.tolist() == [2, 3]
    assert handle.kv_indices.tolist() == [2, 3]
    assert match.node.mamba_value == 11
    assert match.node.mamba_ref_count == 1
    assert match.node.ref_count == 1
    assert match.node.pooled_sums.tolist() == [[3.0]]


def test_growable_manager_copy_failure_does_not_publish_metadata():
    table = torch.zeros((1, 8), dtype=torch.int32)
    manager = CacheManager(8, 1, table, "naive", committed_pages=8,
                           page_index_offset=1)
    req = _req(table_idx=0, cached_len=1, device_len=2)
    table[0, 0] = 8
    manager.free_slots = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    before_free = manager.free_slots.clone()

    def fail_copy(_src, _dst):
        raise RuntimeError("copy failed")

    with pytest.raises(RuntimeError, match="copy failed"):
        manager.compact_active_pages([req], 4, fail_copy)
    assert table[0, 0].item() == 8
    assert torch.equal(manager.free_slots, before_free)


def test_growable_manager_unrepresented_allocated_page_keeps_high_ceiling():
    table = torch.zeros((1, 8), dtype=torch.int32)
    manager = CacheManager(8, 1, table, "naive", committed_pages=8,
                           page_index_offset=1)
    # Page 8 is allocated but deliberately absent from the supplied borrowers.
    manager.free_slots = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    assert manager.compact_active_pages([], 4, lambda _src, _dst: None) == 8


def test_growable_manager_rejects_unsupported_geometry_before_copy():
    calls = []
    table = torch.zeros((1, 8), dtype=torch.int32)
    page_two = CacheManager(4, 2, table, "naive", committed_pages=4)
    with pytest.raises(RuntimeError, match="page_size=1"):
        page_two.compact_active_pages([], 2, lambda *_: calls.append(1))

    swa = CacheManager(8, 1, table, "naive", committed_pages=8,
                       swa_pool=SimpleNamespace(swa_paged=True))
    with pytest.raises(RuntimeError, match="SWA"):
        swa.compact_active_pages([], 4, lambda *_: calls.append(1))
    assert calls == []
