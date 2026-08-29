from __future__ import annotations

from types import SimpleNamespace

import torch

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
