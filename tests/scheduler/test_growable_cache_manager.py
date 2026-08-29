from __future__ import annotations

import torch

from freetoken.scheduler.cache import CacheManager


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
