from __future__ import annotations

import threading
from unittest.mock import patch

import pytest
import torch

from freetoken.distributed.info import DistributedInfo
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.kvcache.quant import Q6_0, Q8_0
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.session_spill import SessionSpillStore


def _linear_pool() -> LinearStatePool:
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(1, 3),
        num_key_heads=2,
        num_value_heads=2,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        output_gate=True,
    )
    return LinearStatePool(group, 8, torch.bfloat16, torch.device("cpu"), tp_size=1)


def _pools():
    with patch(
        "freetoken.kvcache.mha_pool.get_tp_info",
        return_value=DistributedInfo(rank=0, size=1),
    ):
        kv = MHAKVCache(
            num_kv_heads=2,
            num_layers=4,
            head_dim=64,
            num_pages=33,
            page_size=1,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            quant_k=Q8_0,
            quant_v=Q6_0,
        )
    linear = _linear_pool()
    table = torch.zeros((4, 64), dtype=torch.int32)
    manager = CacheManager(
        32,
        1,
        table,
        "hybrid_radix",
        linear_state_pool=linear,
        swa_pool=kv,
    )
    return kv, linear, manager


@pytest.mark.parametrize("tier", ["ram", "disk"])
def test_cold_session_round_trip_is_byte_exact_and_reclaims_gpu_pages(tmp_path, tier):
    kv, linear, manager = _pools()
    store = SessionSpillStore(
        kv,
        linear,
        directory=str(tmp_path),
        ram_budget_bytes=(1 << 30) if tier == "ram" else 0,
        disk_budget_bytes=1 << 30,
        host_reserve_bytes=0,
    )
    tokens = torch.tensor([11, 12, 13, 14, 15], dtype=torch.int32)
    pages = manager._page_to_token(manager._allocate(len(tokens)))
    slot = linear.alloc(1)[0]

    torch.manual_seed(7)
    kv._k_buffer[:, pages] = torch.randint(-128, 127, kv._k_buffer[:, pages].shape, dtype=torch.int8)
    kv._v_buffer[:, pages] = torch.randint(0, 255, kv._v_buffer[:, pages].shape, dtype=torch.uint8)
    kv._scale_buffer[:, :, pages] = torch.randn_like(kv._scale_buffer[:, :, pages])
    linear.conv_states[:, slot] = torch.randn_like(linear.conv_states[:, slot])
    linear.recurrent_states[:, slot] = torch.randn_like(linear.recurrent_states[:, slot])

    expected = {
        "k": kv._k_buffer[:, pages].clone(),
        "v": kv._v_buffer[:, pages].clone(),
        "scale": kv._scale_buffer[:, :, pages].clone(),
        "conv": linear.conv_states[:, slot].clone(),
        "recurrent": linear.recurrent_states[:, slot].clone(),
    }
    manager.prefix_cache.insert(tokens, pages, slot)
    handle = manager.retain_prefix(tokens, len(tokens))
    record = store.spill(tokens, pages, slot)
    assert record is not None and record.tier == tier

    manager.unlock(handle)
    assert manager.evict_all_unlocked_prefixes() == len(tokens)
    assert manager.page_usage()[0] == 0

    restored = manager.restore_hybrid_session_prefix(record, store)
    restored_pages = restored.get_matched_indices().long()
    restored_slot = restored.node.mamba_value
    assert restored.cached_len == len(tokens)
    assert torch.equal(kv._k_buffer[:, restored_pages], expected["k"])
    assert torch.equal(kv._v_buffer[:, restored_pages], expected["v"])
    assert torch.equal(kv._scale_buffer[:, :, restored_pages], expected["scale"])
    assert torch.equal(linear.conv_states[:, restored_slot], expected["conv"])
    assert torch.equal(linear.recurrent_states[:, restored_slot], expected["recurrent"])

    manager.unlock(restored)
    store.discard(record)
    assert store.ram_bytes == 0 and store.disk_bytes == 0
    store.shutdown()


def test_changed_client_prefix_is_not_eligible_for_restore(tmp_path):
    kv, linear, _manager = _pools()
    store = SessionSpillStore(
        kv,
        linear,
        directory=str(tmp_path),
        ram_budget_bytes=1 << 30,
        disk_budget_bytes=0,
        host_reserve_bytes=0,
    )
    # Eligibility itself lives in Scheduler; pin the important representation invariant here:
    # token ids are copied as int32 rather than aliasing a mutable client tensor.
    source = torch.tensor([1, 2, 3], dtype=torch.int64)
    pages = torch.tensor([1, 2, 3], dtype=torch.int32)
    slot = linear.alloc(1)[0]
    record = store.spill(source, pages, slot)
    source[-1] = 99
    assert record is not None
    assert torch.equal(record.token_ids, torch.tensor([1, 2, 3], dtype=torch.int32))
    store.shutdown()


def test_host_pressure_demotes_ram_checkpoint_to_disk(tmp_path, monkeypatch):
    kv, linear, _manager = _pools()
    store = SessionSpillStore(
        kv,
        linear,
        directory=str(tmp_path),
        ram_budget_bytes=1 << 30,
        disk_budget_bytes=1 << 30,
        host_reserve_bytes=3 << 30,
    )
    monkeypatch.setattr(
        "freetoken.scheduler.session_spill._mem_available_bytes",
        lambda: 8 << 30,
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    record = store.spill(tokens, tokens, linear.alloc(1)[0])
    assert record is not None and record.tier == "ram"

    monkeypatch.setattr(
        "freetoken.scheduler.session_spill._mem_available_bytes",
        lambda: 2 << 30,
    )
    assert store.enforce_host_reserve() == (1, 0)
    assert record.tier == "disk"
    assert store.ram_bytes == 0 and store.disk_bytes == record.byte_size
    assert all(chunk.value is None and chunk.file is not None for chunk in record.chunks)
    store.shutdown()


def test_disk_restore_prefetches_exactly_one_chunk_ahead(tmp_path, monkeypatch):
    kv, linear, _manager = _pools()
    store = SessionSpillStore(
        kv,
        linear,
        directory=str(tmp_path),
        ram_budget_bytes=0,
        disk_budget_bytes=1 << 30,
        host_reserve_bytes=0,
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    record = store.spill(tokens, tokens, linear.alloc(1)[0])
    assert record is not None and record.tier == "disk" and len(record.chunks) > 2

    loaded: list[object] = []
    second_started = threading.Event()

    def fake_load(path, **_kwargs):
        loaded.append(path)
        if len(loaded) == 2:
            second_started.set()
        return torch.tensor([len(loaded)])

    monkeypatch.setattr("freetoken.scheduler.session_spill.torch.load", fake_load)
    chunks = store.iter_chunks(record)
    next(chunks)
    assert second_started.wait(timeout=1.0)
    assert len(loaded) == 2
    chunks.close()
    store.shutdown()
