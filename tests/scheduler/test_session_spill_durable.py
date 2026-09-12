from __future__ import annotations

import json

import torch
from unittest.mock import patch

import freetoken.scheduler.session_spill as spill_module
from freetoken.distributed.info import DistributedInfo
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.kvcache.quant import Q6_0, Q8_0
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.session_spill import DurableSessionSource, SessionSpillStore
from freetoken.scheduler.session_spill import DURABLE_GENERATION_OVERHEAD


def _pools():
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(1, 3), num_key_heads=2, num_value_heads=2,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    linear = LinearStatePool(group, 8, torch.bfloat16, torch.device("cpu"), tp_size=1)
    with patch(
        "freetoken.kvcache.mha_pool.get_tp_info",
        return_value=DistributedInfo(rank=0, size=1),
    ):
        kv = MHAKVCache(
            num_kv_heads=2, num_layers=4, head_dim=64, num_pages=33, page_size=1,
            dtype=torch.bfloat16, device=torch.device("cpu"), quant_k=Q8_0, quant_v=Q6_0,
        )
    manager = CacheManager(
        32, 1, torch.zeros((4, 64), dtype=torch.int32), "hybrid_radix",
        linear_state_pool=linear, swa_pool=kv,
    )
    return kv, linear, manager


def _store(tmp_path, **overrides):
    values = dict(
        directory=str(tmp_path), ram_budget_bytes=0, disk_budget_bytes=1 << 30,
        host_reserve_bytes=0, limit_bytes=1 << 30, persist=True, model_id="model-a",
    )
    values.update(overrides)
    return values


def _resident(manager, linear, tokens):
    pages = manager._page_to_token(manager._allocate(len(tokens)))
    slot = linear.alloc(1)[0]
    return pages, slot


def test_durable_barrier_persists_ram_and_resident_then_restart_restores(tmp_path):
    kv, linear, manager = _pools()
    store = SessionSpillStore(
        kv, linear, **_store(tmp_path, ram_budget_bytes=1 << 30)
    )
    tokens_a = torch.tensor([1, 2, 3], dtype=torch.int32)
    pages_a, slot_a = _resident(manager, linear, tokens_a)
    record_a = store.spill("a", tokens_a, pages_a, slot_a)
    assert record_a is not None and record_a.tier == "ram"

    tokens_b = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    pages_b, slot_b = _resident(manager, linear, tokens_b)
    torch.manual_seed(91)
    kv._k_buffer[:, pages_b] = torch.randint(
        -128, 127, kv._k_buffer[:, pages_b].shape, dtype=torch.int8
    )
    linear.conv_states[:, slot_b] = torch.randn_like(linear.conv_states[:, slot_b])
    expected_k = kv._k_buffer[:, pages_b].clone()
    expected_conv = linear.conv_states[:, slot_b].clone()

    result = store.persist_durable(
        [DurableSessionSource("b", tokens_b, pages_b, slot_b)]
    )
    assert result.complete and result.durable_session_ids == ("a", "b")
    assert store.ram_bytes == 0
    manifest = json.loads((store.get("b").directory / "manifest.json").read_text())
    assert manifest["tokens"].endswith("-tokens.pt")

    store.shutdown()

    revived = SessionSpillStore(kv, linear, **_store(tmp_path))
    adopted = revived.get("b")
    assert adopted is not None
    restored = manager.restore_hybrid_session_prefix(adopted, revived)
    restored_pages = restored.get_matched_indices().long()
    assert torch.equal(kv._k_buffer[:, restored_pages], expected_k)
    assert torch.equal(
        linear.conv_states[:, restored.node.mamba_value], expected_conv
    )
    manager.unlock(restored)
    revived.shutdown()


def test_durable_barrier_is_idempotent_for_disk_records(tmp_path):
    kv, linear, manager = _pools()
    store = SessionSpillStore(kv, linear, **_store(tmp_path))
    tokens = torch.tensor([3, 4], dtype=torch.int32)
    pages, slot = _resident(manager, linear, tokens)
    assert store.spill("a", tokens, pages, slot).tier == "disk"
    first = store.persist_durable()
    second = store.persist_durable()
    assert first.complete and second.complete
    assert first.durable_session_ids == second.durable_session_ids == ("a",)


def test_adoption_keeps_backward_compatible_fixed_token_filename(tmp_path):
    kv, linear, manager = _pools()
    store = SessionSpillStore(kv, linear, **_store(tmp_path))
    tokens = torch.tensor([4, 5, 6], dtype=torch.int32)
    pages, slot = _resident(manager, linear, tokens)
    record = store.spill("legacy-layout", tokens, pages, slot)
    manifest_path = record.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("tokens")
    manifest_path.write_text(json.dumps(manifest))

    revived = SessionSpillStore(kv, linear, **_store(tmp_path))
    assert torch.equal(revived.get("legacy-layout").token_ids, tokens)


def test_durable_barrier_refuses_peak_capacity_without_destroying_ram(tmp_path):
    kv, linear, manager = _pools()
    generous = SessionSpillStore(
        kv, linear, **_store(tmp_path, ram_budget_bytes=1 << 30)
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    pages, slot = _resident(manager, linear, tokens)
    record = generous.spill("a", tokens, pages, slot)
    assert record is not None and record.tier == "ram"
    generous.disk_budget_bytes = record.byte_size - 1
    result = generous.persist_durable()
    assert not result.complete
    assert generous.get("a") is record and record.valid and record.tier == "ram"


def test_failure_before_manifest_publish_preserves_old_ram_record(tmp_path, monkeypatch):
    kv, linear, manager = _pools()
    store = SessionSpillStore(
        kv, linear, **_store(tmp_path, ram_budget_bytes=1 << 30)
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    pages, slot = _resident(manager, linear, tokens)
    old = store.spill("a", tokens, pages, slot)
    calls = 0
    real = store._save_bounded

    def fail_second(value, path, limit):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected fsync failure")
        real(value, path, limit)

    monkeypatch.setattr(store, "_save_bounded", fail_second)
    result = store.persist_durable()
    assert not result.complete and "injected fsync failure" in result.error
    assert store.get("a") is old and old.valid and old.tier == "ram"


def test_oversize_serializer_is_stopped_at_write_bound_and_preserves_old(tmp_path, monkeypatch):
    kv, linear, manager = _pools()
    store = SessionSpillStore(
        kv, linear, **_store(tmp_path, ram_budget_bytes=1 << 30)
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    pages, slot = _resident(manager, linear, tokens)
    old = store.spill("a", tokens, pages, slot)
    monkeypatch.setattr(spill_module, "DURABLE_GENERATION_OVERHEAD", 32)

    def oversized_save(_value, writer):
        writer.write(b"x" * (old.byte_size + 33))

    monkeypatch.setattr(spill_module.torch, "save", oversized_save)
    result = store.persist_durable()
    assert not result.complete and "exceeded reserved generation bytes" in result.error
    assert store.get("a") is old and old.valid and old.tier == "ram"
    assert store._owned_physical_bytes() == 0


def test_failure_after_manifest_publish_tracks_new_record_but_cannot_ack(tmp_path, monkeypatch):
    kv, linear, manager = _pools()
    store = SessionSpillStore(
        kv, linear, **_store(tmp_path, ram_budget_bytes=1 << 30)
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    pages, slot = _resident(manager, linear, tokens)
    old = store.spill("a", tokens, pages, slot)

    def fail_directory(_path):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(store, "_fsync_dir", fail_directory)
    result = store.persist_durable()
    current = store.get("a")
    assert not result.complete and result.durable_session_ids == ()
    assert current is not old and current.valid and current.tier == "disk"
    assert not old.valid
    assert json.loads((current.directory / "manifest.json").read_text())["session_id"] == "a"


def test_postpublish_failure_retry_counts_retained_old_generation(tmp_path, monkeypatch):
    kv, linear, manager = _pools()
    store = SessionSpillStore(kv, linear, **_store(tmp_path))
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    pages, slot = _resident(manager, linear, tokens)
    old = store.spill("a", tokens, pages, slot)
    source = DurableSessionSource("a", tokens, pages, slot)
    real_fsync_dir = store._fsync_dir

    monkeypatch.setattr(
        store,
        "_fsync_dir",
        lambda _path: (_ for _ in ()).throw(OSError("postpublish")),
    )
    assert not store.persist_durable([source]).complete
    assert not old.valid
    retained = store._owned_physical_bytes()

    monkeypatch.setattr(store, "_fsync_dir", real_fsync_dir)
    store.disk_budget_bytes = retained + store.get("a").byte_size + DURABLE_GENERATION_OVERHEAD - 1
    before = sorted(path.name for path in store.get("a").directory.iterdir())
    retry = store.persist_durable([source])
    assert not retry.complete and "peak capacity" in retry.error
    assert sorted(path.name for path in store.get("a").directory.iterdir()) == before
