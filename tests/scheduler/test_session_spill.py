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
    record = store.spill("agent-a", tokens, pages, slot)
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
    record = store.spill("agent-a", source, pages, slot)
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
    record = store.spill("agent-a", tokens, tokens, linear.alloc(1)[0])
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
    record = store.spill("agent-a", tokens, tokens, linear.alloc(1)[0])
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


def _store(tmp_path, **overrides):
    kwargs = dict(
        directory=str(tmp_path),
        ram_budget_bytes=0,
        disk_budget_bytes=1 << 30,
        host_reserve_bytes=0,
        limit_bytes=1 << 30,
        persist=True,
        model_id="model-a",
    )
    kwargs.update(overrides)
    return kwargs


def test_restart_adopts_matching_checkpoint_and_restores_it(tmp_path):
    kv, linear, manager = _pools()
    store = SessionSpillStore(kv, linear, **_store(tmp_path))
    tokens = torch.tensor([21, 22, 23, 24, 25], dtype=torch.int32)
    pages = manager._page_to_token(manager._allocate(len(tokens)))
    slot = linear.alloc(1)[0]
    torch.manual_seed(11)
    kv._k_buffer[:, pages] = torch.randint(-128, 127, kv._k_buffer[:, pages].shape, dtype=torch.int8)
    linear.recurrent_states[:, slot] = torch.randn_like(linear.recurrent_states[:, slot])
    expected_k = kv._k_buffer[:, pages].clone()
    expected_recurrent = linear.recurrent_states[:, slot].clone()

    manager.prefix_cache.insert(tokens, pages, slot)
    handle = manager.retain_prefix(tokens, len(tokens))
    record = store.spill("agent-restart", tokens, pages, slot)
    assert record is not None and record.tier == "disk"
    manifest = record.directory / "manifest.json"
    assert manifest.is_file() and (manifest.stat().st_mode & 0o777) == 0o600
    store.shutdown()  # a persisting shutdown flushes manifests and keeps the root

    manager.unlock(handle)
    assert manager.evict_all_unlocked_prefixes() == len(tokens)

    # Simulated restart: a brand-new store over the same root and the same live pool.
    revived = SessionSpillStore(kv, linear, **_store(tmp_path))
    adopted = revived.get("agent-restart")
    assert adopted is not None
    assert torch.equal(adopted.token_ids, tokens)
    assert adopted.num_pages == len(tokens)

    restored = manager.restore_hybrid_session_prefix(adopted, revived)
    restored_pages = restored.get_matched_indices().long()
    assert restored.cached_len == len(tokens)
    assert torch.equal(kv._k_buffer[:, restored_pages], expected_k)
    assert torch.equal(linear.recurrent_states[:, restored.node.mamba_value], expected_recurrent)
    manager.unlock(restored)
    revived.shutdown()


def test_restart_deletes_foreign_or_incompatible_checkpoints(tmp_path):
    kv, linear, _manager = _pools()
    store = SessionSpillStore(kv, linear, **_store(tmp_path))
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    record = store.spill("agent-stale", tokens, tokens, linear.alloc(1)[0])
    assert record is not None
    directory = record.directory
    store.shutdown()
    assert directory.is_dir()

    class _OtherLayout:
        def __getattr__(self, name):
            return getattr(kv, name)

        def session_spill_fingerprint(self):
            return ("mha-kv-v1", "different-layout")

    revived = SessionSpillStore(_OtherLayout(), linear, **_store(tmp_path))
    assert revived.get("agent-stale") is None
    assert not directory.exists()
    assert revived.disk_bytes == 0


def test_restart_rejects_checkpoints_written_by_another_model(tmp_path):
    kv, linear, _manager = _pools()
    store = SessionSpillStore(kv, linear, **_store(tmp_path))
    tokens = torch.tensor([4, 5, 6], dtype=torch.int32)
    record = store.spill("agent-other-model", tokens, tokens, linear.alloc(1)[0])
    assert record is not None
    store.shutdown()

    revived = SessionSpillStore(kv, linear, **_store(tmp_path, model_id="model-b"))
    assert revived.get("agent-other-model") is None
    assert list(tmp_path.iterdir()) == []
    revived.shutdown()


def test_non_persistent_store_wipes_checkpoints_on_exit(tmp_path):
    kv, linear, _manager = _pools()
    store = SessionSpillStore(kv, linear, **_store(tmp_path, persist=False))
    tokens = torch.tensor([7, 8, 9], dtype=torch.int32)
    record = store.spill("agent-ephemeral", tokens, tokens, linear.alloc(1)[0])
    assert record is not None and record.directory.is_dir()
    store.shutdown()
    assert not record.directory.exists()
    assert store.get("agent-ephemeral") is None


def test_startup_collects_leaked_and_unreadable_directories(tmp_path):
    kv, linear, _manager = _pools()
    leaked = tmp_path / "server-abc123"
    leaked.mkdir()
    (leaked / "checkpoint-1").mkdir()
    (tmp_path / "stray.pt").write_bytes(b"junk")
    (tmp_path / "operator-notes.txt").write_text("not ours", encoding="utf-8")
    torn = tmp_path / ("0" * 64)
    torn.mkdir()
    (torn / "manifest.json").write_text("{not json", encoding="utf-8")

    store = SessionSpillStore(kv, linear, **_store(tmp_path))
    # Everything this store's layouts can produce is collected; foreign content is not.
    assert [p.name for p in tmp_path.iterdir()] == ["operator-notes.txt"]
    assert store.disk_bytes == 0
    store.shutdown()


def test_capacity_cap_evicts_least_recently_used_checkpoint(tmp_path):
    kv, linear, _manager = _pools()
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    probe = SessionSpillStore(kv, linear, **_store(tmp_path))
    one = probe._payload_bytes(3, tokens)
    probe.shutdown()

    store = SessionSpillStore(kv, linear, **_store(tmp_path, limit_bytes=2 * one + 16))
    first = store.spill("agent-1", tokens, tokens, linear.alloc(1)[0])
    second = store.spill("agent-2", tokens, tokens, linear.alloc(1)[0])
    assert first is not None and second is not None
    store.touch(first)  # last use, not spill time, is what orders the eviction

    third = store.spill("agent-3", tokens, tokens, linear.alloc(1)[0])
    assert third is not None
    assert store.get("agent-2") is None and not second.valid
    assert {r.session_id for r in store._records} == {"agent-1", "agent-3"}
    assert store.ram_bytes + store.disk_bytes <= store.limit_bytes
    store.shutdown()


def test_record_larger_than_the_whole_cap_is_refused(tmp_path):
    kv, linear, _manager = _pools()
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    store = SessionSpillStore(kv, linear, **_store(tmp_path, limit_bytes=1024))
    resident = store.spill("agent-big", tokens, tokens, linear.alloc(1)[0])
    assert resident is None
    assert store._records == [] and list(tmp_path.iterdir()) == []
    store.shutdown()


# --------------------------------------------------------------- boundary states (3G)


def _boundary_pools(tmp_path, **overrides):
    kv, linear, manager = _pools()
    kwargs = dict(
        directory=str(tmp_path),
        ram_budget_bytes=1 << 30,
        disk_budget_bytes=1 << 30,
        host_reserve_bytes=0,
    )
    kwargs.update(overrides)
    return kv, linear, manager, SessionSpillStore(kv, linear, **kwargs)


def _seeded_session(kv, linear, manager, tokens):
    """Random KV for ``tokens`` plus two distinguishable recurrent states."""
    pages = manager._page_to_token(manager._allocate(len(tokens)))
    final_slot, boundary_slot = linear.alloc(2)
    torch.manual_seed(5)
    kv._k_buffer[:, pages] = torch.randint(
        -128, 127, kv._k_buffer[:, pages].shape, dtype=torch.int8
    )
    for slot in (final_slot, boundary_slot):
        linear.recurrent_states[:, slot] = torch.randn_like(linear.recurrent_states[:, slot])
    return pages, final_slot, boundary_slot


def test_partial_restore_resumes_at_the_deepest_stored_boundary(tmp_path):
    kv, linear, manager, store = _boundary_pools(tmp_path)
    tokens = torch.tensor([11, 12, 13, 14, 15, 16], dtype=torch.int32)
    pages, final_slot, boundary_slot = _seeded_session(kv, linear, manager, tokens)
    expected_k = kv._k_buffer[:, pages[:4]].clone()
    expected_state = linear.recurrent_states[:, boundary_slot].clone()

    record = store.spill(
        "agent-a", tokens, pages, final_slot, extra_states=[(4, boundary_slot)]
    )
    assert record is not None and record.state_boundaries == [4, 6]

    # A drift in the last two tokens: only the prefix through boundary 4 is reusable.
    assert record.restorable_length(5) == 4
    restored = manager.restore_hybrid_session_prefix(record, store, 4)
    assert restored.cached_len == 4
    assert torch.equal(kv._k_buffer[:, restored.get_matched_indices().long()], expected_k)
    # The installed state is the one snapshotted AT the cut, not the checkpoint's end.
    assert torch.equal(linear.recurrent_states[:, restored.node.mamba_value], expected_state)
    manager.unlock(restored)
    store.shutdown()


def test_exact_match_still_restores_the_whole_checkpoint(tmp_path):
    kv, linear, manager, store = _boundary_pools(tmp_path)
    tokens = torch.tensor([21, 22, 23, 24, 25, 26], dtype=torch.int32)
    pages, final_slot, boundary_slot = _seeded_session(kv, linear, manager, tokens)
    expected_k = kv._k_buffer[:, pages].clone()
    expected_state = linear.recurrent_states[:, final_slot].clone()

    record = store.spill(
        "agent-a", tokens, pages, final_slot, extra_states=[(4, boundary_slot)]
    )
    assert record is not None and record.restorable_length(len(tokens)) == len(tokens)

    restored = manager.restore_hybrid_session_prefix(record, store)
    assert restored.cached_len == len(tokens)
    assert torch.equal(kv._k_buffer[:, restored.get_matched_indices().long()], expected_k)
    assert torch.equal(linear.recurrent_states[:, restored.node.mamba_value], expected_state)
    manager.unlock(restored)
    store.shutdown()


def test_a_drift_before_the_first_boundary_leaves_nothing_to_restore(tmp_path):
    kv, linear, manager, store = _boundary_pools(tmp_path)
    tokens = torch.tensor([31, 32, 33, 34, 35, 36], dtype=torch.int32)
    pages, final_slot, boundary_slot = _seeded_session(kv, linear, manager, tokens)
    record = store.spill(
        "agent-a", tokens, pages, final_slot, extra_states=[(4, boundary_slot)]
    )
    assert record is not None
    assert record.restorable_length(3) == 0  # full recompute, no half state to install
    with pytest.raises(ValueError):
        manager.restore_hybrid_session_prefix(record, store, 3)
    store.shutdown()


def test_boundary_states_are_thinned_to_the_stride_then_to_the_freshest(tmp_path):
    _kv, _linear, _manager, store = _boundary_pools(tmp_path, state_stride_tokens=100)
    store.max_states = 3
    # Stride-spaced coverage first, newest end backwards, and the budget stops it at 3.
    assert store._select_state_boundaries([200, 700, 800, 900], 1000) == [800, 900, 1000]
    # A session whose candidates are all closer than one stride still keeps them: the
    # second pass spends what the first could not.
    assert store._select_state_boundaries([2, 4], 6) == [2, 4, 6]
    store.shutdown()


# ------------------------------------------------------------- look-ahead prefetch (3F)


def _disk_record(store, linear, session_id, monkeypatch):
    """Spill one record that the host-reserve guard forces onto the disk tier."""
    monkeypatch.setattr(
        "freetoken.scheduler.session_spill._mem_available_bytes", lambda: 0
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    record = store.spill(session_id, tokens, tokens, linear.alloc(1)[0])
    monkeypatch.setattr(
        "freetoken.scheduler.session_spill._mem_available_bytes", lambda: 8 << 30
    )
    assert record is not None and record.tier == "disk"
    return record


def test_prefetch_promotes_a_queued_session_checkpoint_to_ram(tmp_path, monkeypatch, caplog):
    import logging

    _kv, linear, _manager, store = _boundary_pools(tmp_path)
    record = _disk_record(store, linear, "agent-a", monkeypatch)
    directory = record.directory

    with caplog.at_level(logging.INFO, logger="freetoken.scheduler.session_spill"):
        assert store.start_prefetch("agent-a") is True
        assert store.start_prefetch("agent-b") is False  # one in flight at a time
        assert store.collect_prefetch("agent-a", wait=True) == "agent-a"

    assert record.tier == "ram" and record.directory is None
    assert store.ram_bytes == record.byte_size and store.disk_bytes == 0
    assert all(c.value is not None and c.file is None for c in record.chunks)
    assert not directory.exists()
    assert any(m.startswith("Prefetched cold session agent-a to RAM") for m in caplog.messages)
    store.shutdown()


def test_prefetch_is_refused_when_the_ram_budget_cannot_hold_it(tmp_path, monkeypatch):
    _kv, linear, _manager, store = _boundary_pools(tmp_path, ram_budget_bytes=0)
    record = _disk_record(store, linear, "agent-a", monkeypatch)

    assert store.start_prefetch("agent-a") is False  # refusal, not an error
    assert store.start_prefetch("nobody") is False
    assert store.collect_prefetch() is None
    assert record.tier == "disk" and record.directory.is_dir()
    store.shutdown()


def test_prefetch_demotes_an_lru_ram_record_but_never_a_protected_one(tmp_path, monkeypatch):
    _kv, linear, _manager, store = _boundary_pools(tmp_path)
    monkeypatch.setattr(
        "freetoken.scheduler.session_spill._mem_available_bytes", lambda: 8 << 30
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    resident = store.spill("resident", tokens, tokens, linear.alloc(1)[0])
    assert resident is not None and resident.tier == "ram"
    store.ram_budget_bytes = resident.byte_size  # room for exactly one RAM record
    queued = _disk_record(store, linear, "queued", monkeypatch)

    # The resident session's own checkpoint is never the one that pays for the look-ahead.
    assert store.start_prefetch("queued", protect={"resident"}) is False
    assert resident.tier == "ram"

    assert store.start_prefetch("queued") is True
    assert store.collect_prefetch(wait=True) == "queued"
    assert resident.tier == "disk" and queued.tier == "ram"
    store.shutdown()


def test_a_cancelled_prefetch_leaves_the_checkpoint_on_disk(tmp_path, monkeypatch):
    _kv, linear, _manager, store = _boundary_pools(tmp_path)
    record = _disk_record(store, linear, "agent-a", monkeypatch)

    assert store.start_prefetch("agent-a") is True
    assert store.cancel_prefetch("agent-a") is True
    assert store.collect_prefetch(wait=True) is None
    assert record.tier == "disk" and record.directory.is_dir()
    assert store.ram_bytes == 0
    # The cancelled slot is reusable once its reader has stopped.
    assert store.start_prefetch("agent-a") is True
    assert store.collect_prefetch(wait=True) == "agent-a"
    store.shutdown()


# -------------------------------------------------- prefill-time state capture (3G)


def _capture(stride=64, max_states=8):
    from freetoken.scheduler.session_spill import PrefillStateCapture

    linear = _linear_pool()
    torch.manual_seed(13)
    linear.conv_states.normal_()
    linear.recurrent_states.normal_()
    return linear, PrefillStateCapture(linear, stride=stride, max_states=max_states)


def test_capture_takes_one_state_per_stride_and_never_more_than_the_cap():
    linear, capture = _capture()

    taken = capture.capture([], 64, 1)
    taken = capture.capture(taken, 100, 2)  # 36 tokens on: inside the stride, no copy
    assert [b for b, _c, _r in taken] == [64]
    assert torch.equal(taken[0][1], linear.conv_states[:, 1])
    assert torch.equal(taken[0][2], linear.recurrent_states[:, 1])

    for boundary in range(128, 64 * 21, 64):
        taken = capture.capture(taken, boundary, 1)

    boundaries = [b for b, _c, _r in taken]
    assert len(boundaries) == 8  # the cap holds however long the turn runs
    assert boundaries == sorted(boundaries) and boundaries[-1] == 64 * 20
    assert len(set(boundaries)) == 8


def test_capture_thinning_keeps_stride_spaced_coverage_of_the_whole_prefix():
    _linear, capture = _capture(stride=65_536, max_states=8)
    kept = capture.thin(
        [(b, None, None) for b in range(65_536, 65_536 * 12 + 1, 65_536)]
    )
    boundaries = [b for b, _c, _r in kept]
    assert len(boundaries) == 8
    assert boundaries[-1] == 65_536 * 12  # the newest boundary always survives
    assert all(
        later - earlier >= 65_536
        for earlier, later in zip(boundaries, boundaries[1:])
    )


def test_captured_states_ride_into_the_checkpoint_beside_the_final_one(tmp_path):
    kv, linear, manager, store = _boundary_pools(tmp_path)
    tokens = torch.tensor([41, 42, 43, 44, 45, 46], dtype=torch.int32)
    pages, final_slot, _boundary_slot = _seeded_session(kv, linear, manager, tokens)
    conv = torch.randn_like(linear.conv_states[:, 0])
    recurrent = torch.randn_like(linear.recurrent_states[:, 0])

    record = store.spill(
        "agent-a", tokens, pages, final_slot, captured_states=[(3, conv, recurrent)]
    )
    assert record is not None and record.state_boundaries == [3, 6]

    restored = manager.restore_hybrid_session_prefix(record, store, 3)
    assert restored.cached_len == 3
    assert torch.equal(linear.recurrent_states[:, restored.node.mamba_value], recurrent)
    assert torch.equal(linear.conv_states[:, restored.node.mamba_value], conv)
    manager.unlock(restored)
    store.shutdown()
