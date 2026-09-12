"""CPU integration bridge for elastic KV handoff state preservation.

This exercises real cache/tree/state/spill implementations. Physical commit changes are
allocator-accounting stand-ins only; CUDA VMM mapping is intentionally not claimed here.
"""

from __future__ import annotations

import pytest
import torch

from .test_session_spill import _pools
from freetoken.scheduler.session_spill import SessionSpillStore


@pytest.mark.parametrize("tier", ["ram", "disk"])
def test_checkpoint_compact_shrink_grow_restore_preserves_hybrid_session(
    tmp_path, request, tier
):
    kv, linear, manager = _pools()
    # This is the real Scheduler constructor wiring: its generic plug-in argument carries
    # the primary pool for every model. Identity alone must not misclassify MHA as SWA.
    assert manager.swa_pool is kv
    assert not manager.swa_paged
    store = SessionSpillStore(
        kv,
        linear,
        directory=str(tmp_path),
        ram_budget_bytes=(1 << 30) if tier == "ram" else 0,
        disk_budget_bytes=(1 << 30) if tier == "disk" else 0,
        host_reserve_bytes=0,
    )
    request.addfinalizer(store.shutdown)
    tokens = torch.tensor([101, 102, 103, 104, 105], dtype=torch.int32)

    # Put A at a high physical suffix and leave low holes for ownership-aware compaction.
    allocated = manager._page_to_token(manager._allocate(20))
    manager._free(allocated[:15])
    pages = allocated[15:]
    slot = linear.alloc(1)[0]
    torch.manual_seed(23)
    kv._k_buffer[:, pages] = torch.randint(
        -128, 127, kv._k_buffer[:, pages].shape, dtype=torch.int8
    )
    kv._v_buffer[:, pages] = torch.randint(
        0, 255, kv._v_buffer[:, pages].shape, dtype=torch.uint8
    )
    kv._scale_buffer[:, :, pages] = torch.randn_like(kv._scale_buffer[:, :, pages])
    linear.conv_states[:, slot] = torch.randn_like(linear.conv_states[:, slot])
    linear.recurrent_states[:, slot] = torch.randn_like(
        linear.recurrent_states[:, slot]
    )
    expected = {
        "k": kv._k_buffer[:, pages].clone(),
        "v": kv._v_buffer[:, pages].clone(),
        "scale": kv._scale_buffer[:, :, pages].clone(),
        "conv": linear.conv_states[:, slot].clone(),
        "recurrent": linear.recurrent_states[:, slot].clone(),
    }
    pooled = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    manager.prefix_cache.insert(tokens, pages, slot, pooled_sums=pooled)
    handle = manager.retain_prefix(tokens, len(tokens))
    node = handle.node
    before_refs = (node.ref_count, node.mamba_ref_count)

    # Exercise explicit pin ownership too. The small fixture's default token cap is lower
    # than this synthetic prefix, so widen only the test manager's configured budget.
    manager.pin_prefix_max_tokens = manager.num_pages
    manager.pin_prefix_max_slots = 4
    assert manager.pin_prefix(node)
    pinned = (
        manager.prefix_counters.pinned_prefixes,
        manager.prefix_counters.pinned_tokens,
        manager.prefix_counters.pinned_slots,
    )
    record = store.spill("agent-a", tokens, pages, slot)
    assert record is not None and record.valid and record.tier == tier

    # Fake physical suffix shrink accounting around the real prefix-aware copy/remap.
    target = manager.compact_active_pages([], 8, kv.copy_pages)
    assert target == 8
    compacted_pages = node.value.long()
    assert max(compacted_pages).item() <= 8
    assert torch.equal(kv._k_buffer[:, compacted_pages], expected["k"])
    assert torch.equal(kv._v_buffer[:, compacted_pages], expected["v"])
    assert torch.equal(kv._scale_buffer[:, :, compacted_pages], expected["scale"])
    assert (node.ref_count, node.mamba_ref_count) == (
        before_refs[0] + 1,
        before_refs[1] + 1,
    )
    assert torch.equal(node.pooled_sums, pooled)
    assert node.pooled_count == len(tokens)
    assert (
        manager.prefix_counters.pinned_prefixes,
        manager.prefix_counters.pinned_tokens,
        manager.prefix_counters.pinned_slots,
    ) == pinned
    manager.remove_committed_pages(8)
    assert manager.committed_pages == 8

    # During an eviction interval, the valid checkpoint allows all A GPU ownership to
    # be released while another workload may use the reclaimed physical capacity.
    manager.unlock(handle)
    manager.unpin_all()
    assert manager.evict_all_unlocked_prefixes() == len(tokens)
    assert manager.page_usage()[0] == 0
    manager.remove_committed_pages(4)
    assert manager.committed_pages == 4

    # At A's return, fake one-step physical regrowth, then install the real checkpoint.
    manager.add_committed_pages(8)
    restored = manager.restore_hybrid_session_prefix(record, store)
    restored_pages = restored.get_matched_indices().long()
    restored_node = restored.node
    restored_slot = restored_node.mamba_value
    assert restored.cached_len == len(tokens)
    assert torch.equal(kv._k_buffer[:, restored_pages], expected["k"])
    assert torch.equal(kv._v_buffer[:, restored_pages], expected["v"])
    assert torch.equal(kv._scale_buffer[:, :, restored_pages], expected["scale"])
    assert torch.equal(linear.conv_states[:, restored_slot], expected["conv"])
    assert torch.equal(
        linear.recurrent_states[:, restored_slot], expected["recurrent"]
    )
    assert restored_node.ref_count == 1
    assert restored_node.mamba_ref_count == 1
    manager.check_integrity()

    manager.unlock(restored)


@pytest.mark.parametrize("capability", ["swa_radix", "swa_paged"])
def test_compaction_still_rejects_real_swa_capabilities(capability):
    _kv, _linear, manager = _pools()
    if capability == "swa_radix":
        manager.is_swa = True
    else:
        # HybridSWAKVCache and DSV4 both advertise this capability to CacheManager.
        manager.swa_paged = True
    with pytest.raises(RuntimeError, match="does not support SWA"):
        manager.compact_active_pages([], 8, lambda *_: None)
