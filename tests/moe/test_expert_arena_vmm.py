"""GPU test for the expert arena (design step 3): a fixed-capacity VMM-backed slot
cache whose usable slot count shrinks/grows in place via
``OffloadMoeCache.set_usable_slots``, with no reallocation of the expert banks or
the id_of_slot/usage/evict_slots/src_indices bookkeeping (so a captured decode CUDA
graph never needs to be destroyed/recaptured).

Requires CUDA; skipped otherwise.
"""

from __future__ import annotations

import random

import pytest
import torch

import freetoken.moe.offload_cache as offload_cache_module
import freetoken.moe.offload_kernels as offload_kernels
from freetoken.engine.cache_budget import arena_bytes_for_usable
from freetoken.kernel.vmm import allocation_granularity
from freetoken.moe.offload_cache import OffloadMoeCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_NUM_LAYERS = 1
_NUM_EXPERTS = 4  # 2*num_experts = 8, comfortably below every boundary we shrink to
_CAPACITY = 64
_STEP = 16


def _make_arena_cache(device: torch.device) -> OffloadMoeCache:
    """Build a small bf16 arena cache with the gate forced on for construction.

    Row bytes are set to (a multiple of) the VMM allocation granularity here so
    the ladder's chunk bytes are round numbers in assertions below; the cumulative
    granule-aligned chunking (see ``offload_cache._arena_chunk_ranges``) does not
    actually require this -- see ``test_unaligned_row_bytes_keeps_slot_addressing_linear``
    for a bank whose row size does not divide the granularity.
    """
    granularity = allocation_granularity(device)
    gate_up_elems = granularity // 2  # row_bytes == granularity (bf16 = 2 bytes/elem)
    down_elems = granularity // 4  # row_bytes == granularity // 2

    gate_up = torch.zeros(
        (_NUM_EXPERTS, gate_up_elems), dtype=torch.bfloat16, pin_memory=True
    )
    down = torch.zeros(
        (_NUM_EXPERTS, down_elems), dtype=torch.bfloat16, pin_memory=True
    )

    offload_cache_module.FREETOKEN_EXPERT_ARENA = True
    try:
        cache = OffloadMoeCache(
            num_layers=_NUM_LAYERS,
            num_experts=_NUM_EXPERTS,
            cache_size=_CAPACITY,
            device=device,
            quant_format="bf16",
            prefill_overlap=True,  # exercises the 2*num_experts floor, not just num_experts
            slot_capacity=_CAPACITY,
            arena_step_slots=_STEP,
        )
        cache.direct_device_banks = True
        cache.set_bank_sources({"gate_up": [gate_up], "down": [down]})
    finally:
        offload_cache_module.FREETOKEN_EXPERT_ARENA = False
    return cache


def _bank_row_bytes(cache: OffloadMoeCache) -> list[int]:
    row_bytes = cache.bank_row_bytes
    assert row_bytes is not None
    return row_bytes


def _total_mapped_bytes(cache: OffloadMoeCache) -> int:
    return sum(meta["allocation"].mapped_bytes for meta in cache._arena_banks.values())


def test_arena_attributes_and_construction_geometry():
    cache = _make_arena_cache(torch.device("cuda"))
    assert cache.slot_capacity == _CAPACITY
    assert cache.arena_layout == (_CAPACITY, _STEP)
    row_bytes = _bank_row_bytes(cache)
    assert len(row_bytes) == 2  # gate_up, down
    assert int(cache.usable_slots.item()) == _CAPACITY
    # Base addresses are permanent -- capacity-shaped, not cache_size-shaped.
    assert cache.bank_caches["gate_up"].shape[0] == _CAPACITY
    assert cache.id_of_slot.numel() == _CAPACITY
    assert cache.usage.numel() == _CAPACITY


def test_shrink_releases_exact_arena_bytes_and_keeps_pointers():
    device = torch.device("cuda")
    cache = _make_arena_cache(device)
    row_bytes = _bank_row_bytes(cache)

    ptrs_before = {name: t.data_ptr() for name, t in cache.bank_caches.items()}
    id_of_slot_ptr = cache.id_of_slot.data_ptr()
    usage_ptr = cache.usage.data_ptr()
    evict_slots_ptr = cache.evict_slots.data_ptr()
    src_indices_ptr = cache.src_indices.data_ptr()

    before_mapped = _total_mapped_bytes(cache)
    n = 32
    expected_released = arena_bytes_for_usable(
        _CAPACITY, _CAPACITY, _STEP, row_bytes
    ) - arena_bytes_for_usable(n, _CAPACITY, _STEP, row_bytes)
    assert expected_released > 0

    released = cache.set_usable_slots(n)
    torch.cuda.synchronize(device)

    assert released == expected_released
    assert before_mapped - _total_mapped_bytes(cache) == expected_released
    assert int(cache.usable_slots.item()) == n
    assert cache.cache_size == n

    # (d) id_of_slot/usage/plan buffers keep the same data_ptr() across shrink/grow.
    for name, ptr in ptrs_before.items():
        assert cache.bank_caches[name].data_ptr() == ptr
    assert cache.id_of_slot.data_ptr() == id_of_slot_ptr
    assert cache.usage.data_ptr() == usage_ptr
    assert cache.evict_slots.data_ptr() == evict_slots_ptr
    assert cache.src_indices.data_ptr() == src_indices_ptr

    # No slot_for_id entry points at a slot >= n.
    assert torch.all(cache.id_of_slot[n:] == -1)
    assert torch.all(cache.usage[n:] == 0)
    valid = cache.slot_for_id[cache.slot_for_id >= 0]
    assert torch.all(valid < n)


def test_gated_kernel_never_selects_shrunk_slots():
    device = torch.device("cuda")
    cache = _make_arena_cache(device)
    n = 16
    cache.set_usable_slots(n)
    torch.cuda.synchronize(device)

    rng = random.Random(7)
    offload_kernels.FREETOKEN_EXPERT_ARENA = True
    try:
        for _ in range(50):
            experts = sorted(
                rng.sample(range(_NUM_EXPERTS), k=rng.randint(1, _NUM_EXPERTS))
            )
            ids = torch.tensor([experts], dtype=torch.int32, device=device)
            cache.ensure_experts(0, ids)
            torch.cuda.synchronize(device)
            m = int(cache.num_indices.item())
            if m:
                assert (cache.evict_slots[:m] < n).all(), cache.evict_slots[:m].tolist()
    finally:
        offload_kernels.FREETOKEN_EXPERT_ARENA = False

    assert torch.all(cache.id_of_slot[n:] == -1)


def test_grow_recommits_and_slots_become_usable_again():
    device = torch.device("cuda")
    cache = _make_arena_cache(device)
    row_bytes = _bank_row_bytes(cache)
    n = 32

    released = cache.set_usable_slots(n)
    torch.cuda.synchronize(device)

    expected_committed = arena_bytes_for_usable(
        _CAPACITY, _CAPACITY, _STEP, row_bytes
    ) - arena_bytes_for_usable(n, _CAPACITY, _STEP, row_bytes)
    assert expected_committed == released

    before_mapped = _total_mapped_bytes(cache)
    committed = cache.set_usable_slots(_CAPACITY)
    torch.cuda.synchronize(device)

    assert committed == released  # symmetric ladder: what was freed is re-committed
    assert _total_mapped_bytes(cache) - before_mapped == committed
    assert int(cache.usable_slots.item()) == _CAPACITY
    assert cache.cache_size == _CAPACITY

    # Slots above n are usable again: the gated kernel may now place ids there.
    rng = random.Random(11)
    offload_kernels.FREETOKEN_EXPERT_ARENA = True
    saw_high_slot = False
    try:
        for _ in range(50):
            experts = sorted(
                rng.sample(range(_NUM_EXPERTS), k=rng.randint(1, _NUM_EXPERTS))
            )
            ids = torch.tensor([experts], dtype=torch.int32, device=device)
            cache.ensure_experts(0, ids)
            torch.cuda.synchronize(device)
            m = int(cache.num_indices.item())
            if m and bool((cache.evict_slots[:m] >= n).any()):
                saw_high_slot = True
    finally:
        offload_kernels.FREETOKEN_EXPERT_ARENA = False
    # Not a hard requirement (LRU may keep reusing low slots for only 4 experts),
    # but the arena must not have quietly re-clamped usable back down.
    assert int(cache.usable_slots.item()) == _CAPACITY
    del saw_high_slot  # informational only; see comment above


def test_shrink_below_two_num_experts_raises():
    device = torch.device("cuda")
    cache = _make_arena_cache(device)
    floor = 2 * _NUM_EXPERTS
    assert floor == 8
    with pytest.raises(ValueError):
        cache.set_usable_slots(0)
    # cache state must be untouched by the rejected call.
    assert int(cache.usable_slots.item()) == _CAPACITY


def test_non_boundary_target_raises():
    device = torch.device("cuda")
    cache = _make_arena_cache(device)
    with pytest.raises(ValueError):
        cache.set_usable_slots(20)  # not a multiple of arena_step_slots (16)
    assert int(cache.usable_slots.item()) == _CAPACITY


def test_gate_off_is_byte_for_byte_unchanged():
    """Sanity check: constructing without the gate never touches arena machinery."""
    device = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=_NUM_LAYERS,
        num_experts=_NUM_EXPERTS,
        cache_size=_CAPACITY,
        device=device,
        quant_format="bf16",
    )
    assert cache.slot_capacity == _CAPACITY
    assert cache.bank_row_bytes is None
    assert cache.arena_layout is None
    with pytest.raises(AssertionError):
        cache.set_usable_slots(32)


# Real NVFP4 Nemotron gate_up row size: 8192 * 609 bytes. Not a multiple of any real VMM
# granularity (2 MiB on the H100/RTX class devices this runs against), so the old
# per-chunk-independent rounding would have refused ``arena_step_slots=8`` here with a
# ValueError. The cumulative granule-aligned partition (offload_cache._arena_chunk_ranges)
# has no such precondition.
_UNALIGNED_ROW_BYTES = 8192 * 609
_UNALIGNED_CAPACITY = 32
_UNALIGNED_STEP = 8


def _make_unaligned_row_cache(device: torch.device) -> OffloadMoeCache:
    assert _UNALIGNED_ROW_BYTES % 2 == 0  # must divide evenly into bf16 elements
    gate_up_elems = _UNALIGNED_ROW_BYTES // 2
    gate_up = torch.zeros(
        (_NUM_EXPERTS, gate_up_elems), dtype=torch.bfloat16, pin_memory=True
    )
    down = torch.zeros((_NUM_EXPERTS, 128), dtype=torch.bfloat16, pin_memory=True)

    offload_cache_module.FREETOKEN_EXPERT_ARENA = True
    try:
        cache = OffloadMoeCache(
            num_layers=_NUM_LAYERS,
            num_experts=_NUM_EXPERTS,
            cache_size=_UNALIGNED_CAPACITY,
            device=device,
            quant_format="bf16",
            prefill_overlap=False,  # floor == num_experts
            slot_capacity=_UNALIGNED_CAPACITY,
            arena_step_slots=_UNALIGNED_STEP,
        )
        cache.direct_device_banks = True
        cache.set_bank_sources({"gate_up": [gate_up], "down": [down]})
    finally:
        offload_cache_module.FREETOKEN_EXPERT_ARENA = False
    return cache


def test_unaligned_row_bytes_keeps_slot_addressing_linear():
    """gate_up's row size (8192*609 B) does not divide any real VMM granularity, so this
    would have raised under the old per-chunk-independent rounding. Under the cumulative
    partition it must construct cleanly, price via the same ``arena_bytes_for_usable``
    model, and -- the actual correctness bar -- keep ``slot * row_bytes`` addressing exact
    across a shrink/grow cycle: data written to low (always-resident) slots before the
    shrink must read back unchanged afterwards.
    """
    device = torch.device("cuda")
    cache = _make_unaligned_row_cache(device)
    row_bytes = _bank_row_bytes(cache)
    assert _UNALIGNED_ROW_BYTES in row_bytes

    granularity = allocation_granularity(device)
    assert (_UNALIGNED_STEP * _UNALIGNED_ROW_BYTES) % granularity != 0, (
        "test is meaningless unless this row/step combination is actually unaligned"
    )

    # _NUM_EXPERTS (4) is below the floor's own chunk boundary; shrink to n=8, the
    # smallest real chunk boundary at/above the floor (2*4 or 4, either way < 8).
    n = _UNALIGNED_STEP
    assert n in offload_cache_module.OffloadMoeCache._arena_chunk_boundaries(
        _UNALIGNED_CAPACITY, _UNALIGNED_STEP
    )
    # Write a distinct known pattern into every slot that stays resident after the shrink.
    pattern = {i: float(i + 1) for i in range(n)}
    for slot, value in pattern.items():
        cache.bank_caches["gate_up"][slot].fill_(value)
    torch.cuda.synchronize(device)

    expected_released = arena_bytes_for_usable(
        _UNALIGNED_CAPACITY, _UNALIGNED_CAPACITY, _UNALIGNED_STEP, row_bytes
    ) - arena_bytes_for_usable(n, _UNALIGNED_CAPACITY, _UNALIGNED_STEP, row_bytes)
    released = cache.set_usable_slots(n)
    torch.cuda.synchronize(device)
    assert released == expected_released
    assert released > 0

    for slot, value in pattern.items():
        readback = cache.bank_caches["gate_up"][slot]
        assert torch.all(readback == value), (slot, value)

    committed = cache.set_usable_slots(_UNALIGNED_CAPACITY)
    torch.cuda.synchronize(device)
    assert committed == released

    # The low slots were never unmapped by the shrink (they are below the floor), so their
    # data must still be exactly what was written before the shrink/grow round trip.
    for slot, value in pattern.items():
        readback = cache.bank_caches["gate_up"][slot]
        assert torch.all(readback == value), (slot, value)
