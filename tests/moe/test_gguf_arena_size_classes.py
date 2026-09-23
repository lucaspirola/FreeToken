"""S12b: the expert arena serving mixed-GGUF size classes -- one arena per class.

``FREETOKEN_EXPERT_ARENA=1`` used to refuse ``set_bank_sources`` for a GGUF
checkpoint with >1 distinct row signature (``offload_cache.py:738-742`` before this
step). This now builds one independent VMM arena per signature class instead
(``_set_gguf_size_class_sources``), so growable KV can fund itself from a mixed-GGUF
cache the same way it already does from a uniform one.

Like the single-class arena in real (engine.py) use, ``cache_size == slot_capacity``
at construction: the cache starts fully committed, and growable KV only ever
SHRINKS it (freeing bytes for KV) and grows it back, never past that ceiling --
see ``_set_gguf_size_class_sources``'s assertion of this invariant, and its
docstring for why a mixed-GGUF arena cannot start any other way.

CPU-only: the real ``VMMTensor``/``allocation_granularity`` require CUDA, so this
mocks the VMM layer with a plain in-process fake that tracks committed ranges
without touching a device, the same role the real allocation plays for
``_alloc_arena_bank_cache``/``set_usable_slots`` (see ``tests/moe/test_expert_arena_vmm.py``,
which exercises the real thing under CUDA).
"""

from __future__ import annotations

import pytest
import torch

import freetoken.moe.offload_cache as offload_cache_module
from freetoken.engine.cache_budget import joint_arena_boundaries, joint_arena_bytes_for_usable
from freetoken.moe.offload_cache import OffloadMoeCache


class _FakeVMMAllocation:
    """Records committed ranges without any real CUDA VMM call."""

    def __init__(self, shape, dtype, device, reserved_bytes, initial_ranges):
        self.tensor = torch.zeros(shape, dtype=dtype, device=device)
        self._committed: set[tuple[int, int]] = set(initial_ranges)

    @property
    def mapped_bytes(self) -> int:
        return sum(size for _, size in self._committed)

    def commit_ranges(self, ranges):
        for r in ranges:
            self._committed.add(r)

    def uncommit_ranges(self, ranges):
        for r in ranges:
            self._committed.discard(r)


@pytest.fixture
def fake_vmm(monkeypatch):
    monkeypatch.setattr(
        "freetoken.kernel.vmm.VMMTensor",
        lambda shape, *, dtype, device, reserved_bytes=None, initial_ranges=None: (
            _FakeVMMAllocation(shape, dtype, device, reserved_bytes, initial_ranges)
        ),
    )


def _mixed_sources(num_layers=4, num_experts=2):
    # Two GGUF signatures, same shape as tests/moe/test_gguf_size_classes.py.
    gate = [torch.full((num_experts, 16), layer + 1, dtype=torch.uint8) for layer in range(num_layers)]
    down = [
        torch.full((num_experts, 8 if layer < 2 else 12), layer + 11, dtype=torch.uint8)
        for layer in range(num_layers)
    ]
    return {"gate_up": gate, "down": down}


def _make_arena_class_cache(*, cache_size=12, num_experts=2, num_layers=4, prefill_overlap=False):
    """``slot_capacity`` is always ``cache_size`` -- see the module docstring: a
    mixed-GGUF arena must start fully committed, exactly like engine.py's real
    single-class wiring."""
    offload_cache_module.FREETOKEN_EXPERT_ARENA = True
    try:
        cache = OffloadMoeCache(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            device=torch.device("cpu"),
            quant_format="gguf",
            prefill_overlap=prefill_overlap,
            slot_capacity=cache_size,
            arena_step_slots=cache_size,
        )
        cache.direct_device_banks = True
        cache.set_bank_sources(_mixed_sources(num_layers, num_experts))
    finally:
        offload_cache_module.FREETOKEN_EXPERT_ARENA = False
    return cache


def test_arena_no_longer_refuses_mixed_gguf_signatures(fake_vmm):
    # The refusal this replaces was offload_cache.py:738-742 (NotImplementedError).
    cache = _make_arena_class_cache()
    assert cache._size_class_enabled
    assert cache.class_arena_layouts is not None
    assert len(cache.class_arena_layouts) == 2


def test_one_arena_per_class_with_its_own_ceiling_and_row_bytes(fake_vmm):
    cache = _make_arena_class_cache(cache_size=16, num_experts=2, num_layers=4)
    layouts = cache.class_arena_layouts
    assert layouts is not None
    ceilings = [c for c, _ in layouts]
    assert sum(ceilings) == 16  # the ceilings partition the full slot_capacity
    assert cache._class_ranges == [(0, ceilings[0]), (ceilings[0], 16)]
    row_bytes = cache.class_bank_row_bytes
    assert row_bytes is not None
    assert len(row_bytes) == 2
    for per_class_rows in row_bytes:
        assert len(per_class_rows) == 2  # gate_up, down


def test_mixed_gguf_arena_requires_cache_size_equal_slot_capacity(fake_vmm):
    offload_cache_module.FREETOKEN_EXPERT_ARENA = True
    try:
        cache = OffloadMoeCache(
            num_layers=4,
            num_experts=2,
            cache_size=8,
            device=torch.device("cpu"),
            quant_format="gguf",
            slot_capacity=16,
            arena_step_slots=8,
        )
        cache.direct_device_banks = True
        with pytest.raises(AssertionError, match="fully committed"):
            cache.set_bank_sources(_mixed_sources())
    finally:
        offload_cache_module.FREETOKEN_EXPERT_ARENA = False


def test_n1_uniform_signature_never_touches_the_class_arena(fake_vmm):
    """A uniform-signature checkpoint (never reaching _set_gguf_size_class_sources
    at all) must keep using the single-class arena_layout/bank_row_bytes -- the
    class_arena_layouts/class_bank_row_bytes attributes stay None. This is the
    N=1 regression at the OffloadMoeCache level."""
    offload_cache_module.FREETOKEN_EXPERT_ARENA = True
    try:
        cache = OffloadMoeCache(
            num_layers=2,
            num_experts=4,
            cache_size=8,
            device=torch.device("cpu"),
            quant_format="bf16",
        )
        # bf16 on CPU without direct_device_banks never builds a real arena either
        # (matches the pre-existing single-class behavior) -- the point here is
        # only that the NEW class attributes are untouched/None.
        cache.set_bank_sources(
            {
                "gate_up": [torch.zeros((4, 16), dtype=torch.bfloat16) for _ in range(2)],
                "down": [torch.zeros((4, 8), dtype=torch.bfloat16) for _ in range(2)],
            }
        )
    finally:
        offload_cache_module.FREETOKEN_EXPERT_ARENA = False
    assert cache.class_arena_layouts is None
    assert cache.class_bank_row_bytes is None


def test_set_class_usable_slots_shrinks_and_regrows_exact_bytes(fake_vmm):
    cache = _make_arena_class_cache(cache_size=16, num_experts=2, num_layers=4)
    ceilings = [c for c, _ in cache.class_arena_layouts]
    steps = [s for _, s in cache.class_arena_layouts]
    row_bytes = cache.class_bank_row_bytes
    total = sum(ceilings)
    assert total == 16
    assert int(cache.usable_slots.item()) == total  # starts fully committed

    boundaries = joint_arena_boundaries(ceilings, steps)
    floor = sum(ceilings[:-1]) + cache.num_experts

    target = max(b for b in boundaries if floor <= b < total)
    # CPU builds use granularity=1 (no real VMM page size) -- see
    # _set_gguf_size_class_sources's `allocation_granularity(...) if cuda else 1`.
    # The default 2 MiB granule in cache_budget would round every tiny test row
    # up to the same one granule and hide any real byte delta.
    expected_released = joint_arena_bytes_for_usable(
        total, ceilings, steps, row_bytes, granule=1
    ) - joint_arena_bytes_for_usable(target, ceilings, steps, row_bytes, granule=1)
    released = cache.set_class_usable_slots(target)
    assert released == expected_released
    assert int(cache.usable_slots.item()) == target
    assert torch.all(cache.id_of_slot[target:total] == -1)

    regrown = cache.set_class_usable_slots(total)
    assert regrown == released
    assert int(cache.usable_slots.item()) == total


def test_set_class_usable_slots_rejects_non_boundary_target(fake_vmm):
    cache = _make_arena_class_cache(cache_size=16, num_experts=2, num_layers=4)
    ceilings = [c for c, _ in cache.class_arena_layouts]
    steps = [s for _, s in cache.class_arena_layouts]
    boundaries = set(joint_arena_boundaries(ceilings, steps))
    total = sum(ceilings)
    non_boundary = next(n for n in range(total + 1) if n not in boundaries)
    with pytest.raises(ValueError):
        cache.set_class_usable_slots(non_boundary)


def test_set_class_usable_slots_rejects_below_joint_floor(fake_vmm):
    cache = _make_arena_class_cache(cache_size=16, num_experts=2, num_layers=4)
    ceilings = [c for c, _ in cache.class_arena_layouts]
    total = sum(ceilings)
    # The joint floor is sum(ceilings[:-1]) + num_experts (last class's own
    # floor) -- a lower, valid boundary must be refused, never partially applied.
    floor = sum(ceilings[:-1]) + cache.num_experts
    below = 0
    assert below < floor
    with pytest.raises(ValueError, match="joint floor"):
        cache.set_class_usable_slots(below)
    assert int(cache.usable_slots.item()) == total  # untouched


def test_set_class_usable_slots_off_gate_raises():
    cache = OffloadMoeCache(
        num_layers=4, num_experts=2, cache_size=12, device=torch.device("cpu"), quant_format="gguf",
    )
    cache.set_bank_sources(_mixed_sources(num_layers=4, num_experts=2))
    assert cache._size_class_enabled  # sanity: mixed signatures, non-arena path
    with pytest.raises(AssertionError):
        cache.set_class_usable_slots(4)
