"""S12b: ``GrowableKvController._plan_growable_kv`` over a mixed-GGUF per-class
arena (``moe.class_arena_layouts``/``class_bank_row_bytes``), a 2-class synthetic
fixture -- the joint planner (``cache_budget.plan_joint_arena_usable``) instead of
the uniform-arena's ``prefill_overlap``-toggling binary search below it.

CPU-only, no torch/CUDA: the stub engine supplies only what ``_plan_growable_kv``
reads before it reaches the class-arena branch (it returns before touching
``_growable_moe_ceiling``/``moe.validate_rebuild``, the uniform-arena-only code).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.cache_budget import (
    joint_arena_bytes_for_usable,
    joint_arena_floor,
)
from freetoken.engine.growable_kv import GrowableKvController


class _Pool:
    def __init__(self, mapped_bytes_per_page: int):
        self._per_page = mapped_bytes_per_page

    def mapped_bytes_for_pages(self, pages: int) -> int:
        return pages * self._per_page


class _MixedClassMoe:
    def __init__(self, class_arena_layouts, class_bank_row_bytes, *, num_experts=2, prefill_overlap=False):
        self.class_arena_layouts = class_arena_layouts
        self.class_bank_row_bytes = class_bank_row_bytes
        self.num_experts = num_experts
        self.prefill_overlap = prefill_overlap


def _controller(moe, pool, *, memory_ratio, baseline_free, weights_bytes, fixed_cache_size=0):
    engine = SimpleNamespace(
        kv_cache=pool,
        moe_offload_cache=moe,
        config=SimpleNamespace(
            memory_ratio=memory_ratio,
            model_config=SimpleNamespace(linear_attention_group=lambda: None, slot_states=()),
        ),
        _baseline_free=baseline_free,
        _weights_bytes=weights_bytes,
        linear_state_pool=None,
        _pool_cls=SimpleNamespace(
            kv_cost=lambda config: (1, fixed_cache_size, 1, 0)
        ),
    )
    return GrowableKvController(engine)


def test_plan_growable_kv_multi_class_maximizes_moe_within_budget():
    capacities, steps, rows = [8, 12], [4, 4], [[10], [10]]
    moe = _MixedClassMoe(list(zip(capacities, steps)), rows)
    pool = _Pool(mapped_bytes_per_page=1)
    # A generous budget: memory_ratio*baseline_free - weights - fixed_cache_size,
    # minus the 256 MiB VMM safety margin baked into _plan_growable_kv.
    baseline_free = 10_000_000 + 256 * 1024 * 1024
    ctl = _controller(moe, pool, memory_ratio=1.0, baseline_free=baseline_free, weights_bytes=0)

    target_moe, kv_bytes = ctl._plan_growable_kv(target_pages=100)

    assert kv_bytes == 100
    assert target_moe == sum(capacities)  # whole arena affordable -> take it all


def test_plan_growable_kv_multi_class_shrinks_to_fund_kv_never_below_floor():
    capacities, steps, rows = [8, 12], [4, 4], [[2 * 1024 * 1024], [2 * 1024 * 1024]]
    moe = _MixedClassMoe(list(zip(capacities, steps)), rows, num_experts=2, prefill_overlap=False)
    pool = _Pool(mapped_bytes_per_page=1024 * 1024)  # 1 MiB/page
    floor = joint_arena_floor(capacities, [moe.num_experts, moe.num_experts])
    full_bytes = joint_arena_bytes_for_usable(sum(capacities), capacities, steps, rows)
    floor_bytes = joint_arena_bytes_for_usable(floor, capacities, steps, rows)
    # Budget room for KV = just enough to force a shrink below full, but well
    # above the floor's own footprint.
    kv_pages = 4
    kv_bytes_wanted = kv_pages * pool._per_page
    margin = 256 * 1024 * 1024
    budget_bytes = floor_bytes + kv_bytes_wanted + 1024 * 1024  # a bit of slack
    baseline_free = budget_bytes + margin

    ctl = _controller(moe, pool, memory_ratio=1.0, baseline_free=baseline_free, weights_bytes=0)
    target_moe, kv_bytes = ctl._plan_growable_kv(target_pages=kv_pages)

    assert kv_bytes == kv_pages * pool._per_page
    assert target_moe < sum(capacities)  # it really did shrink
    assert target_moe >= floor
    assert joint_arena_bytes_for_usable(target_moe, capacities, steps, rows) + kv_bytes <= (
        budget_bytes
    )


def test_plan_growable_kv_multi_class_raises_when_floor_does_not_fit():
    capacities, steps, rows = [8, 12], [4, 4], [[10_000_000], [10_000_000]]
    moe = _MixedClassMoe(list(zip(capacities, steps)), rows)
    pool = _Pool(mapped_bytes_per_page=1)
    # Enough baseline_free to clear _plan_growable_kv's own 256 MiB safety-margin
    # check (a separate, earlier RuntimeError) but nowhere near the class floor's
    # own footprint.
    baseline_free = 256 * 1024 * 1024 + 1000
    ctl = _controller(moe, pool, memory_ratio=1.0, baseline_free=baseline_free, weights_bytes=0)

    with pytest.raises(ValueError, match="joint arena floor"):
        ctl._plan_growable_kv(target_pages=1)
