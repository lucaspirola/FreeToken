"""Torch-backed exercise of the growable-KV expert-arena resize path (design step 5).

``test_growable_kv_transaction_source.py`` extracts grow_runtime_kv/shrink_runtime_kv
by AST and runs them against tiny stubs without ever importing ``freetoken.engine.engine``
(the production module pulls in torch/model kernels). This file instead drives the REAL,
non-extracted methods on a minimally-constructed ``Engine`` (``Engine.__new__``, bypassing
``__init__``/model load/GPU allocation -- the same pattern ``test_cache_budget.py`` uses),
so it also catches real attribute-path/decorator mistakes the AST extraction cannot see.

``_plan_growable_kv`` is a large, separately-tested budget planner; it is stubbed here
exactly as the AST test stubs it, since the expert-arena resize only consumes its
``(target_moe, kv_bytes)`` return shape (and, per design step 5, does not use the
``target_moe`` element at all -- only ``kv_bytes``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.cache_budget import arena_bytes_for_usable
from freetoken.engine.engine import Engine

MiB = 1024 * 1024
GRANULE = 2 * MiB
# One VMM granule per slot per bank: chunk_slots * row_bytes is always an exact multiple
# of the granule, so arena_bytes_for_usable never rounds up and the hand-computed
# expectations below are exact.
ROW_BYTES = [GRANULE]


class FakePool:
    def __init__(self, pages: int, *, fail_commit: bool = False):
        self.committed_pages = pages
        self.fail_commit = fail_commit

    def mapped_bytes_for_pages(self, pages: int) -> int:
        return pages * MiB

    def commit_pages(self, pages: int) -> None:
        if self.fail_commit:
            self.fail_commit = False
            raise MemoryError("injected VMM commit failure")
        self.committed_pages = pages

    def decommit_pages(self, pages: int) -> None:
        self.committed_pages = pages


class FakeArenaMoe:
    """Only the surface grow_runtime_kv/shrink_runtime_kv touch under the arena
    branch: ``arena_layout``/``bank_row_bytes`` (fixed), ``cache_size``/
    ``prefill_overlap``/``num_experts`` (read), and ``set_usable_slots`` (the only
    mutator -- records every call so tests can assert it, and never re-implements
    ``rebuild`` as anything but a hard failure)."""

    def __init__(
        self, cache_size: int, capacity: int, step: int, *, num_experts=2, prefill_overlap=True
    ):
        self.cache_size = cache_size
        self.arena_layout = (capacity, step)
        self.bank_row_bytes = ROW_BYTES
        self.num_experts = num_experts
        self.prefill_overlap = prefill_overlap
        self.usable_calls: list[int] = []
        self.rebuild_calls: list[int] = []

    def set_usable_slots(self, n: int) -> int:
        old = self.cache_size
        self.usable_calls.append(n)
        self.cache_size = n
        return abs(
            arena_bytes_for_usable(n, *self.arena_layout, self.bank_row_bytes)
            - arena_bytes_for_usable(old, *self.arena_layout, self.bank_row_bytes)
        )

    def rebuild(self, cache_size: int) -> None:
        # Must never be called from the arena branch -- rebuild collapses the arena
        # and would defeat the entire point of design step 5.
        self.rebuild_calls.append(cache_size)
        self.cache_size = cache_size


def _engine(pool: FakePool, moe: FakeArenaMoe, *, free_bytes_fn=lambda: (0, 0)) -> Engine:
    engine = Engine.__new__(Engine)  # bypass __init__/GPU/model load
    engine.kv_cache = pool
    engine.moe_offload_cache = moe
    engine.num_pages = 4096
    engine.device = torch.device("cuda")
    engine.config = SimpleNamespace(
        kv_grow_step_tokens=8,
        page_size=1,
        moe_cache_size=moe.cache_size,
        tp_info=SimpleNamespace(size=1),
    )
    engine._pending_graph_bs = None
    engine.graph_runner = SimpleNamespace(
        graph_bs_list=[1],
        destroy_cuda_graphs=lambda: pytest.fail(
            "expert-arena resize must never destroy CUDA graphs"
        ),
    )
    engine.attn_backend = SimpleNamespace(
        reset_capture=lambda: pytest.fail("expert-arena resize must never reset capture")
    )
    engine._plan_growable_kv = lambda pages, **kw: (0, pages * MiB)
    engine._sync_get_memory = free_bytes_fn
    engine.sync_all_ranks = lambda: None
    return engine


def _dynamic_free(moe: FakeArenaMoe, capacity: int):
    """Free VRAM grows exactly as much as the arena releases -- lets the grow path's
    live-memory guard observe a real before/after delta across set_usable_slots."""

    def fn():
        freed = (capacity - moe.cache_size) * GRANULE
        return freed, freed

    return fn


def test_grow_funds_kv_via_set_usable_slots_never_rebuild():
    capacity, step = 1024, 64
    moe = FakeArenaMoe(cache_size=capacity, capacity=capacity, step=step)
    pool = FakePool(8)
    engine = _engine(pool, moe, free_bytes_fn=_dynamic_free(moe, capacity))

    old_pages, new_pages = engine.grow_runtime_kv(16)

    assert (old_pages, new_pages) == (8, 16)
    assert pool.committed_pages == 16
    # target_usable is a real chunk boundary (multiple of step) at/above the floor
    # (2*num_experts=4, prefill_overlap on), and strictly below the old usable count.
    assert moe.usable_calls == [768]
    assert moe.usable_calls[0] % step == 0
    assert moe.usable_calls[0] >= 2 * moe.num_experts
    assert moe.rebuild_calls == []
    # Bytes actually freed (old -> target) must cover what the commit needed
    # (commit_bytes=8MiB + 256MiB VMM reserve = 264MiB).
    freed = arena_bytes_for_usable(
        capacity, capacity, step, ROW_BYTES
    ) - arena_bytes_for_usable(moe.usable_calls[0], capacity, step, ROW_BYTES)
    assert freed >= 264 * MiB
    assert engine._pending_graph_bs is None
    assert engine.config.moe_cache_size == 768


def test_grow_rollback_regrows_experts_on_failed_commit():
    capacity, step = 1024, 64
    moe = FakeArenaMoe(cache_size=capacity, capacity=capacity, step=step)
    pool = FakePool(8, fail_commit=True)
    engine = _engine(pool, moe, free_bytes_fn=_dynamic_free(moe, capacity))

    with pytest.raises(MemoryError, match="VMM commit failure"):
        engine.grow_runtime_kv(16)

    # Shrunk to fund the commit, then re-grown back to the original count on rollback.
    assert moe.usable_calls == [768, 1024]
    assert moe.rebuild_calls == []
    assert moe.cache_size == 1024
    assert pool.committed_pages == 8
    assert engine.config.moe_cache_size == 1024
    assert engine._pending_graph_bs is None
    assert getattr(engine, "_growable_transition_failed", False) is False


def test_shrink_regrows_experts_via_set_usable_slots_never_rebuild():
    capacity, step = 1024, 64
    moe = FakeArenaMoe(cache_size=768, capacity=capacity, step=step)
    pool = FakePool(144)
    engine = _engine(pool, moe)

    old_pages, new_pages = engine.shrink_runtime_kv(16)

    assert (old_pages, new_pages) == (144, 16)
    assert pool.committed_pages == 16
    assert moe.usable_calls == [832]
    assert moe.usable_calls[0] % step == 0
    assert moe.rebuild_calls == []
    # The regrow must never claim back more than the released KV bytes actually fund
    # (128 MiB released here: 144 - 16 pages at 1 MiB/page).
    grown = arena_bytes_for_usable(
        832, capacity, step, ROW_BYTES
    ) - arena_bytes_for_usable(768, capacity, step, ROW_BYTES)
    assert grown <= 128 * MiB
    assert engine._pending_graph_bs is None
    assert engine.config.moe_cache_size == 832


def test_shrink_no_regrow_when_released_bytes_are_too_small():
    capacity, step = 1024, 64
    moe = FakeArenaMoe(cache_size=768, capacity=capacity, step=step)
    pool = FakePool(20)  # only 4 MiB released -- not enough for one more 64-slot chunk

    engine = _engine(pool, moe)
    engine.shrink_runtime_kv(16)

    assert moe.usable_calls == []
    assert moe.cache_size == 768
    assert moe.rebuild_calls == []
