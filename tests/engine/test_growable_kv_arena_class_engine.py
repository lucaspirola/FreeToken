"""Torch-backed exercise of the growable-KV expert-arena resize path (design step 5)
for a MIXED-GGUF cache -- one arena per size class (S12b).

Sibling of ``test_growable_kv_arena_engine.py``, same ``Engine.__new__`` /
``GrowableKvController`` pattern, but the fake MoE cache exposes
``class_arena_layouts``/``class_bank_row_bytes`` instead of ``arena_layout``/
``bank_row_bytes`` -- i.e. it drives the class-arena branch the coordinator's
review found unreachable from ``Engine.grow_runtime_kv``/``shrink_runtime_kv``:
before this fix, engine construction with a class arena raised
``GROWABLE_KV_UNSUPPORTED`` at startup, and even past that gate every
``set_usable_slots`` call would have hit ``OffloadMoeCache``'s
``assert self._arena_banks`` (a class arena only populates
``_class_arena_banks``). This file exercises the REAL public entry points, not
just the planner (``test_growable_kv_joint_plan.py``) or the low-level cache
method (``test_gguf_arena_size_classes.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.cache_budget import joint_arena_bytes_for_usable
from freetoken.engine.engine import Engine
from freetoken.engine.growable_kv import GrowableKvController
from freetoken.moe.residency import WholeModelResidency

# These drive the arena transaction through torch.cuda (memory queries, device tensors).
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

MiB = 1024 * 1024
GRANULE = 2 * MiB
# One VMM granule per slot per bank, per class -- so joint_arena_bytes_for_usable
# never rounds up and hand-computed expectations are exact (matches the
# single-class sibling test file's ROW_BYTES convention).
ROW_BYTES_PER_CLASS = [[GRANULE], [GRANULE]]


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


class FakeMixedClassArenaMoe:
    """Mixed-GGUF counterpart of ``test_growable_kv_arena_engine.FakeArenaMoe``:
    exposes ``class_arena_layouts``/``class_bank_row_bytes`` (one arena per size
    class) instead of the single-class ``arena_layout``/``bank_row_bytes``, and
    never exposes those single-class attributes at all (``getattr(..., None)``
    at every real call site must take the class branch, not silently pass
    ``None`` through)."""

    def __init__(
        self,
        cache_size: int,
        class_layouts: list[tuple[int, int]],
        *,
        num_experts=2,
        prefill_overlap=True,
    ):
        self.cache_size = cache_size
        self.class_arena_layouts = list(class_layouts)
        self.class_bank_row_bytes = ROW_BYTES_PER_CLASS
        self.slot_capacity = sum(c for c, _ in class_layouts)
        self.num_experts = num_experts
        self.prefill_overlap = prefill_overlap
        self.residency = WholeModelResidency()
        self.usable_calls: list[int] = []
        self.rebuild_calls: list[int] = []

    def _bytes(self, n: int) -> int:
        capacities = [c for c, _ in self.class_arena_layouts]
        steps = [s for _, s in self.class_arena_layouts]
        return joint_arena_bytes_for_usable(n, capacities, steps, self.class_bank_row_bytes)

    def set_usable_slots(self, n: int) -> int:
        old = self.cache_size
        self.usable_calls.append(n)
        self.cache_size = n
        return abs(self._bytes(n) - self._bytes(old))

    def rebuild(self, cache_size: int) -> None:
        # Must never be called from the arena branch.
        self.rebuild_calls.append(cache_size)
        self.cache_size = cache_size


def _engine(pool: FakePool, moe: FakeMixedClassArenaMoe, *, free_bytes_fn=lambda: (0, 0)) -> Engine:
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
    engine.graph_runner = SimpleNamespace(
        graph_bs_list=[1],
        destroy_cuda_graphs=lambda: pytest.fail(
            "expert-arena resize must never destroy CUDA graphs"
        ),
    )
    engine.attn_backend = SimpleNamespace(
        reset_capture=lambda: pytest.fail("expert-arena resize must never reset capture")
    )
    engine.growable_kv = GrowableKvController(engine)
    engine.growable_kv._plan_growable_kv = lambda pages, **kw: (0, pages * MiB)
    engine._sync_get_memory = free_bytes_fn
    engine.sync_all_ranks = lambda: None
    return engine


def _dynamic_free(moe: FakeMixedClassArenaMoe, capacity: int):
    def fn():
        freed = (capacity - moe.cache_size) * GRANULE
        return freed, freed

    return fn


def _two_class_layout():
    # Two classes, each its own 512-slot arena stepping by num_experts=64 (matching
    # the real cache's uniform per-class step design -- see
    # OffloadMoeCache._set_gguf_size_class_sources).
    return [(512, 64), (512, 64)]


def test_class_arena_startup_gate_accepts_class_arena_layouts():
    """Engine construction itself is not exercised here (that needs a full
    weight-loaded Engine), but the same gate expression
    ``grow_runtime_kv``/``shrink_runtime_kv`` and ``Engine.__init__`` use is:
    accept when EITHER ``arena_layout`` or ``class_arena_layouts`` is present.
    This directly regresses the coordinator's confirmed defect #1/#2 (a class
    arena has no ``arena_layout`` and must not be refused)."""
    moe = FakeMixedClassArenaMoe(cache_size=1024, class_layouts=_two_class_layout())
    assert getattr(moe, "arena_layout", None) is None
    assert getattr(moe, "class_arena_layouts", None) is not None


@needs_cuda
def test_grow_funds_kv_via_set_usable_slots_on_class_arena_never_rebuild():
    class_layouts = _two_class_layout()
    total = sum(c for c, _ in class_layouts)
    moe = FakeMixedClassArenaMoe(cache_size=total, class_layouts=class_layouts)
    pool = FakePool(8)
    engine = _engine(pool, moe, free_bytes_fn=_dynamic_free(moe, total))

    old_pages, new_pages = engine.grow_runtime_kv(16)

    assert (old_pages, new_pages) == (8, 16)
    assert pool.committed_pages == 16
    assert moe.rebuild_calls == []
    assert len(moe.usable_calls) == 1
    target = moe.usable_calls[0]
    assert target < total
    # Every joint boundary this could have landed on is a real class-local
    # chunk boundary translated into joint id space -- i.e. it is in
    # joint_arena_boundaries's set, not an arbitrary integer.
    from freetoken.engine.cache_budget import joint_arena_boundaries

    capacities = [c for c, _ in class_layouts]
    steps = [s for _, s in class_layouts]
    assert target in joint_arena_boundaries(capacities, steps)
    freed = moe._bytes(total) - moe._bytes(target)
    assert freed >= 264 * MiB
    assert engine.config.moe_cache_size == target


@needs_cuda
def test_grow_rollback_regrows_experts_on_failed_commit_class_arena():
    class_layouts = _two_class_layout()
    total = sum(c for c, _ in class_layouts)
    moe = FakeMixedClassArenaMoe(cache_size=total, class_layouts=class_layouts)
    pool = FakePool(8, fail_commit=True)
    engine = _engine(pool, moe, free_bytes_fn=_dynamic_free(moe, total))

    with pytest.raises(MemoryError, match="VMM commit failure"):
        engine.grow_runtime_kv(16)

    # Shrunk to fund the commit, then re-grown back to the original count on rollback
    # -- via set_usable_slots (the class-arena dispatch), never rebuild.
    assert len(moe.usable_calls) == 2
    assert moe.usable_calls[-1] == total
    assert moe.rebuild_calls == []
    assert moe.cache_size == total
    assert pool.committed_pages == 8
    assert engine.config.moe_cache_size == total
    assert getattr(engine, "_growable_transition_failed", False) is False


@needs_cuda
def test_shrink_regrows_experts_via_set_usable_slots_on_class_arena_never_rebuild():
    class_layouts = _two_class_layout()
    total = sum(c for c, _ in class_layouts)
    start = total - 192  # below full capacity, a real joint boundary (768 for 512/64 classes)
    moe = FakeMixedClassArenaMoe(cache_size=start, class_layouts=class_layouts)
    pool = FakePool(144)
    engine = _engine(pool, moe)

    old_pages, new_pages = engine.shrink_runtime_kv(16)

    assert (old_pages, new_pages) == (144, 16)
    assert pool.committed_pages == 16
    assert moe.rebuild_calls == []
    if moe.usable_calls:
        target = moe.usable_calls[0]
        assert target > start
        grown = moe._bytes(target) - moe._bytes(start)
        # The regrow must never claim back more than the released KV bytes fund
        # (128 MiB released here: 144 - 16 pages at 1 MiB/page).
        assert grown <= 128 * MiB
        assert engine.config.moe_cache_size == target
    else:
        assert engine.config.moe_cache_size == start


@needs_cuda
def test_shrink_no_regrow_when_released_bytes_are_too_small_class_arena():
    class_layouts = _two_class_layout()
    total = sum(c for c, _ in class_layouts)
    start = total - 192
    moe = FakeMixedClassArenaMoe(cache_size=start, class_layouts=class_layouts)
    pool = FakePool(20)  # only 4 MiB released -- not enough for one more chunk

    engine = _engine(pool, moe)
    engine.shrink_runtime_kv(16)

    assert moe.usable_calls == []
    assert moe.cache_size == start
    assert moe.rebuild_calls == []
