"""The growable-KV headroom prices one prefill chunk's transient (not just the VMM cushion).

Native Linux OOMed Ornith's 8192-token chunk right after a 65536 -> 131072 KV grow: the
grow's arena shrink guaranteed only the 256 MiB VMM commit cushion + 128 MiB after the
commit, and the chunk needs > 0.61 GiB of transient
(tasks/exclusive-expert-ram/results/ornith-s12a-grow-ab-box/README.md). These pin the
arithmetic that fixes it, CPU-only against stub engines (``torch.cuda`` replaced inside
the controller's module, as in test_growable_kv_transaction.py):

* the headroom is the larger of the cushion and the measured transient;
* the ceiling plan subtracts it, and never plans above the startup's live budget;
* a grow's arena shrink leaves commit + headroom (+ margin) free, and refuses (rolling
  back) when the arena cannot fund that above its floor;
* the startup park/fill leaves exactly the headroom + margin free;
* a KV shrink never regrows the arena above the startup fill.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine import growable_kv
from freetoken.engine.cache_budget import arena_bytes_for_usable
from freetoken.engine.growable_kv import (
    PREFILL_HEADROOM_MARGIN_BYTES,
    VMM_COMMIT_CUSHION_BYTES,
    GrowableKvController,
    growable_headroom_bytes,
)

MiB = 1024 * 1024
GRANULE = 2 * MiB
ROW = [GRANULE]  # one granule per slot: arena bytes are exactly slots * 2 MiB


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(
        growable_kv,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda *_: None,
                memory_allocated=lambda *_: 0,
                memory_reserved=lambda *_: 0,
                mem_get_info=lambda *_: (0, 0),
            )
        ),
    )


class _Pool:
    def __init__(self, pages: int, *, per_page: int = MiB):
        self.committed_pages = pages
        self.per_page = per_page

    def mapped_bytes_for_pages(self, pages: int) -> int:
        return pages * self.per_page

    def commit_pages(self, pages: int) -> None:
        self.committed_pages = pages

    def decommit_pages(self, pages: int) -> None:
        self.committed_pages = pages


class _ArenaMoe:
    def __init__(self, cache_size, capacity, step, *, num_experts=2, min_gpu_slots=0):
        self.cache_size = cache_size
        self.arena_layout = (capacity, step)
        self.bank_row_bytes = ROW
        self.num_experts = num_experts
        self.prefill_overlap = True
        self.residency = SimpleNamespace(min_gpu_slots=lambda: min_gpu_slots)
        self.usable_calls: list[int] = []

    def _bytes(self, n):
        return arena_bytes_for_usable(n, *self.arena_layout, self.bank_row_bytes)

    def set_usable_slots(self, n: int) -> int:
        old = self.cache_size
        self.usable_calls.append(n)
        self.cache_size = n
        return abs(self._bytes(n) - self._bytes(old))


def _engine(pool, moe, *, transient=0, device_free_at_capacity=0, **extra):
    """A stub engine whose live free VRAM is ``device_free_at_capacity`` plus whatever
    the arena has released below its capacity, minus KV committed past its start."""
    capacity = moe.arena_layout[0]
    start_pages = pool.committed_pages

    def free():
        released = moe._bytes(capacity) - moe._bytes(moe.cache_size)
        kv_growth = pool.mapped_bytes_for_pages(pool.committed_pages) - pool.mapped_bytes_for_pages(start_pages)
        f = device_free_at_capacity + released - kv_growth
        return f, f

    engine = SimpleNamespace(
        kv_cache=pool,
        moe_offload_cache=moe,
        num_pages=1 << 20,
        device="cuda",
        config=SimpleNamespace(
            kv_grow_step_tokens=64,
            page_size=1,
            moe_cache_size=moe.cache_size,
            tp_info=SimpleNamespace(size=1),
        ),
        graph_runner=object(),
        _sync_get_memory=free,
        prefill_transient_bytes=transient,
        **extra,
    )
    ctl = GrowableKvController(engine)
    ctl._plan_growable_kv = lambda pages, **kw: (0, pool.mapped_bytes_for_pages(pages))
    engine.growable_kv = ctl
    return ctl, engine


def test_headroom_is_the_larger_of_cushion_and_transient():
    assert growable_headroom_bytes(0) == VMM_COMMIT_CUSHION_BYTES
    assert growable_headroom_bytes(100 * MiB) == VMM_COMMIT_CUSHION_BYTES
    assert growable_headroom_bytes(700 * MiB) == 700 * MiB
    # A stub engine without the attribute prices the bare cushion (pre-fix behavior).
    ctl = GrowableKvController(SimpleNamespace())
    assert ctl.headroom_bytes() == VMM_COMMIT_CUSHION_BYTES


# --- the ceiling plan ------------------------------------------------------------------


class _ClassMoe:
    """Two-class arena: the planner's pure-arithmetic branch (plan_joint_arena_usable)."""

    num_experts = 2
    prefill_overlap = False

    def __init__(self):
        self.class_arena_layouts = [(8, 4), (12, 4)]
        self.class_bank_row_bytes = [ROW, ROW]


def _plan_controller(*, baseline_free, transient=0, live_budget=None):
    engine = SimpleNamespace(
        kv_cache=_Pool(0, per_page=1),
        moe_offload_cache=_ClassMoe(),
        config=SimpleNamespace(
            memory_ratio=1.0,
            model_config=SimpleNamespace(linear_attention_group=lambda: None, slot_states=()),
        ),
        _baseline_free=baseline_free,
        _weights_bytes=0,
        linear_state_pool=None,
        _pool_cls=SimpleNamespace(kv_cost=lambda config: (1, 0, 1, 0)),
        prefill_transient_bytes=transient,
        _growable_live_budget=live_budget,
    )
    return GrowableKvController(engine)


def test_ceiling_plan_prices_the_measured_transient():
    kv = 100
    # Exactly the whole 20-slot arena after the bare cushion.
    baseline = VMM_COMMIT_CUSHION_BYTES + 20 * GRANULE + kv
    assert _plan_controller(baseline_free=baseline)._plan_growable_kv(kv) == (20, kv)
    # A transient 8 MiB above the cushion costs 4 slots (one chunk): 20 -> 16.
    ctl = _plan_controller(
        baseline_free=baseline, transient=VMM_COMMIT_CUSHION_BYTES + 8 * MiB
    )
    assert ctl._plan_growable_kv(kv) == (16, kv)
    # A transient below the cushion changes nothing: the cushion already covers it.
    ctl = _plan_controller(baseline_free=baseline, transient=64 * MiB)
    assert ctl._plan_growable_kv(kv) == (20, kv)


def test_ceiling_plan_never_exceeds_the_startup_live_budget():
    kv = 100
    baseline = VMM_COMMIT_CUSHION_BYTES + 20 * GRANULE + kv
    # Startup measured room for only 12 slots after graphs/workspaces (live budget),
    # though the ratio arithmetic affords all 20.
    live = VMM_COMMIT_CUSHION_BYTES + 12 * GRANULE + kv
    ctl = _plan_controller(baseline_free=baseline, live_budget=live)
    assert ctl._plan_growable_kv(kv) == (12, kv)


def test_ceiling_plan_names_the_headroom_when_nothing_is_left():
    ctl = _plan_controller(baseline_free=600 * MiB, transient=700 * MiB)
    with pytest.raises(RuntimeError, match="VMM commit / prefill headroom"):
        ctl._plan_growable_kv(1)


# --- the runtime grow ------------------------------------------------------------------


def test_grow_leaves_one_chunk_transient_free_after_the_commit():
    capacity, step = 1024, 8
    transient = 700 * MiB
    moe = _ArenaMoe(capacity, capacity, step)
    pool = _Pool(64)
    ctl, engine = _engine(pool, moe, transient=transient)

    ctl.grow_runtime_kv(128)

    assert pool.committed_pages == 128
    free_after_commit = engine._sync_get_memory()[0]
    # The pre-fix guarantee (cushion + 128 MiB = 384 MiB) is what OOMed the next chunk;
    # now the chunk after the grow has its whole measured transient, plus the margin.
    assert free_after_commit >= transient + PREFILL_HEADROOM_MARGIN_BYTES
    # ... and not a chunk more than it needs.
    assert free_after_commit < transient + PREFILL_HEADROOM_MARGIN_BYTES + step * GRANULE


def test_grow_without_a_transient_keeps_the_old_cushion_target():
    capacity, step = 1024, 8
    moe = _ArenaMoe(capacity, capacity, step)
    pool = _Pool(64)
    ctl, engine = _engine(pool, moe, transient=0)

    ctl.grow_runtime_kv(128)

    free_after_commit = engine._sync_get_memory()[0]
    old_target = VMM_COMMIT_CUSHION_BYTES + 128 * MiB
    assert old_target <= free_after_commit < old_target + step * GRANULE


def test_grow_refuses_and_rolls_back_when_the_floor_cannot_fund_the_transient():
    # 64 slots x 2 MiB = 128 MiB of arena above nothing: the arena can never free the
    # 700 MiB transient, so the commit is refused and the arena regrown.
    capacity, step = 64, 8
    moe = _ArenaMoe(capacity, capacity, step)
    pool = _Pool(64)
    ctl, engine = _engine(pool, moe, transient=700 * MiB)

    with pytest.raises(RuntimeError, match="unsafe VMM commit"):
        ctl.grow_runtime_kv(128)

    assert pool.committed_pages == 64
    assert moe.cache_size == capacity
    assert engine.config.moe_cache_size == capacity
    assert getattr(ctl, "_growable_transition_failed", False) is False


# --- startup park / fill ---------------------------------------------------------------


def test_park_goes_to_the_overlap_floor_rounded_to_a_boundary():
    moe = _ArenaMoe(1024, 1024, 8, num_experts=5)  # overlap floor 10 -> boundary 16
    ctl, engine = _engine(_Pool(64), moe)
    assert ctl.park_arena_for_startup() == (1024, 16, (1024 - 16) * GRANULE)
    assert engine.config.moe_cache_size == 16


def test_park_respects_the_mirror_coverage_floor():
    moe = _ArenaMoe(1024, 1024, 64, min_gpu_slots=100)
    ctl, _engine_ = _engine(_Pool(64), moe)
    old, parked, _released = ctl.park_arena_for_startup()
    assert (old, parked) == (1024, 128)


def test_fill_leaves_headroom_plus_margin_free_and_caps_at_capacity():
    capacity, step = 1024, 8
    transient = 700 * MiB
    moe = _ArenaMoe(capacity, capacity, step)
    # At full capacity the device would have 100 MiB free: the ratio plan's arena
    # does not leave one chunk's transient.
    ctl, engine = _engine(_Pool(64), moe, transient=transient, device_free_at_capacity=100 * MiB)
    ctl.park_arena_for_startup()

    usable, free_after, target_free = ctl.fill_arena_to_headroom(capacity)

    assert target_free == transient + PREFILL_HEADROOM_MARGIN_BYTES
    assert free_after >= target_free
    assert free_after - step * GRANULE < target_free  # one more chunk would breach it
    assert usable % step == 0 and usable < capacity
    assert moe.cache_size == usable == engine.config.moe_cache_size

    # With plenty of free VRAM the fill stops at the ratio plan's capacity.
    moe2 = _ArenaMoe(capacity, capacity, step)
    ctl2, _e2 = _engine(_Pool(64), moe2, transient=transient, device_free_at_capacity=4096 * MiB)
    ctl2.park_arena_for_startup()
    assert ctl2.fill_arena_to_headroom(capacity)[0] == capacity


def test_kv_shrink_never_regrows_the_arena_above_the_startup_fill():
    capacity, step = 1024, 8
    moe = _ArenaMoe(512, capacity, step)
    pool = _Pool(1024)
    ctl, engine = _engine(pool, moe, _growable_moe_ceiling=600)

    ctl.shrink_runtime_kv(64)  # releases 960 MiB: would fund all 1024 slots

    assert moe.cache_size == 600
