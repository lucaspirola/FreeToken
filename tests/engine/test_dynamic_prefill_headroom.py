"""Dynamic prefill headroom (engine/growable_kv.py): the expert arena holds one prefill
chunk's transient free only while prefill runs; decode takes the slots back.

Real ``GrowableKvController`` over a stub engine and a byte-exact arena stub (2 MiB rows,
so every boundary's byte count is exact). ``torch.cuda`` is replaced in the controller's
module only: nothing here touches a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine import growable_kv
from freetoken.engine.growable_kv import (
    PREFILL_HEADROOM_MARGIN_BYTES,
    VMM_COMMIT_CUSHION_BYTES,
    GrowableKvController,
)

MIB = 1024 * 1024
GIB = 1024 * MIB
ROW = 2 * MIB


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(
        growable_kv,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda *_: None,
                empty_cache=lambda: calls.append("empty_cache"),
                mem_get_info=lambda *_: (0, 0),
                memory_allocated=lambda *_: 0,
                memory_reserved=lambda *_: 0,
            ),
            inference_mode=lambda: (lambda f: f),
        ),
    )
    monkeypatch.delenv("FREETOKEN_DYNAMIC_PREFILL_HEADROOM", raising=False)
    return calls


class _Arena:
    """Arena of ``capacity`` slots in chunks of ``step``; ``set_usable_slots`` moves the
    shared free-VRAM ledger by the exact bytes it maps or unmaps."""

    num_experts = 8

    def __init__(self, ledger, *, capacity=1024, step=16, size=512, min_gpu=0):
        self.ledger = ledger
        self.arena_layout = (capacity, step)
        self.bank_row_bytes = [ROW]
        self.cache_size = size
        self.prefill_overlap = True
        self.residency = SimpleNamespace(min_gpu_slots=lambda: min_gpu)
        self.calls = []

    def set_usable_slots(self, n):
        assert n % self.arena_layout[1] == 0 or n == 2 * self.num_experts, n
        delta = (self.cache_size - n) * ROW
        self.ledger["free"] += delta
        self.calls.append(n)
        self.cache_size = n
        return abs(delta)


def _controller(*, free_gib=0.5, transient_gib=1.0, **arena_kw):
    ledger = {"free": int(free_gib * GIB)}
    moe = _Arena(ledger, **arena_kw)
    engine = SimpleNamespace(
        moe_offload_cache=moe,
        device="cuda",
        prefill_transient_bytes=int(transient_gib * GIB),
        _growable_moe_ceiling=moe.cache_size,
        config=SimpleNamespace(
            kv_grow_step_tokens=65536,
            moe_cache_size=moe.cache_size,
            tp_info=SimpleNamespace(size=1),
        ),
        _sync_get_memory=lambda: (ledger["free"], ledger["free"]),
    )
    ctl = GrowableKvController(engine)
    engine.growable_kv = ctl
    return ctl, engine, moe, ledger


def test_levels():
    ctl, *_ = _controller()
    assert ctl.decode_free_target_bytes() == VMM_COMMIT_CUSHION_BYTES + PREFILL_HEADROOM_MARGIN_BYTES
    assert ctl.prefill_free_target_bytes() == GIB + PREFILL_HEADROOM_MARGIN_BYTES


def test_transitions_follow_the_batch_kind():
    ctl, *_ = _controller()
    # After the startup fill the arena sits at the prefill level.
    assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True) is None
    assert ctl.prefill_headroom_transition(prefill=False, prefill_pending=True) is None
    assert ctl.prefill_headroom_transition(prefill=False, prefill_pending=False) == "release"
    ctl._decode_level = True
    assert ctl.prefill_headroom_transition(prefill=False, prefill_pending=False) is None
    assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True) == "reserve"


def test_release_then_reserve_round_trip(_no_cuda):
    # Startup fill left the transient + margin free (1.125 GiB).
    ctl, engine, moe, ledger = _controller(free_gib=1.125, size=512)
    before, after = ctl.release_prefill_headroom()
    assert _no_cuda == ["empty_cache"]
    # Decode keeps only cushion + margin (0.375 GiB): 0.75 GiB = 384 rows go back.
    assert (before, after) == (512, 896)
    assert ledger["free"] == ctl.decode_free_target_bytes()
    assert engine.config.moe_cache_size == 896
    assert ctl._decode_level is True

    before, after = ctl.reserve_prefill_headroom()
    assert (before, after) == (896, 512)
    assert ledger["free"] >= ctl.prefill_free_target_bytes()
    assert ctl._decode_level is False
    assert engine.config.moe_cache_size == 512


def test_release_never_leaves_less_than_the_decode_target():
    ctl, _engine, moe, ledger = _controller(free_gib=0.3, size=512)
    ctl.release_prefill_headroom()
    assert moe.calls == []  # 0.3 GiB < the 0.375 GiB decode level: nothing to give
    assert ledger["free"] == int(0.3 * GIB)


def test_release_stops_at_capacity():
    ctl, _engine, moe, ledger = _controller(free_gib=8, size=512, capacity=1024)
    ctl.release_prefill_headroom()
    assert moe.cache_size == 1024


def test_reserve_respects_the_coverage_floor():
    # A bounded mirror needs 900 GPU slots: the arena cannot free the whole transient.
    ctl, _engine, moe, ledger = _controller(free_gib=0.375, size=960, min_gpu=900)
    ctl._decode_level = True
    ctl.reserve_prefill_headroom()
    assert moe.cache_size == 912  # 900 rounded up to the 16-slot chunk
    assert moe.cache_size >= 900


def test_reserve_is_a_no_op_when_free_already_covers_the_chunk():
    ctl, _engine, moe, _ledger = _controller(free_gib=2, size=512)
    ctl._decode_level = True
    assert ctl.reserve_prefill_headroom() == (512, 512)
    assert moe.calls == []
    assert ctl._decode_level is False


@pytest.mark.parametrize("value", ["0", "off"])
def test_env_restores_the_static_reservation(monkeypatch, value):
    monkeypatch.setenv("FREETOKEN_DYNAMIC_PREFILL_HEADROOM", value)
    ctl, *_ = _controller()
    assert ctl.prefill_headroom_transition(prefill=False, prefill_pending=False) is None


def test_off_without_growable_kv_or_a_measured_transient_above_the_cushion():
    ctl, engine, *_ = _controller()
    engine.config.kv_grow_step_tokens = 0
    assert not ctl.dynamic_enabled()
    ctl, engine, *_ = _controller(transient_gib=0.1)  # below the 256 MiB cushion
    assert not ctl.dynamic_enabled()


def test_scheduler_drains_before_moving_the_headroom():
    from freetoken.scheduler.scheduler import Scheduler

    order = []
    sched = SimpleNamespace(
        engine=SimpleNamespace(
            prefill_headroom_transition=lambda prefill, prefill_pending: (
                "reserve" if prefill else None
            ),
            apply_prefill_headroom=lambda kind: order.append(("apply", kind)),
        ),
        prefill_manager=SimpleNamespace(runnable=True),
        _last_data="inflight",
    )
    sched._drain_inflight = lambda last: order.append(("drain", last))
    Scheduler._move_prefill_headroom(sched, SimpleNamespace(is_prefill=True))
    assert order == [("drain", "inflight"), ("apply", "reserve")]
    assert sched._last_data is None

    order.clear()
    Scheduler._move_prefill_headroom(sched, SimpleNamespace(is_prefill=False))
    assert order == []


def test_scheduler_tolerates_stub_engines():
    from freetoken.scheduler.scheduler import Scheduler

    sched = SimpleNamespace(engine=SimpleNamespace(), prefill_manager=None, _last_data=None)
    Scheduler._move_prefill_headroom(sched, SimpleNamespace(is_prefill=True))
