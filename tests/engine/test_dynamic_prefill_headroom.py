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
    DECODE_FREE_TARGET_BYTES,
    PREFILL_HEADROOM_MARGIN_BYTES,
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
    monkeypatch.delenv("FREETOKEN_DECODE_FREE_TARGET_MB", raising=False)
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
    assert ctl.decode_free_target_bytes() == DECODE_FREE_TARGET_BYTES == 128 * MIB
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
    ctl, engine, moe, ledger = _controller(free_gib=1.125, size=512, capacity=2048)
    before, after = ctl.release_prefill_headroom()
    assert _no_cuda == ["empty_cache"]
    # Decode keeps only 128 MiB: 1.0 GiB = 512 rows go back.
    assert (before, after) == (512, 1024)
    assert ledger["free"] == ctl.decode_free_target_bytes()
    assert engine.config.moe_cache_size == 1024
    assert ctl._decode_level is True

    before, after = ctl.reserve_prefill_headroom()
    assert (before, after) == (1024, 512)
    assert ledger["free"] >= ctl.prefill_free_target_bytes()
    assert ctl._decode_level is False
    assert engine.config.moe_cache_size == 512


def test_release_never_leaves_less_than_the_decode_target():
    ctl, _engine, moe, ledger = _controller(free_gib=0.1, size=512)
    ctl.release_prefill_headroom()
    assert moe.calls == []  # 0.1 GiB < the 128 MiB decode level: nothing to give
    assert ledger["free"] == int(0.1 * GIB)


def test_release_keeps_the_override_level(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DECODE_FREE_TARGET_MB", "384")
    ctl, _engine, moe, ledger = _controller(free_gib=1.125, size=512)
    ctl.release_prefill_headroom()
    # The old 0.375 GiB level, by override: 0.75 GiB = 384 rows go back.
    assert moe.cache_size == 896
    assert ledger["free"] == 384 * MIB


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


def test_off_without_growable_kv_or_a_prefill_level_above_the_decode_level(monkeypatch):
    ctl, engine, *_ = _controller()
    engine.config.kv_grow_step_tokens = 0
    assert not ctl.dynamic_enabled()
    # A transient below the 256 MiB cushion still holds the prefill level (cushion +
    # margin, 0.375 GiB) above the 128 MiB decode level: the headroom moves.
    ctl, engine, *_ = _controller(transient_gib=0.1)
    assert ctl.dynamic_enabled()
    # With the decode level overridden up to the prefill level, both are the same.
    monkeypatch.setenv("FREETOKEN_DECODE_FREE_TARGET_MB", "384")
    ctl, engine, *_ = _controller(transient_gib=0.1)
    assert not ctl.dynamic_enabled()


def test_scheduler_drains_before_moving_the_headroom():
    from freetoken.scheduler.scheduler import Scheduler

    order = []
    sched = SimpleNamespace(
        engine=SimpleNamespace(
            prefill_headroom_transition=lambda prefill, prefill_pending, new_tokens=None: (
                order.append(("ask", new_tokens)) or ("reserve" if prefill else None)
            ),
            apply_prefill_headroom=lambda kind: order.append(("apply", kind)),
        ),
        prefill_manager=SimpleNamespace(runnable=True),
        _last_data="inflight",
    )
    sched._drain_inflight = lambda last: order.append(("drain", last))
    Scheduler._move_prefill_headroom(sched, SimpleNamespace(is_prefill=True, log_new_tokens=700))
    assert order == [("ask", 700), ("drain", "inflight"), ("apply", "reserve")]
    assert sched._last_data is None

    order.clear()
    Scheduler._move_prefill_headroom(sched, SimpleNamespace(is_prefill=False, log_new_tokens=0))
    assert order == [("ask", None)]


def test_scheduler_tolerates_stub_engines():
    from freetoken.scheduler.scheduler import Scheduler

    sched = SimpleNamespace(engine=SimpleNamespace(), prefill_manager=None, _last_data=None)
    Scheduler._move_prefill_headroom(sched, SimpleNamespace(is_prefill=True))


def _kv_shrink(ctl, engine, ledger, returned_bytes):
    """Run the real ``_shrink_runtime_kv_arena`` (a request's teardown shrink) over a
    KV pool stub that hands ``returned_bytes`` back to the free-VRAM ledger."""
    engine.kv_cache = SimpleNamespace(
        decommit_pages=lambda _pages: ledger.__setitem__("free", ledger["free"] + returned_bytes)
    )
    engine.graph_runner = object()
    return ctl._shrink_runtime_kv_arena(
        old_pages=131072, target_pages=65536, old_moe=engine.moe_offload_cache.cache_size,
        old_overlap=True, kv_bytes=returned_bytes, old_kv_bytes=2 * returned_bytes,
    )


def test_release_then_teardown_shrink_still_reserves_before_the_next_prefill():
    """dyn-g5 (6ec54b1): 80K request -> release to the decode level -> the teardown KV
    shrink -> the next prefill found no reserve and OOMed in mamba2 prefill on native
    Linux. The shrink must leave the decode level standing."""
    ctl, engine, moe, ledger = _controller(free_gib=1.125, size=512)
    engine._growable_moe_ceiling = 512          # the startup fill
    # A long request's prefill ran at the prefill level; its decode takes the slots back.
    assert ctl.prefill_headroom_transition(prefill=False, prefill_pending=False) == "release"
    ctl.release_prefill_headroom()
    assert moe.cache_size == 1024
    # Teardown: KV 131072 -> 65536 returns 0.21 GiB. The arena is above the startup
    # fill, so it does not move, and free VRAM is well below the prefill target.
    _kv_shrink(ctl, engine, ledger, int(0.21 * GIB))
    assert moe.cache_size == 1024
    assert ledger["free"] < ctl.prefill_free_target_bytes()
    # The next request's first prefill chunk must reserve.
    assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True) == "reserve"
    ctl.reserve_prefill_headroom()
    assert ledger["free"] >= ctl.prefill_free_target_bytes()


def test_teardown_shrink_at_the_prefill_level_needs_no_reserve():
    # Shrink while still at the prefill level (no decode in between): free only rises.
    ctl, engine, moe, ledger = _controller(free_gib=1.125, size=512)
    engine._growable_moe_ceiling = 512
    _kv_shrink(ctl, engine, ledger, int(0.21 * GIB))
    assert ledger["free"] >= ctl.prefill_free_target_bytes()
    assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True) is None


def _small(ctl, engine, tokens=1024, gib=0.1):
    engine.prefill_transient_small = (tokens, int(gib * GIB))


def test_short_batch_reserves_only_the_short_chunk():
    """A batch that forwards <= the small measured chunk (a short prompt, a tool result)
    reserves that chunk's headroom (VMM cushion 256 MiB here, above its 0.1 GiB
    transient) instead of the full chunk's 1 GiB: fewer slots unmapped and written back."""
    ctl, engine, moe, ledger = _controller(free_gib=1.125, size=512, capacity=2048)
    _small(ctl, engine)
    ctl.release_prefill_headroom()
    assert moe.cache_size == 1024
    assert ctl.prefill_free_target_bytes(1000) == 256 * MIB + PREFILL_HEADROOM_MARGIN_BYTES
    assert ctl.prefill_free_target_bytes(None) == GIB + PREFILL_HEADROOM_MARGIN_BYTES
    assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True, new_tokens=1000) == "reserve"
    ctl.reserve_prefill_headroom()
    assert ledger["free"] >= ctl.prefill_free_target_bytes(1000)
    assert ledger["free"] < ctl.prefill_free_target_bytes()
    small_slots = moe.cache_size
    assert 512 < small_slots < 1024
    # Another short batch: already covered.
    assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True, new_tokens=1024) is None
    # A full chunk (or an unknown size) needs the full reserve on top.
    assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True, new_tokens=1025) == "reserve"
    ctl.reserve_prefill_headroom()
    assert ledger["free"] >= ctl.prefill_free_target_bytes()
    assert moe.cache_size < small_slots
    # The full level covers everything, short or not.
    for n in (1, 1024, 8192, None):
        assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True, new_tokens=n) is None
    # Decode takes it all back; the next short batch reserves the short level again.
    assert ctl.prefill_headroom_transition(prefill=False, prefill_pending=False) == "release"
    ctl.release_prefill_headroom()
    assert ctl.prefill_headroom_transition(prefill=True, prefill_pending=True, new_tokens=None) == "reserve"
    ctl.reserve_prefill_headroom()
    assert ledger["free"] >= ctl.prefill_free_target_bytes()


def test_no_small_measurement_prices_the_full_chunk():
    ctl, engine, *_ = _controller()
    assert ctl.prefill_free_target_bytes(10) == ctl.prefill_free_target_bytes()
    # A small measurement above the full one is not trusted either.
    _small(ctl, engine, gib=2.0)
    assert ctl.prefill_free_target_bytes(10) == ctl.prefill_free_target_bytes()
    _small(ctl, engine, tokens=0)
    assert ctl.prefill_free_target_bytes(10) == ctl.prefill_free_target_bytes()
