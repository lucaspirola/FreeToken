"""Pure-stub policy tests for queued-agent KV shrink handoff."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[2] / "python/freetoken/scheduler/scheduler.py"


def _methods(*names):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    nodes = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    ns = {
        "math": math,
        "time": SimpleNamespace(monotonic=lambda: 10.0),
        "logger": SimpleNamespace(
            info_rank0=lambda *a, **k: None,
            debug_rank0=lambda *a, **k: None,
        ),
        "SessionLease": object,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return SimpleNamespace(**{name: ns[name] for name in names})


class _CM:
    page_size = 1
    num_pages = 64
    is_hybrid = True

    def __init__(self, committed=32, used=8, occupied=24):
        self.committed_pages = committed
        self.used = used
        self.free_slots = list(range(committed - occupied))
        self.removed = []
        self.compact_reqs = None

    def page_usage(self):
        return self.used, self.committed_pages

    def compact_active_pages(self, reqs, target, copy, retained_handles=()):
        self.compact_reqs = list(reqs)
        self.compact_handles = list(retained_handles)
        return target

    def remove_committed_pages(self, target):
        self.removed.append(target)
        self.committed_pages = target


def _pending(n, output=1, *, chunked=None, sid="incoming"):
    return SimpleNamespace(
        input_len=n, output_len=output, chunked_req=chunked, session_id=sid
    )


def _scheduler(*, pending, cm=None, shrink_error=None):
    methods = _methods(
        "_maybe_shrink_growable_kv",
        "_growable_handoff_demand_pages",
        "_evict_growable_prefix_pages",
        "_release_soft_session_handle",
        "_elastic_retained_session_handles",
    )
    cm = cm or _CM()
    calls = []

    def shrink(target):
        calls.append(target)
        if shrink_error:
            raise shrink_error
        return cm.committed_pages, target

    obj = SimpleNamespace(
        config=SimpleNamespace(kv_grow_step_tokens=8, page_size=1),
        _growable_shrink_pending=True,
        cache_manager=cm,
        prefill_manager=SimpleNamespace(
            pending_list=pending,
            finishability_reservation=lambda: sum(
                max(0, p.input_len - p.chunked_req.cached_len) + p.output_len
                for p in pending if p.chunked_req is not None
            ),
        ),
        decode_manager=SimpleNamespace(running_reqs=[]),
        engine=SimpleNamespace(
            kv_cache=SimpleNamespace(copy_pages=lambda *_: None),
            moe_offload_cache=SimpleNamespace(cache_size=99),
            shrink_runtime_kv=shrink,
        ),
        _sessions={},
        _last_data=None,
        stream=SimpleNamespace(synchronize=lambda: None),
        _elastic_live_requests=lambda: [
            p.chunked_req for p in pending if p.chunked_req is not None
        ],
        _evict_growable_prefix_pages=lambda pages: (
            setattr(cm, "free_slots", cm.free_slots + list(range(pages))) or pages
        ),
        _release_soft_session_handle=lambda *a, **k: True,
    )
    obj._growable_handoff_demand_pages = methods._growable_handoff_demand_pages.__get__(obj)
    obj._elastic_retained_session_handles = (
        methods._elastic_retained_session_handles.__get__(obj)
    )
    obj.engine.stream = SimpleNamespace(synchronize=lambda: None)
    return obj, methods, calls


def test_long_to_short_queued_handoff_shrinks_one_or_more_steps():
    obj, methods, calls = _scheduler(pending=[_pending(4, 1)])
    methods._maybe_shrink_growable_kv(obj)
    assert calls and calls[0] <= 24
    assert obj.cache_manager.removed == [calls[0]]


def test_shrink_supplies_all_resident_session_handles_including_protected():
    obj, methods, _calls = _scheduler(pending=[])
    idle = object()
    protected = object()
    obj._sessions = {
        "idle": SimpleNamespace(handle=idle),
        "protected": SimpleNamespace(handle=protected),
        "spilled": SimpleNamespace(handle=None),
        "duplicate": SimpleNamespace(handle=idle),
    }

    methods._maybe_shrink_growable_kv(obj)

    assert {id(handle) for handle in obj.cache_manager.compact_handles} == {
        id(idle), id(protected)
    }
    assert len(obj.cache_manager.compact_handles) == 2


def test_large_incoming_agent_avoids_pointless_shrink_and_regrowth():
    obj, methods, calls = _scheduler(pending=[_pending(30, 4)])
    methods._maybe_shrink_growable_kv(obj)
    assert calls == []
    assert obj.cache_manager.removed == []


def test_handoff_releases_long_outgoing_session_to_reach_active_agent_floor():
    cm = _CM(committed=64, used=41, occupied=45)
    obj, methods, calls = _scheduler(pending=[_pending(4, 1)], cm=cm)
    lease = SimpleNamespace(
        last_used_at=1,
        reclaimable=True,
        active_uid=None,
        handle=object(),
    )
    obj._sessions = {"outgoing": lease}
    released = []

    def release(sid, _reason, *, require_checkpoint):
        assert require_checkpoint
        released.append(sid)
        lease.handle = None
        cm.used = 1
        return True

    obj._release_soft_session_handle = release
    methods._maybe_shrink_growable_kv(obj)

    assert released == ["outgoing"]
    assert calls == [8]
    assert cm.removed == [8]


def test_intrinsically_large_incoming_skips_before_spilling_outgoing_session():
    cm = _CM(committed=64, used=41, occupied=45)
    obj, methods, calls = _scheduler(pending=[_pending(60, 4)], cm=cm)
    obj._sessions = {
        "outgoing": SimpleNamespace(
            last_used_at=1, reclaimable=True, active_uid=None, handle=object()
        )
    }
    releases = []
    obj._release_soft_session_handle = lambda *a, **k: releases.append(a) or True

    methods._maybe_shrink_growable_kv(obj)

    assert releases == []
    assert calls == []


def test_undrained_overlap_batch_defers_without_consuming_or_mutating():
    obj, methods, calls = _scheduler(pending=[_pending(4, 1)])
    obj._last_data = object()
    effects = []
    obj.stream = SimpleNamespace(synchronize=lambda: effects.append("scheduler-sync"))
    obj.engine.stream = SimpleNamespace(synchronize=lambda: effects.append("engine-sync"))
    obj._release_soft_session_handle = lambda *a, **k: effects.append("spill") or True
    obj._evict_growable_prefix_pages = lambda n: effects.append("evict") or n
    obj.cache_manager.compact_active_pages = lambda *a: effects.append("copy") or 8

    methods._maybe_shrink_growable_kv(obj)

    assert obj._growable_shrink_pending is True
    assert effects == []
    assert calls == []


def test_supported_growable_dispatch_uses_drained_normal_loop():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_forever")
    normal = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "normal_loop")
    run_source = ast.unparse(run)
    normal_source = ast.unparse(normal)

    assert "self.config.kv_grow_step_tokens" in run_source
    assert "self.normal_loop()" in run_source
    # normal_loop finishes and synchronizes each launched batch in the same iteration;
    # the next iteration's shrink runs before another batch is scheduled.
    assert normal_source.index("self._maybe_shrink_growable_kv()") < normal_source.index(
        "self._schedule_next_batch()"
    )
    assert normal_source.index("self._schedule_next_batch()") < normal_source.index(
        "self._process_last_data(ongoing_data)"
    )


def test_chunked_pending_borrower_is_enumerated_for_compaction():
    chunked = SimpleNamespace(cached_len=10, table_idx=3)
    obj, methods, calls = _scheduler(pending=[_pending(14, 1, chunked=chunked)])
    methods._maybe_shrink_growable_kv(obj)
    assert calls
    assert obj.cache_manager.compact_reqs == [chunked]


def test_failed_shrink_never_advances_cache_manager_accounting():
    obj, methods, _ = _scheduler(
        pending=[_pending(2)], shrink_error=MemoryError("injected shrink failure")
    )
    with pytest.raises(MemoryError, match="shrink failure"):
        methods._maybe_shrink_growable_kv(obj)
    assert obj.cache_manager.committed_pages == 32
    assert obj.cache_manager.removed == []


def test_handoff_checkpoint_failure_keeps_protected_handle():
    methods = _methods("_release_soft_session_handle")
    handle = object()
    lease = SimpleNamespace(
        reclaimable=True,
        active_uid=None,
        handle=handle,
        token_ids=[1],
        protected_until=None,
        ttl_seconds=60,
    )
    unlocks = []
    obj = SimpleNamespace(
        _sessions={"A": lease},
        _spill_soft_session=lambda *_: False,
        cache_manager=SimpleNamespace(unlock=lambda h: unlocks.append(h)),
    )
    assert not methods._release_soft_session_handle(
        obj, "A", "queued-agent handoff", require_checkpoint=True
    )
    assert lease.handle is handle
    assert unlocks == []


def test_a_to_b_to_a_handoff_releases_only_after_valid_checkpoint():
    methods = _methods("_release_soft_session_handle")
    handle = object()
    record = SimpleNamespace(valid=True, token_ids=(1, 2, 3))
    lease = SimpleNamespace(
        reclaimable=True,
        active_uid=None,
        handle=handle,
        spill=None,
        token_ids=[1, 2, 3],
        protected_until=5,
        expires_at=None,
        ttl_seconds=60,
    )
    unlocks = []

    def checkpoint(_sid, session):
        session.spill = record
        return True

    obj = SimpleNamespace(
        _sessions={"A": lease},
        _spill_soft_session=checkpoint,
        cache_manager=SimpleNamespace(unlock=lambda h: unlocks.append(h)),
        _invalidate_match_memo=lambda: None,
        _growable_shrink_pending=False,
    )
    assert methods._release_soft_session_handle(
        obj, "A", "queued-agent handoff", require_checkpoint=True
    )
    assert unlocks == [handle]
    assert lease.handle is None
    assert lease.spill is record and lease.spill.valid
    # The later A turn still has the exact checkpoint tokens needed by restore matching.
    assert lease.spill.token_ids == (1, 2, 3)
