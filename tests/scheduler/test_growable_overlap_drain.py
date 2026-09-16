"""Design step 6: let ``overlap_loop`` host growable-KV grow/shrink behind an explicit
drain, gated by ``FREETOKEN_GROWABLE_OVERLAP`` (default off).

``_drain_inflight`` (factored out of the speculative-decoding drain in ``overlap_loop``)
is the no-forward-in-flight boundary ``grow_runtime_kv``/``shrink_runtime_kv`` need. These
tests exercise the two call sites that now use it -- the growth check inside
``_schedule_next_batch`` and the shrink gate in ``overlap_loop`` -- plus the ``run_forever``
loop-selection gate, on pure stubs in the manner of ``test_growable_shrink_release_guard``'s
``_SchedulerStub`` (only the collaborators the code under test does not own are replaced;
everything else is the genuine ``Scheduler`` method, bound off the class).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from freetoken.scheduler import scheduler as scheduler_mod
from freetoken.scheduler.scheduler import Scheduler


class _SchedulerStub:
    """A ``Scheduler`` with only its collaborators replaced.

    ``__getattr__`` binds every other attribute straight off the real ``Scheduler``, so
    ``overlap_loop``, ``_drain_inflight``, ``_schedule_next_batch``, ``_batch_needs_kv_growth``
    and ``_maybe_shrink_growable_kv`` are the genuine methods under test, not a
    reimplementation of them.
    """

    def __getattr__(self, name: str):
        return getattr(Scheduler, name).__get__(self, type(self))


class _CM:
    """Minimal cache-manager stub covering growth and shrink bookkeeping."""

    page_size = 1
    num_pages = 64
    is_hybrid = True

    def __init__(self, *, committed=32, used=8, occupied=24, required=32):
        self.committed_pages = committed
        self.used = used
        self.free_slots = list(range(committed - occupied))
        self._required = required
        self.added = []
        self.removed = []

    def committed_pages_required(self, _reqs) -> int:
        return self._required

    def add_committed_pages(self, new_pages: int) -> None:
        self.added.append(new_pages)
        self.committed_pages = new_pages

    def page_usage(self):
        return self.used, self.committed_pages

    def compact_active_pages(self, reqs, target, copy, retained_handles=()):
        return target

    def remove_committed_pages(self, target):
        self.removed.append(target)
        self.committed_pages = target


class _Stream:
    def __init__(self, calls: list, tag: str):
        self._calls = calls
        self._tag = tag

    def wait_stream(self, other) -> None:
        self._calls.append((f"{self._tag}.wait_stream",))

    def synchronize(self) -> None:
        self._calls.append((f"{self._tag}.synchronize",))


def _base_overlap_stub(calls: list) -> _SchedulerStub:
    """A stub wired to run ``overlap_loop`` one iteration with nothing scheduled by
    default: no messages, no pending checkpoint/rebuild, no spec, nothing runnable.
    Individual tests layer scheduling/shrink behavior on top."""
    obj = _SchedulerStub()
    obj.stream = _Stream(calls, "stream")
    obj.engine = SimpleNamespace(stream=_Stream(calls, "engine.stream"))
    obj.engine_stream_ctx = contextlib.nullcontext()
    obj._durable_checkpoint_sealed = False
    obj._expire_sessions = lambda: None
    obj._release_due_soft_sessions = lambda: None
    obj._enforce_session_host_reserve = lambda: None
    obj._sessions = {}
    obj._pending_rebuild = None
    obj._growable_shrink_pending = False
    obj._spec = None
    obj._admission_stalled = False
    obj.receive_msg = lambda blocking: []
    obj._only_idle_sessions = lambda last_data: False  # never take the time.sleep(0.01) nap
    obj._process_one_msg = lambda msg: None
    obj._execute_pending_durable_checkpoint = lambda: None
    obj._execute_pending_rebuild = lambda: None
    obj._maybe_resize_elastic_capacity = lambda: None
    obj._publish_scheduler_counters = lambda **kw: None
    obj._flush_abort_acks = lambda: calls.append(("flush_abort_acks",))
    obj._process_last_data = lambda last_data: calls.append(("process_last_data", last_data))
    obj._finalize_growable_handoff = lambda forward_input: None
    obj.prefill_manager = SimpleNamespace(runnable=False, schedule_next_batch=lambda *_: None)
    obj.decode_manager = SimpleNamespace(runnable=False, schedule_next_batch=lambda *_: None)
    obj.prefill_budget = 100
    obj._growable_decode_steps = 0
    obj._growable_decode_burst = 4
    obj._reclaim_for_blocked_prefill = lambda: False
    obj._note_prefix_admissions = lambda batch: None
    obj._report_prompt_admissions = lambda batch: None
    return obj


def test_growing_prefill_drains_before_grow_runtime_kv():
    """A prefill batch whose ``committed_pages_required`` exceeds the committed pages
    must see ``_drain_inflight`` (wait_stream + _process_last_data) run BEFORE
    ``grow_runtime_kv`` -- growth must land at a no-forward-in-flight boundary."""
    calls: list = []
    obj = _base_overlap_stub(calls)
    obj.config = SimpleNamespace(kv_grow_step_tokens=8, adaptive_scheduler=False)
    cm = _CM(committed=32, required=64)  # required > committed: growth needed
    obj.cache_manager = cm

    batch = SimpleNamespace(reqs=[SimpleNamespace(uid=1)], is_decode=False, is_prefill=True)
    obj.prefill_manager = SimpleNamespace(
        runnable=True, schedule_next_batch=lambda _budget: batch
    )
    obj.decode_manager = SimpleNamespace(runnable=False, schedule_next_batch=lambda: None)

    def fake_prepare_batch(b):
        # Stand-in for ``_prepare_batch``'s growth call (the rest of the real method
        # needs a full engine/tensor rig unrelated to this ordering question).
        required = cm.committed_pages_required(b.reqs)
        old, new = obj.engine.grow_runtime_kv(required)
        if new > old:
            cm.add_committed_pages(new)
        calls.append(("grow_runtime_kv", required))
        return SimpleNamespace(batch=b)

    obj._prepare_batch = fake_prepare_batch
    obj.engine.grow_runtime_kv = lambda required: (32, 64)
    obj._forward = lambda forward_input: "output"
    obj._restore_linear_states = lambda batch: None

    last_data = SimpleNamespace(marker="prev-forward")
    Scheduler.overlap_loop(obj, last_data)

    drain_idx = calls.index(("process_last_data", last_data))
    grow_idx = calls.index(("grow_runtime_kv", 64))
    assert drain_idx < grow_idx, (
        f"drain must precede grow_runtime_kv; got order {calls}"
    )
    assert cm.committed_pages == 64


def test_non_growing_decode_does_not_drain():
    """A decode iteration that needs no growth must NOT drain -- overlap is preserved
    on the common (non-resizing) path."""
    calls: list = []
    obj = _base_overlap_stub(calls)
    obj.config = SimpleNamespace(kv_grow_step_tokens=8, adaptive_scheduler=False)
    cm = _CM(committed=64, required=32)  # required <= committed: no growth
    obj.cache_manager = cm

    batch = SimpleNamespace(reqs=[SimpleNamespace(uid=1)], is_decode=True, is_prefill=False)
    obj.decode_manager = SimpleNamespace(runnable=True, schedule_next_batch=lambda: batch)
    obj.prefill_manager = SimpleNamespace(runnable=False, schedule_next_batch=lambda _b: None)

    prepare_calls = []

    def fake_prepare_batch(b):
        prepare_calls.append(b)
        return SimpleNamespace(batch=b)

    obj._prepare_batch = fake_prepare_batch
    obj._forward = lambda forward_input: "output"
    obj._restore_linear_states = lambda batch: None

    last_data = SimpleNamespace(marker="prev-forward")
    Scheduler.overlap_loop(obj, last_data)

    drain_count = sum(1 for c in calls if c[0] == "process_last_data")
    assert drain_count == 1, (
        "exactly one _process_last_data call is expected: the mandatory end-of-iteration "
        f"drain of the batch this same iteration just launched; got {calls}"
    )
    # The one call must be the END-of-iteration drain of ``last_data`` (overlap intact),
    # not an early growth-triggered drain -- there is nothing to grow this iteration.
    assert ("process_last_data", last_data) in calls
    assert calls.index(("process_last_data", last_data)) == len(calls) - 2, (
        f"expected the drain to be the last-but-one call (flush_abort_acks follows); got {calls}"
    )
    assert cm.added == []


def test_shrink_path_drains_before_shrink_runtime_kv():
    """A pending shrink must see the previous forward drained BEFORE
    ``shrink_runtime_kv`` runs -- otherwise ``_maybe_shrink_growable_kv``'s own
    ``_last_data is not None`` guard defers it forever under overlap."""
    calls: list = []
    obj = _base_overlap_stub(calls)
    obj.config = SimpleNamespace(kv_grow_step_tokens=8, page_size=1, adaptive_scheduler=False)
    obj._growable_shrink_pending = True
    cm = _CM(committed=32, used=8, occupied=24)
    obj.cache_manager = cm
    obj.prefill_manager = SimpleNamespace(
        runnable=False, schedule_next_batch=lambda _b: None, pending_list=[]
    )
    obj.decode_manager = SimpleNamespace(
        runnable=False, schedule_next_batch=lambda: None, running_reqs=[]
    )

    def shrink(target):
        calls.append(("shrink_runtime_kv", target))
        return cm.committed_pages, target

    obj.engine.shrink_runtime_kv = shrink
    obj.engine.moe_offload_cache = SimpleNamespace(cache_size=99)
    obj.engine.kv_cache = SimpleNamespace(copy_pages=lambda *_: None)
    obj._evict_growable_prefix_pages = lambda pages: (
        setattr(cm, "free_slots", cm.free_slots + list(range(pages))) or pages
    )

    last_data = SimpleNamespace(marker="prev-forward")
    Scheduler.overlap_loop(obj, last_data)

    drain_idx = calls.index(("process_last_data", last_data))
    shrink_idx = next(i for i, c in enumerate(calls) if c[0] == "shrink_runtime_kv")
    assert drain_idx < shrink_idx, f"drain must precede shrink_runtime_kv; got order {calls}"
    assert obj._growable_shrink_pending is False
    assert cm.removed, "the shrink must actually have run, not been deferred"


def test_shrink_in_flight_guard_still_refuses_release_under_overlap():
    """Requirement 4: ``_growable_shrink_in_flight`` (and the release guard it arms in
    ``_release_soft_session_handle``) must still work when the shrink is reached through
    ``overlap_loop`` (via the new drain), not only through ``normal_loop``."""
    calls: list = []
    obj = _base_overlap_stub(calls)
    obj.config = SimpleNamespace(kv_grow_step_tokens=8, page_size=1, adaptive_scheduler=False)
    obj._growable_shrink_pending = True
    cm = _CM(committed=32, used=8, occupied=24)
    obj.cache_manager = cm
    obj.prefill_manager = SimpleNamespace(
        runnable=False, schedule_next_batch=lambda _b: None, pending_list=[]
    )
    obj.decode_manager = SimpleNamespace(
        runnable=False, schedule_next_batch=lambda: None, running_reqs=[]
    )
    obj.engine.moe_offload_cache = SimpleNamespace(cache_size=99)
    obj.engine.kv_cache = SimpleNamespace(copy_pages=lambda *_: None)

    observed_in_flight = []

    def shrink(target):
        # Reentrant release attempt while the resize is "in flight" -- this is exactly
        # what a mamba_reclaim_hook / admission-pressure callback could try mid-shrink.
        observed_in_flight.append(bool(getattr(obj, "_growable_shrink_in_flight", False)))
        return cm.committed_pages, target

    obj.engine.shrink_runtime_kv = shrink
    obj._evict_growable_prefix_pages = lambda pages: (
        setattr(cm, "free_slots", cm.free_slots + list(range(pages))) or pages
    )

    lease = SimpleNamespace(
        reclaimable=True, active_uid=None, handle=object(),
        ttl_seconds=300.0, protected_until=None,
    )
    obj._sessions = {"idle": lease}
    unlocked = []
    cm.unlock = lambda handle: unlocked.append(handle)
    obj._spill_soft_session = lambda *_: True
    obj._invalidate_match_memo = lambda: None

    # A reentrant release attempt made from INSIDE the shrink (e.g. a mamba_reclaim_hook
    # fired by ``compact_active_pages``/``_evict_growable_prefix_pages``) must be refused
    # while ``_growable_shrink_in_flight`` is set. Exercise it explicitly at the point
    # the shrink is under way, via ``_evict_growable_prefix_pages``.
    release_during_shrink = []

    def evict(pages):
        release_during_shrink.append(
            Scheduler._release_soft_session_handle.__get__(obj, type(obj))(
                "idle", "reentrant", require_checkpoint=False
            )
        )
        cm.free_slots = cm.free_slots + list(range(pages))
        return pages

    obj._evict_growable_prefix_pages = evict

    last_data = SimpleNamespace(marker="prev-forward")
    Scheduler.overlap_loop(obj, last_data)

    assert observed_in_flight == [True], (
        "_growable_shrink_in_flight must be set for the duration of the shrink, "
        "including when reached via overlap_loop's new drain"
    )
    assert getattr(obj, "_growable_shrink_in_flight") is False, "must clear on exit"
    assert release_during_shrink == [False], (
        "a release attempted WHILE the shrink is in flight must be refused, "
        "regardless of which loop reached the shrink"
    )
    assert lease.handle is not None and unlocked == [], (
        "the lease must still be protected; the reentrant release must not have unlocked it"
    )

    # After the shrink window closes, the same release is free to run normally.
    original_handle = lease.handle
    refused = Scheduler._release_soft_session_handle.__get__(obj, type(obj))(
        "idle", "post-shrink", require_checkpoint=False
    )
    assert refused is True
    assert unlocked == [original_handle]
    assert lease.handle is None


def test_flag_off_selects_normal_loop_when_growable(monkeypatch):
    monkeypatch.setattr(scheduler_mod.ENV, "GROWABLE_OVERLAP", False)
    monkeypatch.setattr(scheduler_mod.ENV, "DISABLE_OVERLAP_SCHEDULING", False)

    class NormalSentinel(Exception):
        pass

    class OverlapSentinel(Exception):
        pass

    obj = SimpleNamespace(config=SimpleNamespace(kv_grow_step_tokens=8))
    obj.engine_stream_ctx = contextlib.nullcontext()
    obj.engine = SimpleNamespace(stream=SimpleNamespace(wait_stream=lambda s: None))
    obj.stream = object()

    def raise_normal():
        raise NormalSentinel()

    def raise_overlap(_data):
        raise OverlapSentinel()

    obj.normal_loop = raise_normal
    obj.overlap_loop = raise_overlap

    with pytest.raises(NormalSentinel):
        Scheduler.run_forever(obj)


def test_flag_on_selects_overlap_loop_when_growable(monkeypatch):
    monkeypatch.setattr(scheduler_mod.ENV, "GROWABLE_OVERLAP", True)
    monkeypatch.setattr(scheduler_mod.ENV, "DISABLE_OVERLAP_SCHEDULING", False)

    class OverlapSentinel(Exception):
        pass

    obj = SimpleNamespace(config=SimpleNamespace(kv_grow_step_tokens=8))
    obj.stream = object()
    monkeypatch.setattr(scheduler_mod.torch.cuda, "current_stream", lambda: obj.stream)

    def raise_overlap(_data):
        raise OverlapSentinel()

    obj.overlap_loop = raise_overlap

    with pytest.raises(OverlapSentinel):
        Scheduler.run_forever(obj)


def test_flag_has_no_effect_when_kv_grow_step_tokens_unset(monkeypatch):
    """The flag only relaxes the gate; with growable KV off entirely, overlap_loop is
    still the default regardless of FREETOKEN_GROWABLE_OVERLAP."""
    monkeypatch.setattr(scheduler_mod.ENV, "GROWABLE_OVERLAP", False)
    monkeypatch.setattr(scheduler_mod.ENV, "DISABLE_OVERLAP_SCHEDULING", False)

    class OverlapSentinel(Exception):
        pass

    obj = SimpleNamespace(config=SimpleNamespace(kv_grow_step_tokens=0))
    obj.stream = object()
    monkeypatch.setattr(scheduler_mod.torch.cuda, "current_stream", lambda: obj.stream)

    def raise_overlap(_data):
        raise OverlapSentinel()

    obj.overlap_loop = raise_overlap

    with pytest.raises(OverlapSentinel):
        Scheduler.run_forever(obj)
