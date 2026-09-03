from __future__ import annotations

import time
from types import SimpleNamespace

from freetoken.scheduler.scheduler import Scheduler, SessionLease


class _Cache:
    def __init__(self) -> None:
        self.cached = []
        self.retained = []
        self.unlocked = []

    def cache_req(self, req, *, finished: bool) -> None:
        self.cached.append((req.uid, finished))

    def retain_prefix(self, input_ids, cached_len: int):
        handle = f"handle-{len(self.retained)}"
        self.retained.append((input_ids, cached_len, handle))
        return handle

    def unlock(self, handle) -> None:
        self.unlocked.append(handle)


def _scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = _Cache()
    scheduler.table_manager = SimpleNamespace(free=lambda _idx: None)
    scheduler.config = SimpleNamespace(
        kv_grow_step_tokens=65_536,
        auto_session_grace_seconds=0.0,
        adaptive_scheduler=False,
    )
    scheduler._sessions = {}
    scheduler._growable_shrink_pending = False
    scheduler._pending_abort_acks = set()
    scheduler._last_data = None
    scheduler.prefill_manager = SimpleNamespace(runnable=False, abort_req=lambda _uid: None)
    scheduler.decode_manager = SimpleNamespace(runnable=False, abort_req=lambda _uid: None)
    return scheduler


def _req(uid: int, session_id: str):
    return SimpleNamespace(
        uid=uid,
        session_id=session_id,
        session_ttl_seconds=30.0,
        mm_embeds=None,
        input_ids=[1, 2, 3, 4],
        cached_len=3,
        table_idx=7,
    )


def test_normal_turn_replaces_but_does_not_release_session_lease():
    scheduler = _scheduler()
    scheduler._sessions["agent"] = SessionLease("old", 30.0, active_uid=9)
    req = _req(9, "agent")

    scheduler._free_req_resources(req, retain_session=True)

    lease = scheduler._sessions["agent"]
    assert lease.handle == "handle-0"
    assert lease.active_uid is None
    assert lease.expires_at is not None
    assert scheduler.cache_manager.unlocked == ["old"]
    assert req.table_idx == -1


def test_close_releases_retained_kv_and_requests_growable_shrink():
    scheduler = _scheduler()
    scheduler._sessions["helper"] = SessionLease("helper-handle", 30.0)

    scheduler._close_session("helper")

    assert "helper" not in scheduler._sessions
    assert scheduler.cache_manager.unlocked == ["helper-handle"]
    assert scheduler._growable_shrink_pending is True


def test_idle_expiry_closes_only_inactive_sessions():
    scheduler = _scheduler()
    scheduler._sessions = {
        "expired": SessionLease("expired-handle", 1.0, expires_at=time.monotonic() - 1),
        "active": SessionLease("active-handle", 1.0, expires_at=time.monotonic() - 1, active_uid=2),
    }

    scheduler._expire_sessions()

    assert set(scheduler._sessions) == {"active"}
    assert scheduler.cache_manager.unlocked == ["expired-handle"]


def test_soft_session_releases_protection_after_grace_but_keeps_identity():
    scheduler = _scheduler()
    scheduler.config.auto_session_grace_seconds = 30.0
    scheduler.prefill_manager.runnable = True  # the timer is armed only under demand
    scheduler._sessions["auto"] = SessionLease(
        "soft-handle",
        300.0,
        expires_at=time.monotonic() + 200,
        reclaimable=True,
        protected_until=time.monotonic() - 1,
    )

    scheduler._release_due_soft_sessions()

    lease = scheduler._sessions["auto"]
    assert lease.handle is None
    assert lease.expires_at is not None
    assert scheduler.cache_manager.unlocked == ["soft-handle"]
    assert scheduler._growable_shrink_pending is True


def test_explicit_session_remains_protected_past_soft_grace():
    scheduler = _scheduler()
    scheduler.config.auto_session_grace_seconds = 30.0
    scheduler.prefill_manager.runnable = True
    scheduler._sessions["hard"] = SessionLease(
        "hard-handle",
        300.0,
        expires_at=time.monotonic() + 200,
        reclaimable=False,
        protected_until=time.monotonic() - 1,
    )

    scheduler._release_due_soft_sessions()

    assert scheduler._sessions["hard"].handle == "hard-handle"
    assert scheduler.cache_manager.unlocked == []


def test_admission_pressure_releases_oldest_idle_soft_session_only():
    scheduler = _scheduler()

    class _PressureCache(_Cache):
        is_hybrid = False

        @property
        def available_size(self):
            return 100 if self.unlocked else 0

        def match_req(self, _req):
            return SimpleNamespace(cuda_handle=SimpleNamespace(cached_len=0))

    scheduler.cache_manager = _PressureCache()
    scheduler._sessions = {
        "old-soft": SessionLease(
            "old-handle", 300.0, reclaimable=True, last_used_at=1.0
        ),
        "new-soft": SessionLease(
            "new-handle", 300.0, reclaimable=True, last_used_at=2.0
        ),
        "hard": SessionLease(
            "hard-handle", 300.0, reclaimable=False, last_used_at=0.0
        ),
    }
    msg = SimpleNamespace(
        uid=8,
        session_id="incoming",
        input_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(max_tokens=4),
    )

    scheduler._reclaim_soft_sessions_for_admission(msg)

    assert scheduler.cache_manager.unlocked == ["old-handle"]
    assert scheduler._sessions["old-soft"].handle is None
    assert scheduler._sessions["new-soft"].handle == "new-handle"
    assert scheduler._sessions["hard"].handle == "hard-handle"


class _SpillStore:
    """Minimal cold-tier double: records what the scheduler asks it to do."""

    def __init__(self) -> None:
        self.records: dict[str, object] = {}
        self.discarded: list[object] = []
        self.touched: list[object] = []

    def spill(self, session_id, token_ids, page_indices, linear_slot):
        record = SimpleNamespace(
            session_id=session_id,
            num_pages=len(page_indices),
            tier="ram",
            byte_size=1 << 20,
            token_ids=token_ids,
        )
        self.records[session_id] = record
        return record

    @property
    def num_records(self) -> int:
        return len(self.records)

    def get(self, session_id):
        return self.records.get(session_id)

    def touch(self, record) -> None:
        self.touched.append(record)

    def discard(self, record) -> None:
        if record is None:
            return
        self.discarded.append(record)
        self.records.pop(getattr(record, "session_id", None), None)


class _SessionCache(_Cache):
    """Cache whose KV is exhausted until a session lease is released."""

    is_hybrid = False

    def __init__(self) -> None:
        super().__init__()
        self.free = 0

    @property
    def available_size(self):
        return self.free

    def match_req(self, _req):
        return SimpleNamespace(cuda_handle=SimpleNamespace(cached_len=0))

    def retain_prefix(self, input_ids, cached_len: int):
        handle = SimpleNamespace(
            cached_len=len(input_ids),
            node=SimpleNamespace(mamba_value=3),
            get_matched_indices=lambda: [0, 1, 2, 3],
        )
        self.retained.append((input_ids, cached_len, handle))
        return handle

    def unlock(self, handle) -> None:
        super().unlock(handle)
        if any(handle is retained for _ids, _len, retained in self.retained):
            self.free = 4096  # releasing the retained prefix is what unblocks admission


class _FakePrefillManager:
    def __init__(self, cache) -> None:
        self.cache = cache
        self.pending_list: list = []
        self.admitted: list[int] = []

    @property
    def runnable(self) -> bool:
        return bool(self.pending_list)

    def schedule_next_batch(self, _budget):
        if not self.pending_list:
            return None
        pending = self.pending_list[0]
        if pending.input_len + pending.output_len > self.cache.available_size:
            return None  # admission fails for lack of KV pages
        self.pending_list.pop(0)
        self.admitted.append(pending.uid)
        return SimpleNamespace(uid=pending.uid)

    def abort_req(self, _uid):
        return None


def _queued(uid: int, session_id: str):
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(uid, [1, 2, 3, 4], SimpleNamespace(max_tokens=8), session_id=session_id)


def _demand_scheduler():
    scheduler = _scheduler()
    scheduler.cache_manager = _SessionCache()
    scheduler.prefill_manager = _FakePrefillManager(scheduler.cache_manager)
    scheduler.decode_manager = SimpleNamespace(
        runnable=False, schedule_next_batch=lambda: None, abort_req=lambda _uid: None
    )
    scheduler._session_spill_store = _SpillStore()
    scheduler._session_spill_last_pressure_check = 0.0
    scheduler.prefill_budget = 8192
    scheduler._growable_decode_steps = 0
    scheduler._prepare_batch = lambda batch: batch
    scheduler._report_prompt_admissions = lambda batch: None
    return scheduler


def test_idle_session_is_not_released_while_nothing_is_queued():
    scheduler = _demand_scheduler()
    scheduler.config.auto_session_grace_seconds = 30.0
    scheduler.cache_manager.free = 4096  # room to spare: no memory pressure either
    scheduler._sessions["auto"] = SessionLease(
        "soft-handle", 300.0, reclaimable=True, protected_until=time.monotonic() - 1
    )

    scheduler._release_due_soft_sessions()

    assert scheduler._sessions["auto"].handle == "soft-handle"
    assert scheduler.cache_manager.unlocked == []


def test_grace_timer_is_disabled_by_default_even_under_demand():
    scheduler = _demand_scheduler()
    scheduler.prefill_manager.pending_list.append(_queued(2, "other"))
    scheduler._sessions["auto"] = SessionLease(
        "soft-handle", 300.0, reclaimable=True, protected_until=time.monotonic() - 1
    )

    scheduler._release_due_soft_sessions()  # grace 0 == disabled

    assert scheduler._sessions["auto"].handle == "soft-handle"


def test_finished_turn_arms_no_grace_deadline_by_default():
    scheduler = _demand_scheduler()
    scheduler._sessions["agent"] = SessionLease(None, 300.0, active_uid=9, reclaimable=True)

    scheduler._free_req_resources(_req(9, "agent"), retain_session=True)

    assert scheduler._sessions["agent"].protected_until is None


def test_blocked_admission_checkpoints_the_finished_session_and_admits_the_queued_request():
    scheduler = _demand_scheduler()
    scheduler._sessions["A"] = SessionLease(
        "A-handle", 300.0, active_uid=1, reclaimable=True, last_used_at=1.0
    )
    scheduler.prefill_manager.pending_list.append(_queued(2, "B"))

    # Iteration 1: A is still mid-turn, so nothing can be reclaimed and B waits.
    assert scheduler._schedule_next_batch() is None
    assert scheduler._sessions["A"].handle == "A-handle"
    assert scheduler.prefill_manager.admitted == []

    # A's turn ends: the lease becomes idle (and reclaimable) immediately.
    scheduler._free_req_resources(_req(1, "A"), retain_session=True)
    assert scheduler._sessions["A"].active_uid is None

    # Iteration 2: admission fails once more, which is the demand signal to checkpoint A.
    assert scheduler._schedule_next_batch() is None
    lease = scheduler._sessions["A"]
    assert lease.handle is None
    assert scheduler._session_spill_store.get("A") is not None
    assert scheduler.prefill_manager.admitted == []

    # Iteration 3: the freed pages admit B -- the next loop iteration, not 30 s later.
    batch = scheduler._schedule_next_batch()
    assert batch is not None and scheduler.prefill_manager.admitted == [2]


def test_resident_session_is_never_reclaimed_when_nothing_is_queued():
    scheduler = _demand_scheduler()
    scheduler._sessions["A"] = SessionLease(
        "A-handle", 300.0, reclaimable=True, last_used_at=1.0
    )

    for _ in range(3):
        assert scheduler._schedule_next_batch() is None
        scheduler._release_due_soft_sessions()

    assert scheduler._sessions["A"].handle == "A-handle"
    assert scheduler._session_spill_store.records == {}


def test_idle_expiry_reaps_a_checkpointed_lease_but_keeps_the_checkpoint():
    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    # Already checkpointed (handle released on demand): the TTL now bounds the identity.
    lease = SessionLease(None, 1.0, expires_at=time.monotonic() - 1, reclaimable=True)
    lease.spill = store.spill("A", [1, 2, 3, 4], [0, 1, 2, 3], 3)
    scheduler._sessions["A"] = lease

    scheduler._expire_sessions()

    assert "A" not in scheduler._sessions
    assert store.discarded == [] and store.get("A") is not None


def test_explicit_delete_discards_the_checkpoint():
    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    lease = SessionLease("A-handle", 300.0, reclaimable=True)
    lease.spill = store.spill("A", [1, 2, 3, 4], [0, 1, 2, 3], 3)
    scheduler._sessions["A"] = lease

    scheduler._close_session("A")

    assert store.get("A") is None and len(store.discarded) == 1


def test_restore_finds_a_checkpoint_left_behind_by_a_closed_lease():
    import torch

    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    tokens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    store.spill("A", tokens, [0, 1, 2, 3], 3)
    # A fresh lease (reconnect, or a restarted server) carries no record of its own.
    scheduler._sessions["A"] = SessionLease(None, 300.0, reclaimable=True)
    scheduler.cache_manager.hybrid_session_restore_geometry = lambda _t: (0, 99)
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s: "restored"

    assert scheduler._restore_cold_session("A", torch.tensor([1, 2, 3, 4, 5])) is True
    assert scheduler._sessions["A"].handle == "restored"
    assert store.touched and store.get("A") is None  # consumed by the restore


def test_client_disconnect_ends_the_lease_but_keeps_the_checkpoint():
    from freetoken.message import AbortBackendMsg

    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    lease = SessionLease("A-handle", 300.0, reclaimable=True)
    lease.spill = store.spill("A", [1, 2, 3, 4], [0, 1, 2, 3], 3)
    scheduler._sessions["A"] = lease
    scheduler._abort_tombstones = {}

    scheduler._process_one_msg(AbortBackendMsg(uid=5, session_id="A"))

    assert "A" not in scheduler._sessions
    assert store.discarded == [] and store.get("A") is not None
    assert 5 in scheduler._pending_abort_acks


def test_resident_session_outlives_its_ttl_and_is_checkpointed_only_on_demand():
    """Idle time never costs a session its state; the next admission does, and it restores."""
    import torch

    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    scheduler._sessions["A"] = SessionLease(
        "A-handle", 1.0, active_uid=1, reclaimable=True, last_used_at=1.0
    )

    scheduler._free_req_resources(_req(1, "A"), retain_session=True)
    lease = scheduler._sessions["A"]
    assert lease.active_uid is None
    # A resident automatic lease arms no idle deadline at all...
    assert lease.expires_at is None and lease.protected_until is None
    assert scheduler._sessions_need_service() is False  # ...and lets the loop block

    # ... and even a stale deadline (2x TTL elapsed) cannot take its GPU state.
    lease.expires_at = time.monotonic() - 2 * lease.ttl_seconds
    for _ in range(3):
        scheduler._expire_sessions()
        scheduler._release_due_soft_sessions()
        assert scheduler._schedule_next_batch() is None
    assert scheduler._sessions["A"].handle is not None
    assert store.records == {}

    # B arrives: its blocked admission is what finally checkpoints A.
    scheduler.prefill_manager.pending_list.append(_queued(2, "B"))
    assert scheduler._schedule_next_batch() is None
    assert scheduler._sessions["A"].handle is None
    assert store.get("A") is not None
    assert scheduler._schedule_next_batch() is not None
    assert scheduler.prefill_manager.admitted == [2]

    # Releasing the state arms the TTL for the (now empty) identity; the checkpoint
    # survives it and A's next request restores from the store.
    assert scheduler._sessions["A"].expires_at is not None
    scheduler._sessions["A"].expires_at = time.monotonic() - 1
    scheduler._expire_sessions()
    assert "A" not in scheduler._sessions and store.get("A") is not None

    scheduler._sessions["A"] = SessionLease(None, 300.0, reclaimable=True)
    scheduler.cache_manager.hybrid_session_restore_geometry = lambda _t: (0, 99)
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s: "restored"
    assert scheduler._restore_cold_session("A", torch.tensor([1, 2, 3, 4, 5])) is True
    assert scheduler._sessions["A"].handle == "restored"
