from __future__ import annotations

import re
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
        self.captured: list = []
        self.prefetching: str | None = None
        self.cancelled: str | None = None
        self.protected: set[str] = set()

    def spill(
        self,
        session_id,
        token_ids,
        page_indices,
        linear_slot,
        *,
        extra_states=(),
        captured_states=(),
    ):
        self.captured = list(captured_states)
        boundaries = sorted(
            {
                len(page_indices),
                *(int(b) for b, _slot in extra_states),
                *(int(b) for b, _conv, _rec in captured_states),
            }
        )
        record = SimpleNamespace(
            session_id=session_id,
            num_pages=len(page_indices),
            tier="ram",
            byte_size=1 << 20,
            token_ids=token_ids,
            state_boundaries=boundaries,
            restorable_length=lambda matched, _b=boundaries: max(
                [b for b in _b if b <= matched], default=0
            ),
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

    # Look-ahead promotion (3F): the double records the calls, moves no bytes.
    def start_prefetch(self, session_id, *, protect=()) -> bool:
        record = self.records.get(session_id)
        if record is None or record.tier != "disk" or self.prefetching is not None:
            return False
        self.prefetching = session_id
        self.protected = set(protect)
        return True

    def collect_prefetch(self, session_id=None, *, wait: bool = False):
        if self.prefetching is None or session_id not in (None, self.prefetching):
            return None
        promoted, self.prefetching = self.prefetching, None
        if self.cancelled == promoted:
            return None
        self.records[promoted].tier = "ram"
        return promoted

    def cancel_prefetch(self, session_id=None) -> bool:
        if self.prefetching is None or session_id not in (None, self.prefetching):
            return False
        self.cancelled = self.prefetching
        return True


class _SessionCache(_Cache):
    """Cache whose KV is exhausted until a session lease is released."""

    is_hybrid = False

    def hybrid_session_state_boundaries(self, _handle):
        return []

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
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s, _n=None: "restored"

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
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s, _n=None: "restored"
    assert scheduler._restore_cold_session("A", torch.tensor([1, 2, 3, 4, 5])) is True
    assert scheduler._sessions["A"].handle == "restored"


def test_spill_and_restore_logs_report_sub_second_durations(caplog):
    """The 1M-session gate reads spill/restore cost off these lines; the log formatter
    only stamps whole seconds, so the duration has to be in the message itself."""
    import logging

    import torch

    scheduler = _demand_scheduler()
    tokens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    handle = SimpleNamespace(
        node=SimpleNamespace(mamba_value=3), get_matched_indices=lambda: [0, 1, 2, 3]
    )
    lease = SessionLease(handle, 300.0, reclaimable=True)
    lease.token_ids = tokens
    scheduler._sessions["A"] = lease
    scheduler.cache_manager.hybrid_session_restore_geometry = lambda _t: (0, 99)
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s, _n=None: "restored"

    with caplog.at_level(logging.INFO, logger="freetoken.scheduler.scheduler"):
        scheduler._spill_soft_session("A", lease)
        assert scheduler._restore_cold_session("A", torch.tensor([1, 2, 3, 4, 5])) is True

    spilled = [m for m in caplog.messages if m.startswith("Spilled soft session")]
    restored = [m for m in caplog.messages if m.startswith("Restored cold session")]
    assert len(spilled) == 1 and len(restored) == 1
    for message in (spilled[0], restored[0]):
        seconds = re.search(r"in (\d+\.\d{3}) s \((\d+\.\d\d) GiB/s\)", message)
        assert seconds is not None, message
        assert float(seconds.group(1)) >= 0.0
    assert "1.00 GiB" not in spilled[0]  # the double's 1 MiB record, not a rounded stub


# ------------------------------------------------- look-ahead prefetch (task 3F)


def _queued_disk_checkpoint(scheduler, session_id="B", tokens=(1, 2, 3, 4)):
    """A checkpoint sitting on NVMe for a session whose request is already queued."""
    import torch

    store = scheduler._session_spill_store
    record = store.spill(
        session_id, torch.tensor(tokens, dtype=torch.int32), list(range(len(tokens))), 3
    )
    record.tier = "disk"
    scheduler.prefill_manager.pending_list.append(_queued(2, session_id))
    return record


def test_a_queued_session_prefetches_its_disk_checkpoint_while_the_resident_runs():
    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    scheduler._sessions["A"] = SessionLease(
        "A-handle", 300.0, active_uid=1, reclaimable=True, last_used_at=1.0
    )
    _queued_disk_checkpoint(scheduler)

    # A is mid-turn: B cannot be admitted, which is exactly the window to read it in.
    assert scheduler._schedule_next_batch() is None
    assert store.prefetching == "B"
    assert store.protected == {"A"}  # the resident checkpoint never pays for the look-ahead

    # The next scheduler iteration installs it, so admission finds it in RAM.
    assert scheduler._schedule_next_batch() is None
    assert store.get("B").tier == "ram"
    assert scheduler.prefill_manager.admitted == []


def test_no_look_ahead_for_an_explicit_lease_or_a_session_without_a_checkpoint():
    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    _queued_disk_checkpoint(scheduler, "explicit")
    # An explicit session_id lease is never spilled, so there is nothing to promote.
    scheduler._sessions["explicit"] = SessionLease(None, 300.0, reclaimable=False)
    scheduler.prefill_manager.pending_list.append(_queued(3, "unknown-session"))

    scheduler._reclaim_for_blocked_prefill()

    assert store.prefetching is None
    assert store.get("explicit").tier == "disk"


def test_an_aborted_request_cancels_its_in_flight_look_ahead():
    from freetoken.message import AbortBackendMsg

    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    scheduler._abort_tombstones = {}
    scheduler._sessions["B"] = SessionLease(None, 300.0, reclaimable=True)
    record = _queued_disk_checkpoint(scheduler)

    scheduler._reclaim_for_blocked_prefill()
    assert store.prefetching == "B"

    scheduler._process_one_msg(AbortBackendMsg(uid=2, session_id="B"))

    assert store.cancelled == "B"
    assert store.collect_prefetch() is None  # the bytes read so far are dropped
    assert record.tier == "disk"  # and the checkpoint survives for the reconnect


# ------------------------------------------- partial-prefix cold restore (task 3G)


def _restoring_scheduler(record_tokens, extra_states=()):
    import torch

    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    tokens = torch.tensor(record_tokens, dtype=torch.int32)
    store.spill("A", tokens, list(range(len(tokens))), 3, extra_states=extra_states)
    scheduler._sessions["A"] = SessionLease(None, 300.0, reclaimable=True)
    scheduler.cache_manager.hybrid_session_restore_geometry = lambda _t: (0, 99)
    return scheduler, store


def test_restore_resumes_at_the_deepest_boundary_the_client_tokens_still_match():
    import torch

    scheduler, store = _restoring_scheduler([1, 2, 3, 4, 5, 6], extra_states=[(4, 7)])
    installed = []
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s, n=None: (
        installed.append(n) or "restored"
    )

    # The client echoed its own turn back with one token retokenized at index 4.
    assert scheduler._restore_cold_session("A", torch.tensor([1, 2, 3, 4, 99, 6, 7])) is True

    assert installed == [4]  # 4 restored, the drifting tail re-prefilled
    assert scheduler._sessions["A"].handle == "restored"
    assert store.get("A") is None  # consumed


def test_an_unchanged_prompt_still_restores_the_whole_checkpoint():
    import torch

    scheduler, _store = _restoring_scheduler([1, 2, 3, 4, 5, 6], extra_states=[(4, 7)])
    installed = []
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s, n=None: (
        installed.append(n) or "restored"
    )

    assert scheduler._restore_cold_session("A", torch.tensor([1, 2, 3, 4, 5, 6, 7])) is True
    assert installed == [6]


def test_a_drift_before_the_first_boundary_discards_the_checkpoint():
    import torch

    scheduler, store = _restoring_scheduler([1, 2, 3, 4, 5, 6], extra_states=[(4, 7)])
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s, _n=None: "restored"

    assert scheduler._restore_cold_session("A", torch.tensor([1, 99, 3, 4, 5, 6, 7])) is False
    assert scheduler._sessions["A"].handle is None
    assert store.get("A") is None  # nothing reusable: full recompute


def test_a_restore_blocked_by_the_resident_session_is_retried_before_admission():
    """The pools the restore needs are the ones the resident lease still owns."""
    import torch

    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    store.spill("B", torch.tensor([1, 2, 3, 4], dtype=torch.int32), [0, 1, 2, 3], 3)
    scheduler._sessions["B"] = SessionLease(None, 300.0, reclaimable=True)
    scheduler._sessions["A"] = SessionLease(
        "A-handle", 300.0, reclaimable=True, last_used_at=1.0
    )
    pending = _queued(2, "B")
    # A real PendingReq carries a tensor; the checkpoint covers all but its last token.
    pending.input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    scheduler.prefill_manager.pending_list.append(pending)
    scheduler.cache_manager.hybrid_session_restore_geometry = lambda _t: (0, 99)

    attempts = []

    def _restore(_record, _store, num_tokens=None):
        attempts.append(num_tokens)
        if len(attempts) == 1:
            raise RuntimeError("no GDN snapshot slot available for cold session restore")
        return "restored"

    scheduler.cache_manager.restore_hybrid_session_prefix = _restore

    # Message receipt: A still owns the state slot, so the restore fails -- but the
    # checkpoint is kept rather than thrown away.
    assert scheduler._restore_cold_session("B", torch.tensor([1, 2, 3, 4, 5])) is False
    assert store.get("B") is not None

    # The blocked admission checkpoints A and retries B's restore in the same pass.
    assert scheduler._schedule_next_batch() is None
    assert attempts == [4, 4]
    assert scheduler._sessions["B"].handle == "restored"
    assert store.get("B") is None


# ------------------------------------ prefill-captured state boundaries (task 3G)


def _linear_state_pool(num_slots=8):
    import torch

    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=2,
        key_head_dim=8, value_head_dim=8, conv_kernel_dim=4, output_gate=True,
    )
    return LinearStatePool(group, num_slots, torch.bfloat16, torch.device("cpu"), tp_size=1)


def _long_session(scheduler, tokens, captures):
    """A resident 200K-token lease whose turns captured state at ``captures``."""
    import torch

    handle = SimpleNamespace(
        node=SimpleNamespace(mamba_value=3),
        get_matched_indices=lambda: list(range(len(tokens))),
    )
    lease = SessionLease(handle, 300.0, reclaimable=True)
    lease.token_ids = tokens
    lease.state_captures = [
        (b, torch.zeros(2, dtype=torch.float32), torch.full((2,), float(b)))
        for b in captures
    ]
    scheduler._sessions["A"] = lease
    return lease


def test_a_drift_in_the_last_tokens_of_a_200k_session_resumes_at_the_last_capture():
    """The 1M profile keeps no radix snapshots, so the prefill captures are the only cut
    points a 200K checkpoint has. One retokenized tail must cost the tail, not 200K."""
    import torch

    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    tokens = torch.arange(200_000, dtype=torch.int32)
    lease = _long_session(scheduler, tokens, [65_536, 131_072, 196_608])

    scheduler._spill_soft_session("A", lease)
    record = store.get("A")
    assert record is not None
    assert record.state_boundaries == [65_536, 131_072, 196_608, 200_000]

    # The client echoes its own last turn back with the final 100 tokens retokenized.
    drifted = torch.cat(
        (tokens[:199_900].clone(), torch.full((101,), 7, dtype=torch.int32))
    )
    installed = []
    scheduler.cache_manager.hybrid_session_restore_geometry = lambda _t: (0, 99)
    scheduler.cache_manager.restore_hybrid_session_prefix = lambda _r, _s, n=None: (
        installed.append(n) or "restored"
    )
    scheduler._sessions["A"] = SessionLease(None, 300.0, reclaimable=True)

    assert scheduler._restore_cold_session("A", drifted) is True
    assert installed == [196_608]  # 3 328 tokens re-prefilled instead of 200 000


def test_captured_states_are_released_once_the_checkpoint_owns_them():
    import weakref

    import torch

    scheduler = _demand_scheduler()
    store = scheduler._session_spill_store
    tokens = torch.arange(4, dtype=torch.int32)
    lease = _long_session(scheduler, tokens, [2])
    staged = weakref.ref(lease.state_captures[0][2])

    scheduler._spill_soft_session("A", lease)

    assert [b for b, _c, _r in store.captured] == [2]  # handed to the checkpoint ...
    assert lease.state_captures == []  # ... and the staging is dropped
    del store.captured
    assert staged() is None


def test_state_capture_defaults_on_only_when_snapshot_slots_are_scarce():
    import torch

    def _install(num_slots, running, wanted=None):
        scheduler = Scheduler.__new__(Scheduler)
        pool = SimpleNamespace(
            num_slots=num_slots,
            device=torch.device("cpu"),
            bytes_per_slot=lambda: 1 << 20,
            conv_states=torch.zeros(1, num_slots),
            recurrent_states=torch.zeros(1, num_slots),
        )
        scheduler.cache_manager = SimpleNamespace(
            linear_state_pool=pool, session_state_capture_hook=None
        )
        scheduler._session_spill_store = SimpleNamespace(
            state_stride_tokens=65_536, max_states=8
        )
        scheduler._state_capture = None
        scheduler._install_state_capture(
            SimpleNamespace(
                max_running_req=running, session_spill_capture_states=wanted
            )
        )
        return scheduler

    # The 1M profile: 5 slots for one request cannot spare one to donate.
    assert _install(5, 1)._state_capture is not None
    # The default pool leaves snapshot slots over, so the free radix boundaries suffice.
    assert _install(9, 1)._state_capture is None
    assert _install(16, 3)._state_capture is not None  # scarce again at 3 running requests
    # And the flag overrides the automatic choice either way.
    assert _install(9, 1, wanted=True)._state_capture is not None
    assert _install(5, 1, wanted=False)._state_capture is None


def test_every_prefill_forward_offers_its_boundary_snapshot_to_the_session():
    """Intermediate chunks never reach cache_req, so the drain is the only point that sees
    a long prompt's boundaries -- one capture per stride of prefilled tokens."""
    import torch

    from freetoken.scheduler.session_spill import PrefillStateCapture

    linear = _linear_state_pool()
    scheduler = _demand_scheduler()
    scheduler._state_capture = PrefillStateCapture(linear, stride=64, max_states=8)
    lease = SessionLease(None, 300.0, reclaimable=True)
    scheduler._sessions["A"] = lease

    def _chunk(boundary, next_track_idx, session_id="A"):
        return SimpleNamespace(
            session_id=session_id,
            mamba_last_track_seqlen=boundary,
            mamba_ping_pong=(1, 2),
            mamba_next_track_idx=next_track_idx,
        )

    torch.manual_seed(23)
    linear.recurrent_states.normal_()
    expected = linear.recurrent_states[:, 1].clone()

    # The forward flipped the index, so the state it just wrote is in ping_pong[0].
    scheduler._capture_session_states(SimpleNamespace(reqs=[_chunk(64, 1)]))
    scheduler._capture_session_states(SimpleNamespace(reqs=[_chunk(96, 0)]))  # < 1 stride
    scheduler._capture_session_states(SimpleNamespace(reqs=[_chunk(128, 1)]))
    # A request with no lease (plain completion) and one with no boundary are ignored.
    scheduler._capture_session_states(SimpleNamespace(reqs=[_chunk(192, 1, None)]))
    scheduler._capture_session_states(SimpleNamespace(reqs=[_chunk(None, 1)]))

    assert [b for b, _c, _r in lease.state_captures] == [64, 128]
    assert torch.equal(lease.state_captures[0][2], expected)
