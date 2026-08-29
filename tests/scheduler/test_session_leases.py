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
    scheduler.config = SimpleNamespace(kv_grow_step_tokens=65_536)
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
