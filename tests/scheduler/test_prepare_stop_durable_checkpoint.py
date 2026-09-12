from types import SimpleNamespace

import pytest

from freetoken.message import DurableCheckpointBackendMsg
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.session_spill import DurableSpillResult


class Store:
    persist = True
    disk_budget_bytes = 10**9

    def __init__(self, result):
        self.result = result
        self.current = {}
        self.sources = None

    def persist_durable(self, sources):
        self.sources = sources
        return self.result

    def get(self, session_id):
        return self.current.get(session_id)


def _scheduler(store, sessions=None):
    replies = []
    obj = SimpleNamespace(
        _pending_durable_checkpoint=DurableCheckpointBackendMsg("op-1"),
        _durable_checkpoint_result=None,
        _last_data=None,
        _pending_rebuild=None,
        _session_spill_store=store,
        _sessions=sessions or {},
        config=SimpleNamespace(tp_info=SimpleNamespace(size=1), kv_grow_step_tokens=8),
        prefill_manager=SimpleNamespace(runnable=False),
        decode_manager=SimpleNamespace(runnable=False),
        cache_manager=SimpleNamespace(hybrid_session_state_boundaries=lambda handle: ()),
        send_result=lambda result: replies.extend(result),
        _session_hash=Scheduler._session_hash,
    )
    return obj, replies


def test_complete_barrier_seals_and_keeps_resident_handle():
    handle = SimpleNamespace(
        node=SimpleNamespace(mamba_value=3),
        get_matched_indices=lambda: [1, 2],
    )
    lease = SimpleNamespace(handle=handle, token_ids=[10, 11], active_uid=None,
                            spill=None, state_captures=[])
    store = Store(DurableSpillResult(True, ("session-a",)))
    current = SimpleNamespace(valid=True)
    store.current["session-a"] = current
    scheduler, replies = _scheduler(store, {"session-a": lease})
    Scheduler._execute_pending_durable_checkpoint(scheduler)
    assert replies[0].status == "complete"
    assert scheduler._durable_checkpoint_sealed is True
    assert lease.handle is handle
    assert lease.spill is current
    assert store.sources[0].session_id == "session-a"


def test_partial_replacement_rebinds_handleless_lease_and_can_retry():
    old = SimpleNamespace(valid=True)
    new = SimpleNamespace(valid=True)
    lease = SimpleNamespace(handle=None, token_ids=None, active_uid=None,
                            spill=old, state_captures=[])
    store = Store(DurableSpillResult(False, ("session-a",), "later failure"))
    store.current["session-a"] = new
    scheduler, replies = _scheduler(store, {"session-a": lease})
    Scheduler._execute_pending_durable_checkpoint(scheduler)
    assert replies[0].status == "failed"
    assert lease.spill is new
    assert scheduler._durable_checkpoint_result is None
    scheduler._pending_durable_checkpoint = DurableCheckpointBackendMsg("op-1")
    store.result = DurableSpillResult(True, ("session-a",))
    Scheduler._execute_pending_durable_checkpoint(scheduler)
    assert replies[-1].status == "complete"


def test_persistence_exception_is_failed_reply_not_scheduler_crash():
    store = Store(None)
    lease = SimpleNamespace(handle=None, token_ids=None, active_uid=None,
                            spill=SimpleNamespace(valid=True), state_captures=[])
    replacement = SimpleNamespace(valid=True)
    store.current["session-a"] = replacement
    store.persist_durable = lambda sources: (_ for _ in ()).throw(OSError("disk"))
    scheduler, replies = _scheduler(store, {"session-a": lease})
    Scheduler._execute_pending_durable_checkpoint(scheduler)
    assert replies[0].status == "failed"
    assert "OSError" in replies[0].error
    assert lease.spill is replacement


def test_many_adopted_records_use_bounded_hash_sample_without_failing():
    ids = tuple(f"stored-{i:03d}" for i in range(138))
    store = Store(DurableSpillResult(True, ids))
    scheduler, replies = _scheduler(store)
    Scheduler._execute_pending_durable_checkpoint(scheduler)
    reply = replies[0]
    assert reply.status == "complete"
    assert reply.durable_count == 138
    assert len(reply.durable_hashes) == 64
    assert reply.durable_hashes_truncated is True


def test_sealed_normal_loop_only_receives_same_operation_retry_without_mutation():
    cached = SimpleNamespace(operation_id="op-1", status="complete")
    replies, blocking_values = [], []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sealed loop mutated or forwarded")

    scheduler = SimpleNamespace(
        _durable_checkpoint_sealed=True,
        _durable_checkpoint_result=cached,
        _pending_durable_checkpoint=None,
        _pending_rebuild=None,
        prefill_manager=SimpleNamespace(runnable=False),
        decode_manager=SimpleNamespace(runnable=False),
        _sessions_need_service=lambda: False,
        receive_msg=lambda blocking: (
            blocking_values.append(blocking) or [DurableCheckpointBackendMsg("op-1")]),
        _process_one_msg=None,
        _execute_pending_durable_checkpoint=lambda: None,
        _publish_scheduler_counters=lambda force=False: None,
        send_result=lambda values: replies.extend(values),
        _expire_sessions=forbidden, _release_due_soft_sessions=forbidden,
        _enforce_session_host_reserve=forbidden, _only_idle_sessions=forbidden,
        _maybe_shrink_growable_kv=forbidden, _maybe_resize_elastic_capacity=forbidden,
        _schedule_next_batch=forbidden, _forward=forbidden,
    )
    scheduler._process_one_msg = Scheduler._process_one_msg.__get__(scheduler)
    Scheduler.normal_loop(scheduler)
    assert blocking_values == [True]
    assert replies == [cached]


@pytest.mark.parametrize("gate", [
    "tp", "non_growable", "no_store", "persist_false", "runnable",
    "active_lease", "pending_rebuild",
])
def test_barrier_gates_fail_closed_without_persistence(gate):
    store = Store(DurableSpillResult(True, ()))
    scheduler, replies = _scheduler(store)
    if gate == "tp":
        scheduler.config.tp_info.size = 2
    if gate == "non_growable":
        scheduler.config.kv_grow_step_tokens = 0
    if gate == "no_store":
        scheduler._session_spill_store = None
    if gate == "persist_false":
        store.persist = False
    if gate == "runnable":
        scheduler.prefill_manager.runnable = True
    if gate == "active_lease":
        scheduler._sessions = {"s": SimpleNamespace(active_uid=4)}
    if gate == "pending_rebuild":
        scheduler._pending_rebuild = object()
    Scheduler._execute_pending_durable_checkpoint(scheduler)
    assert replies[0].status == "unsupported"
    assert store.sources is None


def test_state_capture_synchronizes_before_durable_persistence():
    order = []
    store = Store(DurableSpillResult(True, ()))
    original = store.persist_durable
    store.persist_durable = lambda sources: (order.append("persist") or original(sources))
    scheduler, replies = _scheduler(store)
    scheduler._state_capture = SimpleNamespace(synchronize=lambda: order.append("sync"))
    Scheduler._execute_pending_durable_checkpoint(scheduler)
    assert replies[0].status == "complete"
    assert order == ["sync", "persist"]
