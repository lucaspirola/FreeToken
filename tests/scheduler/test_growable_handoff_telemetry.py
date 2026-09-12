from types import SimpleNamespace

from freetoken.scheduler.counters import GrowableHandoffEvents, build_scheduler_counters
from freetoken.scheduler.scheduler import Scheduler


def test_ring_is_monotonic_and_bounded_to_sixteen():
    ring = GrowableHandoffEvents()
    for i in range(20):
        ring.begin({"attempt_monotonic_ns": i})
    doc = ring.as_dict()
    assert doc["next_sequence"] == 21
    assert [event["sequence"] for event in doc["events"]] == list(range(5, 21))
    assert build_scheduler_counters(growable_handoff_events=ring)[
        "growable_handoff"] == doc


def _stub(expected_uid=7):
    ring = GrowableHandoffEvents()
    sequence = ring.begin({
        "attempt_monotonic_ns": 1, "resize_completed_monotonic_ns": 2,
        "head_uid_hash": Scheduler._handoff_identity(expected_uid),
        "before_committed_pages": 32, "after_committed_pages": 8,
        "before_expert_slots": 80, "after_expert_slots": 99,
        "outcome": "awaiting_pre_forward_admission", "qualified": False,
    })
    scheduler = SimpleNamespace(
        _growable_handoff_events=ring,
        _growable_handoff_pending=(sequence, expected_uid),
        _handoff_identity=Scheduler._handoff_identity,
        _publish_scheduler_counters=lambda force=False: setattr(scheduler, "published", force),
        published=False,
        cache_manager=SimpleNamespace(committed_pages=8),
        engine=SimpleNamespace(moe_offload_cache=SimpleNamespace(cache_size=99)),
    )
    return scheduler, ring


def test_matching_head_is_qualified_before_forward():
    scheduler, ring = _stub()
    forward = SimpleNamespace(batch=SimpleNamespace(
        reqs=[SimpleNamespace(uid=7)], is_prefill=True))
    Scheduler._finalize_growable_handoff(scheduler, forward)
    event = ring.as_dict()["events"][0]
    assert event["outcome"] == "success" and event["qualified"] is True
    assert event["admitted_uid_hash"] == event["head_uid_hash"]
    assert event["pre_forward_monotonic_ns"] >= event["resize_completed_monotonic_ns"]
    assert event["admitted_batch_is_prefill"] is True
    assert event["admitted_committed_pages"] == 8
    assert event["admitted_expert_slots"] == 99
    assert scheduler.published is True


def test_nonmatching_admission_is_not_qualified():
    scheduler, ring = _stub()
    forward = SimpleNamespace(batch=SimpleNamespace(
        reqs=[SimpleNamespace(uid=8)], is_prefill=True))
    Scheduler._finalize_growable_handoff(scheduler, forward)
    event = ring.as_dict()["events"][0]
    assert event["outcome"] == "not_qualified" and event["qualified"] is False
    assert event["admitted_uid_hash"] != event["head_uid_hash"]


def test_immediate_scheduling_regrowth_is_captured_at_admission():
    scheduler, ring = _stub()
    scheduler.cache_manager.committed_pages = 32
    scheduler.engine.moe_offload_cache.cache_size = 80
    forward = SimpleNamespace(batch=SimpleNamespace(
        reqs=[SimpleNamespace(uid=7)], is_prefill=True))
    Scheduler._finalize_growable_handoff(scheduler, forward)
    event = ring.as_dict()["events"][0]
    assert event["outcome"] == "success"
    assert event["admitted_committed_pages"] == event["before_committed_pages"]
    assert event["admitted_expert_slots"] == event["before_expert_slots"]


def test_matching_decode_uid_is_not_a_qualified_handoff():
    scheduler, ring = _stub()
    forward = SimpleNamespace(batch=SimpleNamespace(
        reqs=[SimpleNamespace(uid=7)], is_prefill=False))
    Scheduler._finalize_growable_handoff(scheduler, forward)
    event = ring.as_dict()["events"][0]
    assert event["outcome"] == "not_qualified"
    assert event["qualified"] is False
    assert event["admitted_batch_is_prefill"] is False


def test_error_event_cannot_be_success():
    ring = GrowableHandoffEvents()
    sequence = ring.begin({"outcome": "attempting", "qualified": False})
    ring.update(sequence, outcome="error", error_type="MemoryError")
    event = ring.as_dict()["events"][0]
    assert event["outcome"] == "error" and event["qualified"] is False
