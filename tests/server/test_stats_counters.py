"""``/v1/stats`` gains the counters a live soak could not otherwise get.

Two independent channels meet in the ``StatsTracker``:

* **aborts by reason** -- the wire carries one untagged ``AbortMsg`` for a client
  disconnect, a prepare-stop drain and an engine error alike, and the scheduler's own
  ``"Aborting request %d"`` is debug-only, so a soak log cannot count them at all. The
  reason is only knowable at the frontend call site, which is where it is now recorded.
* **the scheduler counters** -- chunked-prefill deferrals and ``max_chunked_prefills``
  hits, the seatable-lanes divisor, finishability-invariant violations, spec-decode
  declines, session spill/restore/prefetch traffic. They live in another process and reach
  the frontend as a ``SchedulerCountersReply``, which ``listen()`` stores whole.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from freetoken.message import SchedulerCountersMsg, SchedulerCountersReply, UserReply
from freetoken.server.api_server import FrontendManager
from freetoken.server.stats import StatsTracker, build_stats


class FakeSendQueue:
    def __init__(self) -> None:
        self.sent: list = []

    async def put(self, msg) -> None:
        self.sent.append(msg)


def _manager() -> FrontendManager:
    return FrontendManager(
        config=SimpleNamespace(served_model_name="unit-model", model_path="/models/unit"),
        send_tokenizer=FakeSendQueue(),
        recv_tokenizer=None,
        maintenance_state="serving",
        initialized=True,
    )


def _config():
    return SimpleNamespace(
        served_model_name="unit-model",
        max_seq_len=4096,
        page_size=1,
        model_config=SimpleNamespace(has_linear_attention=False, has_swa_attention=False),
    )


def _doc(tracker: StatsTracker) -> dict:
    state = SimpleNamespace(
        stats=tracker, config=_config(), instance_id="i-1", ready_at=None
    )
    return build_stats(state, 0, 0)


# --------------------------------------------------------------------------- aborts


def test_aborts_are_counted_by_reason():
    tracker = StatsTracker()
    tracker.on_new_user(1)
    tracker.on_new_user(2)
    tracker.on_abort(1)  # the default: every caller but the prepare-stop drain
    tracker.on_abort(2, "explicit")
    assert _doc(tracker)["requests"]["aborts"] == {
        "client_disconnect": 1, "explicit": 1, "error": 0
    }


def test_a_failed_request_counts_as_an_error_not_an_abort():
    """A request that fails is never aborted -- it finishes with ``error`` set -- so
    without this it is invisible next to the two abort reasons."""
    tracker = StatsTracker()
    tracker.on_new_user(7)
    tracker.observe(
        UserReply(uid=7, incremental_output="", finished=True, error="context too long")
    )
    aborts = _doc(tracker)["requests"]["aborts"]
    assert aborts == {"client_disconnect": 0, "explicit": 0, "error": 1}
    # An errored request is still a completion for the active/completed barrier.
    assert tracker.active == 0


def test_an_aborts_own_terminal_ack_is_not_also_scored_as_an_error():
    """The scheduler acknowledges an abort with ``ErrorReplyMsg("request aborted")``, so a
    naive error count scores every client disconnect twice."""
    tracker = StatsTracker()
    tracker.on_new_user(5)
    tracker.on_abort(5)
    tracker.observe(
        UserReply(uid=5, incremental_output="", finished=True, error="request aborted")
    )
    assert _doc(tracker)["requests"]["aborts"] == {
        "client_disconnect": 1, "explicit": 0, "error": 0
    }
    assert tracker.active == 0
    assert _doc(tracker)["requests"]["completed"] == 0, "an abort is not a completion"


def test_a_clean_finish_counts_nothing():
    tracker = StatsTracker()
    tracker.on_new_user(3)
    tracker.observe(UserReply(uid=3, incremental_output="hi", finished=True))
    assert _doc(tracker)["requests"]["aborts"] == {
        "client_disconnect": 0, "explicit": 0, "error": 0
    }
    assert _doc(tracker)["requests"]["completed"] == 1


def test_abort_user_records_the_reason_it_was_given():
    state = _manager()

    async def go():
        await state.abort_user(11)
        await state.abort_user(12, reason="explicit")

    asyncio.run(go())
    assert state.stats.aborts["client_disconnect"] == 1
    assert state.stats.aborts["explicit"] == 1


def test_an_unknown_reason_does_not_lose_the_abort():
    """The reason is a label, not an enum on the hot path: a caller that invents one must
    still be counted rather than raising inside an abort."""
    tracker = StatsTracker()
    tracker.on_abort(1, "some_new_path")
    assert tracker.aborts["some_new_path"] == 1


# ----------------------------------------------------------------- scheduler counters


def test_scheduler_counters_are_absent_until_the_engine_publishes_one():
    """``None`` is not the same answer as an all-zero document: it says this engine never
    reported (offline, a non-primary TP rank, or an older build)."""
    assert _doc(StatsTracker())["scheduler"] is None


def test_the_frontend_stores_a_counters_reply_whole():
    tracker = StatsTracker()
    doc = {"prefill": {"passes": 3}, "spec": None, "session_spill": None}
    tracker.on_scheduler_counters(doc)
    assert _doc(tracker)["scheduler"] == doc

    # Last-known-value, like the pool gauges: a later snapshot replaces the earlier one.
    tracker.on_scheduler_counters({"prefill": {"passes": 9}})
    assert _doc(tracker)["scheduler"]["prefill"]["passes"] == 9


def test_listen_routes_a_counters_reply_to_the_tracker_and_not_to_a_request():
    """It carries no uid, so it must be consumed before the per-request unwrap -- which
    would otherwise call ``observe`` on it and read ``uid`` off a message that has none."""

    class Recv:
        def __init__(self, items):
            self._items = list(items)

        async def get(self):
            if not self._items:
                raise asyncio.CancelledError
            return self._items.pop(0)

    state = _manager()
    state.recv_tokenizer = Recv([SchedulerCountersReply(counters={"prefill": {"passes": 4}})])

    async def go():
        try:
            await state.listen()
        except asyncio.CancelledError:
            pass

    asyncio.run(go())
    assert state.stats.scheduler_counters == {"prefill": {"passes": 4}}


def test_the_counters_message_round_trips_over_the_wire():
    """The document is a nested plain dict; the serializer has to carry it unchanged."""
    from freetoken.message import BaseFrontendMsg, BaseTokenizerMsg

    doc = {
        "prefill": {"passes": 2, "seatable_lanes": {"1": 1, "5-8": 1}},
        "spec": {"declined": {"budget": 3}, "accepted_hist": {"0": 1, "7": 2}},
        "session_spill": None,
    }
    decoded = BaseTokenizerMsg.decoder(
        BaseTokenizerMsg.encoder(SchedulerCountersMsg(counters=doc))
    )
    assert isinstance(decoded, SchedulerCountersMsg) and decoded.counters == doc

    decoded = BaseFrontendMsg.decoder(
        BaseFrontendMsg.encoder(SchedulerCountersReply(counters=doc))
    )
    assert isinstance(decoded, SchedulerCountersReply) and decoded.counters == doc
