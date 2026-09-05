"""A client that goes away must take its engine resources with it.

The leak these cover: a request dropped while it is still in *prefill* has yielded no
chunk, so the old post-first-chunk ``is_disconnected()`` check never ran and no AbortMsg
was ever sent — the pending entry, the table slot and every forwarded KV page stayed
allocated for the life of the server. The non-streaming paths had the same hole (they
aborted only if something else cancelled them, which nothing does).

Everything here is a fake client + fake engine: no GPU, no model, no sockets.

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_disconnect_abort.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.message import AbortMsg, UserReply  # noqa: E402
from freetoken.server import disconnect as D  # noqa: E402
from freetoken.server.anthropic_api import handle_anthropic_messages  # noqa: E402
from freetoken.server.anthropic_models import AnthropicMessagesRequest  # noqa: E402
from freetoken.server.api_server import FrontendManager  # noqa: E402
from freetoken.server.openai_api import (  # noqa: E402
    ChatCompletionRequest,
    CompletionRequest,
    handle_chat_completion,
    handle_completion,
)
from freetoken.server.responses_api import ResponsesRequest, handle_responses  # noqa: E402


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    """The poll interval only bounds detection latency; 10 ms keeps the suite quick.
    Patching the module constant works because it is read at call time, not bound as a
    default argument."""
    monkeypatch.setattr(D, "POLL_INTERVAL_S", 0.01)


class FakeRequest:
    """Stand-in for a starlette Request: the paths under test touch only
    ``is_disconnected()`` (and ``headers``, for session binding).

    ``drops_after`` is the number of polls answered "still here" before the client is
    reported gone; ``None`` means it never disconnects."""

    def __init__(self, drops_after: int | None = 0):
        self.drops_after = drops_after
        self.polls = 0
        self.headers: dict[str, str] = {}

    async def is_disconnected(self) -> bool:
        gone = self.drops_after is not None and self.polls >= self.drops_after
        self.polls += 1
        return gone


class FakeSendQueue:
    def __init__(self) -> None:
        self.sent: list = []

    async def put(self, msg) -> None:
        self.sent.append(msg)


def _manager() -> FrontendManager:
    """A real FrontendManager (real abort_user / stream_with_cancellation) with the ZMQ
    queues faked out. ``initialized=True`` skips the listener task, which would otherwise
    try to read from the absent recv queue."""
    return FrontendManager(
        config=SimpleNamespace(served_model_name="unit-model", model_path="/models/unit"),
        send_tokenizer=FakeSendQueue(),
        recv_tokenizer=None,
        maintenance_state="serving",
        initialized=True,
    )


async def _drain_aborts(state: FrontendManager) -> None:
    """stream_with_cancellation fires the abort as a background task (it runs while the
    generator is being torn down and cannot be awaited there)."""
    while state.abort_tasks:
        await asyncio.gather(*list(state.abort_tasks))


def _aborts(state: FrontendManager) -> list[AbortMsg]:
    return [m for m in state.send_tokenizer.sent if isinstance(m, AbortMsg)]


async def _collect(stream) -> list:
    return [chunk async for chunk in stream]


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def test_stream_aborts_when_client_drops_during_prefill():
    """The bug: no chunk has been produced yet, so nothing used to check the client."""
    state = _manager()

    async def main():
        uid = state.new_user()
        forever = asyncio.Event()

        async def prefilling():
            await forever.wait()  # the engine is still forwarding chunks; nothing to yield
            yield b"data: never\n\n"

        stream = state.stream_with_cancellation(prefilling(), FakeRequest(0), uid, "sess-1")
        with pytest.raises(asyncio.CancelledError):
            await _collect(stream)
        await _drain_aborts(state)
        return uid

    uid = asyncio.run(main())
    aborts = _aborts(state)
    assert len(aborts) == 1
    assert aborts[0].uid == uid and aborts[0].session_id == "sess-1"
    # Resources released: the ack queue and its wakeup event are gone, and the request is
    # marked aborting (it stays "active" until the scheduler's terminal ack, by design).
    assert uid not in state.ack_map and uid not in state.event_map
    assert uid in state.stats._aborting


def test_stream_abort_sends_exactly_one_abortmsg():
    """A disconnect that keeps being reported must not turn into a storm of aborts."""
    state = _manager()

    async def main():
        uid = state.new_user()
        forever = asyncio.Event()

        async def prefilling():
            await forever.wait()
            yield b"never"

        request = FakeRequest(0)
        stream = state.stream_with_cancellation(prefilling(), request, uid)
        with pytest.raises(asyncio.CancelledError):
            await _collect(stream)
        await _drain_aborts(state)
        # Give any stray background task a chance to run before we count.
        await asyncio.sleep(0)
        return request

    request = asyncio.run(main())
    assert request.polls >= 1
    assert len(_aborts(state)) == 1


def test_stream_aborts_after_first_chunk_disconnect():
    """The case that already worked keeps working (one chunk out, then the client goes)."""
    state = _manager()

    async def main():
        uid = state.new_user()
        forever = asyncio.Event()

        async def one_then_stall():
            yield b"data: one\n\n"
            await forever.wait()
            yield b"data: two\n\n"

        chunks = []
        stream = state.stream_with_cancellation(one_then_stall(), FakeRequest(1), uid)
        with pytest.raises(asyncio.CancelledError):
            async for chunk in stream:
                chunks.append(chunk)
        await _drain_aborts(state)
        return chunks

    chunks = asyncio.run(main())
    assert chunks == [b"data: one\n\n"]
    assert len(_aborts(state)) == 1


def test_stream_completes_normally_without_aborting():
    """No regression: a connected client gets every chunk and no AbortMsg is sent."""
    state = _manager()

    async def main():
        uid = state.new_user()

        async def two_chunks():
            yield b"data: one\n\n"
            yield b"data: [DONE]\n\n"

        chunks = await _collect(
            state.stream_with_cancellation(two_chunks(), FakeRequest(None), uid, "sess-9")
        )
        await _drain_aborts(state)
        return chunks

    assert asyncio.run(main()) == [b"data: one\n\n", b"data: [DONE]\n\n"]
    assert _aborts(state) == []
    assert state.abort_tasks == set()


def test_stream_without_request_object_is_unchanged():
    """In-process callers pass request=None and must not be polled or aborted."""

    async def main():
        return await _collect(D.aiter_or_disconnect(_two_items(), None))

    async def _two_items():
        yield 1
        yield 2

    assert asyncio.run(main()) == [1, 2]


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #
class HangingState:
    """Fake engine that accepts a submission and then never answers: the request is
    parked in prefill exactly as a long prompt would be."""

    def __init__(self, replies: list[UserReply] | None = None):
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser=None,
            reasoning_parser=None,
        )
        self.maintenance_state = "serving"
        self.replies = replies
        self.sent: list = []
        self.aborts: list[tuple[int, str | None]] = []
        self._uid = 0

    def new_user(self) -> int:
        self._uid += 1
        return self._uid

    async def send_one(self, msg) -> None:
        self.sent.append(msg)

    async def wait_for_ack(self, uid: int):
        if self.replies is None:
            await asyncio.Event().wait()  # never answers
        for reply in self.replies or []:
            yield reply

    async def abort_user(self, uid: int, session_id: str | None = None) -> None:
        self.aborts.append((uid, session_id))


def _chat_request(**kwargs) -> ChatCompletionRequest:
    payload = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
    }
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def _run_handler(coro):
    return asyncio.run(coro)


def test_chat_nonstream_aborts_on_disconnect():
    state = HangingState()
    with pytest.raises(asyncio.CancelledError):
        _run_handler(
            handle_chat_completion(_chat_request(), FakeRequest(0), state, {})
        )
    assert state.aborts == [(1, None)]


def test_chat_nonstream_normal_completion_is_unaffected():
    """No regression: a connected client still gets its answer, with no abort."""
    state = HangingState(
        replies=[
            UserReply(
                uid=1,
                incremental_output="hi there",
                finished=True,
                finish_reason="stop",
                prompt_tokens_delta=3,
                completion_tokens_delta=2,
            )
        ]
    )
    result = _run_handler(
        handle_chat_completion(_chat_request(), FakeRequest(None), state, {})
    )
    assert result["choices"][0]["message"]["content"] == "hi there"
    assert state.aborts == []


def test_completions_nonstream_aborts_on_disconnect():
    state = HangingState()
    req = CompletionRequest(model="client-model", prompt="hello", max_tokens=8)
    with pytest.raises(asyncio.CancelledError):
        _run_handler(handle_completion(req, FakeRequest(0), state, {}))
    assert state.aborts == [(1, None)]


def test_anthropic_nonstream_aborts_on_disconnect():
    state = HangingState()
    req = AnthropicMessagesRequest.model_validate(
        {"model": "claude-x", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )
    with pytest.raises(asyncio.CancelledError):
        _run_handler(handle_anthropic_messages(req, FakeRequest(0), state, {}))
    assert state.aborts == [(1, None)]


def test_responses_nonstream_aborts_on_disconnect():
    state = HangingState()
    req = ResponsesRequest.model_validate(
        {"model": "gpt-x", "input": "hi", "max_output_tokens": 16}
    )
    with pytest.raises(asyncio.CancelledError):
        _run_handler(handle_responses(req, FakeRequest(0), state, {}))
    assert state.aborts == [(1, None)]


def test_disconnected_nonstream_request_stops_reading_the_engine():
    """The abort is only half the fix: the handler must also stop awaiting the engine,
    or the coroutine (and its ack queue) lives on."""
    state = HangingState()
    with pytest.raises(asyncio.CancelledError):
        _run_handler(handle_chat_completion(_chat_request(), FakeRequest(0), state, {}))
    # One submission, one abort — nothing left running.
    assert len(state.sent) == 1
    assert len(state.aborts) == 1


def test_probe_failure_counts_as_a_disconnect():
    """A transport that can no longer answer "is the client there?" must not leave the
    request running: the error is raised where the abort would be, so it has to abort."""
    state = _manager()

    class BrokenRequest:
        async def is_disconnected(self):
            raise RuntimeError("receive channel closed")

    async def main():
        uid = state.new_user()
        forever = asyncio.Event()

        async def prefilling():
            await forever.wait()
            yield b"never"

        with pytest.raises(asyncio.CancelledError):
            await _collect(state.stream_with_cancellation(prefilling(), BrokenRequest(), uid))
        await _drain_aborts(state)

    asyncio.run(main())
    assert len(_aborts(state)) == 1

# --------------------------------------------------------------------------- #
# The abort has to be COUNTED, and the count has to survive a cancelled handler
# --------------------------------------------------------------------------- #
# Soak §W5: the disconnect probe took ``/v1/stats.requests.active`` 0 -> 1 -> 0 in two
# seconds on a ~60 K-token request dropped mid-prefill, and
# ``requests.aborts.client_disconnect`` never left 0. The probe was non-streaming
# (``"stream": false``), and that path *awaits* ``abort_user`` from inside its own
# ``except asyncio.CancelledError`` handler -- where the first statement was a 0.1 s
# sleep. Anything that cancels the request task during that window discarded the whole
# coroutine, taking both the counter and the AbortMsg with it, so the disconnect was
# invisible on /v1/stats AND the request kept its slot and its forwarded KV pages.
#
# ``abort_user`` now dispatches a shielded task, which is what the streaming path has
# always had from ``spawn_abort``.


def _hanging_manager() -> FrontendManager:
    """A real FrontendManager (real abort_user, real StatsTracker) whose engine accepts a
    submission and then never answers -- a request parked in prefill."""
    state = _manager()

    async def wait_for_ack(uid: int):
        await asyncio.Event().wait()
        yield None  # pragma: no cover -- unreachable, keeps this an async generator

    state.wait_for_ack = wait_for_ack
    return state


def test_stream_disconnect_is_counted_once():
    state = _manager()

    async def main():
        uid = state.new_user()
        forever = asyncio.Event()

        async def prefilling():
            await forever.wait()
            yield b"never"

        with pytest.raises(asyncio.CancelledError):
            await _collect(state.stream_with_cancellation(prefilling(), FakeRequest(0), uid))
        await _drain_aborts(state)

    asyncio.run(main())
    assert state.stats.aborts["client_disconnect"] == 1
    assert state.stats.aborts["explicit"] == 0 and state.stats.aborts["error"] == 0


def test_nonstream_disconnect_is_counted_once():
    """The path the soak probe took: /v1/chat/completions with stream=false."""
    state = _hanging_manager()

    with pytest.raises(asyncio.CancelledError):
        _run_handler(handle_chat_completion(_chat_request(), FakeRequest(0), state, {}))

    assert state.stats.aborts["client_disconnect"] == 1
    assert len(_aborts(state)) == 1


def test_nonstream_disconnect_is_counted_even_if_the_handler_is_cancelled():
    """The §W5 regression itself.

    The ASGI server tears the request task down while the abort is in flight. Before the
    shield, the 0.1 s settling sleep swallowed the whole abort: no counter, no AbortMsg,
    and a request left holding its KV for the life of the server.
    """
    state = _hanging_manager()

    async def main():
        task = asyncio.ensure_future(
            handle_chat_completion(_chat_request(), FakeRequest(0), state, {})
        )
        await asyncio.sleep(0.05)   # inside the settling sleep
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _drain_aborts(state)

    asyncio.run(main())
    assert state.stats.aborts["client_disconnect"] == 1
    assert len(_aborts(state)) == 1, "the AbortMsg must reach the scheduler too"


def test_a_completed_request_is_not_counted_as_a_disconnect():
    """No regression: a connected client's request leaves every abort counter at 0."""
    state = HangingState(
        replies=[
            UserReply(
                uid=1,
                incremental_output="hi there",
                finished=True,
                finish_reason="stop",
                prompt_tokens_delta=3,
                completion_tokens_delta=2,
            )
        ]
    )
    _run_handler(handle_chat_completion(_chat_request(), FakeRequest(None), state, {}))
    assert state.aborts == []


def test_the_prepare_stop_drain_is_counted_under_its_own_reason():
    """``explicit`` and ``client_disconnect`` must stay distinguishable on /v1/stats --
    a maintenance drain is not a client going away."""
    state = _hanging_manager()

    async def main():
        uid = state.new_user()
        await state.abort_user(uid, reason="explicit")

    asyncio.run(main())
    assert state.stats.aborts == {"client_disconnect": 0, "explicit": 1, "error": 0}
