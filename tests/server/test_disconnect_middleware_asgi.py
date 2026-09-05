"""The disconnect must survive the middleware stack — proven over a real socket.

``tests/server/test_disconnect_abort.py`` covers the abort *logic* with a fake request whose
``is_disconnected()`` answers True on cue. That is exactly the assumption soak §X5 broke: the
logic was right, and the abort still never fired in production, because the request-ring
recorder installed on the app was an ``@app.middleware("http")`` — i.e. a Starlette
``BaseHTTPMiddleware``, which runs the endpoint behind a proxied receive channel and never
forwards ``http.disconnect``. ``is_disconnected()`` therefore read False for the life of the
request and a 60 K-token non-streaming request answered 200 OK into a closed socket.

Nothing below that layer can see this, so these tests run the real thing: a uvicorn server on
an ephemeral port, a raw client socket that sends a request, waits until the endpoint has
admitted it, and then closes — and the assertion is on the abort the scheduler would receive
(``AbortMsg`` on the wire, ``/v1/stats`` ``requests.aborts.client_disconnect`` incremented).

The ``basehttp`` case is the sensitivity check: the same server with the old recorder must NOT
abort. If a future Starlette forwards ``http.disconnect`` through ``BaseHTTPMiddleware``, that
test fails — read this docstring, confirm it, and retire the case rather than the fix.

Run:  PYTHONPATH=python <venv>/bin/python -m pytest \
          tests/server/test_disconnect_middleware_asgi.py -v
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI, Request

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.message import AbortMsg  # noqa: E402
from freetoken.server import disconnect as D  # noqa: E402
from freetoken.server import request_ring  # noqa: E402
from freetoken.server.api_server import FrontendManager  # noqa: E402

# The endpoint polls every POLL_S; the client closes its socket 0.3 s after admission, so a
# working path aborts well inside ABORT_TIMEOUT_S. The negative case waits the same budget
# before concluding the abort never came, so both arms are equally patient.
POLL_S = 0.02
ABORT_TIMEOUT_S = 5.0


class FakeSendQueue:
    """The scheduler-bound ZMQ queue. Everything the abort must reach ends up in ``sent``."""

    def __init__(self) -> None:
        self.sent: list = []

    async def put(self, msg) -> None:
        self.sent.append(msg)


def _manager() -> FrontendManager:
    """A real FrontendManager — real abort_user, real stats — with the queues faked out.
    ``initialized=True`` skips the listener task, which would read from the absent recv queue."""
    return FrontendManager(
        config=SimpleNamespace(served_model_name="unit-model", model_path="/models/unit"),
        send_tokenizer=FakeSendQueue(),
        recv_tokenizer=None,
        maintenance_state="serving",
        initialized=True,
    )


@contextlib.contextmanager
def _serve(app: FastAPI):
    """Run `app` under a real uvicorn on an ephemeral port; yield the port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10.0
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "uvicorn did not come up"
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        with contextlib.suppress(OSError):
            sock.close()


def _post_then_drop(port: int, path: str, admitted: threading.Event) -> None:
    """Send a non-streaming request, wait until the endpoint has admitted it, close the socket.

    A half-close is not enough: uvicorn only reports ``http.disconnect`` once the connection is
    actually gone, which is what an abandoned client does."""
    body = b'{"prompt": "x"}'
    head = (
        f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()
    client = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        client.sendall(head + body)
        assert admitted.wait(timeout=10.0), "endpoint never admitted the request"
        time.sleep(0.3)  # let the request settle into the poll loop, as a real one would
    finally:
        with contextlib.suppress(OSError):
            client.shutdown(socket.SHUT_RDWR)
        client.close()


def _wait_for(pred, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _build_app(recorder: str, state: FrontendManager, admitted: threading.Event) -> FastAPI:
    """A server whose /generate mirrors the real non-streaming path: park on the engine inside
    ``await_or_disconnect``, and send the AbortMsg from the ``CancelledError`` handler."""
    app = FastAPI()
    if recorder == "asgi":
        app.add_middleware(
            request_ring.RequestRingMiddleware, model_name=lambda: "unit-model"
        )
    elif recorder == "basehttp":

        @app.middleware("http")  # what api_server.py installed before the fix
        async def _record(request, call_next):
            return await call_next(request)

    elif recorder != "none":  # pragma: no cover - guards a typo in the parametrize list
        raise AssertionError(f"unknown recorder {recorder!r}")

    @app.post("/generate")
    async def generate(request: Request):
        await request.body()
        uid = state.new_user()
        admitted.set()
        try:
            # Never completes: the request is parked on the engine, exactly where the leak was.
            await D.await_or_disconnect(
                asyncio.Event().wait(), request, poll_interval=POLL_S
            )
        except asyncio.CancelledError:
            await state.abort_user(uid, session_id="sess-x")
            raise
        return {"ok": True}  # pragma: no cover - unreachable in these tests

    @app.post("/generate/quick")
    async def quick():
        return {"ok": True}

    return app


def _aborts(state: FrontendManager) -> list:
    return [m for m in state.send_tokenizer.sent if isinstance(m, AbortMsg)]


# --------------------------------------------------------------------------- #
# The fix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("recorder", ["asgi", "none"])
def test_dropped_nonstream_request_aborts_through_the_middleware(recorder):
    """A client that walks away from a non-streaming request must free the engine.

    ``none`` is the control (no middleware at all); ``asgi`` is the shipping recorder and must
    behave identically — that equivalence is the whole claim of the fix."""
    state = _manager()
    admitted = threading.Event()
    app = _build_app(recorder, state, admitted)

    with _serve(app) as port:
        _post_then_drop(port, "/generate", admitted)
        assert _wait_for(lambda: _aborts(state), ABORT_TIMEOUT_S), (
            "no AbortMsg reached the scheduler: the disconnect never got through"
        )

    aborts = _aborts(state)
    assert len(aborts) == 1
    assert aborts[0].session_id == "sess-x"
    assert state.stats.aborts["client_disconnect"] == 1
    # The abort also has to release the frontend bookkeeping it was holding.
    assert state.ack_map == {} and state.event_map == {}


def test_basehttp_recorder_swallows_the_disconnect():
    """The bug, reproduced: same server, old recorder, no abort. See the module docstring."""
    state = _manager()
    admitted = threading.Event()
    app = _build_app("basehttp", state, admitted)

    with _serve(app) as port:
        _post_then_drop(port, "/generate", admitted)
        fired = _wait_for(lambda: _aborts(state), ABORT_TIMEOUT_S)

    assert not fired, (
        "BaseHTTPMiddleware forwarded http.disconnect — Starlette changed; this test has "
        "served its purpose and can be retired (the fix itself must stay)."
    )


# --------------------------------------------------------------------------- #
# ...without losing what the middleware was there for
# --------------------------------------------------------------------------- #
def test_asgi_recorder_still_records_the_ring_row():
    """The pure-ASGI recorder must log the same row the BaseHTTPMiddleware one did."""
    request_ring.reset()
    state = _manager()
    app = _build_app("asgi", state, threading.Event())
    try:
        with _serve(app) as port:
            client = socket.create_connection(("127.0.0.1", port), timeout=10)
            try:
                client.sendall(
                    f"POST /generate/quick HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                    f"Content-Length: 0\r\nConnection: close\r\n\r\n".encode()
                )
                assert b"200 OK" in client.recv(4096)
            finally:
                client.close()

        rows, _ = request_ring.requests_since(0, 100)
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["path"] == "/generate/quick"
        assert row["method"] == "POST"
        assert row["status"] == 200
        assert row["model"] == "unit-model"
        assert row["stream"] is False
        assert row["duration_ms"] >= 0
        assert row["prompt_tokens"] is None and row["completion_tokens"] is None
    finally:
        request_ring.reset()


@pytest.mark.parametrize(
    "path, tracked",
    [
        ("/v1/completions", True),
        ("/generate", True),
        ("/generate/quick", True),
        ("/v1/messages/count_tokens", False),  # shares no tracked prefix; never a ring row
        ("/v1/chat/completions", False),  # recorded by generation.py, with real token totals
        ("/v1/stats", False),
    ],
)
def test_tracked_prefixes_are_unchanged(path, tracked):
    """The path filter is the one piece of policy the rewrite had to carry over verbatim."""
    scope = {"type": "http", "path": path, "method": "POST"}
    assert request_ring.RequestRingMiddleware.tracks(scope) is tracked


def test_non_http_scopes_pass_straight_through():
    assert request_ring.RequestRingMiddleware.tracks({"type": "lifespan"}) is False
    assert (
        request_ring.RequestRingMiddleware.tracks({"type": "websocket", "path": "/generate"})
        is False
    )
