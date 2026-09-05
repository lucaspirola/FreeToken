"""In-memory ring of recent API requests for the desktop Logs tab.

A bounded deque + a monotonic all-time cursor: clients pull incrementally with
``?since=<next_cursor>``. Records are appended by :class:`RequestRingMiddleware` below (and,
for the chat protocols, by generation.py); p95 (for /v1/stats) reads the same ring. Purely
in-process — request_logger.py still owns the on-disk JSONL.

Stdlib-only on purpose: importing ``freetoken.server.api_server`` drags in torch and the
engine, so the recorder lives here where a disconnect probe or an ASGI-level test can import
it on a box with no CUDA."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class RequestRecord:
    ts: str
    method: str
    path: str
    status: int
    model: str | None
    duration_ms: int
    ttft_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    stream: bool | None
    error: str | None


class RequestRing:
    def __init__(self, capacity: int = 512) -> None:
        self._buf: "deque[tuple[int, RequestRecord]]" = deque(maxlen=capacity)
        self._next = 0  # all-time monotonic cursor (survives eviction)

    def add(self, rec: RequestRecord) -> None:
        self._buf.append((self._next, rec))
        self._next += 1

    def since(self, cursor: int, limit: int) -> tuple[list[dict], int]:
        """Return records with id >= cursor (up to limit) and the next cursor to poll with.
        If limit truncates the result set, next_cursor is the cursor of the last returned record.
        Only when all matched records are returned may next_cursor be self._next (the all-time count)."""
        matched = [(idx, rec) for idx, rec in self._buf if idx >= cursor]
        out = [asdict(rec) for idx, rec in matched[:limit]]

        # If we got fewer results than matched, we were truncated by limit
        if len(out) < len(matched):
            # next_cursor is the cursor of the last returned record + 1
            last_returned_idx = matched[len(out) - 1][0]
            next_cursor = last_returned_idx + 1
        else:
            # All matched records were returned
            next_cursor = self._next

        return out, next_cursor

    def p95_ms(self) -> int:
        durs = sorted(rec.duration_ms for _idx, rec in self._buf)
        if not durs:
            return 0
        k = max(0, math.ceil(0.95 * len(durs)) - 1)
        return int(durs[k])

    def ttft_mean_ms(self) -> int:
        """Mean TTFT over the records that have one."""
        vals = [rec.ttft_ms for _idx, rec in self._buf if rec.ttft_ms is not None]
        if not vals:
            return 0
        return int(round(sum(vals) / len(vals)))

    def count(self) -> int:
        return self._next


# ------------------------------------------------------------------ module singleton
_RING = RequestRing()


def record_request(rec: RequestRecord) -> None:
    _RING.add(rec)


def requests_since(cursor: int, limit: int) -> tuple[list[dict], int]:
    return _RING.since(cursor, limit)


def requests_p95_ms() -> int:
    return _RING.p95_ms()


def requests_ttft_mean_ms() -> int:
    return _RING.ttft_mean_ms()


def requests_count() -> int:
    return _RING.count()


def reset() -> None:
    """Test helper: clear the singleton ring."""
    global _RING
    _RING = RequestRing()


# ------------------------------------------------------------------ ASGI recorder

# Paths the recorder logs into the ring. The three chat protocols funnel through the shared
# generation layer and are recorded there instead (with real token totals) — kept out here to
# avoid a duplicate token-less row. These two run their own ack loop, so they stay logged here
# as before (without per-request tokens). See generation.py `_record_generation`.
TRACKED_REQUEST_PREFIXES = (
    "/v1/completions",
    "/generate",
)

# Subpaths that share a tracked prefix but are NOT generation requests. count_tokens never
# enters generation accounting, and its first-touch tokenizer load would otherwise dominate the
# /v1/stats p95 and pollute /v1/requests — exclude it before the prefix check below.
UNTRACKED_REQUEST_PREFIXES = ("/v1/messages/count_tokens",)


class RequestRingMiddleware:
    """Pure-ASGI recorder: times tracked generation requests into the ring.

    Pure ASGI is the whole point. This was an ``@app.middleware("http")`` decorator, which
    Starlette turns into a ``BaseHTTPMiddleware``; that class runs the downstream app in its
    own task behind a *proxied* receive channel that never forwards ``http.disconnect``. So
    ``Request.is_disconnected()`` — polled by ``server/disconnect.py`` while a handler is
    parked on the engine — read False forever, and the ``except asyncio.CancelledError`` arm
    that sends the AbortMsg was unreachable for every non-streaming request. Soak §X5 is that
    silence: a request whose socket was closed 5 s into its prefill ran to completion and
    answered 200 OK into a dead socket, while the identical *streaming* request aborted at
    once (its abort comes from the send side, not the poll).
    ``benchmarks/probe_disconnect_middleware.py`` reproduces both arms in ~10 s, no model.

    ``receive`` is therefore passed through untouched; only ``send`` is observed. The row is
    written on ``http.response.start``, which is exactly where the old ``call_next`` returned:
    same status, same content-type, same duration (for a stream, the time to the response
    head; for a non-stream, the whole handler). A request that never produces a response head
    — i.e. one aborted by the disconnect path this fix exists to unblock — logs no row, where
    before it logged a misleading "200 OK" for an answer nobody received.

    ``model_name`` is a callable because the served model is only known after startup; single-
    model server, so it is the whole answer. Token counts are P3 (SSE usage arrives after the
    handler returns) and stay None.
    """

    def __init__(self, app: Any, model_name: Callable[[], str | None]) -> None:
        self.app = app
        self.model_name = model_name

    @staticmethod
    def tracks(scope: dict) -> bool:
        if scope.get("type") != "http":
            return False
        path = scope.get("path", "")
        return path.startswith(TRACKED_REQUEST_PREFIXES) and not path.startswith(
            UNTRACKED_REQUEST_PREFIXES
        )

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if not self.tracks(scope):
            await self.app(scope, receive, send)
            return

        start = time.monotonic()
        recorded = False

        async def send_wrapper(message: dict) -> None:
            nonlocal recorded
            if message.get("type") == "http.response.start" and not recorded:
                recorded = True
                ctype = ""
                for key, value in message.get("headers") or ():
                    if key.lower() == b"content-type":
                        ctype = value.decode("latin-1")
                        break
                record_request(
                    RequestRecord(
                        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        method=scope.get("method", ""),
                        path=scope.get("path", ""),
                        status=int(message.get("status", 0)),
                        model=self.model_name(),
                        duration_ms=int((time.monotonic() - start) * 1000),
                        ttft_ms=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                        stream=ctype.startswith("text/event-stream"),
                        error=None,
                    )
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)
