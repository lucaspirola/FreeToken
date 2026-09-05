"""Client-disconnect detection for the request paths that park on the engine.

An abandoned request only stops costing the server once an ``AbortMsg`` reaches the
scheduler: until then its pending entry, its table slot and every KV page already
forwarded for it stay allocated. Starlette hands us ``Request.is_disconnected()``, but
nothing polled it while a handler was parked waiting for the engine — the streaming path
checked only *after* a chunk had been yielded, and the non-streaming path only if
something else cancelled it. A request dropped during prefill yields no chunk and is
never cancelled, so it leaked for the life of the server.

Both helpers below race the awaited work against a periodic poll of the connection and
raise ``asyncio.CancelledError`` the moment the client is gone — the signal every caller
already handles by aborting the request. The generation code stays HTTP-unaware: only
the duck-typed ``is_disconnected()`` is used, so a caller with no request object (the
in-process callers, and the tests) passes ``None`` and gets a plain await.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator, Awaitable, TypeVar

T = TypeVar("T")

# How often to ask the transport whether the client is still there while we wait. The
# poll costs one non-blocking ASGI receive; the interval only bounds how long an
# abandoned prefill keeps running, so it is deliberately much coarser than a chunk.
POLL_INTERVAL_S = 0.25

# nginx's "client closed request". Never reaches a socket -- uvicorn drops a send on a
# disconnected transport -- so it exists purely to give the endpoint something to RETURN.
CLIENT_CLOSED_REQUEST = 499


class ClientGone(asyncio.CancelledError):
    """The poll below observed the client had left; nobody cancelled this task.

    A subclass of ``CancelledError`` on purpose: every caller's existing
    ``except asyncio.CancelledError`` still catches it and still sends the AbortMsg, so
    the abort stays in one place per endpoint. What the distinct type buys is the answer
    to the question the endpoint could not previously ask -- "is there still a client to
    raise at?" -- and the answer is no, so re-raising is wrong: it leaves uvicorn to log
    ``ERROR: Exception in ASGI application`` with a full traceback for a request that
    ended exactly as designed (soak §Y8.4, 10 of them in one phase). Return
    :func:`client_gone_response` instead; a genuine outer cancellation (shutdown) is a
    plain ``CancelledError`` and must still propagate.
    """


def client_gone_response():
    """The quiet end of a request whose client is already gone.

    Built lazily so this module stays importable without a web framework (the in-process
    callers and most of the unit tests never touch it).
    """
    from fastapi import Response

    return Response(status_code=CLIENT_CLOSED_REQUEST)


async def client_gone(request: Any) -> bool:
    """True if the client has disconnected (or its receive channel is already dead)."""
    if request is None:
        return False
    try:
        return bool(await request.is_disconnected())
    except Exception:  # noqa: BLE001 -- see below
        # The probe is a pure read of the ASGI receive channel, so an error out of it
        # (ClientDisconnect, a torn-down transport, a closed channel) means we can no
        # longer establish that anyone is listening. Treating that as "gone" is also the
        # only safe answer: letting it propagate would kill the request WITHOUT ever
        # sending the abort, which is exactly the leak this module exists to close.
        return True


async def _drain_cancelled(task: asyncio.Task) -> None:
    """Cancel `task` and wait for it to finish unwinding.

    The wait is what makes cleanup reliable: it lets the cancelled coroutine run its own
    ``finally`` (``wait_for_ack`` drops the ack/event maps, ``generate_events`` records
    the row) before we propagate. Whatever it raises on the way out is discarded — the
    client that would have received it is gone.
    """
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _wait_or_disconnect(
    task: asyncio.Task, request: Any, poll_interval: float | None
) -> None:
    """Wait for `task`, polling the connection. On disconnect, cancel it and raise."""
    # Read at call time, not as a default argument, so the module constant stays the one
    # place the interval is set (and a test can turn it down).
    interval = POLL_INTERVAL_S if poll_interval is None else poll_interval
    while True:
        done, _ = await asyncio.wait({task}, timeout=interval)
        if done:
            return
        if await client_gone(request):
            await _drain_cancelled(task)
            raise ClientGone


async def await_or_disconnect(
    awaitable: Awaitable[T], request: Any, poll_interval: float | None = None
) -> T:
    """Await `awaitable`, raising ``CancelledError`` as soon as the client disconnects.

    Used by the non-streaming paths around ``generate_full``: the caller's existing
    ``except asyncio.CancelledError`` is what sends the AbortMsg, so the abort stays in
    exactly one place per endpoint.
    """
    if request is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    await _wait_or_disconnect(task, request, poll_interval)
    return task.result()


async def aiter_or_disconnect(
    iterable: Any, request: Any, poll_interval: float | None = None
) -> AsyncIterator[Any]:
    """Iterate `iterable`, polling the connection *while* each item is awaited.

    The poll covers the wait for the first item — the prefill window, where the leak
    lived — as well as every later one, and the connection is re-checked after an item
    arrives so a chunk is never written to a socket already known to be gone.
    """
    if request is None:
        async for item in iterable:
            yield item
        return
    iterator = iterable.__aiter__()
    step: asyncio.Task | None = None
    try:
        while True:
            step = asyncio.ensure_future(iterator.__anext__())
            await _wait_or_disconnect(step, request, poll_interval)
            try:
                item = step.result()
            except StopAsyncIteration:
                return
            finally:
                step = None
            if await client_gone(request):
                raise asyncio.CancelledError
            yield item
    finally:
        # Only reachable when the *caller* was cancelled (server shutdown, uvicorn
        # tearing the cycle down) while an item was in flight: that step still owns the
        # generator, so it has to be unwound before we leave.
        if step is not None and not step.done():
            await _drain_cancelled(step)
