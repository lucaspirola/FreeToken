#!/usr/bin/env python3
"""Why ``requests.aborts.client_disconnect`` never counts a NON-streaming abort.

Soak §X5: a 60 K-token non-streaming request whose socket was closed 5 s into its prefill
ran to completion and answered 200 OK into the dead socket, while the identical streaming
request was aborted the instant its socket closed. ``server/disconnect.py`` polls
``Request.is_disconnected()`` every 0.25 s and is correct; what blinds it is
``api_server.py``'s ``@app.middleware("http")`` request-ring recorder. Starlette turns that
decorator into a ``BaseHTTPMiddleware``, which proxies the ASGI receive channel through its
own task and never forwards ``http.disconnect`` to the downstream ``Request`` -- so the
poll reads False forever and the ``except asyncio.CancelledError`` handler that sends the
AbortMsg is never entered. The streaming path is immune because its abort comes from the
*send* side (the write to a closed socket), not from the poll -- so the ``/stream`` arm
here reports NO RESULT in **both** arms and is not the evidence; it is there to show that
the poll alone is blind on that path too. The streaming abort was verified live instead
(soak §X5: ``Client disconnected ... user 3407`` the second the socket closed).

    uv run benchmarks/probe_disconnect_middleware.py            # middleware ON  -> never seen
    FREETOKEN_MW=0 uv run benchmarks/probe_disconnect_middleware.py   # OFF -> seen in 2.01 s

No GPU, no model, ~10 s. Measured 2026-09-05 on starlette 1.6.0 / uvicorn 0.52.4 /
fastapi 0.141.1.
"""

import asyncio
import json
import os
import socket
import threading
import time
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse

MW = os.environ.get("FREETOKEN_MW", "1") == "1"
POLL = 0.25
app = FastAPI()
seen = {}

if MW:
    @app.middleware("http")            # exactly what api_server.py installs
    async def _record(request, call_next):
        return await call_next(request)

async def client_gone(request):
    try:
        return bool(await request.is_disconnected())
    except Exception:
        return True

@app.post("/nonstream")
async def nonstream(request: Request):
    await request.body()
    task = asyncio.ensure_future(asyncio.sleep(12))
    t0 = time.time()
    while True:
        done, _ = await asyncio.wait({task}, timeout=POLL)
        if done:
            seen["nonstream"] = "COMPLETED -- disconnect NEVER seen"
            return {"ok": True}
        if await client_gone(request):
            seen["nonstream"] = "disconnect seen after %.2f s" % (time.time() - t0)
            task.cancel()
            raise asyncio.CancelledError

@app.post("/stream")
async def stream(request: Request):
    await request.body()
    async def gen():
        t0 = time.time()
        for _ in range(60):
            await asyncio.sleep(0.2)
            if await client_gone(request):
                seen["stream"] = "disconnect seen after %.2f s" % (time.time() - t0)
                raise asyncio.CancelledError
            yield b"data: x\n\n"
        seen["stream"] = "COMPLETED -- disconnect NEVER seen"
    return StreamingResponse(gen(), media_type="text/event-stream")

def probe(path):
    body = json.dumps({"x": "y" * 1000}).encode()
    req = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:8731\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    s = socket.create_connection(("127.0.0.1", 8731), timeout=5)
    s.sendall(req); time.sleep(2.0)
    s.shutdown(socket.SHUT_RDWR); s.close(); time.sleep(5.0)
    print(f"  {path}: {seen.get(path.strip('/'), 'NO RESULT')}")

cfg = uvicorn.Config(app, host="127.0.0.1", port=8731, log_level="critical")
srv = uvicorn.Server(cfg)
threading.Thread(target=srv.run, daemon=True).start()
time.sleep(2)
print(f"BaseHTTPMiddleware installed: {MW}")
probe("/nonstream"); probe("/stream")
srv.should_exit = True
