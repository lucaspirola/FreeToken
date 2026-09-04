"""Read-only control-plane endpoints consumed by the desktop app: /health (lifecycle),
/v1/stats (runtime metrics, Task 6), /v1/requests (request log ring, Task 5).

All handlers read a shared FrontendManager snapshot via ``get_state``; nothing here touches
the scheduler or blocks. Registered on the app alongside the OpenAI/Anthropic/Responses routes.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Health is a liveness probe as much as a lifecycle one: answer 503 whenever the doc says
# "error", so a router or supervisor polling it sees a failing upstream instead of a healthy
# one that answers nothing.
HEALTH_ERROR_STATUS = 503


def dead_backend_reason(state: Any) -> str | None:
    """Name the first backend worker process that is no longer running, else None.

    ``fatal_error`` alone is not a liveness signal: the supervisor thread latches it up to one
    poll after the fact and then *returns*, so any later death (or a wedged supervisor) leaves
    it None forever. In the Switchyard soak that produced a nine-minute stall in which every
    ``/health`` probe answered ``{"status": "ok"}`` with a dead scheduler behind it. The
    ``multiprocessing.Process`` handles are already on the frontend state; ask them directly.
    """
    for proc in getattr(state, "backend_processes", None) or ():
        try:
            alive = proc.is_alive()
        except Exception:  # noqa: BLE001 -- unqueryable handle: assume alive (as the supervisor does)
            continue
        if not alive:
            return f"backend worker {getattr(proc, 'name', '?')} is not running"
    return None


def build_health(state: Any, version: str) -> dict:
    """Full-lifecycle health doc: loading -> ok -> error."""
    instance_id = getattr(state, "instance_id", None)
    fatal = getattr(state, "fatal_error", None) or dead_backend_reason(state)
    if fatal:
        return {"status": "error", "message": fatal, "instance_id": instance_id}

    mstate = getattr(state, "maintenance_state", "serving")
    config = getattr(state, "config", None)
    model = getattr(config, "served_model_name", None)

    if mstate == "loading":
        lp = getattr(state, "load_progress", None)
        return {
            "status": "loading",
            "phase": lp.phase if lp is not None else "other",
            "progress": {
                "done_bytes": lp.done_bytes if lp is not None else 0,
                "total_bytes": lp.total_bytes if lp is not None else 0,
            },
            "model": model,
            "instance_id": instance_id,
        }

    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    return {
        "status": "ok",
        "model": model,
        "instance_id": instance_id,
        "uptime_s": uptime_s,
        "maintenance": mstate,
        "version": version,
    }


def register_control_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict] | None = None,
) -> None:
    @app.get("/health")
    async def health():
        doc = build_health(get_state(), app.version)
        if doc.get("status") == "error":
            return JSONResponse(status_code=HEALTH_ERROR_STATUS, content=doc)
        return doc

    from . import request_ring

    @app.get("/v1/requests")
    async def list_requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        entries, next_cursor = request_ring.requests_since(since, limit)
        return {"entries": entries, "next_cursor": next_cursor}

    from .stats import build_stats

    @app.get("/v1/stats")
    async def stats():
        doc = build_stats(
            get_state(), request_ring.requests_p95_ms(), request_ring.requests_ttft_mean_ms()
        )
        # Surface the model's recommended sampling (from its generation_config.json / GGUF
        # metadata) so clients can seed their sampling controls per-model instead of guessing.
        if get_model_sampling is not None:
            doc["model"]["sampling"] = get_model_sampling() or {}
        return doc
