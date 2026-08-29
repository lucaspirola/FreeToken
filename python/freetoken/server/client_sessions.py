"""Translate agent-client conversation identifiers into FreeToken session leases."""

from __future__ import annotations

import hashlib
from typing import Any


def _header(request: Any | None, name: str) -> str | None:
    if request is None:
        return None
    value = request.headers.get(name)
    return value.strip() if value and value.strip() else None


def _implicit_id(client: str, *parts: str | None) -> str | None:
    values = [part for part in parts if part]
    if not values:
        return None
    digest = hashlib.sha256("\0".join(values).encode()).hexdigest()
    return f"auto:{client}:{digest}"


def anthropic_session_id(req: Any, request: Any | None) -> str | None:
    """Use Claude Code's stable session header, splitting child agents when present."""
    if req.session_id is not None:
        return req.session_id
    session = _header(request, "x-claude-code-session-id")
    agent = _header(request, "x-claude-code-agent-id")
    return _implicit_id("claude-code", session, agent)


def responses_session_id(req: Any, request: Any | None) -> str | None:
    """Use Codex's stable prompt-cache/thread identity without client customization."""
    if req.session_id is not None:
        return req.session_id

    metadata = req.client_metadata or {}
    identity = (
        req.prompt_cache_key
        or metadata.get("session_id")
        or metadata.get("thread_id")
        or _header(request, "session-id")
        or _header(request, "thread-id")
    )
    return _implicit_id("codex", str(identity) if identity is not None else None)
