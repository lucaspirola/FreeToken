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


#: Conversation-id headers for ``/v1/chat/completions``, in precedence order with
#: the label each derived id is namespaced under. Switchyard's own override header
#: wins over the harness-native ids it forwards verbatim (``crates/protocol/
#: src/metadata.rs``: the ``x-switchyard-*`` headers are explicit overrides).
_CHAT_SESSION_HEADERS = (
    ("switchyard", "x-switchyard-session-id"),
    ("claude-code", "x-claude-code-session-id"),
    ("codex", "x-codex-session-id"),
)

#: Sub-agent lineage headers. A child agent of one conversation gets its own lease
#: so a parent and its sub-agent neither serialize on nor evict each other's prefix.
_CHAT_AGENT_HEADERS = ("x-switchyard-agent-id", "x-claude-code-agent-id")

#: Generic correlation headers, below OpenAI's own ``prompt_cache_key``.
_CHAT_FALLBACK_HEADERS = ("session-id", "x-session-id")


def _first_header(request: Any | None, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _header(request, name)
        if value is not None:
            return value
    return None


def chat_session_id(req: Any, request: Any | None) -> str | None:
    """Bind a ``/v1/chat/completions`` turn to a KV session lease.

    Precedence: an explicit ``session_id`` (the client opted in and owns the
    lease) beats every inferred identity; then Switchyard's own session header,
    the two harness-native ids it forwards, OpenAI's ``prompt_cache_key``
    affinity hint, and finally the generic correlation headers. Everything below
    the explicit field is *auto-bound*: the caller marks those leases reclaimable
    (as the Anthropic path does) so an idle one can be taken back without a close
    the client never learned to send.
    """
    if req.session_id is not None:
        return req.session_id
    agent = _first_header(request, _CHAT_AGENT_HEADERS)
    for client, name in _CHAT_SESSION_HEADERS:
        session = _header(request, name)
        if session is not None:
            return _implicit_id(client, session, agent)
    cache_key = getattr(req, "prompt_cache_key", None)
    identity = cache_key.strip() if isinstance(cache_key, str) and cache_key.strip() else None
    identity = identity or _first_header(request, _CHAT_FALLBACK_HEADERS)
    if identity is None:
        return None
    return _implicit_id("openai-chat", identity, agent)
