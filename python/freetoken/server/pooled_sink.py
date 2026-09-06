"""Pooled hidden-state sink: one JSON line per pooled probe, appended server-side.

``kv_transfer_params.pooling`` returns the pooled prompt vectors inline
(:mod:`freetoken.hidden_states`); with ``--pooled-sink-dir`` set the same vectors are
also appended to ``<dir>/<pooled_sink or "default">/pooled.jsonl`` so a collector can
pick them up without holding the HTTP response. The inline response is unchanged.

Line schema (keys in this order):

``ts`` (unix seconds, float), ``request_id`` (the response id), ``session_id`` (the
FreeToken session lease the turn was bound to, or null -- a probe binds none, so this
is null today), ``x_switchyard_session_id`` (the raw request header, or null),
``model`` (the id the client named), ``prompt_tokens``, ``layer_ids``, ``hidden``,
``dtype``, ``mean`` / ``last`` (base64 float32, present only when requested),
``prompt_sha256`` (hex SHA-256 of the *rendered* chat-template prompt, UTF-8, from the
frontend tokenizer's ``render_prompt`` -- the exact string the worker encodes; null when
this server has no frontend tokenizer or the render fails).

Writes never fail the response: any error is logged and swallowed. The append runs in a
worker thread under an exclusive ``flock`` on the file, so concurrent probes from
several uvicorn workers interleave whole lines, never partial ones.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.utils import init_logger

logger = init_logger(__name__)

__all__ = [
    "DEFAULT_SINK",
    "POOLED_SINK_FILE",
    "PooledSink",
    "pooled_sink_for",
    "validate_pooled_sink",
    "write_pooled_line",
]

#: File name inside the per-sink subdirectory.
POOLED_SINK_FILE = "pooled.jsonl"
#: Subdirectory used when the request names no ``pooled_sink``.
DEFAULT_SINK = "default"

_SINK_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_DISABLED = "pooled sink is disabled; start the server with --pooled-sink-dir"


def validate_pooled_sink(params: Any, root: str | None) -> str | None:
    """Validate ``params.pooled_sink`` (``kv_transfer_params``) and return it, or None
    when the request names no sink.

    Naming a sink needs ``pooling`` (there is nothing else to record) and a server
    started with ``--pooled-sink-dir``. The name is a single path segment: letters,
    digits, ``.``, ``_``, ``-``, at most 64 characters, never ``.`` or ``..``.
    ``ValueError`` -> HTTP 400 on ``kv_transfer_params``. A ``pooling`` request that
    names no sink is still recorded (under ``"default"``) when the server has the
    directory, and silently not recorded when it does not.
    """
    name = getattr(params, "pooled_sink", None)
    if name is None:
        return None
    if getattr(params, "pooling", None) is None:
        raise ValueError("kv_transfer_params.pooled_sink requires kv_transfer_params.pooling")
    if not isinstance(name, str) or not _SINK_NAME.match(name) or name in (".", ".."):
        raise ValueError(
            "kv_transfer_params.pooled_sink must be a single path segment matching "
            "^[A-Za-z0-9._-]{1,64}$ (not '.' or '..')"
        )
    if root is None:
        raise ValueError(_DISABLED)
    return name


def write_pooled_line(root: str, sink: str, record: dict[str, Any]) -> str:
    """Append ``record`` as one JSON line to ``<root>/<sink>/pooled.jsonl``; return
    the path. Creates the subdirectory. The file is created 0o666 & ~umask (like the
    hidden-state artifact, so a collector under another uid can truncate/rotate it) and
    the write of the line happens under ``LOCK_EX``."""
    directory = os.path.join(root, sink)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, POOLED_SINK_FILE)
    line = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            os.write(fd, line)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return path


@dataclass
class PooledSink:
    """Everything a sink line needs except the pooled vectors, fixed at request time;
    :meth:`record` adds them once the engine has answered."""

    root: str
    sink: str
    request_id: str
    session_id: str | None
    x_switchyard_session_id: str | None
    model: str
    spec: Any  # GenSpec: the rendered prompt is hashed from its messages/tools/kwargs
    state: Any

    async def record(self, pooled: dict[str, Any]) -> None:
        """Append one line for ``pooled`` (the response's ``kv_transfer_params.pooled``).
        Never raises: a failed render or write is logged and the response goes out
        unchanged."""
        try:
            digest = await self._prompt_sha256()
            record = {
                "ts": time.time(),
                "request_id": self.request_id,
                "session_id": self.session_id,
                "x_switchyard_session_id": self.x_switchyard_session_id,
                "model": self.model,
                "prompt_tokens": pooled.get("prompt_tokens"),
                "layer_ids": pooled.get("layer_ids"),
                "hidden": pooled.get("hidden"),
                "dtype": pooled.get("dtype"),
            }
            for key in ("mean", "last"):
                if key in pooled:
                    record[key] = pooled[key]
            record["prompt_sha256"] = digest
            await asyncio.to_thread(write_pooled_line, self.root, self.sink, record)
        except Exception as exc:  # noqa: BLE001 -- never fail the response
            logger.warning(
                "pooled sink write failed for %s (%s/%s): %s",
                self.request_id, self.root, self.sink, exc,
            )

    async def _prompt_sha256(self) -> str | None:
        build = getattr(self.state, "frontend_tokenizer", None)
        if build is None:
            return None
        try:
            manager = await asyncio.to_thread(build)
            rendered = await asyncio.to_thread(
                manager.render_prompt,
                TokenizeMsg(
                    uid=0,
                    text=self.spec.messages,
                    sampling_params=SamplingParams(),
                    chat_template_kwargs=self.spec.chat_template_kwargs,
                    tools=self.spec.template_tools,
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- the hash is best-effort metadata
            logger.warning("pooled sink: prompt render failed for %s: %s", self.request_id, exc)
            return None
        if not isinstance(rendered, str):
            return None
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def pooled_sink_for(
    req: Any,
    request: Any,
    state: Any,
    spec: Any,
    request_id: str,
) -> PooledSink | None:
    """Build the sink for a chat request: every ``pooling`` request is recorded once
    the server has ``--pooled-sink-dir``, under ``pooled_sink`` or ``"default"``.
    None otherwise. ``validate_pooled_sink`` has already run by the time this is
    called, so a ``pooled_sink`` name here is well-formed."""
    params = req.kv_transfer_params
    if params is None or params.pooling is None:
        return None
    root = getattr(state.config, "pooled_sink_dir", None)
    if root is None:
        return None
    header = None
    if request is not None:
        value = request.headers.get("x-switchyard-session-id")
        header = value.strip() if value and value.strip() else None
    return PooledSink(
        root=root,
        sink=params.pooled_sink or DEFAULT_SINK,
        request_id=request_id,
        session_id=spec.session_id,
        x_switchyard_session_id=header,
        model=req.model,
        spec=spec,
        state=state,
    )
