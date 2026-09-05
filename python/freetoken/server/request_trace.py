"""Request tracing: one JSON line per completed request, for replaying real traffic.

Enabled by ``--trace-dir DIR`` (off by default). Every request that reaches one of the
completion handlers appends one record to a per-process
``trace-<stamp>-<pid>.jsonl`` under that directory when it finishes -- successfully,
with an error, or by client disconnect.

Why it exists: the Switchyard soak drives *synthetic* traffic (fixed scenario mix, fixed
prompt lengths, a closed 16-client loop). A trace is the real thing -- real arrival
times, real session interleaving, real prompt-length and cached-fraction distributions --
and ``benchmarks/trace_replay.py`` plays it back against a server while
``benchmarks/trace_to_profile.py`` turns it into a ``benchmarks/scheduler_replay.py``
profile, so the CPU scheduler gate can run on the traffic shape that actually occurred.

**No prompt text is written by default.** What is written instead is the *prefix chain*:

    chain[i] = sha256(chain[i-1] || canonical(message_i))     (16 hex chars kept)

so two requests share exactly the first ``m`` messages iff their chains agree for ``m``
entries -- which is precisely the structure the radix prefix cache keys on. Together with
``msg_chars`` (per-message character counts) and the request's own ``prompt_tokens``, a
replay can synthesize filler prompts that reproduce both the length profile and the
prefix-sharing graph without ever holding the text. ``--trace-include-text`` adds the
messages themselves, for a private replay of one's own traffic.

Overhead: the record is built from values the handler already has and handed to a
dedicated writer thread through a bounded queue (the ``request_logger`` pattern), so the
event loop never touches the disk. Disabled, ``record()`` is a single ``if`` on a module
global.

Stdlib only, on purpose: ``benchmarks/`` and the tests read this format on machines with
no torch and no CUDA, and this module must stay importable there.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Iterable, Iterator

logger = logging.getLogger("freetoken.request_trace")

#: Bumped whenever a field changes meaning. Readers must refuse an unknown major version.
TRACE_VERSION = 1

#: Truncated sha256 width for the prefix chain. 16 hex chars = 64 bits: a collision needs
#: ~4e9 distinct messages before it is even likely, and a collision only ever makes the
#: replay *over*-share a prefix, never mis-order a request.
_HASH_HEX = 16

_MAX_QUEUE = 10_000

_dir: str | None = None
_include_text = False
_path: str | None = None
_fh = None
_queue: "queue.Queue[str | None]" = queue.Queue(maxsize=_MAX_QUEUE)
_worker: threading.Thread | None = None
_lock = threading.Lock()
_failed = False
_warned = False
_dropped = 0


def enabled() -> bool:
    return _dir is not None and not _failed


def trace_path() -> str | None:
    return _path


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        logger.warning("%s", msg)
        _warned = True


def configure(trace_dir: str | None, include_text: bool = False) -> None:
    """Turn tracing on (idempotent). Called once at server startup so a bad path fails
    loudly there rather than silently on the first request."""
    global _dir, _include_text
    if not trace_dir:
        return
    with _lock:
        if _dir is not None:
            return
        _dir = os.path.expanduser(os.path.expandvars(trace_dir))
        _include_text = bool(include_text)
    _open()


def _open() -> None:
    global _path, _fh, _worker, _failed
    try:
        os.makedirs(_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        _path = os.path.join(_dir, f"trace-{stamp}-{os.getpid()}.jsonl")
        # 0o600: with --trace-include-text these lines carry prompts.
        fd = os.open(_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        _fh = os.fdopen(fd, "a", encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 -- never fail serving over a trace path
        _failed = True
        _warn_once(f"cannot open request trace under {_dir!r}: {exc}")
        return
    _worker = threading.Thread(target=_writer_loop, name="reqtrace-writer", daemon=True)
    _worker.start()
    atexit.register(close)
    logger.info("Tracing requests to %s (text=%s)", _path, _include_text)


def _writer_loop() -> None:
    fh = _fh
    while True:
        line = _queue.get()
        try:
            if line is None:
                return
            fh.write(line)
        except Exception as exc:  # noqa: BLE001 -- a write failure must not kill the thread
            _warn_once(f"failed to write request trace: {exc}")
        finally:
            _queue.task_done()
        # Flush only when the queue has drained: a burst of finishes costs one fsync-less
        # flush, not one per record.
        if _queue.empty():
            try:
                fh.flush()
            except Exception:  # noqa: BLE001
                pass


def flush(timeout: float = 5.0) -> bool:
    """Block until queued records are written. Tests and shutdown only."""
    if _worker is None:
        return True
    done = threading.Event()
    threading.Thread(target=lambda: (_queue.join(), done.set()), daemon=True).start()
    ok = done.wait(timeout)
    try:
        _fh.flush()
    except Exception:  # noqa: BLE001
        pass
    return ok


def close(timeout: float = 5.0) -> None:
    """Flush and stop the writer. Idempotent; safe from atexit and from a lifespan hook."""
    global _worker
    w = _worker
    if w is None:
        return
    _worker = None
    try:
        flush(timeout)
        _queue.put(None, timeout=1.0)
        w.join(timeout=2.0)
        if _fh is not None:
            _fh.flush()
            _fh.close()
    except Exception:  # noqa: BLE001
        pass
    if _dropped:
        logger.warning("request trace dropped %d records (slow disk?)", _dropped)


# --------------------------------------------------------------------------- hashing


def as_jsonable(msg: Any) -> Any:
    """Normalize one prompt element to plain JSON data, once per request.

    Chat messages arrive as pydantic ``Message`` models; a ``/v1/completions`` prompt is a
    string or a list of token ids. Duck-typed rather than importing pydantic, so this
    module stays stdlib-only and importable by ``benchmarks/`` and the tests.
    """
    if isinstance(msg, (str, int, float, bool)) or msg is None:
        return msg
    dump = getattr(msg, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(msg, dict):
        return msg
    return str(msg)


def _canonical(msg: Any) -> str:
    """A stable string for one already-normalized prompt element.

    ``sort_keys`` makes the hash independent of field order, so a client that reorders
    keys between turns still reports the same shared prefix.
    """
    if isinstance(msg, str):
        return msg
    try:
        return json.dumps(msg, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(msg)


def prefix_chain(messages: Iterable[Any]) -> tuple[list[str], list[int]]:
    """``(chain, chars)`` for a prompt: cumulative per-message hashes and lengths.

    ``chain[i]`` covers messages ``0..i`` inclusive, so ``chain[:m]`` of two requests are
    equal exactly when their first ``m`` messages are identical -- the prefix-cache
    boundary, expressed without the text.
    """
    chain: list[str] = []
    chars: list[int] = []
    h = b""
    for msg in messages:
        s = _canonical(msg)
        chars.append(len(s))
        h = hashlib.sha256(h + s.encode("utf-8")).digest()
        chain.append(h.hex()[:_HASH_HEX])
    return chain, chars


def message_roles(messages: Iterable[Any]) -> list[str]:
    """Per-message role, or ``"prompt"`` for a raw ``/v1/completions`` string.

    Roles carry no user content and the chat template's own token cost depends on them, so
    a replay that reproduces the role sequence reproduces the template overhead too.
    """
    out: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append(str(msg.get("role") or "user"))
        else:
            role = getattr(msg, "role", None)
            out.append(str(role) if role else "prompt")
    return out


def prompt_hash(messages: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for msg in messages:
        h.update(_canonical(msg).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:_HASH_HEX]


# --------------------------------------------------------------------------- recording


def record(
    *,
    route: str,
    arrival: float,
    messages: Iterable[Any] | None = None,
    request_id: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    stream: bool | None = None,
    prompt_tokens: int | None = None,
    cached_tokens: int | None = None,
    max_tokens: int | None = None,
    sampling: dict[str, Any] | None = None,
    output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    ttft: float | None = None,
    finished: float | None = None,
    finish_reason: str | None = None,
    status: str = "ok",
    error_code: str | None = None,
) -> None:
    """Append one trace record. No-op unless ``--trace-dir`` was given; never raises.

    ``arrival``/``ttft``/``finished`` are ``time.time()`` values (``ttft`` is the absolute
    time of the first token, not a delta -- the record stores the delta).
    """
    if _dir is None or _failed:
        return
    try:
        msgs = [as_jsonable(m) for m in messages] if messages is not None else []
        chain, chars = prefix_chain(msgs)
        now = finished if finished is not None else time.time()
        rec: dict[str, Any] = {
            "v": TRACE_VERSION,
            "t": round(arrival, 6),
            "route": route,
            "rid": request_id,
            "model": model,
            "session": session_id,
            "stream": stream,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "max_tokens": max_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "ttft_ms": None if ttft is None else round((ttft - arrival) * 1e3, 3),
            "duration_ms": round((now - arrival) * 1e3, 3),
            "finish_reason": finish_reason,
            "status": status,
            "error_code": error_code,
            "sampling": sampling or {},
            "prompt_sha256": prompt_hash(msgs),
            "msg_chain": chain,
            "msg_chars": chars,
            "msg_roles": message_roles(msgs),
        }
        if _include_text:
            rec["messages"] = msgs
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    except Exception as exc:  # noqa: BLE001 -- tracing must never break a request
        _warn_once(f"failed to build trace record: {exc}")
        return
    try:
        _queue.put_nowait(line)
    except queue.Full:
        global _dropped
        _dropped += 1
        _warn_once("request trace queue full; dropping records (slow disk?)")
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"failed to enqueue trace record: {exc}")


# --------------------------------------------------------------------------- context


class _NullTrace:
    """What ``start()`` returns when tracing is off: every call site stays unconditional
    and costs one attribute lookup plus an empty call, with no allocation per request."""

    __slots__ = ()

    def first_token(self) -> None:
        pass

    def seal(self, **_: Any) -> None:
        pass

    def __bool__(self) -> bool:
        return False


NULL = _NullTrace()


class Trace:
    """One in-flight request. Built at the handler entry, sealed wherever it ends.

    The handler holds the messages by reference and nothing is serialized until
    :meth:`seal`, so an aborted request pays only the arrival timestamp.
    """

    __slots__ = ("route", "arrival", "messages", "model", "session_id", "stream",
                 "max_tokens", "sampling", "ttft", "_sealed")

    def __init__(self, route: str, messages: Any, model: str | None,
                 session_id: str | None, stream: bool | None,
                 max_tokens: int | None, sampling: dict[str, Any] | None):
        self.route = route
        self.arrival = time.time()
        self.messages = messages
        self.model = model
        self.session_id = session_id
        self.stream = stream
        self.max_tokens = max_tokens
        self.sampling = sampling
        self.ttft: float | None = None
        self._sealed = False

    def first_token(self) -> None:
        if self.ttft is None:
            self.ttft = time.time()

    def seal(self, **kw: Any) -> None:
        """Write the record. Idempotent: a request that ends twice (a stream that raises
        inside its own ``finally``) must not produce two rows."""
        if self._sealed:
            return
        self._sealed = True
        kw.setdefault("session_id", self.session_id)
        record(
            route=self.route,
            arrival=self.arrival,
            messages=self.messages,
            model=self.model,
            stream=self.stream,
            max_tokens=self.max_tokens,
            sampling=self.sampling,
            ttft=self.ttft,
            **kw,
        )

    def __bool__(self) -> bool:
        return True


def start(route: str, *, messages: Any = None, model: str | None = None,
          session_id: str | None = None, stream: bool | None = None,
          max_tokens: int | None = None,
          sampling: dict[str, Any] | None = None) -> Trace | _NullTrace:
    if _dir is None or _failed:
        return NULL
    return Trace(route, messages, model, session_id, stream, max_tokens, sampling)


# --------------------------------------------------------------------------- reading


def read_trace(path: str) -> Iterator[dict[str, Any]]:
    """Yield records from one ``.jsonl`` file or every ``trace-*.jsonl`` under a directory,
    in arrival order. Malformed lines are skipped (a trace of a killed server ends in a
    partial line often enough to be worth tolerating)."""
    files: list[str]
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.startswith("trace-") and f.endswith(".jsonl")
        )
    else:
        files = [path]
    recs: list[dict[str, Any]] = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict) or rec.get("v") != TRACE_VERSION:
                    continue
                recs.append(rec)
    recs.sort(key=lambda r: r.get("t") or 0.0)
    return iter(recs)


def _reset_for_tests() -> None:
    """Tear the module state down so a test can configure a fresh directory."""
    global _dir, _include_text, _path, _fh, _queue, _worker, _failed, _warned, _dropped
    close()
    _dir = None
    _include_text = False
    _path = None
    _fh = None
    _queue = queue.Queue(maxsize=_MAX_QUEUE)
    _worker = None
    _failed = False
    _warned = False
    _dropped = 0
