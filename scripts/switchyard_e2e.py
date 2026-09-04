#!/usr/bin/env python3
"""End-to-end checks for FreeToken as a NVIDIA Switchyard ``openai_chat`` upstream.

Three subcommands, deliberately separable because they need different things:

``contract``
    Talks *directly* to a running FreeToken server and asserts every wire promise
    Switchyard depends on (``docs/switchyard.md`` lists them). No Switchyard build
    required.

``soak``
    Renders a ``routes.toml`` pointing at that FreeToken server, starts
    ``switchyard-server`` on it, waits for health, and runs ``switchyard-soak``
    through the router. Needs the Rust binaries built:
    ``cargo build --release -p switchyard-server -p switchyard-soak``.

``agents``
    Prints the exact environment lines for the manual Claude Code / Codex smoke
    tests. Nothing to run automatically -- those are interactive.

Neither ``contract`` nor ``soak`` starts FreeToken: the GPU server is launched by
hand (the P2 profile in ``docs/switchyard.md``) so that a failed check never leaves
a 16 GiB process behind.

Dependencies: stdlib plus ``httpx`` (already in the FreeToken venv). Run with
``uv run python scripts/switchyard_e2e.py ...`` or through ``scripts/switchyard_e2e.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

try:  # httpx is in the venv; keep the import soft so the helpers stay unit-testable
    import httpx
except ImportError:  # pragma: no cover - exercised only outside the venv
    httpx = None  # type: ignore[assignment]

DEFAULT_FREETOKEN_URL = "http://127.0.0.1:1919"  # ft serve default port
DEFAULT_ROUTER_URL = "http://127.0.0.1:4000"
DEFAULT_MODEL = "nemotron-3.5-lightning"
#: Route ids from the generated routes.toml, soaked in this order.
DEFAULT_ROUTES = ("switchyard/passthrough", "switchyard/stage")
SWITCHYARD_DIR = os.path.expanduser("~/ai/Switchyard")

#: Soak scenarios from the Phase 3D plan. Every id is a value ``switchyard-soak
#: --scenario`` accepts; keep this list in sync with ``docs/switchyard.md``.
SOAK_SCENARIOS = (
    "prefix-reuse",
    "growing-conversation",
    "tool-call-burst",
    "large-tool-catalog",
    "long-context",
)

# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/server/test_switchyard_e2e.py)
# --------------------------------------------------------------------------- #

#: Switchyard's escalation verdict, copied verbatim from
#: ``crates/libsy/src/prompts/escalation/schema.json``. A ``stage_router``
#: classifier sends exactly this as ``response_format.json_schema`` and
#: deserializes ``escalate`` out of the answer; an unparseable verdict degrades
#: to "ambiguous" (stay on the efficient tier), so the model that fails this
#: check silently disables escalation rather than erroring.
ESCALATION_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "escalate": {
            "type": "boolean",
            "description": (
                "True when the run is likely doomed without escalation to the "
                "strong tier."
            ),
        },
        "reason": {
            "type": "string",
            "description": (
                "One short sentence naming the trouble pattern, or stating why "
                "the run is progressing."
            ),
        },
    },
    "required": ["escalate", "reason"],
    "additionalProperties": False,
}

#: The wrapper Switchyard puts around it on the wire.
ESCALATION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "EscalationVerdict",
        "strict": True,
        "schema": ESCALATION_VERDICT_SCHEMA,
    },
}

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate ``value`` against the subset of JSON Schema Switchyard emits.

    Deliberately small (type / required / properties / enum / additionalProperties
    / items): the point is to check what the router itself would reject, not to be
    a general validator, and the e2e script must not grow a dependency the
    FreeToken venv does not already have.
    """
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str):
        allowed = _JSON_TYPES.get(expected, ())
        # JSON booleans are not numbers, however much Python disagrees.
        ok = isinstance(value, allowed) and not (
            expected in ("number", "integer") and isinstance(value, bool)
        )
        if not ok:
            return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']!r}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}: unexpected property {name!r}")
        for name, sub in properties.items():
            if name in value:
                errors.extend(validate_json_schema(value[name], sub, f"{path}.{name}"))
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(validate_json_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


def sse_events(raw: str) -> list[dict[str, Any]]:
    """Split a raw ``text/event-stream`` body into its decoded JSON payloads.

    ``[DONE]`` becomes ``{"__done__": True}`` so a caller can assert on stream
    termination; a payload that is not JSON is surfaced as ``{"__raw__": ...}``
    rather than raising, because a malformed event is itself a finding.
    """
    events: list[dict[str, Any]] = []
    for block in re.split(r"\r?\n\r?\n", raw):
        data = [
            line[5:].strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data:
            continue
        payload = "\n".join(data)
        if payload == "[DONE]":
            events.append({"__done__": True})
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            events.append({"__raw__": payload})
    return events


def concat_stream(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fold chat-completion chunks back into one message.

    Streaming answers must be reassembled *before* they are graded -- a JSON
    verdict or a tool call arrives split across arbitrary token boundaries, and
    grading per-chunk is how a passing model looks like a failing one. Tool-call
    arguments concatenate by ``index``, matching the OpenAI delta protocol.
    """
    out: dict[str, Any] = {
        "content": "",
        "reasoning_content": "",
        "finish_reason": None,
        "tool_calls": [],
        "usage": None,
        "error": None,
        "done": False,
        "first_event": None,
    }
    tools: dict[int, dict[str, Any]] = {}
    for i, event in enumerate(events):
        if i == 0:
            out["first_event"] = event
        if event.get("__done__"):
            out["done"] = True
            continue
        if "error" in event:
            out["error"] = event["error"]
            continue
        if event.get("usage"):
            out["usage"] = event["usage"]
        for choice in event.get("choices") or []:
            if choice.get("finish_reason"):
                out["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                out["content"] += delta["content"]
            if delta.get("reasoning_content"):
                out["reasoning_content"] += delta["reasoning_content"]
            if delta.get("reasoning"):  # Switchyard reads either spelling
                out["reasoning_content"] += delta["reasoning"]
            for call in delta.get("tool_calls") or []:
                index = call.get("index", 0)
                slot = tools.setdefault(
                    index, {"id": None, "name": "", "arguments": ""}
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    out["tool_calls"] = [tools[i] for i in sorted(tools)]
    return out


def extract_json_object(text: str) -> Any:
    """Parse the first balanced top-level JSON object out of a completion.

    FreeToken's JSON mode already strips think residue and fences, but a judge
    answer that arrives with a stray prefix must still grade as valid -- the
    router itself is this forgiving.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response content")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object in response content")


#: ``switchyard-soak``'s terminal verdict (crates/switchyard-soak/src/lib.rs):
#: ``Soak PASS: requests=1234 error_rate=0.0000% p95_ms=812.5 summary=<path>``,
#: followed by one ``- <reason>`` line per failure reason.
_SOAK_VERDICT_RE = re.compile(r"^Soak (?P<verdict>PASS|FAIL):(?P<rest>.*)$", re.M)
_SOAK_FIELD_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_soak_report(text: str) -> dict[str, Any]:
    """Extract the verdict from ``switchyard-soak``'s stdout.

    Used when no ``summary.json`` is available. ``error_rate`` is normalized to a
    fraction (the printed form is a percentage). A run whose verdict line never
    appeared -- the soak died, or the output was truncated -- reports
    ``passed=None``, which the caller must not treat as a pass.
    """
    report: dict[str, Any] = {
        "passed": None,
        "requests": None,
        "errors": None,
        "error_rate": None,
        "p95_ms": None,
        "summary_path": None,
        "reasons": [],
    }
    match = _SOAK_VERDICT_RE.search(text)
    if match is None:
        return report
    report["passed"] = match.group("verdict") == "PASS"
    for key, value in _SOAK_FIELD_RE.findall(match.group("rest")):
        if key == "requests":
            report["requests"] = int(value)
        elif key == "error_rate":
            report["error_rate"] = float(value.rstrip("%")) / 100.0
        elif key == "p95_ms":
            report["p95_ms"] = None if value in ("-", "n/a") else float(value)
        elif key == "summary":
            report["summary_path"] = value
    if isinstance(report["requests"], int) and isinstance(report["error_rate"], float):
        report["errors"] = round(report["requests"] * report["error_rate"])
    for line in text[match.end() :].splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            report["reasons"].append(stripped[2:])
        elif stripped:
            break
    return report


def soak_summary_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    """Normalize a soak ``summary.json`` into the same verdict shape.

    Preferred over :func:`parse_soak_report` whenever ``--results-dir`` was used:
    the structured summary carries ``passed`` and ``failure_reasons`` directly
    (``crates/switchyard-soak/src/stats.rs``) and cannot be misread the way a
    prose line can.
    """
    passed = summary.get("passed")
    requests = summary.get("requests")
    errors = summary.get("failures")
    rate = summary.get("error_rate")
    if rate is None and isinstance(errors, int) and isinstance(requests, int) and requests:
        rate = errors / requests
    return {
        "passed": passed if isinstance(passed, bool) else None,
        "requests": requests,
        "errors": errors,
        "error_rate": rate,
        "p95_ms": summary.get("latency_p95_ms"),
        "summary_path": None,
        "reasons": list(summary.get("failure_reasons") or []),
    }


def render_routes_toml(
    *,
    freetoken_url: str,
    model: str,
    context_window: int = 131072,
    api_key_env: str | None = None,
) -> str:
    """The Switchyard config ``soak`` runs against; mirrors ``docs/switchyard.md``.

    Two targets over one client -- there is only one 16 GiB card, so both tiers
    are the same GPU process and this exercises Switchyard's routing, not model
    selection: ``lightning`` (thinking on, the capable tier) and
    ``lightning_fast`` (thinking off + ``force_nonempty_content``, the efficient
    tier and the classifier). Every key here is in
    ``switchyard-runner/src/{config,algorithm}.rs``, which uses
    ``deny_unknown_fields`` -- an invented key fails at startup, so validate any
    edit with ``switchyard-server --config <path> --dry-run``.
    """
    base = freetoken_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    # api_key_env names an environment variable, never the secret itself, and
    # switchyard-server refuses to start if that variable is unset. FreeToken needs
    # no bearer token locally, so the line is emitted only when one is asked for.
    return _ROUTES_TEMPLATE.format(
        base_url=base,
        model=model,
        api_key_line=f'api_key_env = "{api_key_env}"\n' if api_key_env else "",
        context_window=context_window,
    )


_ROUTES_TEMPLATE = """\
# Generated by scripts/switchyard_e2e.py -- see docs/switchyard.md.
schema_version = 1

[llm_clients.freetoken]
format = "openai_chat"
base_url = "{base_url}"
{api_key_line}max_retries = 2

# Capable tier: thinking on (the checkpoint's default).
[targets.lightning]
id = "{model}"
llm_client = "freetoken"

[targets.lightning.extra_body.chat_template_kwargs]
enable_thinking = true

# Efficient tier and classifier: thinking off, and answer with the reasoning
# text rather than an empty message if the turn produces only reasoning.
# The id MUST differ from the capable target's: Switchyard keeps one target per
# (llm_client, model id) pair and drops the other. FreeToken echoes `model` back
# without validating it, so any distinct string reaches the same process.
[targets.lightning_fast]
id = "{model}-fast"
llm_client = "freetoken"

[targets.lightning_fast.extra_body.chat_template_kwargs]
enable_thinking = false
force_nonempty_content = true

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "lightning"
context_window = {context_window}
tool_calling = true
reasoning = true

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
picker = "efficient_first"
capable_target = "lightning"
efficient_target = "lightning_fast"
confidence_threshold = 0.6
recent_turn_window = 28
context_window = {context_window}
tool_calling = true
reasoning = true

[routes.stage.classifier]
target = "lightning_fast"
base_threshold = 0.6
classify_trigger = "user_turn"
response_format_type = "json_schema"
max_output_tokens = 512
"""


# --------------------------------------------------------------------------- #
# Contract checks
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Checks:
    results: list[Result] = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append(Result(name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}{(': ' + detail) if detail else ''}", flush=True)
        return bool(ok)

    def failed(self) -> list[Result]:
        return [r for r in self.results if not r.ok]


class Client:
    """Minimal chat client. Raw ``httpx`` rather than the ``openai`` SDK: the
    checks are about wire details the SDK normalizes away (unknown headers, the
    first SSE event, a 400 body)."""

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        if httpx is None:  # pragma: no cover
            raise SystemExit("httpx is required: run through `uv run`")
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.http.close()

    def models(self) -> dict[str, Any]:
        r = self.http.get(f"{self.base}/v1/models")
        r.raise_for_status()
        return r.json()

    def chat(self, body: dict[str, Any], headers: dict[str, str] | None = None):
        return self.http.post(
            f"{self.base}/v1/chat/completions", json=body, headers=headers or {}
        )

    def chat_stream(self, body: dict[str, Any], headers: dict[str, str] | None = None):
        """Return ``(status, response_headers, raw_body)`` for a streaming call.

        An overflow that the frontend preflight catches answers with a plain 400
        JSON body *before* the stream opens; one caught later rides as the first
        SSE event. Both are contract-conformant, so the raw body is returned
        unparsed and the caller decides.
        """
        body = {**body, "stream": True}
        with self.http.stream(
            "POST", f"{self.base}/v1/chat/completions", json=body, headers=headers or {}
        ) as r:
            chunks = [c for c in r.iter_text()]
            return r.status_code, dict(r.headers), "".join(chunks)


def _served_max_len(client: Client, checks: Checks) -> int:
    data = client.models()
    card = (data.get("data") or [{}])[0]
    ctx = card.get("max_model_len") or card.get("context_length") or 0
    checks.record(
        "models: served id + context window",
        bool(card.get("id")) and int(ctx) > 0,
        f"id={card.get('id')!r} context_length={ctx}",
    )
    return int(ctx)


def check_max_completion_tokens(client: Client, model: str, checks: Checks) -> None:
    """Switchyard sends ``max_completion_tokens`` and never ``max_tokens``."""
    r = client.chat(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_completion_tokens": 16,
            "temperature": 0.0,
        }
    )
    ok = r.status_code == 200
    detail = f"HTTP {r.status_code}"
    if ok:
        body = r.json()
        completion = body["usage"]["completion_tokens"]
        ok = completion <= 16
        detail = f"completion_tokens={completion} (<= 16)"
    checks.record("max_completion_tokens alias honored", ok, detail)


def check_cached_tokens(client: Client, model: str, checks: Checks) -> None:
    """``usage.prompt_tokens_details.cached_tokens`` must grow on a repeat prompt
    (requires ``--enable-cache-report``); Switchyard's prefix-reuse scenario
    grades on it."""
    prompt = "You are a helpful assistant.\n" + ("The quick brown fox. " * 400)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Say ok."},
        ],
        "max_completion_tokens": 8,
        "temperature": 0.0,
    }
    first = client.chat(body).json()
    second = client.chat(body).json()
    cached = ((second.get("usage") or {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens", 0
    )
    first_cached = ((first.get("usage") or {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens", 0
    )
    checks.record(
        "prefix cache reported (cached_tokens > 0 on repeat)",
        isinstance(cached, int) and cached > 0,
        f"first={first_cached} second={cached}",
    )


def check_json_schema_verdict(client: Client, model: str, checks: Checks) -> None:
    """A ``json_schema`` response_format must come back as schema-valid JSON in
    ``content`` -- this is the judge/classifier contract of a stage route."""
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You grade whether a cheap model's answer needs escalation to a "
                    "stronger model. Answer with the verdict object only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Question: prove that every finite group of order 255 is cyclic.\n"
                    "Cheap answer: 'yes it is'. Should this escalate?"
                ),
            },
        ],
        "response_format": ESCALATION_RESPONSE_FORMAT,
        "max_completion_tokens": 256,
        "temperature": 0.0,
    }
    for stream in (False, True):
        label = "stream" if stream else "non-stream"
        if stream:
            status, _, raw = client.chat_stream(body)
            merged = concat_stream(sse_events(raw))
            content, ok_http = merged["content"], status == 200
        else:
            r = client.chat(body)
            ok_http = r.status_code == 200
            content = (
                r.json()["choices"][0]["message"].get("content") or "" if ok_http else ""
            )
        if not ok_http:
            checks.record(f"json_schema verdict ({label})", False, "non-200")
            continue
        try:
            value = extract_json_object(content)
        except (ValueError, json.JSONDecodeError) as exc:
            checks.record(f"json_schema verdict ({label})", False, f"unparseable: {exc}")
            continue
        errors = validate_json_schema(value, ESCALATION_VERDICT_SCHEMA)
        checks.record(
            f"json_schema verdict ({label})",
            not errors,
            "; ".join(errors) if errors else f"escalate={value.get('escalate')}",
        )


def check_context_overflow(
    client: Client, model: str, served_max: int, checks: Checks
) -> None:
    """Overflow must be HTTP 400 with ``error.code == context_length_exceeded``
    non-stream, and the *first* thing a streaming client sees (either the same
    400 -- the frontend preflight rejects before the stream opens -- or an error
    SSE event ahead of any role chunk). Anything else makes a Switchyard route
    fall through instead of retargeting."""
    # One repetition of "token " is ONE token for every tokenizer this targets, so
    # the count is the token count -- halving it (as an earlier version did) builds a
    # prompt that fits and the server correctly answers 200, which reads as a
    # contract failure that is really a test bug.
    filler = "token " * (served_max + 4096)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": filler}],
        "max_completion_tokens": 16,
    }
    r = client.chat(body)
    err = (r.json().get("error") or {}) if r.headers.get("content-type", "").startswith(
        "application/json"
    ) else {}
    checks.record(
        "context overflow: non-stream 400 context_length_exceeded",
        r.status_code == 400 and err.get("code") == "context_length_exceeded",
        f"HTTP {r.status_code} code={err.get('code')!r}",
    )

    status, headers, raw = client.chat_stream(body)
    if status == 400:
        try:
            err = json.loads(raw).get("error") or {}
        except json.JSONDecodeError:
            err = {}
        checks.record(
            "context overflow: streaming rejected before the stream opens",
            err.get("code") == "context_length_exceeded",
            f"HTTP 400 code={err.get('code')!r} (frontend preflight)",
        )
        return
    events = sse_events(raw)
    first = events[0] if events else {}
    code = (first.get("error") or {}).get("code")
    checks.record(
        "context overflow: error is the FIRST SSE event",
        code == "context_length_exceeded",
        f"first event={json.dumps(first)[:160]}",
    )


def check_session_header(client: Client, model: str, checks: Checks) -> None:
    """``x-switchyard-session-id`` must bind the turn to a KV lease and be echoed
    as ``X-FreeToken-Session-Id``, stably across turns of one conversation."""
    session = f"e2e-{int(time.time())}"
    headers = {"x-switchyard-session-id": session}
    seen: list[str | None] = []
    for turn in ("Remember the number 41.", "What number did I give you?"):
        r = client.chat(
            {
                "model": model,
                "messages": [{"role": "user", "content": turn}],
                "max_completion_tokens": 32,
                "temperature": 0.0,
            },
            headers=headers,
        )
        seen.append(r.headers.get("X-FreeToken-Session-Id"))
    ok = all(s for s in seen) and seen[0] == seen[1]
    checks.record(
        "x-switchyard-session-id -> stable X-FreeToken-Session-Id",
        ok,
        f"turn1={seen[0]!r} turn2={seen[1]!r}",
    )

    status, stream_headers, _ = client.chat_stream(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_completion_tokens": 8,
        },
        headers=headers,
    )
    echoed = stream_headers.get("x-freetoken-session-id")
    checks.record(
        "session id echoed on the streaming response too",
        echoed == seen[0] and echoed is not None,
        f"streamed={echoed!r}",
    )


#: The tool-call burst Switchyard's soak sends: a small catalog, history that
#: re-sends the previous call, and a question only the tool can answer.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def check_tool_call(client: Client, model: str, checks: Checks) -> None:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Use a tool when one applies."},
            {"role": "user", "content": "List the files under /var/log, then stop."},
        ],
        "tools": _TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_completion_tokens": 512,
        "temperature": 0.0,
    }
    r = client.chat(body)
    ok = r.status_code == 200
    detail = f"HTTP {r.status_code}"
    if ok:
        choice = r.json()["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        args_ok = False
        if calls:
            try:
                json.loads(calls[0]["function"]["arguments"])
                args_ok = True
            except (json.JSONDecodeError, KeyError, TypeError):
                args_ok = False
        ok = choice.get("finish_reason") == "tool_calls" and args_ok
        detail = (
            f"finish_reason={choice.get('finish_reason')!r} "
            f"calls={[c['function']['name'] for c in calls]} args_parse={args_ok}"
        )
    checks.record("tool-call burst -> finish_reason=tool_calls", ok, detail)

    status, _, raw = client.chat_stream(body)
    merged = concat_stream(sse_events(raw))
    args_ok = False
    if merged["tool_calls"]:
        try:
            json.loads(merged["tool_calls"][0]["arguments"])
            args_ok = True
        except json.JSONDecodeError:
            args_ok = False
    checks.record(
        "tool call reassembles from SSE deltas",
        status == 200
        and merged["finish_reason"] == "tool_calls"
        and args_ok
        and merged["done"],
        f"finish_reason={merged['finish_reason']!r} "
        f"names={[c['name'] for c in merged['tool_calls']]} args_parse={args_ok}",
    )


def check_reasoning_fields(client: Client, model: str, checks: Checks) -> None:
    """Switchyard reads ``reasoning_content`` (or ``reasoning``) and
    ``completion_tokens_details.reasoning_tokens``."""
    r = client.chat(
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is 17 * 23? Think first."}],
            "max_completion_tokens": 512,
            "temperature": 0.0,
        }
    )
    ok = r.status_code == 200
    detail = f"HTTP {r.status_code}"
    if ok:
        body = r.json()
        message = body["choices"][0]["message"]
        reasoning_tokens = (
            (body.get("usage") or {}).get("completion_tokens_details") or {}
        ).get("reasoning_tokens")
        has_content = bool(message.get("content"))
        ok = has_content and isinstance(reasoning_tokens, int)
        detail = (
            f"content={len(message.get('content') or '')}ch "
            f"reasoning={len(message.get('reasoning_content') or '')}ch "
            f"reasoning_tokens={reasoning_tokens}"
        )
    checks.record("reasoning_content + reasoning_tokens reported", ok, detail)


def cmd_contract(args: argparse.Namespace) -> int:
    client = Client(args.base_url, timeout=args.timeout)
    checks = Checks()
    try:
        served_max = _served_max_len(client, checks)
        check_max_completion_tokens(client, args.model, checks)
        check_reasoning_fields(client, args.model, checks)
        check_cached_tokens(client, args.model, checks)
        check_json_schema_verdict(client, args.model, checks)
        check_session_header(client, args.model, checks)
        check_tool_call(client, args.model, checks)
        if served_max:
            check_context_overflow(client, args.model, served_max, checks)
        else:
            checks.record("context overflow", False, "no context window advertised")
    finally:
        client.close()
    failed = checks.failed()
    print(
        f"\n{len(checks.results) - len(failed)}/{len(checks.results)} contract checks passed"
    )
    for r in failed:
        print(f"  FAILED: {r.name}: {r.detail}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Soak
# --------------------------------------------------------------------------- #


def _binary(name: str, switchyard_dir: str) -> str:
    path = os.path.join(switchyard_dir, "target", "release", name)
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(
        f"{name} not found. Build it first:\n"
        f"  cd {switchyard_dir} && cargo build --release "
        f"-p switchyard-server -p switchyard-soak"
    )


def _wait_healthy(url: str, timeout: float, proc: subprocess.Popen) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            r = httpx.get(f"{url.rstrip('/')}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 -- still starting
            pass
        time.sleep(0.5)
    return False


def cmd_soak(args: argparse.Namespace) -> int:
    server_bin = _binary("switchyard-server", args.switchyard_dir)
    soak_bin = _binary("switchyard-soak", args.switchyard_dir)

    workdir = args.workdir or tempfile.mkdtemp(prefix="switchyard-e2e-")
    os.makedirs(workdir, exist_ok=True)
    config_path = args.config or os.path.join(workdir, "routes.toml")
    if args.config is None:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(render_routes_toml(freetoken_url=args.base_url, model=args.model))
        print(f"wrote {config_path}")

    port = args.router_port
    router_url = f"http://127.0.0.1:{port}"
    log_path = os.path.join(workdir, "switchyard-server.log")
    log = open(log_path, "w", encoding="utf-8")
    print(f"starting {server_bin} --config {config_path} --port {port}")
    proc = subprocess.Popen(
        [server_bin, "--config", config_path, "--host", "127.0.0.1", "--port", str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    rc = 1
    try:
        if not _wait_healthy(router_url, args.health_timeout, proc):
            print(f"switchyard-server never became healthy; see {log_path}")
            return 1
        print(f"switchyard-server healthy at {router_url}")
        for route in args.route or list(DEFAULT_ROUTES):
            results_dir = os.path.join(workdir, f"results-{route.replace('/', '_')}")
            cmd = [
                soak_bin,
                "--base-url",
                router_url,
                "--model",
                route,
                "--duration",
                args.duration,
                "--concurrency",
                str(args.concurrency),
                "--max-output-tokens",
                str(args.max_output_tokens),
                "--prompt-bytes",
                str(args.prompt_bytes),
                "--context-window-tokens",
                str(args.context_window_tokens),
                "--max-error-rate",
                str(args.max_error_rate),
                "--request-timeout",
                str(args.request_timeout),
                "--results-dir",
                results_dir,
            ]
            if args.scenario_set:
                cmd += ["--scenario-set", args.scenario_set]
            else:
                for scenario in args.scenario or SOAK_SCENARIOS:
                    cmd += ["--scenario", scenario]
            print("\n$ " + " ".join(cmd), flush=True)
            run = subprocess.run(cmd, capture_output=True, text=True)
            sys.stdout.write(run.stdout)
            sys.stderr.write(run.stderr)
            verdict = _verdict(results_dir, run.stdout + run.stderr)
            passed = verdict["passed"] if verdict["passed"] is not None else run.returncode == 0
            print(
                f"\n[{'PASS' if passed and run.returncode == 0 else 'FAIL'}] soak {route}: "
                f"exit={run.returncode} requests={verdict['requests']} "
                f"errors={verdict['errors']} error_rate={verdict['error_rate']}"
            )
            if run.returncode != 0 or not passed:
                return run.returncode or 1
        rc = 0
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        print(f"switchyard-server log: {log_path}")
    return rc


def _verdict(results_dir: str, output: str) -> dict[str, Any]:
    """Prefer the structured summary the soak writes; fall back to its stdout."""
    for name in ("summary.json", "results.json", "report.json"):
        path = os.path.join(results_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return soak_summary_verdict(json.load(f))
            except (json.JSONDecodeError, OSError):
                break
    return parse_soak_report(output)


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

AGENTS_TEXT = """\
Manual agent smoke tests (run each in its own terminal, one at a time).

Claude Code through Switchyard:
  export ANTHROPIC_BASE_URL={router}
  export ANTHROPIC_AUTH_TOKEN=sk-switchyard-local
  export ANTHROPIC_MODEL=switchyard/passthrough
  export ANTHROPIC_SMALL_FAST_MODEL=switchyard/passthrough
  export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
  claude

Codex through Switchyard (OpenAI-compatible):
  export OPENAI_BASE_URL={router}/v1
  export OPENAI_API_KEY=sk-switchyard-local
  codex --config model_provider=openai --config model=switchyard/passthrough

Straight at FreeToken (bypassing the router, to isolate a failure):
  export OPENAI_BASE_URL={freetoken}/v1
  export OPENAI_API_KEY=sk-freetoken-local
  codex --config model_provider=openai --config model={model}

What to watch for:
  - the FreeToken log shows one session id per agent conversation
    (X-FreeToken-Session-Id derived from x-claude-code-session-id / x-codex-session-id);
  - cached_tokens climbs turn over turn inside a conversation;
  - no "session is busy" surfaces to the agent (the server retries unbound once);
  - tool calls round-trip: the agent's edit/read tools execute rather than being
    echoed as prose.
"""


def cmd_agents(args: argparse.Namespace) -> int:
    print(
        AGENTS_TEXT.format(
            router=f"http://127.0.0.1:{args.router_port}",
            freetoken=args.base_url,
            model=args.model,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="switchyard_e2e",
        description="FreeToken <-> Switchyard end-to-end checks",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("FREETOKEN_URL", DEFAULT_FREETOKEN_URL),
        help="FreeToken server base URL (no /v1 suffix)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("FREETOKEN_MODEL", DEFAULT_MODEL),
        help="served model id (--served-model-name)",
    )
    p.add_argument("--router-port", type=int, default=4000)
    p.add_argument("--switchyard-dir", default=SWITCHYARD_DIR)

    # The same four connection options again, accepted *after* the subcommand:
    # `switchyard_e2e.sh contract --base-url ...` is the spelling docs/switchyard.md
    # and everyone's muscle memory use. argparse.SUPPRESS keeps the top-level value
    # (or its default) whenever the subcommand does not set one.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default=argparse.SUPPRESS)
    common.add_argument("--model", default=argparse.SUPPRESS)
    common.add_argument("--router-port", type=int, default=argparse.SUPPRESS)
    common.add_argument("--switchyard-dir", default=argparse.SUPPRESS)

    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "contract",
        parents=[common],
        help="wire-contract checks against FreeToken",
    )
    c.add_argument("--timeout", type=float, default=300.0)
    c.set_defaults(func=cmd_contract)

    s = sub.add_parser(
        "soak", parents=[common], help="run switchyard-server + switchyard-soak"
    )
    s.add_argument("--config", default=None, help="existing routes.toml (else generated)")
    s.add_argument("--workdir", default=None)
    s.add_argument("--duration", default="20m")
    s.add_argument("--concurrency", type=int, default=16)
    s.add_argument("--max-output-tokens", type=int, default=256)
    s.add_argument("--prompt-bytes", type=int, default=16384)
    s.add_argument("--context-window-tokens", type=int, default=131072)
    s.add_argument("--max-error-rate", type=float, default=0.0)
    # The soak's own client timeout (seconds), not a server promise. Its default of
    # 120 s is shorter than a 118K-token prefill queued behind others on one 16 GiB
    # card, so the long-context / context-overflow scenarios need it raised or they
    # report client timeouts as upstream errors.
    s.add_argument("--request-timeout", type=float, default=600.0)
    s.add_argument("--health-timeout", type=float, default=60.0)
    s.add_argument("--scenario", action="append", default=None)
    s.add_argument("--scenario-set", default=None)
    s.add_argument(
        "--route",
        action="append",
        default=None,
        help="route id to soak; repeat (default: passthrough then stage)",
    )
    s.set_defaults(func=cmd_soak)

    a = sub.add_parser(
        "agents", parents=[common], help="print the manual agent smoke-test env"
    )
    a.set_defaults(func=cmd_agents)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
