"""Protocol-neutral generation core.

The narrow interface every wire protocol (OpenAI chat, Anthropic messages, OpenAI
Responses) sits on. A protocol adapter converts its request into a ``GenSpec``
(messages + sampling + tools), submits it, and formats the resulting semantic
events (``GenEvent``) / ``GenResult`` into its own wire shape. This mirrors vLLM's
split of per-request ``to_sampling_params`` + shared preprocess + a neutral
``SamplingParams`` — no wire request type reaches the engine path.

This module is imported by ``openai_api`` / ``anthropic_api`` / ``responses_api``;
it depends on none of them.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from . import request_ring
from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.tokenize import resolve_thinking_mode

try:
    # Chat templates render through jinja2 (a transformers dependency): a TemplateError means
    # the template rejected the specific conversation (bad role ordering, an unmatched
    # tool_result, an explicit raise_exception) — an input-driven, client-classifiable failure.
    from jinja2 import TemplateError as _TemplateError
except Exception:  # pragma: no cover — jinja2 always ships with transformers
    _TemplateError = ()

from .function_call_parser import FunctionCallParser, TOOLS_TAG_LIST, ToolCallItem
from .reasoning_parser import (
    DSV4_SPECIAL_TOKENS,
    ReasoningParser,
    build_reasoning_parser,
    strip_special_tokens,
)


class GenerationError(Exception):
    """A request failed before producing output (surfaced via ``UserReply.error`` — e.g. a
    chat template the tokenizer cannot render, or a prompt that exceeds the KV budget). Each
    adapter turns this into its own wire-level error instead of hanging on a reply that never
    arrives. ``code`` carries the stable class (``UserReply.error_code``) when the failure has
    one, so an adapter can emit it as OpenAI's error ``code`` and clients need not parse prose."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# Protocol-neutral generation events.
#
# generate_events() / generate_full() yield these instead of OpenAI wire chunks.
# The OpenAI, Anthropic, and Responses streamers are thin formatters over them,
# so generation/parsing logic lives in exactly one place and no adapter has to
# re-parse a serialized OpenAI stream.
# --------------------------------------------------------------------------- #
@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ContentDelta:
    text: str


@dataclass
class ToolCallStart:
    """A tool call opened mid-stream: its name is known, arguments follow as
    ToolCallArgsDelta fragments, and the matching ToolCallsDelta closes it with the
    authoritative final arguments. tool_index is the output ordinal (0, 1, ...).
    args_prefix_stable tells adapters whether the fragments are safe to forward to
    clients that concatenate them (else: send full args once at close)."""

    tool_index: int
    name: str | None
    args_prefix_stable: bool = True


@dataclass
class ToolCallArgsDelta:
    """An incremental fragment of the open call's arguments JSON. Fragments always
    concatenate to a prefix of the final arguments (detectors skip non-prefix
    diffs), so adapters may stream them and top up the remainder at close."""

    tool_index: int
    fragment: str


@dataclass
class ToolCallsDelta:
    """Complete call(s). On the streaming path this closes a ToolCallStart-opened
    call (same tool_index) carrying the final arguments; on the buffered fallback
    path it arrives standalone, without a preceding start/args events."""

    calls: list[ToolCallItem]


@dataclass
class GenDone:
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    matched_stop: str | None = None
    cached_tokens: int = 0
    #: Completion tokens produced while the reasoning parser was inside a reasoning
    #: block (OpenAI's ``usage.completion_tokens_details.reasoning_tokens``). 0 when
    #: no reasoning parser is configured.
    reasoning_tokens: int = 0


GenEvent = ReasoningDelta | ContentDelta | ToolCallStart | ToolCallArgsDelta | ToolCallsDelta | GenDone


@dataclass
class GenResult:
    reasoning: str
    content: str
    tool_calls: list[ToolCallItem]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    matched_stop: str | None = None
    cached_tokens: int = 0
    #: See ``GenDone.reasoning_tokens``.
    reasoning_tokens: int = 0


@dataclass
class GenSpec:
    """Protocol-neutral 'what to generate'. Each wire protocol converts its request
    into a GenSpec (like vLLM's to_sampling_params + _preprocess_chat) and the
    primitive consumes ONLY this — no wire request type reaches the engine path."""

    messages: list[dict[str, Any]]                       # normalized, template-ready
    sampling_params: SamplingParams
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    template_tools: list[dict[str, Any]] | None = None   # tools the model sees (TokenizeMsg.tools)
    parser_tools: list[dict[str, Any]] | None = None     # tools for FunctionCallParser; None disables parsing
    session_id: str | None = None
    session_ttl_seconds: float | None = None
    session_reclaimable: bool = False
    #: Answer with the reasoning text when the turn produced reasoning but no
    #: visible content and no tool call. Set by the adapter from the request's
    #: ``chat_template_kwargs.force_nonempty_content`` / the server flag; a
    #: thinking-off turn defaults it on, because a model whose think block was
    #: pre-closed by the template and that still writes only a thought would
    #: otherwise answer with an empty message.
    force_nonempty_content: bool = False

    @property
    def parse_tools(self) -> bool:
        return self.parser_tools is not None


# --------------------------------------------------------------------------- #
# Wire-neutral builders (shared by every protocol's request->GenSpec converter).
# --------------------------------------------------------------------------- #
# Default max output (decode) tokens when a request omits one. Overridable per server via
# --max-output-tokens (the Responses adapter passes that through); clamped to the remaining
# context by the scheduler regardless.
DEFAULT_MAX_OUTPUT_TOKENS = 32768


def resolve_sampling(
    *,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    max_tokens: int | None,
    ignore_eos: bool,
    model_sampling: dict[str, Any],
    stop: str | list[str] | None = None,
) -> SamplingParams:
    """Map a protocol's sampling fields onto the engine's neutral SamplingParams,
    filling unspecified fields from the checkpoint's recommended defaults."""

    def pick(value, key, framework):
        return value if value is not None else model_sampling.get(key, framework)

    stop_list = [stop] if isinstance(stop, str) else list(stop or [])
    # `is not None`, not truthiness: an explicit max_tokens=0 must not read as "unset" and
    # silently become the 32k default. The engine cannot serve a zero-token budget either
    # (the request would never become decodable, so the client would wait forever), so a
    # non-positive value is a client error.
    if max_tokens is not None and max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    return SamplingParams(
        ignore_eos=ignore_eos,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens,
        temperature=pick(temperature, "temperature", 0.0),
        top_k=pick(top_k, "top_k", -1),
        top_p=pick(top_p, "top_p", 1.0),
        stop_strs=[s for s in stop_list if s],  # drop empty strings (would match everything)
    )


def render_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI-shaped message dicts for the chat template: flatten text
    content parts to a string and decode tool-call arguments from JSON. Raises
    ValueError on a non-text content part (text-only server). Shared by all adapters."""
    return [_render_message(m) for m in messages]


def _render_message(message: dict[str, Any]) -> dict[str, Any]:
    m = dict(message)
    content = m.get("content")
    if isinstance(content, list):
        m["content"] = _flatten_text_parts(content)
    # Templates read different reasoning keys (reasoning_content: most; reasoning:
    # gemma4; thinking: gpt-oss) — accept any, emit both.
    reasoning = m.get("reasoning_content") or m.get("reasoning") or m.get("thinking")
    if reasoning:
        m.setdefault("reasoning_content", reasoning)
        # gpt-oss's template raises when a tool-call assistant turn carries BOTH content and
        # thinking -- it renders one or the other. Visible text wins: dropping it would lose
        # what the user saw, and every other family reads `content` too.
        if not (m.get("tool_calls") and m.get("content")):
            m.setdefault("thinking", reasoning)
        if m.get("role") == "assistant" and m.get("content") is None:
            # gpt-oss templates concatenate message.content unconditionally.
            m["content"] = ""
    tool_calls = m.get("tool_calls")
    if tool_calls:
        rendered = []
        for tc in tool_calls:
            tc = dict(tc)
            fn = dict(tc.get("function") or {})
            arguments = fn.get("arguments")
            if isinstance(arguments, str):
                try:
                    fn["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            tc["function"] = fn
            rendered.append(tc)
        m["tool_calls"] = rendered
    return m


def _flatten_text_parts(parts: list[Any]) -> str:
    texts: list[str] = []
    for part in parts:
        ptype = part.get("type") if isinstance(part, dict) else None
        if ptype == "text":
            texts.append((part.get("text") if isinstance(part, dict) else None) or "")
        else:
            raise ValueError(f"Unsupported content part type for text-only server: {ptype}")
    return "".join(texts)


def split_tool_lists(
    all_tool_dicts: list[dict[str, Any]] | None, selected_name: str | None = None
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    """(template_tools, parser_tools): parser sees all tools; the template sees only
    the selected one when tool_choice forces a specific function. Shared by adapters."""
    if not all_tool_dicts:
        return None, None
    if selected_name:
        template = [t for t in all_tool_dicts if (t.get("function") or {}).get("name") == selected_name]
    else:
        template = all_tool_dicts
    return template, all_tool_dicts


# --------------------------------------------------------------------------- #
# The primitive: submit + generate (consume a GenSpec, drive the engine waist).
# --------------------------------------------------------------------------- #
async def submit_generation(spec: GenSpec, state: Any) -> int:
    """Enqueue one generation from a GenSpec; return its uid. Every protocol adapter
    calls this — it takes the neutral spec, not a wire request type."""
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=spec.messages,
            sampling_params=spec.sampling_params,
            chat_template_kwargs=spec.chat_template_kwargs,
            tools=spec.template_tools,
            session_id=spec.session_id,
            session_ttl_seconds=spec.session_ttl_seconds,
            session_reclaimable=spec.session_reclaimable,
        )
    )
    return uid


async def count_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
    state: Any,
) -> int:
    """Token count of an already-converted (messages, tools, chat_template_kwargs) prompt,
    using the frontend's own tokenizer (``state.frontend_tokenizer()``) so the count equals the
    ``usage.input_tokens`` a real generation of the same prompt would report. The neutral
    counterpart to ``submit_generation`` — any protocol's count endpoint converts to this triple
    and calls it. The caller validates the prompt first (non-empty, has tokenizable content).

    Failure classification mirrors ``/v1/messages``: a chat template that rejects the specific
    conversation (bad role ordering, an unmatched tool_result, an explicit raise_exception) is
    re-raised as ``GenerationError`` — an input-driven, client-classifiable failure, the same
    class the generation path maps to a 400. A tokenizer *initialization* failure (missing
    template, load error) propagates as its original exception, a server fault. Load + tokenize
    run in a worker thread so the event loop is never blocked."""
    msg = TokenizeMsg(
        uid=0,
        text=messages,
        sampling_params=SamplingParams(),
        chat_template_kwargs=chat_template_kwargs,
        tools=tools,
    )
    manager = await asyncio.to_thread(state.frontend_tokenizer)  # init failure -> server fault
    try:
        input_ids = (await asyncio.to_thread(manager.tokenize, [msg]))[0]
    except _TemplateError as exc:
        raise GenerationError(str(exc)) from exc
    return int(input_ids.numel())


CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
"""OpenAI's stable error class for a prompt that does not fit the context window.
Switchyard routes on this code (``error.code``): without it a route falls through
instead of retargeting, so every overflow path — preflight, scheduler, stream and
non-stream — must carry it."""


def context_overflow_message(prompt_tokens: int, max_seq_len: int) -> str:
    """The single phrasing every overflow surface uses. Contains BOTH "maximum
    context length" (what OpenAI clients and Switchyard match on) and "prompt is
    too long" (what Claude Code / OpenClaw match on), so no client has to read a
    code it does not know about."""
    return (
        f"This model's maximum context length is {max_seq_len} tokens, but the "
        f"prompt is too long: {prompt_tokens} tokens (prompt + generation budget). "
        f"Shorten the prompt or increase the KV cache budget."
    )


def context_preflight_enabled(state: Any) -> bool:
    """Whether the preflight tokenizes (and length-checks) as well as renders.
    Default on; ``FREETOKEN_CONTEXT_PREFLIGHT=0`` (or a falsey
    ``config.context_preflight``) drops back to render-only validation, which is
    what an operator wants if the extra ~1.2 µs/token encode is not worth paying
    on every request."""
    env = os.environ.get("FREETOKEN_CONTEXT_PREFLIGHT")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "off", "no")
    return bool(getattr(getattr(state, "config", None), "context_preflight", True))


def served_max_seq_len(state: Any) -> int | None:
    """The context window in force, or None when it cannot be read (never fail a
    request over a missing limit — the scheduler still enforces the real one)."""
    try:
        value = int(state.config.max_seq_len)
    except Exception:  # noqa: BLE001
        return None
    return value if value > 0 else None


async def preflight_error(spec: GenSpec, state: Any) -> GenerationError | None:
    """Validate ``spec``'s prompt frontend-side, returning the failure an adapter
    should surface as an HTTP 400 *before* committing an SSE stream — once headers
    go out, a rejection can only ride in-stream, where some agents show nothing but
    "empty response".

    Two checks: the chat template must render, and (unless preflight is disabled)
    the rendered prompt must fit ``max_seq_len``. The overflow answer is the same
    ``context_length_exceeded`` the scheduler would produce, just without paying a
    queue slot for it. The worker still renders and encodes authoritatively.
    Best-effort: a state without a frontend tokenizer, or one that fails to
    *initialize*, skips validation rather than blocking the generation path.
    """
    return await _preflight(
        TokenizeMsg(
            uid=0,
            text=spec.messages,
            sampling_params=SamplingParams(),
            chat_template_kwargs=spec.chat_template_kwargs,
            tools=spec.template_tools,
        ),
        state,
    )


async def preflight_text_error(text: str, state: Any) -> GenerationError | None:
    """``preflight_error`` for a raw-text prompt (/v1/completions), which has no
    chat template to render — only the length check applies."""
    return await _preflight(
        TokenizeMsg(uid=0, text=text, sampling_params=SamplingParams()), state
    )


async def _preflight(msg: TokenizeMsg, state: Any) -> GenerationError | None:
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return None
    try:
        manager = await asyncio.to_thread(build)
    except Exception:  # noqa: BLE001 -- server fault, not this request's problem
        return None
    if not context_preflight_enabled(state):
        try:
            await asyncio.to_thread(manager.render_prompt, msg)
        except Exception as exc:  # noqa: BLE001 -- mirror the worker's classification
            return GenerationError(f"could not encode request: {exc}")
        return None
    try:
        input_ids = (await asyncio.to_thread(manager.tokenize, [msg]))[0]
    except Exception as exc:  # noqa: BLE001 -- mirror the worker's classification
        return GenerationError(f"could not encode request: {exc}")
    limit = served_max_seq_len(state)
    prompt_tokens = int(input_ids.numel())
    # `>=`, not `>`: the scheduler drops a request whose prompt leaves zero decode
    # budget (max_seq_len - input_len <= 0), so the preflight must reject exactly
    # the same set or a request would pass here and fail there.
    if limit is not None and prompt_tokens >= limit:
        return GenerationError(
            context_overflow_message(prompt_tokens, limit), CONTEXT_LENGTH_EXCEEDED
        )
    return None


def _ack_is_reasoning(was_reasoning: bool, reasoning_delta: str) -> bool:
    """Whether one ack's output counts toward ``reasoning_tokens``: the parser was
    already inside a reasoning block when the chunk arrived, or the chunk itself
    opened one (an explicit ``<think>``). The chunk that carries ``</think>`` still
    counts — its tokens were generated inside the block; the next one does not."""
    return bool(was_reasoning or reasoning_delta)


def _make_reasoning_parser(spec: GenSpec, state: Any) -> ReasoningParser | None:
    """Build a reasoning parser for this generation, or None if the server has no
    reasoning parser configured. ``force_reasoning`` matches the encode-side
    thinking mode so chat-mode content is never mislabeled as reasoning."""
    parser_name = getattr(state.config, "reasoning_parser", None)
    if parser_name in ("qwen3", "nemotron_v3"):
        # The qwen3 chat template opens an implicit <think> (thinking on) unless
        # enable_thinking is explicitly false, so the model emits only the closing
        # </think>. Mirror that default here, else the chain-of-thought leaks into content.
        # Nemotron-3.x is the same contract: `enable_thinking` defaults true and ends the
        # generation prompt with "<think>\n"; false pre-closes it with "<think></think>",
        # so the completion is pure content.
        force_reasoning = (spec.chat_template_kwargs or {}).get("enable_thinking") is not False
    elif parser_name == "glm":
        # GLM's template honors enable_thinking (default on) even with tools; the
        # generic fallback would force thinking and mislabel disabled output as reasoning.
        force_reasoning = (spec.chat_template_kwargs or {}).get("enable_thinking") is not False
    elif parser_name == "gemma4":
        # Gemma4 defaults thinking off even when tools are present: its template injects an
        # empty thought channel before generation. Do not let Codex tool definitions make all
        # visible text look like hidden reasoning.
        ctk = spec.chat_template_kwargs or {}
        force_reasoning = (
            ctk.get("thinking_mode") == "thinking"
            or bool(ctk.get("enable_thinking"))
            or bool(ctk.get("thinking"))
        )
    elif parser_name == "minimax_m3":
        # M3's template pre-opens <mm:think> only in thinking_mode "enabled" (the
        # model then emits just the closing tag); "adaptive" (default) leaves the
        # model to open the tag itself and "disabled" pre-closes it.
        force_reasoning = (spec.chat_template_kwargs or {}).get("thinking_mode") == "enabled"
    else:
        force_reasoning = (
            resolve_thinking_mode(spec.chat_template_kwargs, spec.template_tools) == "thinking"
        )
    return build_reasoning_parser(state.config, force_reasoning)


def _split_reasoning(text: str, spec: GenSpec, state: Any) -> tuple[str, str]:
    """Return ``(reasoning, content)``. A no-op (``("", text)``) when no reasoning
    parser is configured, preserving the original behavior for other models."""
    parser = _make_reasoning_parser(spec, state)
    if parser is None:
        return "", text
    return parser.parse_non_stream(text)


def _leaked_special_tokens(state: Any) -> list[str]:
    """Special-token strings to strip from output. Empty (no-op) unless the dsv4
    reasoning parser is configured, so non-dsv4 output is untouched."""
    return DSV4_SPECIAL_TOKENS if getattr(state.config, "reasoning_parser", None) == "deepseekv32" else []


def _make_tool_parser(spec: GenSpec, state: Any) -> FunctionCallParser:
    """Build the tool-call parser with its turn-start state read from the prompt:
    a muse detector receives the raw turn bytes (opened by the template's
    ``<|start|>assistant``) only when its reasoning parser is not stacked above,
    which otherwise delivers tool slices with full headers."""
    return FunctionCallParser(
        spec.parser_tools or [],
        getattr(state.config, "tool_call_parser", "llama3"),
        turn_starts_open=getattr(state.config, "reasoning_parser", None) != "muse_glimmer",
    )


def _parse_tool_response(
    text: str,
    spec: GenSpec,
    state: Any,
) -> tuple[str, list[ToolCallItem]] | None:
    if not spec.parse_tools:
        return None
    if not any(tag in text for tag in TOOLS_TAG_LIST):
        return None
    parser = _make_tool_parser(spec, state)
    result = parser.parse_non_stream(text)
    if not result.calls:
        return None
    return result.normal_text, result.calls


def _valid_json(text: str) -> bool:
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


KEEPALIVE = object()
"""Sentinel yielded by with_keepalive() when the event stream has been silent."""


async def with_keepalive(events: AsyncIterator[GenEvent], interval: float):
    """Yield events from ``events``, interspersing the KEEPALIVE sentinel whenever
    ``interval`` seconds pass without one (covers queue/prefill silence before the
    first event too). Exceptions propagate unchanged; the pending read is cancelled
    when the consumer closes."""
    aiter = events.__aiter__()
    task = None
    try:
        while True:
            if task is None:
                task = asyncio.ensure_future(aiter.__anext__())
            try:
                ev = await asyncio.wait_for(asyncio.shield(task), interval)
            except asyncio.TimeoutError:
                yield KEEPALIVE
                continue
            except StopAsyncIteration:
                return
            task = None
            yield ev
    finally:
        if task is not None:
            task.cancel()


def _record_generation(
    *,
    source: str | None,
    stream: bool,
    start: float,
    prompt_tokens: int,
    completion_tokens: int,
    error: str | None,
    first_token_at: float | None = None,
) -> None:
    """Log one generation request into the request ring. Every protocol adapter converges here,
    so token totals are captured whatever endpoint served the request — unlike the HTTP
    middleware, which for a stream records before the totals are known. `source is None` opts
    out (those paths stay logged by the middleware)."""
    if source is None:
        return
    from .api_server import _served_model_name  # lazy: api_server imports this module

    request_ring.record_request(
        request_ring.RequestRecord(
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            method="POST",
            path=source,
            status=500 if error else 200,
            model=_served_model_name(),
            duration_ms=int((time.monotonic() - start) * 1000),
            ttft_ms=int((first_token_at - start) * 1000) if first_token_at is not None else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            stream=stream,
            error=error,
        )
    )


async def generate_events(
    uid: int, spec: GenSpec, state: Any, *, source: str | None = None
) -> AsyncIterator[GenEvent]:
    """Wraps `_generate_events_impl` to log the request with its totals, read off the terminal
    `GenDone`. The `finally` still records the row on a mid-stream disconnect — but with 0 tokens
    if the drop lands before `GenDone`, the only event carrying the totals."""
    start = time.monotonic()
    prompt_tokens = 0
    completion_tokens = 0
    first_token_at: float | None = None
    error: str | None = None
    try:
        async for ev in _generate_events_impl(uid, spec, state):
            if isinstance(ev, GenDone):
                prompt_tokens = ev.prompt_tokens
                completion_tokens = ev.completion_tokens
            elif first_token_at is None:
                first_token_at = time.monotonic()
            yield ev
    except GenerationError as exc:
        error = str(exc)
        raise
    finally:
        _record_generation(
            source=source, stream=True, start=start,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, error=error,
            first_token_at=first_token_at,
        )


async def generate_full(
    uid: int, spec: GenSpec, state: Any, *, source: str | None = None
) -> GenResult:
    """Wraps `_generate_full_impl` to log the request with its totals; the `finally` also records
    a `GenerationError` as a failed row."""
    start = time.monotonic()
    result: GenResult | None = None
    error: str | None = None
    try:
        result = await _generate_full_impl(uid, spec, state)
        return result
    except GenerationError as exc:
        error = str(exc)
        raise
    finally:
        _record_generation(
            source=source, stream=False, start=start,
            prompt_tokens=result.prompt_tokens if result else 0,
            completion_tokens=result.completion_tokens if result else 0,
            error=error,
        )


async def _generate_events_impl(uid: int, spec: GenSpec, state: Any) -> AsyncIterator[GenEvent]:
    """``_generate_events_core`` plus the ``force_nonempty_content`` swap: a turn
    that streamed reasoning but no visible content and no tool call emits one
    trailing content delta carrying the reasoning text, just before the terminal
    ``GenDone``. The non-streaming path does the same thing in one shot."""
    reasoning: list[str] = []
    content_seen = False
    tool_seen = False
    async for ev in _generate_events_core(uid, spec, state):
        if isinstance(ev, ReasoningDelta):
            reasoning.append(ev.text)
        elif isinstance(ev, ContentDelta):
            content_seen = content_seen or bool(ev.text.strip())
        elif isinstance(ev, (ToolCallStart, ToolCallArgsDelta, ToolCallsDelta)):
            tool_seen = True
        elif isinstance(ev, GenDone):
            text = "".join(reasoning).strip()
            if spec.force_nonempty_content and text and not content_seen and not tool_seen:
                yield ContentDelta(text)
        yield ev


async def _generate_events_core(uid: int, spec: GenSpec, state: Any) -> AsyncIterator[GenEvent]:
    """Protocol-neutral streaming generation. Yields semantic events (reasoning /
    content / tool-call deltas) terminated by exactly one GenDone. Produces no wire
    format — the OpenAI/Anthropic/Responses streamers format these into their own.

    With tools configured, content is parsed incrementally: plain text streams live
    (the detector holds back only a suffix that could still grow into a tool-call
    tag) and each tool call is emitted as one complete ToolCallsDelta as soon as it
    closes — mid-generation, not after the whole response. Emitted ToolCallItems
    carry the output ordinal (0, 1, ...) in tool_index. Formats whose detector
    cannot parse incrementally (supports_streaming=False) keep the previous
    buffer-everything-then-parse-at-the-end behavior."""
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    pending = ""
    parse_tools = spec.parse_tools
    reasoning_parser = _make_reasoning_parser(spec, state)
    specials = _leaked_special_tokens(state)

    tool_parser: FunctionCallParser | None = None
    if parse_tools:
        try:
            candidate = _make_tool_parser(spec, state)
        except ValueError:
            candidate = None  # unsupported parser name: keep the buffered path's behavior
        if candidate is not None and candidate.supports_streaming():
            tool_parser = candidate
    frag_stable = tool_parser.args_fragments_prefix_stable() if tool_parser else True

    # Streaming tool-call assembly: detectors emit fragments (name first, then
    # argument diffs); they accumulate here and the call is emitted complete when
    # the next call starts, trailing text arrives, or the stream ends.
    open_call: dict[str, Any] | None = None
    calls_emitted = 0
    # Swallow whitespace-only text only between a call's close and the next real
    # text (markup separators) — NOT inside post-call prose, where a lone " "
    # chunk is a legitimate word gap.
    suppress_ws = False

    def _close_open_call() -> ToolCallsDelta | None:
        nonlocal open_call, calls_emitted
        if open_call is None:
            return None
        # Streamed fragments concatenate to the exact final arguments (detectors
        # emit prefix-stable fragments and close the JSON before the call ends);
        # fall back to the detector's parse state when the stream was cut short.
        params = open_call["params"]
        if not _valid_json(params):
            params = tool_parser.unstreamed_arguments(open_call["detector_index"]) or params
        call = ToolCallItem(
            tool_index=open_call["ordinal"], name=open_call["name"], parameters=params or "{}"
        )
        open_call = None
        calls_emitted += 1
        nonlocal suppress_ws
        suppress_ws = True
        return ToolCallsDelta([call])

    def _route_tool_text(piece: str) -> list[GenEvent]:
        nonlocal open_call, suppress_ws
        out: list[GenEvent] = []
        if not piece:
            return out
        for kind, payload in tool_parser.parse_stream_events(piece):
            if kind == "text":
                # Text arriving while a call is open means the call finished:
                # close it first so wire order matches generation order.
                done = _close_open_call()
                if done is not None:
                    out.append(done)
                stripped = strip_special_tokens(payload, specials)
                if stripped and not (stripped.strip() == "" and suppress_ws):
                    out.append(ContentDelta(stripped))
                    if stripped.strip():
                        suppress_ws = False
                continue
            for frag in payload:
                starts_new = frag.name is not None and (
                    open_call is None or frag.tool_index != open_call["detector_index"]
                )
                if starts_new:
                    done = _close_open_call()
                    if done is not None:
                        out.append(done)
                if open_call is None or starts_new:
                    open_call = {
                        "detector_index": frag.tool_index,
                        "name": frag.name,
                        "params": "",
                        "ordinal": calls_emitted,
                    }
                    out.append(ToolCallStart(
                        tool_index=calls_emitted, name=frag.name, args_prefix_stable=frag_stable,
                    ))
                if frag.parameters:
                    open_call["params"] += frag.parameters
                    out.append(
                        ToolCallArgsDelta(tool_index=open_call["ordinal"], fragment=frag.parameters)
                    )
        return out

    engine_finish_reason: str | None = None
    engine_matched_stop: str | None = None
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            raise GenerationError(ack.error, getattr(ack, "error_code", None))
        prompt_tokens += ack.prompt_tokens_delta
        completion_tokens += ack.completion_tokens_delta
        cached_tokens += ack.cached_tokens
        content_delta = ack.incremental_output
        if reasoning_parser is not None and content_delta:
            was_reasoning = reasoning_parser.in_reasoning
            reasoning_delta, content_delta = reasoning_parser.parse_stream_chunk(content_delta)
            if _ack_is_reasoning(was_reasoning, reasoning_delta):
                reasoning_tokens += ack.completion_tokens_delta
            if reasoning_delta:
                stripped_reasoning = strip_special_tokens(reasoning_delta, specials)
                if stripped_reasoning:  # a bare special token must not open a thinking block
                    yield ReasoningDelta(stripped_reasoning)
        if content_delta:
            if tool_parser is not None:
                for ev in _route_tool_text(content_delta):
                    yield ev
            elif parse_tools:
                pending += content_delta
            else:
                yield ContentDelta(strip_special_tokens(content_delta, specials))
        if ack.finished:
            engine_finish_reason = getattr(ack, "finish_reason", None)
            engine_matched_stop = getattr(ack, "matched_stop", None)
            break

    # Drain residue held in the reasoning parser (a deferred tool block, or a
    # trailing partial token) so it is not silently dropped.
    if reasoning_parser is not None:
        flush_reasoning, flush_content = reasoning_parser.flush()
        if flush_reasoning:
            stripped_reasoning = strip_special_tokens(flush_reasoning, specials)
            if stripped_reasoning:
                yield ReasoningDelta(stripped_reasoning)
        if flush_content:
            if tool_parser is not None:
                for ev in _route_tool_text(flush_content):
                    yield ev
            elif parse_tools:
                pending += flush_content
            else:
                yield ContentDelta(strip_special_tokens(flush_content, specials))

    # Engine reason ("stop"/"length"); a tool call overrides it, but a truncation (length) wins.
    finish_reason = engine_finish_reason or "stop"

    if tool_parser is not None:
        # End-of-stream drain: let the detector finalize a call cut off mid-arguments
        # (closing fragments keep the client's concatenated JSON valid), close a call
        # whose end marker never arrived (truncated generation), best-effort recover a
        # call cut off inside an unterminated tag block, then release text still held
        # back for tag disambiguation.
        for frag in tool_parser.finalize_stream():
            if open_call is not None and frag.parameters:
                open_call["params"] += frag.parameters
                yield ToolCallArgsDelta(tool_index=open_call["ordinal"], fragment=frag.parameters)
        done = _close_open_call()
        if done is not None:
            yield done
        for item in tool_parser.recover_truncated_call():
            yield ToolCallStart(tool_index=calls_emitted, name=item.name)
            yield ToolCallsDelta(
                [ToolCallItem(tool_index=calls_emitted, name=item.name, parameters=item.parameters)]
            )
            calls_emitted += 1
        residual = tool_parser.finish_stream()
        if residual:
            stripped = strip_special_tokens(residual, specials)
            if stripped and not (stripped.strip() == "" and suppress_ws):
                yield ContentDelta(stripped)
        if calls_emitted and finish_reason != "length":
            finish_reason = "tool_calls"
    else:
        parsed = _parse_tool_response(pending, spec, state) if parse_tools else None
        if parsed is not None:
            normal_text, tool_calls = parsed
            normal_text = strip_special_tokens(normal_text, specials)
            if normal_text:
                yield ContentDelta(normal_text)
            yield ToolCallsDelta(tool_calls)
            if finish_reason != "length":
                finish_reason = "tool_calls"
        elif parse_tools and pending:
            yield ContentDelta(strip_special_tokens(pending, specials))

    yield GenDone(
        finish_reason, prompt_tokens, completion_tokens,
        matched_stop=engine_matched_stop, cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


async def _generate_full_impl(uid: int, spec: GenSpec, state: Any) -> GenResult:
    """Protocol-neutral non-streaming generation: accumulate, split reasoning, parse
    tool calls, strip special tokens. The adapters format the GenResult into their wire."""
    full_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    engine_finish_reason: str | None = None
    engine_matched_stop: str | None = None
    # The split itself is one-shot over the whole completion (below); this second,
    # streaming parser exists only to attribute each ack's tokens to reasoning or
    # content, which a one-shot parse cannot recover.
    reasoning_meter = _make_reasoning_parser(spec, state)
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            raise GenerationError(ack.error, getattr(ack, "error_code", None))
        prompt_tokens += ack.prompt_tokens_delta
        completion_tokens += ack.completion_tokens_delta
        cached_tokens += ack.cached_tokens
        if reasoning_meter is not None and ack.incremental_output:
            was_reasoning = reasoning_meter.in_reasoning
            meter_delta, _ = reasoning_meter.parse_stream_chunk(ack.incremental_output)
            if _ack_is_reasoning(was_reasoning, meter_delta):
                reasoning_tokens += ack.completion_tokens_delta
        full_content += ack.incremental_output
        if ack.finished:
            engine_finish_reason = getattr(ack, "finish_reason", None)
            engine_matched_stop = getattr(ack, "matched_stop", None)
            break

    reasoning_text, content_text = _split_reasoning(full_content, spec, state)
    # Engine reason ("stop"/"length"); a tool call overrides it, but a truncation (length) wins.
    finish_reason = engine_finish_reason or "stop"
    tool_calls: list[ToolCallItem] = []
    parsed = _parse_tool_response(content_text, spec, state)
    if parsed is not None:
        content_text, tool_calls = parsed
        if finish_reason != "length":
            finish_reason = "tool_calls"

    specials = _leaked_special_tokens(state)
    reasoning = strip_special_tokens(reasoning_text, specials).strip()
    content = strip_special_tokens(content_text, specials)
    if spec.force_nonempty_content and not content.strip() and not tool_calls and reasoning:
        # The turn produced a thought and nothing else. A client that renders only
        # `content` (Switchyard's judge/classifier targets read nothing else) would
        # show an empty answer, so the thought becomes the answer.
        content = reasoning
    return GenResult(
        reasoning=reasoning,
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        matched_stop=engine_matched_stop,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )
