from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.effort import EFFORT_SCALE, KNOWN_REASONING_EFFORTS

from .api_models import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    ToolChoiceObject,
)
from .client_sessions import chat_session_id
from .function_call_parser import ToolCallItem
from .json_output import apply_json_instruction, schema_instruction
from .request_logger import log_request
from .generation import (
    ContentDelta,
    GenDone,
    GenEvent,
    GenerationError,
    GenSpec,
    ReasoningDelta,
    ToolCallArgsDelta,
    ToolCallsDelta,
    ToolCallStart,
    generate_events,
    generate_full,
    preflight_error,
    preflight_text_error,
    render_messages,
    resolve_sampling,
    submit_generation,
)

#: The wire superset plus "off", DeepSeek's disable synonym that
#: effort_toggle_kwargs has always honored.
_ACCEPTED_EFFORTS = (*KNOWN_REASONING_EFFORTS, "off")


def _thinking_type(req: Any) -> str | None:
    """The DeepSeek-wire thinking toggle, or None for absent/foreign shapes
    (which stay ignored, as extra="allow" ignored them before the field existed)."""
    if isinstance(req.thinking, dict):
        value = req.thinking.get("type")
        if value in ("enabled", "disabled"):
            return value
    return None



#: Roles a non-Harmony chat template can be expected to render. `developer` is
#: gpt-oss/Harmony's own role name (OpenAI renamed `system` there); every other
#: family's template either raises on it or drops the turn silently, which is
#: worse — the instructions vanish. Map it onto `system` for those templates.
_DEVELOPER_ROLE = "developer"


def _harmony_chat_template(state: Any) -> bool:
    """Whether the served model's chat template speaks Harmony (gpt-oss), which has
    a real `developer` role. Read off the configured parsers first (args.py picks
    `gpt_oss` for both from the model marker) and the model path as a fallback."""
    config = getattr(state, "config", None)
    for attr in ("reasoning_parser", "tool_call_parser"):
        value = getattr(config, attr, None)
        if isinstance(value, str) and value.replace("-", "_") == "gpt_oss":
            return True
    path = str(getattr(config, "model_path", "") or "").lower()
    return "gpt-oss" in path or "gpt_oss" in path or "gptoss" in path


def _map_developer_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        if message.get("role") == _DEVELOPER_ROLE:
            message = {**message, "role": "system"}
        out.append(message)
    return out


def chat_request_to_genspec(
    req: ChatCompletionRequest,
    model_sampling: dict[str, Any],
    *,
    map_developer_role: bool = True,
    force_nonempty_content: bool = False,
) -> GenSpec:
    """OpenAI ChatCompletionRequest -> GenSpec (the OpenAI 'to_sampling_params').

    ``force_nonempty_content`` is the server default (--force-nonempty-content);
    the request can override it per call through ``chat_template_kwargs``.

    A JSON-mode ``response_format`` also rewrites the prompt here (schema into the
    system block, thinking off by default); the output half lives in
    ``generation``/``json_output``.
    """
    from .model_meta import effort_toggle_kwargs

    ctk = req.chat_template_kwargs
    thinking_type = _thinking_type(req)
    if req.reasoning_effort or thinking_type:
        ctk = effort_toggle_kwargs(req.reasoning_effort, ctk, thinking_type=thinking_type)
    json_mode = req.response_format is not None and req.response_format.json_mode
    json_schema = req.response_format.schema_dict if json_mode else None
    if json_mode:
        # Thinking off by default for a JSON call: the caller wants an object, not a
        # thought, and a think block is decode tokens the answer never uses (and one
        # more thing to strip). An explicit enable_thinking — direct, or via
        # reasoning_effort/`thinking`, which have already written it — wins.
        ctk = dict(ctk or {})
        ctk.setdefault("enable_thinking", False)
    ctk, force_nonempty = _force_nonempty_content(ctk, force_nonempty_content)
    wire_messages = [m.model_dump(exclude_none=True) for m in req.messages]
    if map_developer_role:
        wire_messages = _map_developer_role(wire_messages)
    messages = render_messages(wire_messages)
    if json_mode:
        messages = apply_json_instruction(messages, schema_instruction(json_schema))
    return GenSpec(
        messages=messages,
        sampling_params=resolve_sampling(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            ignore_eos=req.ignore_eos,
            model_sampling=model_sampling,
            stop=req.stop,
        ),
        chat_template_kwargs=ctk,
        template_tools=_tools_for_template(req),
        parser_tools=(_all_tool_dicts(req.tools) if _should_parse_tools(req) else None),
        session_id=req.session_id,
        session_ttl_seconds=req.session_ttl_seconds,
        force_nonempty_content=force_nonempty,
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _force_nonempty_content(
    ctk: dict[str, Any] | None, server_default: bool
) -> tuple[dict[str, Any] | None, bool]:
    """Split ``force_nonempty_content`` out of the template kwargs.

    It is a FreeToken serving knob, not a template variable: Jinja renders with
    ``**chat_template_kwargs`` and an unknown name either raises or silently
    changes what the model sees, so it must never reach the template. Precedence:
    an explicit request value wins; otherwise a thinking-OFF turn enables it (the
    template pre-closes the think block, so a model that still writes only a
    thought would answer with an empty message), else the server default.
    """
    if not ctk:
        return ctk, server_default
    if "force_nonempty_content" not in ctk:
        thinking_off = ctk.get("enable_thinking") is False
        return ctk, True if thinking_off else server_default
    ctk = dict(ctk)
    requested = ctk.pop("force_nonempty_content")
    return ctk, bool(requested)


def _all_tool_dicts(tools) -> list[dict[str, Any]]:
    return [_tool_dict(t) for t in (tools or [])]


def _tool_dict(tool) -> dict[str, Any]:
    """A tool as the template and the tool-call parser see it: OpenAI's
    structured-outputs `strict` flag is dropped (nothing downstream reads it, and
    leaving it in would change the rendered catalog for clients that send it)."""
    dumped = tool.model_dump(exclude_none=True)
    function = dumped.get("function")
    if isinstance(function, dict):
        function.pop("strict", None)
    return dumped


def _maintenance_gate(state: Any) -> JSONResponse | None:
    """503 while the engine is not serving. Distinguishes the startup "loading" phase from a
    runtime cache "rebuild"/"failed" so clients (and the desktop) get an actionable message.
    None when serving."""
    mstate = getattr(state, "maintenance_state", "serving")
    if mstate == "serving":
        return None
    if mstate == "loading":
        msg = "model is still loading"
    elif mstate == "failed":
        msg = "server unavailable: maintenance failed (restart required)"
    else:
        msg = "server unavailable: cache rebuild in progress"
    return JSONResponse({"error": msg}, status_code=503)


def register_openai_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    @app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
    async def v1_root():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(req: ChatCompletionRequest, request: Request):
        log_request("/v1/chat/completions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_chat_completion(req, request, state, get_model_sampling())

    @app.post("/v1/completions")
    async def v1_completions(req: CompletionRequest, request: Request):
        log_request("/v1/completions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_completion(req, request, state, get_model_sampling())

    @app.get("/v1/models")
    async def v1_models():
        state = get_state()
        model_id = _served_model_name(state)
        ctx = _model_context_length(state)
        efforts, default_effort = await _effort_fields(state)
        return ModelList(data=[ModelCard(
            id=model_id,
            root=state.config.model_path,
            max_model_len=ctx,
            context_length=ctx,
            supported_reasoning_efforts=efforts,
            default_reasoning_effort=default_effort,
        )])


async def handle_chat_completion(
    req: ChatCompletionRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    if req.function_call is not None:
        return create_error_response("function_call is not supported; use tools/tool_choice instead")
    if req.logit_bias is not None:
        return create_error_response("logit_bias is not supported")
    if req.n != 1:
        return create_error_response("Only n=1 is supported", param="n")
    # Switchyard forwards top_logprobs verbatim; 0 (or absent) asks for nothing, so
    # only a positive value is a request we cannot serve.
    if req.top_logprobs is not None and req.top_logprobs > 0:
        return create_error_response(
            "logprobs are not supported; omit top_logprobs or send 0",
            param="top_logprobs",
        )
    # Case/whitespace and the "off" disable synonym stay accepted here because
    # effort_toggle_kwargs normalizes and honors them downstream.
    effort = req.reasoning_effort.strip().lower() if isinstance(req.reasoning_effort, str) else None
    if effort and effort not in _ACCEPTED_EFFORTS:
        return create_error_response(
            f"reasoning_effort must be one of {', '.join(_ACCEPTED_EFFORTS)}; "
            f"got {req.reasoning_effort!r}",
            param="reasoning_effort",
        )
    if isinstance(req.thinking, dict):
        thinking_type = req.thinking.get("type")
        if thinking_type is not None and thinking_type not in ("enabled", "disabled"):
            return create_error_response(
                f"thinking.type must be 'enabled' or 'disabled'; got {thinking_type!r}",
                param="thinking",
            )

    try:
        spec = chat_request_to_genspec(
            req,
            model_sampling,
            map_developer_role=not _harmony_chat_template(state),
            force_nonempty_content=getattr(state.config, "force_nonempty_content", False),
        )
    except ValueError as exc:
        return create_error_response(str(exc))

    # Bind the turn to a KV session lease before submit. An id FreeToken inferred
    # from the client's own conversation headers is reclaimable (the client never
    # learned to close it); an explicit session_id stays the client's to own.
    explicit_session = req.session_id is not None
    spec.session_id = chat_session_id(req, request)
    spec.session_reclaimable = spec.session_id is not None and not explicit_session
    session_headers = (
        {"X-FreeToken-Session-Id": spec.session_id} if spec.session_id is not None else None
    )

    # Both paths preflight: the stream path because an error after the headers go
    # out can only ride in-stream, the non-stream path because a context overflow
    # answered here never costs a queue slot and reads identically to the
    # scheduler's own rejection.
    err = await preflight_error(spec, state)
    if err is not None:
        return create_error_response(str(err), code=err.code)

    uid = await submit_generation(spec, state)

    if req.stream:
        chunks = stream_chat_completion_chunks(uid, req, state, spec)
        if request is not None:
            chunks = (
                state.stream_with_cancellation(chunks, request, uid, spec.session_id)
                if spec.session_id is not None
                else state.stream_with_cancellation(chunks, request, uid)
            )
        return StreamingResponse(
            chunks, media_type="text/event-stream", headers=session_headers
        )

    try:
        result = await generate_full(uid, spec, state, source="/v1/chat/completions")
    except asyncio.CancelledError:
        await state.abort_user(uid, session_id=spec.session_id)
        raise
    except GenerationError as exc:
        if not _auto_session_busy(exc, spec):
            return create_error_response(str(exc), code=exc.code)
        uid = await _resubmit_unbound(spec, state)
        try:
            result = await generate_full(uid, spec, state, source="/v1/chat/completions")
        except asyncio.CancelledError:
            await state.abort_user(uid)
            raise
        except GenerationError as retry_exc:
            return create_error_response(str(retry_exc), code=retry_exc.code)
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.reasoning:
        message["reasoning_content"] = result.reasoning
    if result.tool_calls:
        message["tool_calls"] = _tool_calls_to_openai(result.tool_calls)

    payload = {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": _usage(
            result.prompt_tokens,
            result.completion_tokens,
            _reported_cached(state, result.cached_tokens),
            result.reasoning_tokens,
        ),
    }
    # A plain dict when there is no header to carry (FastAPI serializes it to the
    # same JSONResponse); the session id has to ride on the response itself, which
    # is the one thing a returned dict cannot express.
    if session_headers is None:
        return payload
    return JSONResponse(content=payload, headers=session_headers)


#: The scheduler's rejection when a session lease is already serving a turn
#: (``scheduler.py``: ``f"session {id!r} is busy"``).
_SESSION_BUSY_PREFIX = "session "
_SESSION_BUSY_SUFFIX = " is busy"


def _auto_session_busy(exc: GenerationError, spec: GenSpec) -> bool:
    """Whether ``exc`` is the scheduler refusing a session id FreeToken bound itself.

    Sessions serialize their turns, and Switchyard legitimately runs a
    judge/classifier call on the same conversation as the turn it is judging — so
    an id inferred from conversation headers can collide with an in-flight one.
    The request must still be served (unbound, losing only that call's prefix
    reuse) rather than failing on affinity the client never asked for. An explicit
    ``session_id`` keeps the error: serialization is exactly what it requested.
    """
    if not spec.session_reclaimable or spec.session_id is None:
        return False
    message = str(exc)
    return message.startswith(_SESSION_BUSY_PREFIX) and message.endswith(_SESSION_BUSY_SUFFIX)


async def _resubmit_unbound(spec: GenSpec, state: Any) -> int:
    """Resubmit ``spec`` once with no session lease. Mutates the spec, so a second
    busy reply is impossible (``_auto_session_busy`` is False without a session)."""
    spec.session_id = None
    spec.session_reclaimable = False
    return await submit_generation(spec, state)


async def _open_chat_events(
    uid: int, spec: GenSpec, state: Any
) -> tuple[AsyncIterator[GenEvent], Any, bool, GenerationError | None]:
    """Open the event stream and pull its FIRST event, returning
    ``(events, first_event, have_first, error)``.

    A scheduler-side failure (context overflow, a template the worker rejects, a
    busy session) must be the first SSE event on the wire: Switchyard reads event
    0 to decide whether to retarget, and a role chunk ahead of it reads as a
    successful stream that then goes silent.
    """
    events = generate_events(uid, spec, state, source="/v1/chat/completions")
    try:
        return events, await events.__anext__(), True, None
    except StopAsyncIteration:
        return events, None, False, None
    except GenerationError as exc:
        return events, None, False, exc


async def stream_chat_completion_chunks(
    uid: int,
    req: ChatCompletionRequest,
    state: Any,
    spec: GenSpec | None = None,
) -> AsyncIterator[bytes]:
    """Format generate_events() into the OpenAI chat.completion.chunk SSE stream."""
    if spec is None:
        spec = chat_request_to_genspec(req, {})

    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    tool_calls_sent = 0
    open_tool: dict[str, Any] | None = None

    events, first_event, have_first, error = await _open_chat_events(uid, spec, state)
    if error is not None and _auto_session_busy(error, spec):
        # The auto-bound lease is serving another turn of the same conversation.
        # Resubmit unbound rather than failing the request. (The cancellation
        # wrapper this generator runs under was bound to the pre-fallback uid; a
        # disconnect after a fallback therefore aborts that one — the fallback
        # request finishes on its own.)
        uid = await _resubmit_unbound(spec, state)
        events, first_event, have_first, error = await _open_chat_events(uid, spec, state)
    if error is not None:
        yield _sse(
            {"error": {
                "message": str(error), "type": "invalid_request_error", "code": error.code,
            }}
        )
        yield b"data: [DONE]\n\n"
        return

    yield _sse(
        _chat_chunk(
            req,
            uid,
            [{"delta": {"role": "assistant", "content": ""}, "index": 0, "finish_reason": None}],
        )
    )

    while True:
        if have_first:
            ev, have_first = first_event, False
        else:
            try:
                ev = await events.__anext__()
            except StopAsyncIteration:
                break
            except GenerationError as exc:
                # Request failed mid-stream — emit an error chunk + [DONE] so the
                # client gets a terminal signal instead of a stalled stream.
                yield _sse(
                    {"error": {
                        "message": str(exc), "type": "invalid_request_error", "code": exc.code,
                    }}
                )
                break
        if isinstance(ev, ReasoningDelta):
            yield _sse(
                _chat_chunk(
                    req,
                    uid,
                    [{"delta": {"reasoning_content": ev.text}, "index": 0, "finish_reason": None}],
                )
            )
        elif isinstance(ev, ContentDelta):
            yield _sse(
                _chat_chunk(
                    req,
                    uid,
                    [{"delta": {"content": ev.text}, "index": 0, "finish_reason": None}],
                )
            )
        elif isinstance(ev, ToolCallStart):
            open_tool = {
                "index": tool_calls_sent,
                "ordinal": ev.tool_index,
                "sent": "",
                "stable": ev.args_prefix_stable,
            }
            yield _sse(
                _chat_chunk(
                    req, uid,
                    [{
                        "delta": {"tool_calls": [{
                            "index": open_tool["index"],
                            "id": _tool_call_id(ev.name, open_tool["index"]),
                            "type": "function",
                            "function": {"name": ev.name, "arguments": ""},
                        }]},
                        "index": 0, "finish_reason": None,
                    }],
                )
            )
        elif isinstance(ev, ToolCallArgsDelta):
            # Clients concatenate argument fragments, so only prefix-stable
            # fragments stream; otherwise the full arguments arrive at close.
            if open_tool is not None and open_tool["stable"] and ev.fragment:
                open_tool["sent"] += ev.fragment
                yield _sse(
                    _chat_chunk(
                        req, uid,
                        [{
                            "delta": {"tool_calls": [{
                                "index": open_tool["index"],
                                "function": {"arguments": ev.fragment},
                            }]},
                            "index": 0, "finish_reason": None,
                        }],
                    )
                )
        elif isinstance(ev, ToolCallsDelta):
            for call in ev.calls:
                if open_tool is not None and open_tool["ordinal"] == call.tool_index:
                    # Close of a ToolCallStart-opened call: send whatever of the
                    # final (authoritative) arguments wasn't streamed yet.
                    final = call.parameters or ""
                    remainder = (
                        final[len(open_tool["sent"]):]
                        if final.startswith(open_tool["sent"])
                        else final if not open_tool["sent"] else ""
                    )
                    if remainder:
                        yield _sse(
                            _chat_chunk(
                                req, uid,
                                [{
                                    "delta": {"tool_calls": [{
                                        "index": open_tool["index"],
                                        "function": {"arguments": remainder},
                                    }]},
                                    "index": 0, "finish_reason": None,
                                }],
                            )
                        )
                    open_tool = None
                    tool_calls_sent += 1
                    continue
                # Standalone complete call (buffered fallback path).
                for delta in _tool_call_deltas([call], start_index=tool_calls_sent):
                    yield _sse(
                        _chat_chunk(
                            req, uid,
                            [{"delta": {"tool_calls": [delta]}, "index": 0, "finish_reason": None}],
                        )
                    )
                tool_calls_sent += 1
        elif isinstance(ev, GenDone):
            prompt_tokens = ev.prompt_tokens
            completion_tokens = ev.completion_tokens
            cached_tokens = ev.cached_tokens
            reasoning_tokens = ev.reasoning_tokens
            yield _sse(_chat_chunk(req, uid, [{"delta": {}, "index": 0, "finish_reason": ev.finish_reason}]))

    if req.stream_options and req.stream_options.include_usage:
        yield _sse(
            {
                "id": f"chatcmpl-{uid}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [],
                "usage": _usage(
                    prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens),
                    reasoning_tokens,
                ),
            }
        )

    yield b"data: [DONE]\n\n"


async def handle_completion(
    req: CompletionRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    unsupported = _completion_unsupported_reason(req)
    if unsupported is not None:
        return create_error_response(unsupported)
    try:  # surfaces an out-of-range max_tokens as a 400 rather than a 500 from the worker
        _resolve_sampling(req, model_sampling)
    except ValueError as exc:
        return create_error_response(str(exc), param="max_tokens")

    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    assert isinstance(prompts, list)
    # Same overflow contract as chat: a 400 with `context_length_exceeded` before a
    # queue slot (and, on the stream path, before the response headers commit).
    for prompt in prompts:
        err = await preflight_text_error(prompt, state)
        if err is not None:
            return create_error_response(str(err), code=err.code)
    if req.stream:
        if len(prompts) != 1:
            return create_error_response("Streaming completions only support a single text prompt")
        uid = state.new_user()
        await state.send_one(
            TokenizeMsg(
                uid=uid,
                text=prompts[0],
                sampling_params=_resolve_sampling(req, model_sampling),
                session_id=req.session_id,
                session_ttl_seconds=req.session_ttl_seconds,
            )
        )
        chunks = stream_completion_chunks(uid, req, state)
        if request is not None:
            chunks = (
                state.stream_with_cancellation(chunks, request, uid, req.session_id)
                if req.session_id is not None
                else state.stream_with_cancellation(chunks, request, uid)
            )
        return StreamingResponse(chunks, media_type="text/event-stream")

    choices: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    for index, prompt in enumerate(prompts):
        uid = state.new_user()
        await state.send_one(
            TokenizeMsg(
                uid=uid,
                text=prompt,
                sampling_params=_resolve_sampling(req, model_sampling),
                session_id=req.session_id,
                session_ttl_seconds=req.session_ttl_seconds,
            )
        )
        text = ""
        finish_reason = "stop"
        try:
            async for ack in state.wait_for_ack(uid):
                if getattr(ack, "error", None):
                    return create_error_response(
                        ack.error, code=getattr(ack, "error_code", None)
                    )
                prompt_tokens += ack.prompt_tokens_delta
                completion_tokens += ack.completion_tokens_delta
                cached_tokens += ack.cached_tokens
                text += ack.incremental_output
                if ack.finished:
                    finish_reason = getattr(ack, "finish_reason", None) or "stop"
                    break
        except asyncio.CancelledError:
            await state.abort_user(uid, session_id=req.session_id)
            raise
        choices.append({"index": index, "text": text, "finish_reason": finish_reason, "logprobs": None})

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
        "usage": _usage(prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens)),
    }


async def stream_completion_chunks(uid: int, req: CompletionRequest, state: Any) -> AsyncIterator[bytes]:
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    finish_reason = "stop"
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            # Carry the stable class (context_length_exceeded) so a router reading
            # error.code sees the same thing here as on the chat stream.
            yield _sse({"error": {
                "message": ack.error,
                "type": "invalid_request_error",
                "code": getattr(ack, "error_code", None),
            }})
            yield b"data: [DONE]\n\n"
            return
        prompt_tokens += ack.prompt_tokens_delta
        completion_tokens += ack.completion_tokens_delta
        cached_tokens += ack.cached_tokens
        if ack.incremental_output:
            yield _sse(
                {
                    "id": f"cmpl-{uid}",
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "text": ack.incremental_output,
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": None,
                        }
                    ],
                }
            )
        if ack.finished:
            finish_reason = getattr(ack, "finish_reason", None) or "stop"
            break

    yield _sse(
        {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"text": "", "index": 0, "finish_reason": finish_reason, "logprobs": None}],
        }
    )
    if req.stream_options and req.stream_options.include_usage:
        yield _sse(
            {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [],
                "usage": _usage(
                    prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens)
                ),
            }
        )
    yield b"data: [DONE]\n\n"


def create_error_response(
    message: str,
    status_code: int = 400,
    err_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
    )


def _resolve_sampling(
    req: ChatCompletionRequest | CompletionRequest,
    model_sampling: dict[str, Any],
) -> SamplingParams:
    return resolve_sampling(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        ignore_eos=req.ignore_eos,
        model_sampling=model_sampling,
        stop=req.stop,
    )


def _tools_for_template(req: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    if not _should_parse_tools(req):
        return None

    tools = req.tools or []
    if isinstance(req.tool_choice, ToolChoiceObject):
        selected = req.tool_choice.function.name
        tools = [tool for tool in tools if tool.function.name == selected]

    return [_tool_dict(tool) for tool in tools]


def _should_parse_tools(req: ChatCompletionRequest) -> bool:
    return bool(req.tools) and req.tool_choice != "none"


def _tool_calls_to_openai(calls: list[ToolCallItem]) -> list[dict[str, Any]]:
    result = []
    for index, call in enumerate(calls):
        result.append(
            {
                "id": _tool_call_id(call.name, index),
                "index": index,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.parameters,
                },
            }
        )
    return result


def _tool_call_deltas(calls: list[ToolCallItem], start_index: int = 0) -> list[dict[str, Any]]:
    """OpenAI tool_calls stream deltas. ``start_index`` offsets the slot index so
    calls arriving across multiple ToolCallsDelta events (streamed one per call as
    each closes) don't all collapse into slot 0."""
    deltas: list[dict[str, Any]] = []
    for offset, call in enumerate(calls):
        index = start_index + offset
        call_id = _tool_call_id(call.name, index)
        deltas.append(
            {
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": ""},
            }
        )
        deltas.append({"index": index, "function": {"arguments": call.parameters}})
    return deltas


def _tool_call_id(name: str | None, index: int) -> str:
    prefix = (name or "tool").replace("_", "-")[:24]
    return f"call_{prefix}_{index}_{uuid.uuid4().hex[:8]}"


def _chat_chunk(req: ChatCompletionRequest, uid: int, choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
    }


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _reported_cached(state: Any, cached_tokens: int) -> int:
    """The prefix-cache hit to report; 0 unless --enable-cache-report is set."""
    return cached_tokens if getattr(state.config, "enable_cache_report", False) else 0


def _usage(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    # sglang convention: the details object appears only for a nonzero hit, so a
    # disabled report and a 0-token hit serialize identically.
    if cached_tokens > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    # Same rule for the reasoning half (Switchyard treats an absent details object
    # as zero), so a non-reasoning model's usage is byte-identical to before.
    if reasoning_tokens > 0:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return usage


def _response_format_unsupported(response_format: dict[str, Any] | None) -> bool:
    """/v1/completions only: JSON mode is a chat feature.

    Chat serves `json_object`/`json_schema` by instruction + validation + retry
    (`server/json_output.py`), which needs a system block and a repair turn — a raw
    text completion has neither, and no client asks for it there. So the legacy
    rejection stays on this route rather than silently returning unshaped text."""
    return response_format is not None and response_format.get("type") not in (None, "text")


def _completion_unsupported_reason(req: CompletionRequest) -> str | None:
    if _is_token_prompt(req.prompt):
        return "OpenAI token-id prompt inputs are not supported; pass text prompt strings instead"
    if req.logprobs is not None:
        return "logprobs is not supported"
    if req.echo:
        return "echo is not supported"
    if req.suffix is not None:
        return "suffix is not supported"
    if req.logit_bias is not None:
        return "logit_bias is not supported"
    if _response_format_unsupported(req.response_format):
        return "response_format json_object/json_schema is not supported (no constrained decoding)"
    return None


def _is_token_prompt(prompt: Any) -> bool:
    return (
        isinstance(prompt, list)
        and bool(prompt)
        and (
            all(isinstance(item, int) for item in prompt)
            or all(isinstance(item, list) and all(isinstance(token, int) for token in item) for item in prompt)
        )
    )


async def _effort_fields(state: Any) -> tuple[list[str] | None, str | None]:
    """The checkpoint's probed effort vocabulary for /v1/models, or (None, None)
    when there is no frontend tokenizer, it fails to build, or the model has no
    effort knob — a metadata route must never 500 over this."""
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return None, None
    try:
        manager = await asyncio.to_thread(build)
        profile = await asyncio.to_thread(manager.effort_profile)
    except Exception:  # noqa: BLE001 -- metadata only; the generation path reports real faults
        return None, None
    from freetoken.tokenizer.effort import effective_efforts

    served = effective_efforts(profile)
    if not served:
        return None, None
    ordered = sorted(served, key=lambda name: -EFFORT_SCALE.get(name, 0.0))
    return ordered, profile.default


def _served_model_name(state: Any) -> str:
    return getattr(state.config, "served_model_name", None) or state.config.model_path


def _model_context_length(state: Any) -> int | None:
    """The model ceiling, not `min(ceiling, KV budget)`: a rebuild moves the latter, and agents
    read this once at startup."""
    try:  # never 500 a metadata route: max_seq_len walks into the HF config on some builds
        value = int(state.config.max_seq_len)
    except Exception:  # noqa: BLE001
        return None
    return value if value > 0 else None
