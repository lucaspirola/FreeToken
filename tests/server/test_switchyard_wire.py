"""Switchyard's `openai_chat` upstream contract on /v1/chat/completions.

Switchyard routes on the wire, not on prose: a context overflow must be an HTTP 400
whose `error.code` is `context_length_exceeded` (and, on a stream, the FIRST SSE
event), usage must carry `completion_tokens_details.reasoning_tokens`, and the ten
optional fields its codec forwards verbatim must all be accepted. These tests pin
that surface. See tasks/nemotron35-plan.md, "Switchyard contract".
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import freetoken
from fastapi.responses import JSONResponse
from freetoken.message.frontend import UserReply
from freetoken.server import generation as G
from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
from freetoken.server.openai_api import (
    handle_chat_completion,
    handle_completion,
    stream_chat_completion_chunks,
    stream_completion_chunks,
)

from .test_openai_api import FakeState, parse_sse, run, tool_schema

OVERFLOW_CODE = "context_length_exceeded"


class TokenizingState(FakeState):
    """A FakeState with a frontend tokenizer, so the preflight actually runs.

    ``prompt_tokens`` is what the fake tokenizer reports for any prompt; the served
    window is ``max_seq_len``.
    """

    def __init__(self, *args, prompt_tokens: int = 8, max_seq_len: int = 4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.max_seq_len = max_seq_len
        self.rendered: list = []
        self._prompt_tokens = prompt_tokens

    def frontend_tokenizer(self):
        state = self

        class _Manager:
            def render_prompt(self, msg):
                state.rendered.append(msg)
                return "rendered"

            def tokenize(self, msgs):
                state.rendered.extend(msgs)
                return [SimpleNamespace(numel=lambda: state._prompt_tokens) for _ in msgs]

        return _Manager()


def simple_request(**kwargs) -> ChatCompletionRequest:
    payload = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def error_reply(message: str, code: str | None = OVERFLOW_CODE) -> UserReply:
    return UserReply(
        uid=42, incremental_output="", finished=True, error=message, error_code=code
    )


# --------------------------------------------------------------------------- #
# 1. Context overflow: preflight 400 + code, on both paths.
# --------------------------------------------------------------------------- #
def _assert_overflow_response(response) -> dict:
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    import json

    error = json.loads(bytes(response.body))["error"]
    assert error["code"] == OVERFLOW_CODE
    # Both phrasings ride in the one message: OpenAI clients / Switchyard match the
    # first, Claude Code and OpenClaw the second.
    assert "maximum context length" in error["message"]
    assert "prompt is too long" in error["message"]
    return error


def test_preflight_rejects_overflow_before_the_stream_starts():
    state = TokenizingState([], prompt_tokens=9000, max_seq_len=4096)
    response = run(
        handle_chat_completion(
            simple_request(stream=True), request=None, state=state, model_sampling={}
        )
    )
    _assert_overflow_response(response)
    # Never submitted: the 400 costs no queue slot and no KV.
    assert state.sent is None


def test_preflight_rejects_overflow_on_the_non_stream_path():
    state = TokenizingState([], prompt_tokens=9000, max_seq_len=4096)
    response = run(
        handle_chat_completion(
            simple_request(), request=None, state=state, model_sampling={}
        )
    )
    _assert_overflow_response(response)
    assert state.sent is None


def test_preflight_rejects_a_prompt_that_exactly_fills_the_window():
    # The scheduler drops a request that leaves zero decode budget, so the
    # preflight must reject at `>=`, not `>`.
    state = TokenizingState([], prompt_tokens=4096, max_seq_len=4096)
    response = run(
        handle_chat_completion(simple_request(), request=None, state=state, model_sampling={})
    )
    _assert_overflow_response(response)


def test_preflight_passes_a_prompt_that_fits():
    state = TokenizingState(
        [UserReply(uid=42, incremental_output="ok", finished=True)],
        prompt_tokens=100,
        max_seq_len=4096,
    )
    response = run(
        handle_chat_completion(simple_request(), request=None, state=state, model_sampling={})
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    assert state.sent is not None


def test_preflight_disabled_falls_back_to_render_only(monkeypatch):
    monkeypatch.setenv("FREETOKEN_CONTEXT_PREFLIGHT", "0")
    state = TokenizingState(
        [UserReply(uid=42, incremental_output="ok", finished=True)],
        prompt_tokens=9000,
        max_seq_len=4096,
    )
    response = run(
        handle_chat_completion(simple_request(), request=None, state=state, model_sampling={})
    )
    # No length check: the scheduler stays the authority.
    assert response["choices"][0]["message"]["content"] == "ok"


def test_completions_preflight_rejects_overflow():
    state = TokenizingState([], prompt_tokens=9000, max_seq_len=4096)
    req = CompletionRequest(model="client-model", prompt="a long prompt", max_tokens=8)
    response = run(handle_completion(req, request=None, state=state, model_sampling={}))
    _assert_overflow_response(response)


# --------------------------------------------------------------------------- #
# 2. A scheduler-side failure is the FIRST SSE event.
# --------------------------------------------------------------------------- #
def test_scheduler_overflow_is_the_first_sse_event():
    message = (
        "prompt is too long: 9000 tokens > 4096 maximum "
        "(this model's maximum context length, prompt + generation)"
    )
    state = FakeState([error_reply(message)])
    req = simple_request(stream=True)
    chunks = run(_collect(stream_chat_completion_chunks(42, req, state)))
    events = parse_sse(chunks)

    first = events[0]
    assert isinstance(first, dict) and "error" in first, events[:2]
    assert first["error"]["code"] == OVERFLOW_CODE
    assert "maximum context length" in first["error"]["message"]
    assert events[-1] == "[DONE]"
    # No role chunk ahead of it: a client reading event 0 sees the failure.
    assert not any(
        isinstance(e, dict) and e.get("choices") for e in events if e != "[DONE]"
    )


def test_successful_stream_still_opens_with_the_role_chunk():
    state = FakeState(
        [
            UserReply(uid=42, incremental_output="hello", finished=False),
            UserReply(uid=42, incremental_output="", finished=True),
        ]
    )
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, simple_request(stream=True), state))))
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert events[1]["choices"][0]["delta"] == {"content": "hello"}
    assert events[-1] == "[DONE]"


def test_completions_stream_error_carries_the_code():
    state = FakeState([error_reply("prompt is too long: 9000 tokens > 4096 maximum")])
    req = CompletionRequest(model="client-model", prompt="x", stream=True)
    events = parse_sse(run(_collect(stream_completion_chunks(42, req, state))))
    assert events[0]["error"]["code"] == OVERFLOW_CODE
    assert events[-1] == "[DONE]"


def test_completions_non_stream_error_carries_the_code():
    import json

    state = FakeState([error_reply("prompt is too long: 9000 tokens > 4096 maximum")])
    req = CompletionRequest(model="client-model", prompt="x")
    response = run(handle_completion(req, request=None, state=state, model_sampling={}))
    assert isinstance(response, JSONResponse) and response.status_code == 400
    assert json.loads(bytes(response.body))["error"]["code"] == OVERFLOW_CODE


def test_scheduler_overflow_message_carries_both_phrasings():
    """The scheduler's own rejection (which the non-preflight paths still hit)
    must read like the preflight's."""
    source = (
        pathlib.Path(freetoken.__file__).parent / "scheduler" / "scheduler.py"
    ).read_text()
    index = source.index(f'code="{OVERFLOW_CODE}"')
    block = source[max(0, index - 1200) : index]
    assert "maximum context length" in block
    assert "prompt is too long" in block


def test_context_overflow_message_helper():
    message = G.context_overflow_message(9000, 4096)
    assert "maximum context length" in message and "prompt is too long" in message
    assert "4096" in message and "9000" in message


# --------------------------------------------------------------------------- #
# 3. reasoning_tokens in usage.
# --------------------------------------------------------------------------- #
def _split_replies(pieces: list[str]) -> list[UserReply]:
    replies = []
    for index, piece in enumerate(pieces):
        replies.append(
            UserReply(
                uid=42,
                incremental_output=piece,
                finished=index == len(pieces) - 1,
                prompt_tokens_delta=5 if index == 0 else 0,
                completion_tokens_delta=1,
            )
        )
    return replies


def test_reasoning_tokens_non_stream_three_ack_split():
    # "a b</think>c" over three acks: the first two were generated inside the
    # reasoning block (the one carrying </think> included), the third was not.
    state = FakeState(_split_replies(["a ", "b</think>", "c"]), reasoning_parser="qwen3")
    response = run(
        handle_chat_completion(simple_request(), request=None, state=state, model_sampling={})
    )
    assert response["choices"][0]["message"]["reasoning_content"] == "a b"
    assert response["choices"][0]["message"]["content"] == "c"
    assert response["usage"]["completion_tokens"] == 3
    assert response["usage"]["completion_tokens_details"] == {"reasoning_tokens": 2}


def test_reasoning_tokens_in_the_final_stream_usage_chunk():
    state = FakeState(_split_replies(["a ", "b</think>", "c"]), reasoning_parser="qwen3")
    req = simple_request(stream=True, stream_options={"include_usage": True})
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
    usage = [e for e in events if isinstance(e, dict) and e.get("usage")][-1]["usage"]
    assert usage["completion_tokens"] == 3
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 2}


def test_no_reasoning_details_without_a_reasoning_parser():
    state = FakeState(_split_replies(["a ", "b</think>", "c"]))
    response = run(
        handle_chat_completion(simple_request(), request=None, state=state, model_sampling={})
    )
    assert "completion_tokens_details" not in response["usage"]


def test_cached_tokens_details_survive():
    replies = [
        UserReply(
            uid=42, incremental_output="x", finished=True,
            prompt_tokens_delta=100, completion_tokens_delta=1, cached_tokens=64,
        )
    ]
    state = FakeState(replies)
    state.config.enable_cache_report = True
    response = run(
        handle_chat_completion(simple_request(), request=None, state=state, model_sampling={})
    )
    assert response["usage"]["prompt_tokens_details"] == {"cached_tokens": 64}


# --------------------------------------------------------------------------- #
# 4. developer -> system for non-Harmony templates.
# --------------------------------------------------------------------------- #
def _developer_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="client-model",
        messages=[
            {"role": "developer", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
        max_tokens=4,
    )


def test_developer_role_is_mapped_to_system_for_non_harmony_templates():
    state = FakeState([UserReply(uid=42, incremental_output="ok", finished=True)])
    run(handle_chat_completion(_developer_request(), request=None, state=state, model_sampling={}))
    assert [m["role"] for m in state.sent.text] == ["system", "user"]
    assert state.sent.text[0]["content"] == "be terse"


def test_developer_role_is_preserved_for_harmony():
    state = FakeState(
        [UserReply(uid=42, incremental_output="ok", finished=True)],
        tool_call_parser="gpt_oss",
        reasoning_parser="gpt_oss",
    )
    run(handle_chat_completion(_developer_request(), request=None, state=state, model_sampling={}))
    assert [m["role"] for m in state.sent.text] == ["developer", "user"]


def test_developer_role_is_preserved_when_the_model_path_says_gpt_oss():
    state = FakeState([UserReply(uid=42, incremental_output="ok", finished=True)])
    state.config.model_path = "/models/gpt-oss-20b"
    run(handle_chat_completion(_developer_request(), request=None, state=state, model_sampling={}))
    assert [m["role"] for m in state.sent.text] == ["developer", "user"]


# --------------------------------------------------------------------------- #
# 5. Typed optional fields.
# --------------------------------------------------------------------------- #
def test_max_completion_tokens_wins_over_max_tokens():
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16,
        max_completion_tokens=99,
    )
    assert req.max_tokens == 99

    state = FakeState([UserReply(uid=42, incremental_output="ok", finished=True)])
    run(handle_chat_completion(req, request=None, state=state, model_sampling={}))
    assert state.sent.sampling_params.max_tokens == 99


def test_max_tokens_alone_is_still_honored():
    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=16
    )
    assert req.max_tokens == 16


def test_completion_request_alias_precedence():
    req = CompletionRequest(model="m", prompt="hi", max_tokens=16, max_completion_tokens=99)
    assert req.max_tokens == 99


def test_every_optional_switchyard_field_is_accepted():
    """The ten optional fields the openai_chat codec forwards verbatim."""
    req = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=64,          # 1
        tools=tool_schema(),               # 2
        tool_choice="auto",                # 3
        temperature=0.7,                   # 4
        top_p=0.95,                        # 5
        reasoning_effort="medium",         # 6
        response_format={"type": "text"},  # 7
        parallel_tool_calls=True,          # 8
        prompt_cache_key="sy-session-1",   # 9
        stream_options={"include_usage": True},  # 10
        top_logprobs=0,
        stop=["</s>"],
        user="switchyard",
    )
    assert req.prompt_cache_key == "sy-session-1"
    assert req.user == "switchyard"
    assert req.top_logprobs == 0

    state = FakeState([UserReply(uid=42, incremental_output="ok", finished=True)])
    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))
    assert not isinstance(response, JSONResponse), response
    assert response["choices"][0]["message"]["content"] == "ok"


def test_positive_top_logprobs_is_rejected():
    state = FakeState([])
    response = run(
        handle_chat_completion(
            simple_request(top_logprobs=3), request=None, state=state, model_sampling={}
        )
    )
    assert isinstance(response, JSONResponse) and response.status_code == 400
    import json

    assert json.loads(bytes(response.body))["error"]["param"] == "top_logprobs"


def test_function_strict_is_stripped_before_templating():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        }
    ]
    state = FakeState([UserReply(uid=42, incremental_output="ok", finished=True)])
    run(
        handle_chat_completion(
            simple_request(tools=tools), request=None, state=state, model_sampling={}
        )
    )
    assert state.sent.tools == [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        }
    ]


def test_large_tool_catalog_renders():
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i:02d}",
                "description": f"Tool number {i}.",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                    "required": ["arg"],
                },
                "strict": False,
            },
        }
        for i in range(64)
    ]
    state = FakeState([UserReply(uid=42, incremental_output="ok", finished=True)])
    response = run(
        handle_chat_completion(
            simple_request(tools=tools), request=None, state=state, model_sampling={}
        )
    )
    assert not isinstance(response, JSONResponse), response
    assert len(state.sent.tools) == 64
    assert state.sent.tools[63]["function"]["name"] == "tool_63"
    assert all("strict" not in t["function"] for t in state.sent.tools)


async def _collect(stream) -> list[bytes]:
    return [chunk async for chunk in stream]
