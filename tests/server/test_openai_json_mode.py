"""JSON mode (`response_format`) on /v1/chat/completions.

FreeToken has no constrained decoding, so JSON mode is prompt + repair: the
schema goes into the system block with thinking off, the completion is buffered
and canonicalized, a schema failure is retried once at temperature 0, and a final
failure returns the raw text with HTTP 200 (Switchyard falls through to a strong
target on an unusable verdict; a 4xx would break the route instead).

The schemas exercised here are Switchyard's own judge/classifier verdicts
(`crates/libsy/src/prompts/{capability-classifier,escalation}/schema.json`).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from freetoken.message import UserReply
from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
from freetoken.server.json_output import coerce_json_content, schema_error
from freetoken.server.openai_api import (
    chat_request_to_genspec,
    handle_chat_completion,
    handle_completion,
    stream_chat_completion_chunks,
)

from .test_openai_api import parse_sse, run

ESCALATION_SCHEMA = {
    "type": "object",
    "properties": {
        "escalate": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["escalate", "reason"],
    "additionalProperties": False,
}

CAPABILITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["crux", "primary_rule", "capability_boundary", "p_solve"],
    "properties": {
        "crux": {"type": "string", "minLength": 1},
        "primary_rule": {"type": "string", "enum": ["SUP-1", "UNC-1", "none"]},
        "capability_boundary": {
            "type": "string",
            "enum": ["supported", "uncertain", "unsupported", "unmatched"],
        },
        "p_solve": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def json_schema_format(schema: dict, name: str = "EscalationVerdict") -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


class JsonFakeState:
    """A state that can serve MORE THAN ONE submission: attempt N reads batch N.
    Every TokenizeMsg is kept so a test can assert what the retry actually sent."""

    def __init__(self, batches: list[list[UserReply]], reasoning_parser: str | None = None):
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="qwen3_coder",
            reasoning_parser=reasoning_parser,
        )
        self.batches = batches
        self.sent: list = []
        self._uid = 0

    def new_user(self) -> int:
        self._uid += 1
        return self._uid

    async def send_one(self, msg):
        self.sent.append(msg)

    async def wait_for_ack(self, uid: int):
        assert 1 <= uid <= len(self.batches), f"unexpected attempt {uid}"
        for reply in self.batches[uid - 1]:
            yield reply

    async def abort_user(self, uid: int, session_id: str | None = None):
        return None


def stream_replies(*chunks: str) -> list[UserReply]:
    """One reply per chunk; the last one closes the turn."""
    replies = [
        UserReply(
            uid=0,
            incremental_output=chunk,
            finished=False,
            prompt_tokens_delta=5 if i == 0 else 0,
            completion_tokens_delta=1,
        )
        for i, chunk in enumerate(chunks)
    ]
    replies.append(
        UserReply(uid=0, incremental_output="", finished=True, finish_reason="stop")
    )
    return replies


def chat_request(**kwargs) -> ChatCompletionRequest:
    payload = {
        "model": "client-model",
        "messages": [
            {"role": "system", "content": "You judge runs."},
            {"role": "user", "content": "should we escalate?"},
        ],
        "max_tokens": 64,
    }
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def content_of(payload) -> str:
    return payload["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Output repair
# ---------------------------------------------------------------------------
def test_think_residue_and_fences_are_stripped_to_canonical_json():
    """The Lightning shape: a think block the parser did not claim, then a fenced
    object. Switchyard's decoder strips at most ONE fence and never scans prose,
    so what lands in `content` has to be the bare object."""
    reply = '<think>weigh it</think>```json\n{"escalate": true, "reason": "stuck"}\n```'
    state = JsonFakeState([stream_replies(reply)])
    payload = run(
        handle_chat_completion(
            chat_request(response_format=json_schema_format(ESCALATION_SCHEMA)),
            None,
            state,
            {},
        )
    )
    assert content_of(payload) == '{"escalate": true, "reason": "stuck"}'
    assert json.loads(content_of(payload))["escalate"] is True
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_object_embedded_in_prose_is_extracted():
    state = JsonFakeState([stream_replies('Sure! {"ok": true} Hope that helps.')])
    payload = run(
        handle_chat_completion(
            chat_request(response_format={"type": "json_object"}), None, state, {}
        )
    )
    assert content_of(payload) == '{"ok": true}'


def test_text_response_format_is_untouched():
    """`{"type": "text"}` (the only response_format Switchyard's passthrough route
    sends) must not turn on JSON mode."""
    state = JsonFakeState([stream_replies("plain answer")])
    payload = run(
        handle_chat_completion(
            chat_request(response_format={"type": "text"}), None, state, {}
        )
    )
    assert content_of(payload) == "plain answer"
    assert state.sent[0].chat_template_kwargs.get("enable_thinking") is None


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------
def test_invalid_then_valid_retries_once_at_temperature_zero():
    state = JsonFakeState(
        [
            stream_replies("I think you should escalate."),
            stream_replies('{"escalate": true, "reason": "loop"}'),
        ]
    )
    payload = run(
        handle_chat_completion(
            chat_request(response_format=json_schema_format(ESCALATION_SCHEMA)),
            None,
            state,
            {"temperature": 1.0, "top_p": 0.95},
        )
    )
    assert content_of(payload) == '{"escalate": true, "reason": "loop"}'
    assert len(state.sent) == 2, "the repair attempt is a second submission"

    first, second = state.sent
    assert first.sampling_params.temperature == 1.0
    assert second.sampling_params.temperature == 0.0
    assert second.sampling_params.top_p == 1.0  # or "temperature 0" is not greedy
    # The repair turn is the model's own error, appended as a user message.
    assert len(second.text) == len(first.text) + 1
    repair = second.text[-1]
    assert repair["role"] == "user"
    assert "JSON" in repair["content"]
    # Thinking is off by default on a JSON call, on both attempts.
    assert first.chat_template_kwargs["enable_thinking"] is False
    assert second.chat_template_kwargs["enable_thinking"] is False


def test_schema_violation_retries_and_a_final_failure_passes_the_raw_text_through():
    """Two schema-invalid replies: 200 with the raw content, never a 400 — an
    unusable verdict is a soft failure Switchyard routes around."""
    bad = '{"escalate": "probably", "reason": "unsure"}'
    state = JsonFakeState([stream_replies(bad), stream_replies(bad)])
    payload = run(
        handle_chat_completion(
            chat_request(response_format=json_schema_format(ESCALATION_SCHEMA)),
            None,
            state,
            {},
        )
    )
    assert len(state.sent) == 2
    assert content_of(payload) == bad
    assert payload["choices"][0]["finish_reason"] == "stop"
    # The failure the model was shown names the offending field.
    assert "escalate" in state.sent[1].text[-1]["content"]


def test_a_failed_repair_keeps_the_first_answer():
    """The repair turn is longer than the original; if it overflows (or the lease
    goes away) the earlier raw answer is still returned, not a 400."""
    failure = [
        UserReply(
            uid=0,
            incremental_output="",
            finished=True,
            error="This model's maximum context length is 8 tokens",
            error_code="context_length_exceeded",
        )
    ]
    state = JsonFakeState([stream_replies("no json here"), failure])
    payload = run(
        handle_chat_completion(
            chat_request(response_format={"type": "json_object"}), None, state, {}
        )
    )
    assert content_of(payload) == "no json here"
    assert len(state.sent) == 2


def test_json_object_mode_accepts_any_object_without_retrying():
    state = JsonFakeState([stream_replies('{"anything": [1, 2]}')])
    payload = run(
        handle_chat_completion(
            chat_request(response_format={"type": "json_object"}), None, state, {}
        )
    )
    assert content_of(payload) == '{"anything": [1, 2]}'
    assert len(state.sent) == 1


def test_retry_budget_zero_disables_the_repair_turn(monkeypatch):
    monkeypatch.setenv("FREETOKEN_JSON_RETRY", "0")
    state = JsonFakeState([stream_replies("nope")])
    payload = run(
        handle_chat_completion(
            chat_request(response_format={"type": "json_object"}), None, state, {}
        )
    )
    assert len(state.sent) == 1
    assert content_of(payload) == "nope"


# ---------------------------------------------------------------------------
# Prompt side
# ---------------------------------------------------------------------------
def test_schema_instruction_lands_in_the_system_block():
    verdict = (
        '{"crux": "long sweep", "primary_rule": "UNC-1", '
        '"capability_boundary": "uncertain", "p_solve": 0.4}'
    )
    state = JsonFakeState([stream_replies(verdict)])
    run(
        handle_chat_completion(
            chat_request(response_format=json_schema_format(CAPABILITY_SCHEMA)),
            None,
            state,
            {},
        )
    )
    system = state.sent[0].text[0]
    assert system["role"] == "system"
    assert system["content"].startswith("You judge runs.")
    assert "single JSON object only" in system["content"]
    assert "must validate against this JSON Schema" in system["content"]
    assert '"capability_boundary"' in system["content"]
    # The user turn is untouched.
    assert state.sent[0].text[1] == {"role": "user", "content": "should we escalate?"}


def test_json_object_instruction_carries_no_schema():
    spec = chat_request_to_genspec(
        chat_request(response_format={"type": "json_object"}), {}
    )
    system = spec.messages[0]["content"]
    assert "single JSON object only" in system
    assert "JSON Schema" not in system
    assert spec.json_mode is True
    assert spec.json_schema is None


def test_a_system_block_is_created_when_the_conversation_has_none():
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    spec = chat_request_to_genspec(req, {})
    assert spec.messages[0]["role"] == "system"
    assert spec.messages[1]["role"] == "user"


def test_caller_set_enable_thinking_wins():
    spec = chat_request_to_genspec(
        chat_request(
            response_format={"type": "json_object"},
            chat_template_kwargs={"enable_thinking": True},
        ),
        {},
    )
    assert spec.chat_template_kwargs["enable_thinking"] is True


def test_reasoning_effort_still_drives_the_toggle():
    """`reasoning_effort` writes enable_thinking before JSON mode sees the kwargs,
    so an explicit effort is not silently overridden."""
    spec = chat_request_to_genspec(
        chat_request(response_format={"type": "json_object"}, reasoning_effort="high"),
        {},
    )
    assert spec.chat_template_kwargs.get("enable_thinking") is not False


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def test_stream_emits_exactly_one_content_delta_before_the_finish_chunk():
    state = JsonFakeState(
        [stream_replies('{"escalate"', ": true, ", '"reason": "loop"}')]
    )
    req = chat_request(response_format=json_schema_format(ESCALATION_SCHEMA), stream=True)
    spec = chat_request_to_genspec(req, {})
    uid = 0

    async def collect():
        nonlocal uid
        from freetoken.server.generation import submit_generation

        uid = await submit_generation(spec, state)
        return [chunk async for chunk in stream_chat_completion_chunks(uid, req, state, spec)]

    events = parse_sse(run(collect()))
    assert events[-1] == "[DONE]"
    contents = [
        e["choices"][0]["delta"]["content"]
        for e in events
        if isinstance(e, dict) and e["choices"] and "content" in e["choices"][0]["delta"]
    ]
    # The role chunk carries an empty content; the JSON arrives as ONE delta.
    assert contents == ["", '{"escalate": true, "reason": "loop"}']
    finishes = [
        e["choices"][0]["finish_reason"]
        for e in events
        if isinstance(e, dict) and e["choices"] and e["choices"][0]["finish_reason"]
    ]
    assert finishes == ["stop"]
    # ... and the content delta precedes the finish chunk.
    order = [
        i
        for i, e in enumerate(events)
        if isinstance(e, dict)
        and e["choices"]
        and (e["choices"][0]["delta"].get("content") or e["choices"][0]["finish_reason"])
    ]
    assert order == sorted(order)


def test_stream_retries_and_still_emits_one_content_delta():
    state = JsonFakeState([stream_replies("not json"), stream_replies('{"a": 1}')])
    req = chat_request(response_format={"type": "json_object"}, stream=True)
    spec = chat_request_to_genspec(req, {})

    async def collect():
        from freetoken.server.generation import submit_generation

        uid = await submit_generation(spec, state)
        return [chunk async for chunk in stream_chat_completion_chunks(uid, req, state, spec)]

    events = parse_sse(run(collect()))
    contents = [
        e["choices"][0]["delta"]["content"]
        for e in events
        if isinstance(e, dict) and e["choices"] and "content" in e["choices"][0]["delta"]
    ]
    assert contents == ["", '{"a": 1}']
    assert len(state.sent) == 2


# ---------------------------------------------------------------------------
# /v1/completions keeps the rejection
# ---------------------------------------------------------------------------
def test_completions_still_rejects_json_mode():
    req = CompletionRequest(
        model="m", prompt="hi", response_format={"type": "json_object"}, max_tokens=4
    )
    response = run(handle_completion(req, None, JsonFakeState([]), {}))
    assert response.status_code == 400
    body = json.loads(response.body)
    assert "response_format" in body["error"]["message"]


# ---------------------------------------------------------------------------
# The validator itself (Switchyard's real verdict schemas)
# ---------------------------------------------------------------------------
def test_validator_accepts_a_well_formed_capability_verdict():
    verdict = {
        "crux": "needs a 200K-token repo sweep",
        "primary_rule": "UNC-1",
        "capability_boundary": "uncertain",
        "p_solve": 0.42,
    }
    assert schema_error(verdict, CAPABILITY_SCHEMA) is None


def test_validator_catches_enum_missing_field_extra_field_range_and_length():
    base = {
        "crux": "x",
        "primary_rule": "SUP-1",
        "capability_boundary": "supported",
        "p_solve": 0.9,
    }
    assert schema_error({**base, "primary_rule": "SUP-9"}, CAPABILITY_SCHEMA)
    assert schema_error({k: v for k, v in base.items() if k != "p_solve"}, CAPABILITY_SCHEMA)
    assert schema_error({**base, "extra": 1}, CAPABILITY_SCHEMA)
    assert schema_error({**base, "p_solve": 1.5}, CAPABILITY_SCHEMA)
    assert schema_error({**base, "crux": ""}, CAPABILITY_SCHEMA)
    # A bool is not a number, an int is (JSON has one numeric type).
    assert schema_error({**base, "p_solve": True}, CAPABILITY_SCHEMA)
    assert schema_error({**base, "p_solve": 1}, CAPABILITY_SCHEMA) is None


def test_validator_handles_anyof_arrays_and_nesting():
    schema = {
        "type": "object",
        "properties": {
            "score": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
        },
        "required": ["score", "tags", "meta"],
    }
    ok = {"score": None, "tags": ["a"], "meta": {"n": 3}}
    assert schema_error(ok, schema) is None
    assert schema_error({**ok, "score": "high"}, schema)
    assert schema_error({**ok, "tags": ["a", 2]}, schema)
    assert schema_error({**ok, "meta": {"n": 1.5}}, schema)


def test_coerce_reports_why_it_failed():
    _, error = coerce_json_content("no object here", None)
    assert error and "JSON" in error
    content, error = coerce_json_content('{"escalate": true}', ESCALATION_SCHEMA)
    assert error and "reason" in error
    assert content == '{"escalate": true}'  # the raw text, untouched
