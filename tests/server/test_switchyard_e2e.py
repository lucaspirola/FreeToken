"""GPU-free tests for the ``scripts/switchyard_e2e.py`` helpers.

The e2e script's judgement is only as good as three pure functions: the SSE
reassembler (a tool call or a JSON verdict graded per-chunk looks broken when it
is not), the verdict validator (must accept exactly what Switchyard's
``EscalationVerdict`` schema accepts), and the soak report parser (a report the
parser cannot read must never read as a pass). Those are tested here; the parts
that need a GPU server are not.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "switchyard_e2e.py"


def _load():
    # scripts/ is not a package; load the file by path. It must be registered in
    # sys.modules before exec_module because @dataclass resolves annotations
    # through the defining module.
    spec = importlib.util.spec_from_file_location("switchyard_e2e", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["switchyard_e2e"] = module
    spec.loader.exec_module(module)
    return module


sy = _load()


def _sse(*payloads: object) -> str:
    return "".join(f"data: {json.dumps(p)}\n\n" for p in payloads) + "data: [DONE]\n\n"


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


# --------------------------------------------------------------------------- #
# SSE parsing / concatenation
# --------------------------------------------------------------------------- #


def test_sse_events_decodes_payloads_and_done():
    raw = _sse({"a": 1}, {"b": 2})
    events = sy.sse_events(raw)
    assert events == [{"a": 1}, {"b": 2}, {"__done__": True}]


def test_sse_events_tolerates_crlf_and_trailing_noise():
    raw = "data: {\"a\": 1}\r\n\r\n: keepalive\r\n\r\ndata: [DONE]\r\n\r\n"
    assert sy.sse_events(raw) == [{"a": 1}, {"__done__": True}]


def test_sse_events_surfaces_unparseable_payload():
    assert sy.sse_events("data: not json\n\n") == [{"__raw__": "not json"}]


def test_concat_stream_joins_content_across_chunks():
    raw = _sse(
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": '{"escal'}),
        _chunk({"content": 'ate": true, "reason": "hard"}'}),
        _chunk({}, finish="stop"),
    )
    merged = sy.concat_stream(sy.sse_events(raw))
    assert merged["content"] == '{"escalate": true, "reason": "hard"}'
    assert merged["finish_reason"] == "stop"
    assert merged["done"] is True
    # Graded whole, the split verdict is valid; graded per-chunk it is not.
    assert sy.validate_json_schema(
        sy.extract_json_object(merged["content"]), sy.ESCALATION_VERDICT_SCHEMA
    ) == []


def test_concat_stream_splits_reasoning_from_content():
    raw = _sse(
        _chunk({"reasoning_content": "let me "}),
        _chunk({"reasoning_content": "think"}),
        _chunk({"content": "42"}),
        _chunk({}, finish="stop"),
    )
    merged = sy.concat_stream(sy.sse_events(raw))
    assert merged["reasoning_content"] == "let me think"
    assert merged["content"] == "42"


def test_concat_stream_accepts_the_reasoning_alias():
    # Switchyard reads `reasoning_content` or `reasoning`; both fold into one field.
    merged = sy.concat_stream(sy.sse_events(_sse(_chunk({"reasoning": "hm"}))))
    assert merged["reasoning_content"] == "hm"


def test_concat_stream_reassembles_tool_call_arguments_by_index():
    raw = _sse(
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "list_files", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path":'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "/var/log"}'}}]}),
        _chunk({}, finish="tool_calls"),
    )
    merged = sy.concat_stream(sy.sse_events(raw))
    assert merged["finish_reason"] == "tool_calls"
    assert len(merged["tool_calls"]) == 1
    call = merged["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["name"] == "list_files"
    assert json.loads(call["arguments"]) == {"path": "/var/log"}


def test_concat_stream_keeps_parallel_tool_calls_separate():
    raw = _sse(
        _chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "a", "function": {"name": "x", "arguments": "{}"}},
                    {"index": 1, "id": "b", "function": {"name": "y", "arguments": "{}"}},
                ]
            }
        ),
        _chunk({}, finish="tool_calls"),
    )
    merged = sy.concat_stream(sy.sse_events(raw))
    assert [c["name"] for c in merged["tool_calls"]] == ["x", "y"]


def test_concat_stream_reports_an_error_event_first():
    # The context-overflow contract: the error must be the FIRST event, never
    # after a role chunk, or the router cannot retarget.
    raw = (
        'data: {"error": {"message": "too long", "type": "invalid_request_error", '
        '"code": "context_length_exceeded"}}\n\ndata: [DONE]\n\n'
    )
    merged = sy.concat_stream(sy.sse_events(raw))
    assert merged["error"]["code"] == "context_length_exceeded"
    assert merged["first_event"]["error"]["code"] == "context_length_exceeded"
    assert merged["content"] == ""


def test_concat_stream_captures_usage():
    raw = _sse(
        _chunk({"content": "ok"}),
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 96},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    )
    merged = sy.concat_stream(sy.sse_events(raw))
    assert merged["usage"]["prompt_tokens_details"]["cached_tokens"] == 96


# --------------------------------------------------------------------------- #
# Verdict schema
# --------------------------------------------------------------------------- #


def test_escalation_schema_matches_switchyards_asset():
    # Guards against drift from crates/libsy/src/prompts/escalation/schema.json:
    # a verdict FreeToken satisfies but Switchyard rejects is a silent no-escalate.
    schema = sy.ESCALATION_VERDICT_SCHEMA
    assert schema["required"] == ["escalate", "reason"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["escalate"]["type"] == "boolean"
    assert schema["properties"]["reason"]["type"] == "string"
    wrapper = sy.ESCALATION_RESPONSE_FORMAT
    assert wrapper["type"] == "json_schema"
    assert wrapper["json_schema"]["name"] == "EscalationVerdict"
    assert wrapper["json_schema"]["strict"] is True
    assert wrapper["json_schema"]["schema"] is schema


def test_valid_verdict_passes():
    value = {"escalate": True, "reason": "the run is stuck retrying the same edit"}
    assert sy.validate_json_schema(value, sy.ESCALATION_VERDICT_SCHEMA) == []


@pytest.mark.parametrize(
    "value, needle",
    [
        ({"escalate": True}, "missing required property 'reason'"),
        ({"reason": "x"}, "missing required property 'escalate'"),
        ({"escalate": "yes", "reason": "x"}, "expected boolean"),
        ({"escalate": True, "reason": 3}, "expected string"),
        ({"escalate": True, "reason": "x", "extra": 1}, "unexpected property 'extra'"),
        ([], "expected object"),
    ],
)
def test_invalid_verdicts_are_rejected(value, needle):
    errors = sy.validate_json_schema(value, sy.ESCALATION_VERDICT_SCHEMA)
    assert errors, f"expected a rejection for {value!r}"
    assert any(needle in e for e in errors), errors


def test_booleans_are_not_numbers():
    schema = {"type": "object", "properties": {"p": {"type": "number"}}, "required": ["p"]}
    assert sy.validate_json_schema({"p": True}, schema)
    assert sy.validate_json_schema({"p": 0.5}, schema) == []
    assert sy.validate_json_schema({"p": 3}, schema) == []


def test_enum_and_nested_items_are_checked():
    schema = {
        "type": "object",
        "properties": {
            "rule": {"type": "string", "enum": ["SUP-1", "none"]},
            "items": {"type": "array", "items": {"type": "integer"}},
        },
    }
    assert sy.validate_json_schema({"rule": "none", "items": [1, 2]}, schema) == []
    assert sy.validate_json_schema({"rule": "SUP-9"}, schema)
    assert sy.validate_json_schema({"items": [1, "x"]}, schema)


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        '{"escalate": false, "reason": "fine"}',
        '  {"escalate": false, "reason": "fine"}  ',
        'Here is the verdict: {"escalate": false, "reason": "fine"}',
        '{"escalate": false, "reason": "fine"} trailing prose',
    ],
)
def test_extract_json_object_finds_the_verdict(text):
    assert sy.extract_json_object(text) == {"escalate": False, "reason": "fine"}


def test_extract_json_object_respects_braces_inside_strings():
    text = 'note: {"escalate": true, "reason": "saw a } brace"}'
    assert sy.extract_json_object(text)["reason"] == "saw a } brace"


def test_extract_json_object_raises_when_there_is_none():
    with pytest.raises(ValueError):
        sy.extract_json_object("no object here")
    with pytest.raises(ValueError):
        sy.extract_json_object('{"escalate": true')


# --------------------------------------------------------------------------- #
# Soak report parsing
# --------------------------------------------------------------------------- #

_PASS_REPORT = """\
Soak started: model=switchyard/passthrough duration=1200s concurrency=16 scenarios=5 results=/tmp/r
[2026-09-04T01:00:00Z] progress=600s/1200s(50%) reqs=812 interval=812 errors=0(0.0000%) rps=1.4 \
p95_ms=901.2 health=ok rss_mib=412.0 status=OK
Soak PASS: requests=1624 error_rate=0.0000% p95_ms=912.5 summary=/tmp/r/summary.json
"""

_FAIL_REPORT = """\
Soak FAIL: requests=200 error_rate=2.5000% p95_ms=1804.0 summary=/tmp/r/summary.json
- request error rate 2.5000% exceeded the 0.0000% limit
- 3 liveness checks failed
"""


def test_parse_soak_report_reads_a_pass():
    report = sy.parse_soak_report(_PASS_REPORT)
    assert report["passed"] is True
    assert report["requests"] == 1624
    assert report["error_rate"] == 0.0
    assert report["errors"] == 0
    assert report["p95_ms"] == 912.5
    assert report["summary_path"] == "/tmp/r/summary.json"
    assert report["reasons"] == []


def test_parse_soak_report_reads_a_fail_with_reasons():
    report = sy.parse_soak_report(_FAIL_REPORT)
    assert report["passed"] is False
    assert report["requests"] == 200
    assert report["error_rate"] == pytest.approx(0.025)
    assert report["errors"] == 5
    assert report["reasons"] == [
        "request error rate 2.5000% exceeded the 0.0000% limit",
        "3 liveness checks failed",
    ]


def test_parse_soak_report_without_a_verdict_is_not_a_pass():
    # A soak that died mid-run prints progress lines and nothing else. The word
    # "errors=0" on a progress line must not be mistaken for a passing verdict.
    truncated = "\n".join(_PASS_REPORT.splitlines()[:2])
    report = sy.parse_soak_report(truncated)
    assert report["passed"] is None
    assert report["requests"] is None


def test_soak_summary_verdict_reads_the_structured_summary():
    summary = {
        "passed": False,
        "failure_reasons": ["no inference requests completed"],
        "requests": 40,
        "successes": 38,
        "failures": 2,
        "error_rate": 0.05,
        "latency_p95_ms": 1234.5,
    }
    verdict = sy.soak_summary_verdict(summary)
    assert verdict["passed"] is False
    assert verdict["requests"] == 40
    assert verdict["errors"] == 2
    assert verdict["error_rate"] == 0.05
    assert verdict["p95_ms"] == 1234.5
    assert verdict["reasons"] == ["no inference requests completed"]


def test_soak_summary_verdict_derives_a_missing_rate():
    verdict = sy.soak_summary_verdict({"passed": True, "requests": 100, "failures": 1})
    assert verdict["error_rate"] == pytest.approx(0.01)


def test_soak_summary_verdict_without_passed_is_unknown():
    assert sy.soak_summary_verdict({"requests": 10, "failures": 0})["passed"] is None


# --------------------------------------------------------------------------- #
# routes.toml
# --------------------------------------------------------------------------- #


def test_render_routes_toml_shape():
    tomllib = pytest.importorskip("tomllib")
    text = sy.render_routes_toml(
        freetoken_url="http://127.0.0.1:30000", model="nemotron-3.5-lightning"
    )
    config = tomllib.loads(text)
    assert config["schema_version"] == 1
    client = config["llm_clients"]["freetoken"]
    assert client["format"] == "openai_chat"
    assert client["base_url"] == "http://127.0.0.1:30000/v1"
    # No api_key_env by default: switchyard-server refuses to start when the named
    # variable is unset, and FreeToken needs no bearer token locally.
    assert "api_key_env" not in client

    targets = config["targets"]
    # Switchyard keeps one target per (client, model id); duplicate ids silently
    # drop a tier, so the two tiers must not share an id.
    assert targets["lightning"]["id"] != targets["lightning_fast"]["id"]
    assert (
        targets["lightning"]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        is True
    )
    fast = targets["lightning_fast"]["extra_body"]["chat_template_kwargs"]
    assert fast == {"enable_thinking": False, "force_nonempty_content": True}

    routes = config["routes"]
    assert routes["passthrough"]["type"] == "passthrough"
    assert routes["passthrough"]["target"] == "lightning"
    stage = routes["stage"]
    assert stage["type"] == "stage_router"
    assert stage["picker"] == "efficient_first"
    assert stage["capable_target"] == "lightning"
    assert stage["efficient_target"] == "lightning_fast"
    assert stage["classifier"]["target"] == "lightning_fast"
    assert stage["classifier"]["response_format_type"] == "json_schema"
    assert {r["id"] for r in routes.values()} == set(sy.DEFAULT_ROUTES)


def test_render_routes_toml_appends_v1_once():
    for url in ("http://h:1", "http://h:1/", "http://h:1/v1"):
        text = sy.render_routes_toml(freetoken_url=url, model="m")
        assert 'base_url = "http://h:1/v1"' in text


def test_render_routes_toml_emits_api_key_env_when_asked():
    text = sy.render_routes_toml(
        freetoken_url="http://h:1", model="m", api_key_env="FREETOKEN_API_KEY"
    )
    assert 'api_key_env = "FREETOKEN_API_KEY"' in text


def test_soak_scenarios_are_the_planned_set():
    assert sy.SOAK_SCENARIOS == (
        "prefix-reuse",
        "growing-conversation",
        "tool-call-burst",
        "large-tool-catalog",
        "long-context",
    )


def test_parser_defaults_and_subcommands():
    parser = sy.build_parser()
    args = parser.parse_args(["contract"])
    assert args.command == "contract" and args.func is sy.cmd_contract
    args = parser.parse_args(["soak", "--duration", "5m", "--concurrency", "8"])
    assert args.duration == "5m" and args.concurrency == 8
    assert args.context_window_tokens == 131072 and args.max_error_rate == 0.0
    assert parser.parse_args(["agents"]).func is sy.cmd_agents
