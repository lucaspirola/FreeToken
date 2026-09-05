"""The --trace-dir request trace writer.

CPU-only and torch-free on purpose: ``request_trace`` is loaded straight off the file
rather than as ``freetoken.server.request_trace``, because importing the package would run
``freetoken/server/__init__.py`` -> ``launch`` -> torch. The module is stdlib-only so that
``benchmarks/trace_replay.py`` and this test can read the format on a box with no CUDA and
no engine, and this import is what pins that property: the day someone adds
``from freetoken.core import ...`` to it, this file stops collecting.

What is worth testing here is not "does it write JSON" but the two properties a replay
depends on:

  * the **prefix chain** shares exactly as far as the messages do -- turn k+1 of a
    conversation must agree with turn k for the messages they have in common, and must
    disagree from message 0 when the system prompt differs. That is what lets
    trace_replay regenerate prompts whose prefix-cache hits land where the traced ones did.
  * **no prompt text escapes** unless --trace-include-text was passed.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    path = _ROOT / "python" / "freetoken" / "server" / "request_trace.py"
    spec = importlib.util.spec_from_file_location("_ft_request_trace_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def rt():
    mod = _load()
    yield mod
    mod._reset_for_tests()


def _msgs(*contents, system="you are a helpful assistant"):
    out = [{"role": "system", "content": system}]
    for i, c in enumerate(contents):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": c})
    return out


def _read(rt, tmp_path):
    assert rt.flush(timeout=5.0)
    return list(rt.read_trace(str(tmp_path)))


# --------------------------------------------------------------------------- off


def test_disabled_writes_nothing_and_costs_no_object(rt, tmp_path):
    trace = rt.start("/v1/chat/completions", messages=_msgs("hi"))
    assert trace is rt.NULL
    assert not trace  # falsy, so a call site can branch without a None check
    trace.first_token()
    trace.seal(prompt_tokens=10)
    rt.record(route="/v1/chat/completions", arrival=0.0, messages=_msgs("hi"))
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- writing


def test_round_trip_records_every_field_the_replay_needs(rt, tmp_path):
    rt.configure(str(tmp_path))
    trace = rt.start(
        "/v1/chat/completions",
        messages=_msgs("summarize this"),
        model="nemotron-3.5-lightning",
        session_id="auto:switchyard:abc",
        stream=True,
        max_tokens=256,
        sampling={"temperature": 0.6, "top_p": 0.95, "tools": 12},
    )
    trace.first_token()
    trace.seal(request_id="chatcmpl-7", prompt_tokens=2048, cached_tokens=1536,
               output_tokens=231, reasoning_tokens=40, finish_reason="stop")
    recs = _read(rt, tmp_path)
    assert len(recs) == 1
    r = recs[0]
    assert r["v"] == rt.TRACE_VERSION
    assert r["route"] == "/v1/chat/completions"
    assert r["rid"] == "chatcmpl-7"
    assert r["model"] == "nemotron-3.5-lightning"
    assert r["session"] == "auto:switchyard:abc"
    assert r["stream"] is True
    assert (r["prompt_tokens"], r["cached_tokens"], r["output_tokens"]) == (2048, 1536, 231)
    assert r["reasoning_tokens"] == 40
    assert r["max_tokens"] == 256
    assert r["sampling"]["tools"] == 12
    assert r["finish_reason"] == "stop"
    assert r["status"] == "ok"
    assert r["ttft_ms"] is not None and r["ttft_ms"] >= 0.0
    assert r["duration_ms"] >= r["ttft_ms"]
    assert len(r["msg_chain"]) == len(r["msg_chars"]) == len(r["msg_roles"]) == 2
    assert r["msg_roles"] == ["system", "user"]


def test_no_prompt_text_by_default(rt, tmp_path):
    rt.configure(str(tmp_path))
    secret = "the launch code is hunter2"
    rt.start("/v1/chat/completions", messages=_msgs(secret)).seal(prompt_tokens=9)
    assert rt.flush(timeout=5.0)
    blob = "".join(p.read_text() for p in tmp_path.iterdir())
    assert secret not in blob
    assert "hunter2" not in blob
    assert "messages" not in json.loads(blob.splitlines()[0])


def test_trace_include_text_opts_the_text_in(rt, tmp_path):
    rt.configure(str(tmp_path), include_text=True)
    rt.start("/v1/chat/completions", messages=_msgs("hello there")).seal(prompt_tokens=9)
    r = _read(rt, tmp_path)[0]
    assert r["messages"][1]["content"] == "hello there"


def test_file_is_owner_only(rt, tmp_path):
    rt.configure(str(tmp_path))
    rt.start("/v1/chat/completions", messages=_msgs("x")).seal()
    assert rt.flush(timeout=5.0)
    mode = os.stat(rt.trace_path()).st_mode & 0o777
    assert mode == 0o600, f"trace file is {oct(mode)}; --trace-include-text writes prompts"


def test_seal_is_idempotent(rt, tmp_path):
    """A stream that ends and is then closed must not produce two rows for one request."""
    rt.configure(str(tmp_path))
    trace = rt.start("/v1/chat/completions", messages=_msgs("x"))
    trace.seal(prompt_tokens=5, finish_reason="stop")
    trace.seal(status="abort", error_code="client_disconnect")
    recs = _read(rt, tmp_path)
    assert len(recs) == 1
    assert recs[0]["status"] == "ok"


def test_abort_and_error_rows(rt, tmp_path):
    rt.configure(str(tmp_path))
    rt.start("/v1/chat/completions", messages=_msgs("a")).seal(
        status="abort", error_code="client_disconnect")
    rt.start("/v1/chat/completions", messages=_msgs("b")).seal(
        status="error", error_code="context_length_exceeded")
    recs = _read(rt, tmp_path)
    assert [r["status"] for r in recs] == ["abort", "error"]
    assert recs[1]["error_code"] == "context_length_exceeded"


# --------------------------------------------------------------------------- the chain


def test_chain_shares_exactly_as_far_as_the_messages_do():
    rt = _load()
    turn1 = _msgs("what is 2+2")
    turn2 = _msgs("what is 2+2", "4", "and 3+3")
    c1, _ = rt.prefix_chain(turn1)
    c2, _ = rt.prefix_chain(turn2)
    assert c2[: len(c1)] == c1, "turn 2 must extend turn 1's prefix, not restart it"
    assert len(c2) == len(turn2)

    other = _msgs("what is 2+2", system="you are a terse assistant")
    c3, _ = rt.prefix_chain(other)
    assert c3[0] != c1[0], "a different system prompt must break sharing at message 0"


def test_chain_is_order_and_keyorder_stable():
    rt = _load()
    a = [{"role": "user", "content": "hi"}]
    b = [{"content": "hi", "role": "user"}]
    assert rt.prefix_chain(a)[0] == rt.prefix_chain(b)[0]
    c = [{"role": "user", "content": "ho"}]
    assert rt.prefix_chain(a)[0] != rt.prefix_chain(c)[0]


def test_chars_are_the_canonical_length_not_the_content_length():
    """msg_chars must measure what the hash covers, or trace_replay's length rule and its
    sharing rule would disagree about the same message."""
    rt = _load()
    msg = {"role": "user", "content": "abc"}
    chain, chars = rt.prefix_chain([msg])
    assert chars == [len(json.dumps(msg, sort_keys=True, ensure_ascii=False))]
    assert len(chain[0]) == 16


def test_pydantic_like_messages_are_normalized():
    """Chat messages arrive as pydantic models; the module must not need pydantic to hash
    them, and two models with the same fields must hash the same."""
    rt = _load()

    class FakeMessage:
        def __init__(self, role, content):
            self._d = {"role": role, "content": content}

        def model_dump(self, **_):
            return dict(self._d)

    chain_model, _ = rt.prefix_chain([rt.as_jsonable(FakeMessage("user", "hi"))])
    chain_dict, _ = rt.prefix_chain([{"role": "user", "content": "hi"}])
    assert chain_model == chain_dict
    assert rt.message_roles([rt.as_jsonable(FakeMessage("system", "s"))]) == ["system"]


def test_completions_prompt_is_a_one_message_chain():
    rt = _load()
    chain, chars, roles = (*rt.prefix_chain(["a raw prompt"]), rt.message_roles(["x"]))
    assert len(chain) == 1 and chars == [len("a raw prompt")]
    assert roles == ["prompt"]


# --------------------------------------------------------------------------- reading


def test_read_trace_sorts_by_arrival_and_skips_junk(rt, tmp_path):
    (tmp_path / "trace-a.jsonl").write_text(
        json.dumps({"v": rt.TRACE_VERSION, "t": 20.0, "rid": "b"}) + "\n"
        "not json\n"
        + json.dumps({"v": 999, "t": 1.0, "rid": "wrong-version"}) + "\n"
    )
    (tmp_path / "trace-b.jsonl").write_text(
        json.dumps({"v": rt.TRACE_VERSION, "t": 10.0, "rid": "a"}) + "\n"
    )
    recs = list(rt.read_trace(str(tmp_path)))
    assert [r["rid"] for r in recs] == ["a", "b"]
