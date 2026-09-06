"""``--pooled-sink-dir`` / ``kv_transfer_params.pooled_sink``: the JSONL side channel
for inline pooled hidden states.

Pinned: the request-field rules (one path segment, needs ``pooling``, needs the server
flag), the line written for the plain and the streaming path (keys, decodable base64,
rendered-prompt hash, session ids), the ``default`` subdirectory, that a write failure
is logged and never fails the response, and that the append happens under ``flock``.
Everything drives the real adapter and sink code; only the engine is faked.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from freetoken.message.frontend import UserReply
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.openai_api import handle_chat_completion, stream_chat_completion_chunks
from freetoken.server.pooled_sink import validate_pooled_sink, write_pooled_line

from .test_hidden_states_pooled import HIDDEN, _pooled_payload, decode
from .test_hidden_states_probe import ProbeState, final_reply, probe_request
from .test_openai_api import _collect, parse_sse, run

NUM_LAYERS = 12


def _state(replies=None, sink_dir=None) -> ProbeState:
    state = ProbeState(replies or [final_reply()], num_layers=NUM_LAYERS, hidden_states_dir=None)
    state.config.pooled_sink_dir = sink_dir
    return state


def _request(headers: dict[str, str] | None = None):
    return SimpleNamespace(headers=headers or {})


def _lines(root, sink="default") -> list[dict]:
    with open(os.path.join(root, sink, "pooled.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_pooled_sink_is_a_typed_optional_field():
    assert probe_request(pooling="mean", pooled_sink="run-1").kv_transfer_params.pooled_sink == "run-1"
    assert probe_request(pooling="mean").kv_transfer_params.pooled_sink is None


@pytest.mark.parametrize("bad", [".", "..", "", "a/b", "../x", "a b", "x" * 65, "é"])
def test_pooled_sink_must_be_one_path_segment(bad, tmp_path):
    params = SimpleNamespace(pooling="mean", pooled_sink=bad)
    with pytest.raises(ValueError, match="single path segment"):
        validate_pooled_sink(params, str(tmp_path))
    state = _state(sink_dir=str(tmp_path))
    response = run(handle_chat_completion(
        probe_request(pooling="mean", pooled_sink=bad), None, state, {}
    ))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["param"] == "kv_transfer_params"
    assert state.sent is None  # refused before submit


def test_pooled_sink_requires_pooling(tmp_path):
    state = _state(sink_dir=str(tmp_path))
    response = run(handle_chat_completion(
        probe_request(pooled_sink="run-1", layer_ids=[0, 1]), None, state, {}
    ))
    assert response.status_code == 400
    assert b"requires kv_transfer_params.pooling" in response.body


def test_pooled_sink_needs_the_server_flag():
    state = _state(sink_dir=None)
    response = run(handle_chat_completion(
        probe_request(pooling="mean", pooled_sink="run-1"), None, state, {}
    ))
    assert response.status_code == 400
    assert b"--pooled-sink-dir" in response.body
    assert json.loads(response.body)["error"]["param"] == "kv_transfer_params"


def test_no_flag_and_no_sink_name_writes_nothing(tmp_path):
    """Pooling alone on a server without the flag is the plain inline feature."""
    pooled = _pooled_payload()
    state = _state([final_reply(kv_transfer_params={"pooled": pooled})], sink_dir=None)
    payload = run(handle_chat_completion(
        probe_request(pooling="mean", layer_ids=[3, 7]), None, state, {}
    ))
    assert payload["kv_transfer_params"] == {"pooled": pooled}
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------------- #
# The line: plain and streaming
# --------------------------------------------------------------------------- #
EXPECTED_KEYS = [
    "ts", "request_id", "session_id", "x_switchyard_session_id", "model", "prompt_tokens",
    "layer_ids", "hidden", "dtype", "mean", "prompt_sha256",
]


def test_non_stream_writes_one_line_and_keeps_the_response(tmp_path):
    pooled = _pooled_payload()
    state = _state([final_reply(kv_transfer_params={"pooled": pooled})], sink_dir=str(tmp_path))
    payload = run(handle_chat_completion(
        probe_request(pooling="mean", layer_ids=[3, 7], pooled_sink="run-1"),
        _request({"x-switchyard-session-id": " sess-9 "}), state, {},
    ))
    # Inline response unchanged.
    assert payload["kv_transfer_params"] == {"pooled": pooled}

    lines = _lines(tmp_path, "run-1")
    assert len(lines) == 1
    line = lines[0]
    assert list(line) == EXPECTED_KEYS
    assert line["request_id"] == payload["id"] == "chatcmpl-42"
    assert line["session_id"] is None  # a probe binds no lease
    assert line["x_switchyard_session_id"] == "sess-9"
    assert line["model"] == "client-model"
    assert line["prompt_tokens"] == 5 and line["layer_ids"] == [3, 7]
    assert line["hidden"] == HIDDEN and line["dtype"] == "float32"
    assert "last" not in line
    np.testing.assert_array_equal(
        decode(line["mean"], line["layer_ids"]), np.arange(2 * HIDDEN, dtype="<f4").reshape(2, HIDDEN)
    )
    # The hash is of the rendered prompt string the frontend tokenizer produces.
    assert line["prompt_sha256"] == hashlib.sha256(b"rendered").hexdigest()
    assert isinstance(line["ts"], float)


def test_stream_writes_the_terminal_chunks_pooled(tmp_path):
    pooled = {**_pooled_payload(), "last": _pooled_payload()["mean"]}
    state = _state(
        [
            UserReply(uid=42, incremental_output="The", finished=False),
            UserReply(
                uid=42, incremental_output="", finished=True, finish_reason="length",
                kv_transfer_params={"pooled": pooled},
            ),
        ],
        sink_dir=str(tmp_path),
    )
    req = ChatCompletionRequest(
        model="client-model", messages=[{"role": "user", "content": "score me"}],
        max_tokens=1, stream=True,
        kv_transfer_params={"pooling": "both", "layer_ids": [3, 7], "pooled_sink": "s.1"},
    )
    async def passthrough(gen, request, uid, session_id=None):
        async for chunk in gen:
            yield chunk

    state.stream_with_cancellation = passthrough
    response = run(handle_chat_completion(
        req, _request({"x-switchyard-session-id": "sess-1"}), state, {}
    ))
    events = parse_sse(run(_collect(response.body_iterator)))
    terminal = [e for e in events if isinstance(e, dict) and "kv_transfer_params" in e]
    assert len(terminal) == 1 and terminal[0]["kv_transfer_params"] == {"pooled": pooled}

    lines = _lines(tmp_path, "s.1")
    assert len(lines) == 1
    line = lines[0]
    assert line["request_id"] == terminal[0]["id"]
    assert line["x_switchyard_session_id"] == "sess-1"
    assert "mean" in line and "last" in line
    assert base64.b64decode(line["last"]) == base64.b64decode(pooled["last"])
    assert line["prompt_sha256"] == hashlib.sha256(b"rendered").hexdigest()


def test_stream_generator_without_a_sink_writes_nothing(tmp_path):
    state = _state(
        [final_reply(kv_transfer_params={"pooled": _pooled_payload()})], sink_dir=str(tmp_path)
    )
    req = ChatCompletionRequest(
        model="client-model", messages=[{"role": "user", "content": "score me"}],
        max_tokens=1, stream=True, kv_transfer_params={"pooling": "mean", "layer_ids": [3, 7]},
    )
    run(_collect(stream_chat_completion_chunks(42, req, state)))
    assert os.listdir(tmp_path) == []


def test_sink_subdirectory_defaults_and_lines_append(tmp_path):
    pooled = _pooled_payload()
    for _ in range(2):
        state = _state([final_reply(kv_transfer_params={"pooled": pooled})], sink_dir=str(tmp_path))
        run(handle_chat_completion(
            probe_request(pooling="mean", layer_ids=[3, 7]), None, state, {}
        ))
    assert os.listdir(tmp_path) == ["default"]
    lines = _lines(tmp_path)
    assert len(lines) == 2 and lines[0]["x_switchyard_session_id"] is None
    mode = os.stat(tmp_path / "default" / "pooled.jsonl").st_mode & 0o777
    assert mode == 0o666 & ~_umask()


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


def test_prompt_hash_is_null_without_a_frontend_tokenizer(tmp_path):
    state = _state([final_reply(kv_transfer_params={"pooled": _pooled_payload()})], sink_dir=str(tmp_path))
    from .test_switchyard_wire import TokenizingState

    # The tokenizer is a method of the class; hide it for this one call.
    with patch.object(TokenizingState, "frontend_tokenizer", None):
        run(handle_chat_completion(
            probe_request(pooling="mean", layer_ids=[3, 7]), None, state, {}
        ))
    assert _lines(tmp_path)[0]["prompt_sha256"] is None


# --------------------------------------------------------------------------- #
# Failure isolation and locking
# --------------------------------------------------------------------------- #
def test_write_failure_is_logged_and_the_response_is_intact(tmp_path, caplog):
    pooled = _pooled_payload()
    root = tmp_path / "root"
    root.mkdir()
    state = _state([final_reply(kv_transfer_params={"pooled": pooled})], sink_dir=str(root))
    root.rmdir()
    (tmp_path / "root").write_text("not a directory")  # makedirs fails: NotADirectoryError
    payload = run(handle_chat_completion(
        probe_request(pooling="mean", layer_ids=[3, 7], pooled_sink="x"), None, state, {}
    ))
    assert payload["kv_transfer_params"] == {"pooled": pooled}
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert any("pooled sink write failed" in r.getMessage() for r in caplog.records)


def test_the_line_is_written_under_an_exclusive_flock(tmp_path):
    calls: list[tuple[int, int]] = []
    real_flock = fcntl.flock

    def spy(fd, op):
        calls.append((fd, op))
        return real_flock(fd, op)

    with patch("freetoken.server.pooled_sink.fcntl.flock", side_effect=spy):
        path = write_pooled_line(str(tmp_path), "k", {"a": 1})
    assert path == str(tmp_path / "k" / "pooled.jsonl")
    assert [op for _, op in calls] == [fcntl.LOCK_EX, fcntl.LOCK_UN]
    assert len({fd for fd, _ in calls}) == 1
    assert _lines(tmp_path, "k") == [{"a": 1}]
