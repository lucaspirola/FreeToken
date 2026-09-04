"""The hidden-state probe's wire contract (Switchyard's prefill router).

Three surfaces are pinned here: the typed ``kv_transfer_params`` request field and the
validation it drives (server root, path escape, layer-id contiguity, prompt cap), the
artifact writer's round trip through ``safetensors``, and the response echo -- the path
must come back under ``kv_transfer_params.hidden_states_path``, because that is the only
thing Switchyard's scorer reads. See docs/switchyard.md, "Hidden-state probe target".
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from freetoken.hidden_states import (
    DEFAULT_MAX_TOKENS,
    HiddenStateCapture,
    HiddenStateCollector,
    HiddenStateSpec,
    resolve_hidden_states_dir,
    validate_layer_ids,
    write_hidden_states,
)
from freetoken.message.frontend import UserReply
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.openai_api import handle_chat_completion
from safetensors import safe_open

from .test_openai_api import run
from .test_switchyard_wire import TokenizingState

OVERFLOW_CODE = "context_length_exceeded"


class ProbeState(TokenizingState):
    """A frontend state with ``--hidden-states-dir`` and a 52-block model."""

    def __init__(self, *args, hidden_states_dir=None, max_tokens=DEFAULT_MAX_TOKENS,
                 num_layers: int = 52, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.hidden_states_dir = hidden_states_dir
        self.config.hidden_states_max_tokens = max_tokens
        self.config.model_config = SimpleNamespace(num_layers=num_layers)


def probe_request(**kv) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "score me"}],
        max_tokens=1,
        kv_transfer_params=kv,
    )


def final_reply(**kwargs) -> UserReply:
    return UserReply(
        uid=42, incremental_output="ok", finished=True, finish_reason="stop", **kwargs
    )


# --------------------------------------------------------------------------- #
# Request model
# --------------------------------------------------------------------------- #
def test_kv_transfer_params_is_typed_not_swallowed_by_extra():
    req = probe_request(
        hidden_states_path="/tmp/probe", layer_ids=[0, 1], include_output_tokens=False
    )
    assert req.kv_transfer_params is not None
    assert req.kv_transfer_params.hidden_states_path == "/tmp/probe"
    assert req.kv_transfer_params.layer_ids == [0, 1]
    assert req.kv_transfer_params.include_output_tokens is False


def test_kv_transfer_params_absent_is_none():
    assert ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    ).kv_transfer_params is None


def test_kv_transfer_params_rejects_unknown_keys():
    with pytest.raises(Exception):
        probe_request(hidden_states_path="/tmp/probe", shared_storage_path="/tmp/x")


def test_other_endpoints_ignore_kv_transfer_params():
    """Only chat completions declares it; elsewhere it lands in ``extra`` untyped."""
    from freetoken.server.api_models import CompletionRequest

    req = CompletionRequest(
        model="m", prompt="hi", kv_transfer_params={"hidden_states_path": "/tmp/probe"}
    )
    assert "kv_transfer_params" not in type(req).model_fields
    assert req.model_extra["kv_transfer_params"] == {"hidden_states_path": "/tmp/probe"}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_layer_ids_must_be_contiguous_from_zero():
    assert validate_layer_ids([0, 1, 2, 3]) == [0, 1, 2, 3]
    for bad in ([1, 2, 3], [0, 2, 3], [3, 2, 1, 0], [0, 1, 1], []):
        with pytest.raises(ValueError):
            validate_layer_ids(bad)


def test_layer_ids_reject_non_integers_and_bools():
    for bad in (["0"], [True, False], "012", 3):
        with pytest.raises(ValueError):
            validate_layer_ids(bad)


def test_layer_ids_reject_more_layers_than_the_model_has():
    assert validate_layer_ids([0, 1], num_layers=2) == [0, 1]
    with pytest.raises(ValueError):
        validate_layer_ids([0, 1, 2], num_layers=2)


def test_directory_requires_the_server_flag(tmp_path):
    with pytest.raises(ValueError, match="--hidden-states-dir"):
        resolve_hidden_states_dir(str(tmp_path), None)


def test_directory_defaults_to_the_root_and_accepts_a_subdirectory(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    assert resolve_hidden_states_dir(None, str(root)) == os.path.realpath(root)
    assert resolve_hidden_states_dir(str(root / "sub"), str(root)) == os.path.realpath(
        root / "sub"
    )
    assert resolve_hidden_states_dir("sub", str(root)) == os.path.realpath(root / "sub")


def test_directory_refuses_escaping_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    for escape in (str(outside), str(root / ".." / "outside"), "../outside"):
        with pytest.raises(ValueError, match="outside"):
            resolve_hidden_states_dir(escape, str(root))


def test_directory_refuses_a_symlink_out_of_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        resolve_hidden_states_dir(str(root / "link"), str(root))


def test_directory_refuses_a_file_and_a_missing_path(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file").write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        resolve_hidden_states_dir(str(root / "file"), str(root))
    with pytest.raises(ValueError):
        resolve_hidden_states_dir(str(root / "nope"), str(root))


# --------------------------------------------------------------------------- #
# Writer round trip
# --------------------------------------------------------------------------- #
def test_artifact_round_trips_through_safetensors(tmp_path):
    hidden = torch.randn(7, 4, 16).to(torch.bfloat16)
    tokens = torch.arange(100, 107, dtype=torch.int64)
    path = write_hidden_states(str(tmp_path), hidden, tokens)

    assert path.startswith(str(tmp_path) + os.sep)
    assert path.endswith(".safetensors")
    with safe_open(path, framework="pt") as handle:
        assert sorted(handle.keys()) == ["hidden_states", "token_ids"]
        loaded = handle.get_tensor("hidden_states")
        ids = handle.get_tensor("token_ids")
    # Shape/dtype are exactly what Switchyard's token_mean_per_layer asserts.
    assert loaded.shape == (7, 4, 16)
    assert loaded.dtype is torch.bfloat16
    assert ids.dtype is torch.int64
    assert ids.shape == (7,)
    assert torch.equal(loaded, hidden)
    assert torch.equal(ids, tokens)


def test_writer_rejects_a_mismatched_token_count(tmp_path):
    with pytest.raises(ValueError, match="token_ids"):
        write_hidden_states(
            str(tmp_path), torch.zeros(3, 2, 4), torch.zeros(4, dtype=torch.int64)
        )
    with pytest.raises(ValueError, match="prompt_tokens"):
        write_hidden_states(
            str(tmp_path), torch.zeros(3, 4), torch.zeros(3, dtype=torch.int64)
        )


def test_writer_names_every_artifact_uniquely(tmp_path):
    hidden = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
    tokens = torch.zeros(1, dtype=torch.int64)
    paths = {write_hidden_states(str(tmp_path), hidden, tokens) for _ in range(4)}
    assert len(paths) == 4


# --------------------------------------------------------------------------- #
# Capture: chunk concatenation
# --------------------------------------------------------------------------- #
def test_capture_concatenates_chunks_in_order():
    spec = HiddenStateSpec(directory="/tmp", layer_ids=[0, 1, 2])
    capture = HiddenStateCapture(spec, hidden_size=5)

    first = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5)
    second = torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5) + 100
    capture.begin_chunk(torch.tensor([7, 8], dtype=torch.int32))
    for layer in range(3):
        capture.write(layer, first + layer)
    capture.begin_chunk(torch.tensor([9, 10, 11], dtype=torch.int32))
    for layer in range(3):
        capture.write(layer, second + layer)

    hidden, tokens = capture.finish()
    assert hidden.shape == (5, 3, 5)
    assert hidden.dtype is torch.bfloat16
    assert torch.equal(tokens, torch.tensor([7, 8, 9, 10, 11], dtype=torch.int64))
    # Token 0 layer 1 came from the first chunk, token 2 layer 1 from the second.
    assert torch.equal(hidden[0, 1], (first[0] + 1).to(torch.bfloat16))
    assert torch.equal(hidden[2, 1], (second[0] + 1).to(torch.bfloat16))


def test_capture_keeps_only_the_requested_layers():
    capture = HiddenStateCapture(
        HiddenStateSpec(directory="/tmp", layer_ids=[0, 1]), hidden_size=3
    )
    capture.begin_chunk(torch.tensor([1], dtype=torch.int32))
    capture.write(0, torch.ones(1, 3))
    capture.write(1, torch.full((1, 3), 2.0))
    capture.write(2, torch.full((1, 3), 99.0))  # beyond layer_ids: dropped
    hidden, _ = capture.finish()
    assert hidden.shape == (1, 2, 3)
    assert hidden.max().item() == 2.0


# --------------------------------------------------------------------------- #
# Handler: end-to-end through the OpenAI adapter
# --------------------------------------------------------------------------- #
def test_probe_response_carries_the_written_path(tmp_path):
    state = ProbeState(
        [final_reply(hidden_states_path=str(tmp_path / "abc.safetensors"))],
        hidden_states_dir=str(tmp_path),
    )
    payload = run(
        handle_chat_completion(probe_request(hidden_states_path=str(tmp_path)), None, state, {})
    )
    assert payload["kv_transfer_params"] == {
        "hidden_states_path": str(tmp_path / "abc.safetensors")
    }


def test_probe_defaults_to_every_block_and_forces_a_full_recompute(tmp_path):
    state = ProbeState([final_reply()], hidden_states_dir=str(tmp_path), num_layers=52)
    run(handle_chat_completion(probe_request(hidden_states_path=str(tmp_path)), None, state, {}))
    sent = state.sent
    assert sent.hidden_states.layer_ids == list(range(52))
    assert sent.hidden_states.directory == os.path.realpath(tmp_path)
    assert sent.no_prefix_cache is True
    # A probe binds no session lease: it refuses prefix reuse and has no next turn.
    assert sent.session_id is None


def test_probe_layer_subset_is_forwarded(tmp_path):
    state = ProbeState([final_reply()], hidden_states_dir=str(tmp_path))
    run(
        handle_chat_completion(
            probe_request(hidden_states_path=str(tmp_path), layer_ids=[0, 1, 2]),
            None, state, {},
        )
    )
    assert state.sent.hidden_states.layer_ids == [0, 1, 2]


def test_ordinary_request_stays_untouched(tmp_path):
    state = ProbeState([final_reply()], hidden_states_dir=str(tmp_path))
    payload = run(
        handle_chat_completion(
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=4
            ),
            None, state, {},
        )
    )
    assert "kv_transfer_params" not in payload
    assert state.sent.hidden_states is None
    assert state.sent.no_prefix_cache is False


def test_probe_without_the_server_flag_is_a_400(tmp_path):
    state = ProbeState([final_reply()], hidden_states_dir=None)
    response = run(
        handle_chat_completion(probe_request(hidden_states_path=str(tmp_path)), None, state, {})
    )
    assert response.status_code == 400
    assert b"--hidden-states-dir" in response.body


def test_probe_outside_the_root_is_a_400(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    state = ProbeState([final_reply()], hidden_states_dir=str(root))
    response = run(
        handle_chat_completion(probe_request(hidden_states_path=str(outside)), None, state, {})
    )
    assert response.status_code == 400
    assert b"outside" in response.body


def test_probe_with_non_contiguous_layer_ids_is_a_400(tmp_path):
    state = ProbeState([final_reply()], hidden_states_dir=str(tmp_path))
    response = run(
        handle_chat_completion(
            probe_request(hidden_states_path=str(tmp_path), layer_ids=[10, 11]),
            None, state, {},
        )
    )
    assert response.status_code == 400
    assert b"contiguous" in response.body


def test_probe_over_the_token_cap_is_a_400_context_length_exceeded(tmp_path):
    state = ProbeState(
        [final_reply()], hidden_states_dir=str(tmp_path), max_tokens=16,
        prompt_tokens=17, max_seq_len=131072,
    )
    response = run(
        handle_chat_completion(probe_request(hidden_states_path=str(tmp_path)), None, state, {})
    )
    assert response.status_code == 400
    assert OVERFLOW_CODE.encode() in response.body
    assert b"--hidden-states-max-tokens" in response.body


def test_probe_at_the_token_cap_is_served(tmp_path):
    state = ProbeState(
        [final_reply()], hidden_states_dir=str(tmp_path), max_tokens=16,
        prompt_tokens=16, max_seq_len=131072,
    )
    payload = run(
        handle_chat_completion(probe_request(hidden_states_path=str(tmp_path)), None, state, {})
    )
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_probe_cap_applies_even_with_the_context_preflight_disabled(tmp_path):
    state = ProbeState(
        [final_reply()], hidden_states_dir=str(tmp_path), max_tokens=16,
        prompt_tokens=17, max_seq_len=131072,
    )
    state.config.context_preflight = False
    response = run(
        handle_chat_completion(probe_request(hidden_states_path=str(tmp_path)), None, state, {})
    )
    assert response.status_code == 400
    assert b"--hidden-states-max-tokens" in response.body


# --------------------------------------------------------------------------- #
# Collector lifecycle
# --------------------------------------------------------------------------- #
def _batch(reqs):
    from freetoken.core import Batch

    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    return batch


def _req(uid, input_ids, cached_len, spec):
    from freetoken.core import Req

    return Req(
        input_ids=torch.tensor(input_ids, dtype=torch.int32),
        table_idx=0,
        cached_len=cached_len,
        output_len=1,
        uid=uid,
        sampling_params=None,
        cache_handle=None,
        hidden_states=spec,
    )


def test_collector_writes_only_for_opted_in_requests(tmp_path):
    spec = HiddenStateSpec(directory=str(tmp_path), layer_ids=[0, 1])
    collector = HiddenStateCollector(hidden_size=4, num_layers=2)
    probe = _req(1, [5, 6, 7], 0, spec)
    plain = _req(2, [8, 9], 0, None)

    sink = collector.begin_batch(_batch([probe, plain]))
    assert sink is not None
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    sink.capture(0, hidden)
    sink.capture(1, hidden + 1)

    assert collector.begin_batch(_batch([plain])) is None
    assert collector.finish(plain.uid) is None

    path = collector.finish(probe.uid)
    with safe_open(path, framework="pt") as handle:
        loaded = handle.get_tensor("hidden_states")
        ids = handle.get_tensor("token_ids")
    assert loaded.shape == (3, 2, 4)
    # The probe owns the FIRST three rows of the batch (request order).
    assert torch.equal(loaded[:, 0], hidden[:3].to(torch.bfloat16))
    assert torch.equal(ids, torch.tensor([5, 6, 7], dtype=torch.int64))
    assert len(collector) == 0


def test_collector_discard_drops_a_partial_capture(tmp_path):
    spec = HiddenStateSpec(directory=str(tmp_path), layer_ids=[0])
    collector = HiddenStateCollector(hidden_size=2, num_layers=1)
    req = _req(3, [1, 2], 0, spec)
    collector.begin_batch(_batch([req]))
    assert len(collector) == 1
    collector.discard(req.uid)
    assert len(collector) == 0
    assert collector.finish(req.uid) is None


def test_collector_is_a_noop_for_decode_batches(tmp_path):
    from freetoken.core import Batch

    spec = HiddenStateSpec(directory=str(tmp_path), layer_ids=[0])
    collector = HiddenStateCollector(hidden_size=2, num_layers=1)
    req = _req(4, [1, 2], 0, spec)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    assert collector.begin_batch(batch) is None


def test_capture_refuses_to_finish_when_a_layer_was_never_written():
    """A model with no hidden-state hook must fail loudly, not ship uninitialized memory."""
    capture = HiddenStateCapture(
        HiddenStateSpec(directory="/tmp", layer_ids=[0, 1]), hidden_size=3
    )
    capture.begin_chunk(torch.tensor([1], dtype=torch.int32))
    capture.write(0, torch.ones(1, 3))
    with pytest.raises(ValueError, match="hidden-state hook"):
        capture.finish()
