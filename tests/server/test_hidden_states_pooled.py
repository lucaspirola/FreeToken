"""Inline pooled hidden states (``kv_transfer_params.pooling``).

The file probe (test_hidden_states_probe.py) ships every prompt position; this variant
ships one ``mean`` and/or ``last`` vector per layer, base64 float32, on the response
itself. Pinned here: the request rules that differ from the file path (no
``--hidden-states-dir``, any ascending layer subset, no token cap -- all of which still
apply the moment a file is also requested), the O(layers x hidden) accumulation across
prefill chunks, the response object and its base64 encoding on both the plain and the
streaming path, and parity with client-side pooling of the artifact for the same prompt.
Everything drives the real collector, sink, scheduler chunking and adapter code.
"""

from __future__ import annotations

import base64
import json
import os

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from freetoken.hidden_states import (
    POOLINGS,
    HiddenStateCapture,
    HiddenStateCollector,
    HiddenStateSink,
    HiddenStateSpec,
    validate_layer_ids,
)
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.openai_api import handle_chat_completion, stream_chat_completion_chunks
from safetensors import safe_open

from .test_hidden_states_probe import ProbeState, final_reply, probe_request
from .test_openai_api import _collect, parse_sse, run

HIDDEN = 8
NUM_LAYERS = 12


def decode(vector: str, layer_ids: list[int], hidden: int = HIDDEN) -> np.ndarray:
    """What the consumer does with the response."""
    return np.frombuffer(base64.b64decode(vector), dtype="<f4").reshape(len(layer_ids), hidden)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- #
# Request model + validation
# --------------------------------------------------------------------------- #
def test_pooling_is_an_enum():
    for value in ("mean", "last", "both"):
        assert probe_request(pooling=value).kv_transfer_params.pooling == value
    assert probe_request(hidden_states_path="/tmp/x").kv_transfer_params.pooling is None
    with pytest.raises(Exception):
        probe_request(pooling="max")


def test_layer_subsets_are_accepted_only_when_not_contiguous_is_relaxed():
    assert validate_layer_ids([3, 7, 11], 12, contiguous=False) == [3, 7, 11]
    with pytest.raises(ValueError, match="contiguous"):
        validate_layer_ids([3, 7, 11], 12)
    for bad in ([7, 3], [3, 3, 7], [-1, 0]):
        with pytest.raises(ValueError, match="ascending, unique and non-negative"):
            validate_layer_ids(bad, 12, contiguous=False)
    with pytest.raises(ValueError, match="layer 12"):
        validate_layer_ids([0, 12], 12, contiguous=False)


def _pooled_state(replies=None, **kwargs) -> ProbeState:
    return ProbeState(replies or [final_reply()], num_layers=NUM_LAYERS, **kwargs)


def test_explicit_layer_ids_are_served_when_the_state_has_no_model_config(tmp_path):
    state = _pooled_state(hidden_states_dir=str(tmp_path))
    del state.config.model_config
    run(handle_chat_completion(
        probe_request(hidden_states_path=str(tmp_path), layer_ids=[0, 1]), None, state, {}
    ))
    assert state.sent.hidden_states.layer_ids == [0, 1]
    response = run(handle_chat_completion(probe_request(pooling="mean"), None, state, {}))
    assert response.status_code == 400
    assert b"send them explicitly" in response.body


def test_pooled_probe_needs_no_server_directory():
    """The hook is model-side; nothing is written, so --hidden-states-dir is not needed."""
    state = _pooled_state(hidden_states_dir=None)
    payload = run(handle_chat_completion(probe_request(pooling="mean"), None, state, {}))
    assert payload["choices"][0]["finish_reason"] == "stop"
    sent = state.sent
    assert sent.hidden_states.directory is None
    assert sent.hidden_states.pooling == ("mean",)
    assert sent.hidden_states.layer_ids == list(range(NUM_LAYERS))
    # Same serving rules as the file probe: full recompute, no session lease.
    assert sent.no_prefix_cache is True
    assert sent.session_id is None


def test_pooled_probe_accepts_any_ascending_layer_subset():
    state = _pooled_state(hidden_states_dir=None)
    run(handle_chat_completion(
        probe_request(pooling="both", layer_ids=[3, 7, 11]), None, state, {}
    ))
    assert state.sent.hidden_states.layer_ids == [3, 7, 11]
    assert state.sent.hidden_states.pooling == ("mean", "last")


def test_file_probe_still_refuses_a_layer_subset(tmp_path):
    state = _pooled_state(hidden_states_dir=str(tmp_path))
    response = run(handle_chat_completion(
        probe_request(hidden_states_path=str(tmp_path), layer_ids=[3, 7, 11]),
        None, state, {},
    ))
    assert response.status_code == 400
    assert b"contiguous" in response.body


def test_pooled_probe_rejects_unsorted_or_out_of_range_layers():
    state = _pooled_state(hidden_states_dir=None)
    for bad, needle in (([7, 3], b"ascending"), ([0, NUM_LAYERS], b"layer 12")):
        response = run(handle_chat_completion(
            probe_request(pooling="last", layer_ids=bad), None, state, {}
        ))
        assert response.status_code == 400
        assert needle in response.body


def test_pooled_probe_is_not_capped():
    state = _pooled_state(
        hidden_states_dir=None, max_tokens=16, prompt_tokens=17, max_seq_len=131072
    )
    payload = run(handle_chat_completion(probe_request(pooling="mean"), None, state, {}))
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_pooled_probe_works_with_the_context_preflight_disabled():
    state = _pooled_state(hidden_states_dir=None, prompt_tokens=17, max_seq_len=131072)
    state.config.context_preflight = False
    payload = run(handle_chat_completion(probe_request(pooling="mean"), None, state, {}))
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_pooling_plus_a_path_writes_the_file_under_the_file_rules(tmp_path):
    # No directory on the server: the file half needs it, pooling or not.
    state = _pooled_state(hidden_states_dir=None)
    response = run(handle_chat_completion(
        probe_request(pooling="mean", hidden_states_path=str(tmp_path)), None, state, {}
    ))
    assert response.status_code == 400
    assert b"--hidden-states-dir" in response.body

    # Over the cap: the file half is capped.
    state = _pooled_state(
        hidden_states_dir=str(tmp_path), max_tokens=16, prompt_tokens=17,
        max_seq_len=131072,
    )
    response = run(handle_chat_completion(
        probe_request(pooling="mean", hidden_states_path=str(tmp_path)), None, state, {}
    ))
    assert response.status_code == 400
    assert b"--hidden-states-max-tokens" in response.body

    # A subset: the file half needs the contiguous run.
    state = _pooled_state(hidden_states_dir=str(tmp_path))
    response = run(handle_chat_completion(
        probe_request(pooling="mean", hidden_states_path=str(tmp_path), layer_ids=[3, 7]),
        None, state, {},
    ))
    assert response.status_code == 400
    assert b"contiguous" in response.body

    # Well-formed: both halves are requested.
    state = _pooled_state(hidden_states_dir=str(tmp_path))
    run(handle_chat_completion(
        probe_request(pooling="both", hidden_states_path=str(tmp_path), layer_ids=[0, 1]),
        None, state, {},
    ))
    assert state.sent.hidden_states.directory == os.path.realpath(tmp_path)
    assert state.sent.hidden_states.pooling == ("mean", "last")
    assert state.sent.hidden_states.layer_ids == [0, 1]


# --------------------------------------------------------------------------- #
# Capture: accumulation across chunks
# --------------------------------------------------------------------------- #
def _chunks(seed: int = 0, sizes=(3, 5, 2)) -> list[torch.Tensor]:
    """Per-chunk ``[layers, rows, hidden]`` bf16 residuals -- the dtype the model hands
    the sink -- so an fp32 accumulation of them has no rounding of its own to blame."""
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randn(NUM_LAYERS, rows, HIDDEN, generator=gen).to(torch.bfloat16)
        for rows in sizes
    ]


def _feed(capture: HiddenStateCapture, chunks, layer_ids=range(NUM_LAYERS)) -> int:
    token = 100
    for chunk in chunks:
        rows = chunk.shape[1]
        capture.begin_chunk(torch.arange(token, token + rows, dtype=torch.int32))
        token += rows
        for layer_id in layer_ids:
            capture.write(layer_id, chunk[layer_id])
    return token - 100


def test_pooled_mean_and_last_accumulate_across_chunks():
    layer_ids = [1, 4, 11]
    capture = HiddenStateCapture(
        HiddenStateSpec(layer_ids=layer_ids, pooling=("mean", "last")), hidden_size=HIDDEN
    )
    chunks = _chunks()
    total = _feed(capture, chunks)
    hidden, token_ids = capture.finish()
    assert hidden is None and token_ids is None  # nothing to write
    assert capture._chunks == [] and capture._token_chunks == []  # nothing per-token kept

    pooled = capture.pooled()
    assert pooled["layer_ids"] == layer_ids
    assert pooled["hidden"] == HIDDEN
    assert pooled["prompt_tokens"] == total == 10
    assert pooled["dtype"] == "float32"
    assert set(pooled) == {"layer_ids", "hidden", "prompt_tokens", "dtype", "mean", "last"}

    whole = torch.cat(chunks, dim=1).float()  # [layers, tokens, hidden]
    mean = decode(pooled["mean"], layer_ids)
    last = decode(pooled["last"], layer_ids)
    assert mean.dtype == np.dtype("<f4") and mean.shape == (3, HIDDEN)
    np.testing.assert_allclose(mean, whole[layer_ids].mean(dim=1).numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(last, whole[layer_ids][:, -1].numpy())


def test_only_the_requested_poolings_are_present():
    for pooling in POOLINGS.values():
        capture = HiddenStateCapture(
            HiddenStateSpec(layer_ids=[0], pooling=pooling), hidden_size=HIDDEN
        )
        _feed(capture, _chunks(sizes=(4,)), layer_ids=[0])
        pooled = capture.pooled()
        assert {k for k in ("mean", "last") if k in pooled} == set(pooling)
    assert HiddenStateCapture(HiddenStateSpec(directory="/tmp", layer_ids=[0]), HIDDEN).pooled() is None


def test_a_layer_written_twice_in_one_chunk_is_refused():
    capture = HiddenStateCapture(
        HiddenStateSpec(layer_ids=[0], pooling=("mean",)), hidden_size=HIDDEN
    )
    capture.begin_chunk(torch.tensor([1], dtype=torch.int32))
    capture.write(0, torch.ones(1, HIDDEN))
    with pytest.raises(AssertionError, match="twice"):
        capture.write(0, torch.ones(1, HIDDEN))


def test_pooled_refuses_to_finish_when_a_layer_was_never_written():
    capture = HiddenStateCapture(
        HiddenStateSpec(layer_ids=[0, 1], pooling=("mean",)), hidden_size=HIDDEN
    )
    capture.begin_chunk(torch.tensor([1], dtype=torch.int32))
    capture.write(0, torch.ones(1, HIDDEN))
    with pytest.raises(ValueError, match="hidden-state hook"):
        capture.pooled()


def test_pooled_alongside_the_file_keeps_both():
    capture = HiddenStateCapture(
        HiddenStateSpec(directory="/tmp", layer_ids=[0, 1], pooling=("last",)), HIDDEN
    )
    chunks = _chunks(sizes=(2, 3))
    _feed(capture, chunks, layer_ids=[0, 1])
    hidden, _ = capture.finish()
    assert hidden.shape == (5, 2, HIDDEN)
    last = decode(capture.pooled()["last"], [0, 1])
    np.testing.assert_array_equal(last, hidden[-1].float().numpy())


# --------------------------------------------------------------------------- #
# Parity with client-side pooling of the artifact, single and chunked prefill
# --------------------------------------------------------------------------- #
def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build_prefill(width: int = 64, max_running: int = 4):
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    page_table = torch.zeros((max_running + 1, width), dtype=torch.int32)
    cache_manager = CacheManager(num_pages=256, page_size=1, page_table=page_table, type="radix")
    table_manager = TableManager(max_running_reqs=max_running, page_table=page_table)
    return cache_manager, PrefillManager(
        cache_manager, table_manager, DecodeManager(page_size=1)
    )


class _Model:
    """A stand-in residual stream: block ``i`` leaves ``table[i][token]`` behind.

    Deterministic per (layer, token id) so a chunk boundary cannot change a value;
    bf16 like the real residual, so the artifact is lossless and the pooled float32
    accumulation is the only arithmetic under test.
    """

    def __init__(self, vocab: int = 64, seed: int = 1):
        gen = torch.Generator().manual_seed(seed)
        self.table = torch.randn(NUM_LAYERS, vocab, HIDDEN, generator=gen).to(torch.bfloat16)

    def forward(self, batch, sink) -> None:
        ids = torch.cat([req.input_ids[req.cached_len : req.device_len] for req in batch.padded_reqs])
        for layer_id in range(NUM_LAYERS):
            sink.capture(layer_id, self.table[layer_id][ids.long()])


def _drive(prompt: list[int], spec: HiddenStateSpec, chunk: int, model: _Model, uid: int = 1):
    """Admit ``prompt`` through the real PrefillManager in ``chunk``-token pieces and run
    every chunk through the collector; return the collector's finish() result."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cache_manager, prefill_manager = _build_prefill()
    collector = HiddenStateCollector(hidden_size=HIDDEN, num_layers=NUM_LAYERS)
    prefill_manager.pending_list = [
        PendingReq(
            uid=uid, input_ids=torch.tensor(prompt, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=1), hidden_states=spec,
            no_prefix_cache=True,
        )
    ]
    forwards = 0
    while prefill_manager.runnable:
        batch = prefill_manager.schedule_next_batch(chunk)
        assert batch is not None
        batch.padded_reqs = batch.reqs
        sink = collector.begin_batch(batch)
        assert sink is not None
        model.forward(batch, sink)
        cache_manager.allocate_paged(batch.reqs)
        for req in batch.reqs:
            req.complete_one()
        forwards += 1
    result = collector.finish(uid)
    assert len(collector) == 0
    return result, forwards


def _client_pool(path: str):
    with safe_open(path, framework="pt") as handle:
        hidden = handle.get_tensor("hidden_states").float()  # [tokens, layers, hidden]
        token_ids = handle.get_tensor("token_ids")
    return hidden.mean(dim=0).numpy(), hidden[-1].numpy(), token_ids


@pytest.mark.parametrize("chunk", [64, 4])
def test_pooled_matches_client_side_pooling_of_the_artifact(tmp_path, chunk):
    """Same prompt, same stand-in model: pooled vectors == pooling the file."""
    prompt = list(range(1, 15))
    model = _Model()
    layer_ids = list(range(NUM_LAYERS))
    filed, forwards = _drive(
        prompt, HiddenStateSpec(directory=str(tmp_path), layer_ids=layer_ids), chunk, model
    )
    pooled, pooled_forwards = _drive(
        prompt, HiddenStateSpec(layer_ids=layer_ids, pooling=("mean", "last")), chunk, model, uid=2
    )
    assert forwards == pooled_forwards == (1 if chunk == 64 else 4)
    assert set(filed) == {"hidden_states_path"}
    assert set(pooled) == {"pooled"}
    assert os.listdir(tmp_path) == [os.path.basename(filed["hidden_states_path"])]

    ref_mean, ref_last, token_ids = _client_pool(filed["hidden_states_path"])
    assert token_ids.tolist() == prompt
    got = pooled["pooled"]
    assert got["prompt_tokens"] == len(prompt)
    assert got["layer_ids"] == layer_ids
    mean = decode(got["mean"], layer_ids)
    last = decode(got["last"], layer_ids)
    for i in layer_ids:
        assert cosine(mean[i], ref_mean[i]) >= 0.9999
        assert cosine(last[i], ref_last[i]) >= 0.9999
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(last, ref_last)
    # `last` really is the final prompt position, not the last of some chunk.
    np.testing.assert_array_equal(last[3], model.table[3][prompt[-1]].float().numpy())


def test_chunked_prefill_pools_the_same_as_a_single_forward():
    prompt = list(range(20, 34))
    model = _Model(seed=7)
    spec = HiddenStateSpec(layer_ids=[2, 5, 9], pooling=("mean", "last"))
    single, one = _drive(prompt, spec, 64, model)
    chunked, many = _drive(prompt, spec, 3, model, uid=2)
    assert one == 1 and many == 5
    for key in ("mean", "last"):
        np.testing.assert_allclose(
            decode(chunked["pooled"][key], spec.layer_ids),
            decode(single["pooled"][key], spec.layer_ids), rtol=1e-6, atol=1e-6,
        )
    assert chunked["pooled"]["prompt_tokens"] == single["pooled"]["prompt_tokens"] == 14


def test_both_file_and_pooled_from_one_capture(tmp_path):
    prompt = list(range(5, 12))
    model = _Model(seed=3)
    result, _ = _drive(
        prompt,
        HiddenStateSpec(directory=str(tmp_path), layer_ids=[0, 1, 2], pooling=("mean",)),
        4, model,
    )
    assert set(result) == {"hidden_states_path", "pooled"}
    ref_mean, _, _ = _client_pool(result["hidden_states_path"])
    np.testing.assert_allclose(
        decode(result["pooled"]["mean"], [0, 1, 2]), ref_mean, rtol=1e-5, atol=1e-6
    )


def _req(uid, input_ids, spec):
    from freetoken.core import Req

    return Req(
        input_ids=torch.tensor(input_ids, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=1, uid=uid, sampling_params=None, cache_handle=None, hidden_states=spec,
    )


def _batch(reqs):
    from freetoken.core import Batch

    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    return batch


def test_collector_pools_only_the_probe_rows_behind_a_plain_request():
    """Offsets come from begin_batch: a plain request ahead of the probe shifts them."""
    collector = HiddenStateCollector(hidden_size=HIDDEN, num_layers=2)
    plain = _req(1, [8, 9, 10], None)
    probe = _req(2, [5, 6], HiddenStateSpec(layer_ids=[0, 1], pooling=("mean", "last")))
    sink = collector.begin_batch(_batch([plain, probe]))
    rows = torch.arange(5 * HIDDEN, dtype=torch.float32).reshape(5, HIDDEN)
    sink.capture(0, rows)
    sink.capture(1, rows * 10)

    result = collector.finish(probe.uid)
    assert collector.finish(plain.uid) is None
    got = result["pooled"]
    assert got["prompt_tokens"] == 2
    np.testing.assert_array_equal(decode(got["mean"], [0, 1])[0], rows[3:].mean(0).numpy())
    np.testing.assert_array_equal(decode(got["mean"], [0, 1])[1], (rows[3:] * 10).mean(0).numpy())
    np.testing.assert_array_equal(decode(got["last"], [0, 1])[0], rows[4].numpy())


def test_write_failure_keeps_the_pooled_half(tmp_path, caplog):
    """The scheduler's drain must neither crash nor lose the vectors when the artifact
    cannot be written; only the path is missing."""
    from freetoken.scheduler.scheduler import Scheduler

    gone = tmp_path / "gone"
    gone.mkdir()
    collector = HiddenStateCollector(hidden_size=HIDDEN, num_layers=1)
    probe = _req(7, [1, 2, 3], HiddenStateSpec(directory=str(gone), layer_ids=[0], pooling=("mean",)))
    sink = collector.begin_batch(_batch([probe]))
    rows = torch.ones(3, HIDDEN)
    sink.capture(0, rows)
    gone.rmdir()  # the directory vanishes before the drain

    stub = SimpleNamespace(engine=SimpleNamespace(hidden_states=collector))
    result = Scheduler._finish_hidden_states(stub, probe)
    assert set(result) == {"pooled"}
    np.testing.assert_array_equal(decode(result["pooled"]["mean"], [0])[0], rows[0].numpy())
    assert len(collector) == 0
    assert any("artifact write failed" in r.getMessage() for r in caplog.records)


def test_sink_pools_only_its_own_rows_out_of_a_mixed_batch():
    first = HiddenStateCapture(HiddenStateSpec(layer_ids=[0], pooling=("mean", "last")), HIDDEN)
    second = HiddenStateCapture(HiddenStateSpec(layer_ids=[0], pooling=("mean", "last")), HIDDEN)
    first.begin_chunk(torch.tensor([10], dtype=torch.int32))
    second.begin_chunk(torch.tensor([20, 21], dtype=torch.int32))
    rows = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN)
    HiddenStateSink([(first, 0, 1), (second, 1, 3)]).capture(0, rows)
    np.testing.assert_array_equal(decode(first.pooled()["mean"], [0])[0], rows[0].numpy())
    np.testing.assert_array_equal(decode(second.pooled()["mean"], [0])[0], rows[1:].mean(0).numpy())
    np.testing.assert_array_equal(decode(second.pooled()["last"], [0])[0], rows[2].numpy())


# --------------------------------------------------------------------------- #
# Response placement: plain and streaming
# --------------------------------------------------------------------------- #
def _pooled_payload() -> dict:
    vectors = np.arange(2 * HIDDEN, dtype="<f4").reshape(2, HIDDEN)
    return {
        "layer_ids": [3, 7], "hidden": HIDDEN, "prompt_tokens": 5, "dtype": "float32",
        "mean": base64.b64encode(vectors.tobytes()).decode("ascii"),
    }


def test_non_stream_response_carries_pooled_verbatim():
    pooled = _pooled_payload()
    state = _pooled_state(
        [final_reply(kv_transfer_params={"pooled": pooled})], hidden_states_dir=None
    )
    payload = run(handle_chat_completion(
        probe_request(pooling="mean", layer_ids=[3, 7]), None, state, {}
    ))
    assert payload["kv_transfer_params"] == {"pooled": pooled}
    assert "hidden_states_path" not in payload["kv_transfer_params"]
    # It is JSON all the way down and decodes to the vectors the engine pooled.
    back = json.loads(json.dumps(payload))["kv_transfer_params"]["pooled"]
    np.testing.assert_array_equal(
        decode(back["mean"], back["layer_ids"]), np.arange(2 * HIDDEN, dtype="<f4").reshape(2, HIDDEN)
    )


def test_stream_carries_pooled_on_the_terminal_chunk():
    from freetoken.message.frontend import UserReply

    pooled = _pooled_payload()
    state = _pooled_state(
        [
            UserReply(uid=42, incremental_output="The", finished=False),
            UserReply(
                uid=42, incremental_output="", finished=True, finish_reason="length",
                kv_transfer_params={"pooled": pooled},
            ),
        ],
        hidden_states_dir=None,
    )
    req = ChatCompletionRequest(
        model="client-model", messages=[{"role": "user", "content": "score me"}],
        max_tokens=1, stream=True, kv_transfer_params={"pooling": "mean", "layer_ids": [3, 7]},
    )
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
    terminal = [e for e in events if isinstance(e, dict) and "kv_transfer_params" in e]
    assert len(terminal) == 1
    assert terminal[0]["choices"][0]["finish_reason"] == "length"
    assert terminal[0]["kv_transfer_params"] == {"pooled": pooled}
    assert events[-1] == "[DONE]"
