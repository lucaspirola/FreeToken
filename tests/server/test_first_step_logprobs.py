"""First-step logprobs on /v1/chat/completions (docs/switchyard.md, "First-step logprobs").

Pinned here: request validation (``logprobs`` / ``top_logprobs`` bounds, the unchanged
/v1/completions rejection), the pure-tensor helper (full-vocab log_softmax, sampled token
first, top-k of the same distribution), the scheduler drain (a ChunkedReq row contributes
nothing, the final-chunk row does), the wire round trip of the new field, and the OpenAI
shaping on the plain response and on the stream's terminal chunk -- ``logprobs: null``
when not requested.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
from fastapi.responses import JSONResponse
from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.sample import FirstStepLogprobs, Sampler, first_step_logprobs
from freetoken.message import BaseFrontendMsg, BaseTokenizerMsg, DetokenizeMsg, UserReply
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.scheduler import Scheduler
from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
from freetoken.server.openai_api import (
    _completion_unsupported_reason,
    handle_chat_completion,
    stream_chat_completion_chunks,
)

from tests.scheduler.test_abort_inflight_prefill import _launch_req, _setup
from .test_openai_api import _collect, parse_sse, run
from .test_switchyard_wire import TokenizingState

VOCAB = 11


class _Tokenizer:
    """A stand-in HF tokenizer: id -> 't<id>' (id 3 decodes to a two-byte string)."""

    def decode(self, ids):
        (token_id,) = ids
        return "é" if token_id == 3 else f"t{token_id}"


class LogprobState(TokenizingState):
    """The preflight's tokenizing state, plus the HF tokenizer the shaping decodes with."""

    def frontend_tokenizer(self):
        manager = super().frontend_tokenizer()
        manager.tokenizer = _Tokenizer()
        return manager


def request(**kwargs) -> ChatCompletionRequest:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def _error(response) -> dict:
    assert isinstance(response, JSONResponse) and response.status_code == 400, response
    return json.loads(bytes(response.body))["error"]


FIRST = {"token_ids": [3, 3, 7], "logprobs": [-0.1, -0.1, -2.5]}


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
def test_top_logprobs_over_twenty_is_a_400():
    err = _error(run(handle_chat_completion(request(logprobs=True, top_logprobs=21), None, LogprobState([]), {})))
    assert err["param"] == "top_logprobs"


def test_top_logprobs_without_logprobs_is_a_400():
    err = _error(run(handle_chat_completion(request(top_logprobs=3), None, LogprobState([]), {})))
    assert err["param"] == "top_logprobs"


def test_switchyard_top_logprobs_zero_is_still_a_noop():
    state = LogprobState([UserReply(uid=42, incremental_output="ok", finished=True, finish_reason="stop")])
    payload = run(handle_chat_completion(request(top_logprobs=0), None, state, {}))
    assert payload["choices"][0]["logprobs"] is None
    assert state.sent.sampling_params.logprobs is False
    assert state.sent.sampling_params.top_logprobs == 0


@pytest.mark.parametrize("top_logprobs, expected_k", [(None, 0), (0, 0), (5, 5), (20, 20)])
def test_logprobs_request_rides_sampling_params(top_logprobs, expected_k):
    state = LogprobState([UserReply(uid=42, incremental_output="ok", finished=True, finish_reason="stop")])
    run(handle_chat_completion(request(logprobs=True, top_logprobs=top_logprobs), None, state, {}))
    assert state.sent.sampling_params.logprobs is True
    assert state.sent.sampling_params.top_logprobs == expected_k


def test_completions_route_still_rejects_logprobs():
    req = CompletionRequest(model="m", prompt="hi", logprobs=1)
    assert _completion_unsupported_reason(req) == "logprobs is not supported"


# --------------------------------------------------------------------------- #
# The tensor helper
# --------------------------------------------------------------------------- #
def test_helper_is_full_vocab_log_softmax_sampled_first_then_top_k():
    logits = torch.randn(3, VOCAB, dtype=torch.bfloat16) * 4
    sampled = torch.tensor([0, 5, 10], dtype=torch.int32)
    ids, vals = first_step_logprobs(logits, sampled, 4)
    assert ids.shape == vals.shape == (3, 5) and vals.dtype == torch.float32
    ref = torch.log_softmax(logits.float(), dim=-1)
    assert torch.equal(ids[:, 0], sampled.long())
    torch.testing.assert_close(vals[:, 0], ref[torch.arange(3), sampled.long()])
    top_vals, top_ids = torch.topk(ref, 4, dim=-1)
    assert torch.equal(ids[:, 1:], top_ids)
    torch.testing.assert_close(vals[:, 1:], top_vals)
    # A distribution, not raw logits: the exp of every row sums to 1.
    torch.testing.assert_close(ref.exp().sum(-1), torch.ones(3))


def test_helper_with_k_zero_is_the_sampled_logprob_only():
    logits = torch.zeros(1, VOCAB)  # uniform -> every logprob is -log(V)
    ids, vals = first_step_logprobs(logits, torch.tensor([4]), 0)
    assert ids.tolist() == [[4]]
    assert vals.tolist()[0] == pytest.approx([-math.log(VOCAB)])


def test_sampler_selects_plain_rows_on_prefill_only():
    """Only plain Reqs with logprobs on a prefill batch produce rows: ChunkedReq rows
    (a continuation) and decode batches never do, and a row's k is its own."""
    def req(cls, logprobs, k):
        return cls(
            input_ids=torch.tensor([1]), table_idx=0, cached_len=0, output_len=1, uid=1,
            sampling_params=SamplingParams(logprobs=logprobs, top_logprobs=k), cache_handle=None,
        )
    reqs = [req(Req, False, 0), req(ChunkedReq, True, 3), req(Req, True, 2), req(Req, True, 0)]
    sampler = Sampler(torch.device("cpu"), VOCAB)
    logits = torch.randn(4, VOCAB)
    sampled = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

    out = sampler.first_step_logprobs(Batch(reqs=reqs, phase="prefill"), logits, sampled)
    assert out.rows == [2, 3] and out.ks == [2, 0]
    assert out.token_ids.shape == (2, 3)
    assert out.row(0)["token_ids"][0] == 3 and len(out.row(0)["token_ids"]) == 3
    assert out.row(1) == {
        "token_ids": [4],
        "logprobs": [pytest.approx(torch.log_softmax(logits[3], -1)[4].item())],
    }
    assert sampler.first_step_logprobs(Batch(reqs=reqs, phase="decode"), logits, sampled) is None
    assert sampler.first_step_logprobs(Batch(reqs=reqs[:2], phase="prefill"), logits[:2], sampled[:2]) is None


# --------------------------------------------------------------------------- #
# Scheduler drain (real _process_last_data against CPU-built managers)
# --------------------------------------------------------------------------- #
def _drain(stub, batch, logprobs):
    """Drain ``batch`` through the real _process_last_data with a stand-in ForwardOutput."""
    from freetoken.engine import ForwardOutput

    out = ForwardOutput(
        None, torch.tensor([42] * len(batch.reqs), dtype=torch.int32),
        SimpleNamespace(synchronize=lambda: None), logprobs,
    )
    Scheduler._process_last_data(stub, (SimpleNamespace(batch=batch), out))


def test_drain_ships_the_final_chunk_row_and_nothing_for_a_chunk():
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    prompt = torch.arange(1, 13, dtype=torch.int32)
    final = _launch_req(pool, cm, tm, prompt, track_seqlen=8)
    final.sampling_params = SamplingParams(max_tokens=4, logprobs=True, top_logprobs=1)
    dm.filter_reqs([final])
    batch = Batch(reqs=[final], phase="prefill")
    inflight = FirstStepLogprobs(
        rows=[0], ks=[1], token_ids=torch.tensor([[42, 42]]), logprobs=torch.tensor([[-0.5, -0.5]]),
    )
    _drain(stub, batch, inflight)
    (msg,) = [m for m in sent if isinstance(m, DetokenizeMsg)]
    assert msg.first_logprobs == {"token_ids": [42, 42], "logprobs": [-0.5, -0.5]}

    # The next drain (a decode step) carries none.
    _drain(stub, Batch(reqs=[final], phase="decode"), None)
    assert [m.first_logprobs for m in sent if isinstance(m, DetokenizeMsg)] == [msg.first_logprobs, None]


def test_drain_of_an_intermediate_chunk_replies_nothing():
    pool, cm, tm, _dm, pm, sent, stub = _setup()
    prompt = torch.arange(1, 13, dtype=torch.int32)
    chunk = _launch_req(pool, cm, tm, prompt[:8], cls=ChunkedReq)
    chunk.sampling_params = SamplingParams(max_tokens=4, logprobs=True, top_logprobs=1)
    _drain(stub, Batch(reqs=[chunk], phase="prefill"), None)
    assert sent == []


# --------------------------------------------------------------------------- #
# Wire
# --------------------------------------------------------------------------- #
def test_first_logprobs_survive_the_wire():
    detok = DetokenizeMsg(uid=5, next_token=3, finished=False, first_logprobs=FIRST)
    assert BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(detok)).first_logprobs == FIRST
    reply = UserReply(uid=5, incremental_output="x", finished=False, first_logprobs=FIRST)
    assert BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply)).first_logprobs == FIRST
    plain = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(DetokenizeMsg(uid=5, next_token=3, finished=True)))
    assert plain.first_logprobs is None


# --------------------------------------------------------------------------- #
# Response shaping
# --------------------------------------------------------------------------- #
EXPECTED = {
    "content": [{
        "token": "é", "logprob": -0.1, "bytes": [0xC3, 0xA9],
        "top_logprobs": [
            {"token": "é", "logprob": -0.1, "bytes": [0xC3, 0xA9]},
            {"token": "t7", "logprob": -2.5, "bytes": [116, 55]},
        ],
    }]
}


def _replies():
    return [
        UserReply(uid=42, incremental_output="é", finished=False, first_logprobs=FIRST),
        UserReply(uid=42, incremental_output=" more", finished=True, finish_reason="length"),
    ]


def test_non_stream_response_shapes_the_first_token_only():
    state = LogprobState(_replies())
    payload = run(handle_chat_completion(request(logprobs=True, top_logprobs=2), None, state, {}))
    assert payload["choices"][0]["message"]["content"] == "é more"
    assert payload["choices"][0]["logprobs"] == EXPECTED
    json.dumps(payload)


def test_non_stream_logprobs_is_null_when_not_requested():
    state = LogprobState(_replies()[1:])
    payload = run(handle_chat_completion(request(), None, state, {}))
    assert "logprobs" in payload["choices"][0] and payload["choices"][0]["logprobs"] is None


def test_stream_carries_logprobs_on_the_terminal_chunk():
    state = LogprobState(_replies())
    req = request(logprobs=True, top_logprobs=2, stream=True)
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
    chunks = [e for e in events if isinstance(e, dict)]
    with_logprobs = [c for c in chunks if c["choices"] and c["choices"][0].get("logprobs")]
    assert len(with_logprobs) == 1
    assert with_logprobs[0]["choices"][0]["finish_reason"] == "length"
    assert with_logprobs[0]["choices"][0]["logprobs"] == EXPECTED
    assert events[-1] == "[DONE]"


def test_stream_terminal_chunk_logprobs_is_null_when_not_requested():
    state = LogprobState(_replies()[1:])
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, request(stream=True), state))))
    terminal = [e for e in events if isinstance(e, dict) and e["choices"] and e["choices"][0]["finish_reason"]]
    assert len(terminal) == 1 and terminal[0]["choices"][0]["logprobs"] is None
