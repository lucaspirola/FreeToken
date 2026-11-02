"""repetition_penalty: the sampler matches transformers, and requests pick up the model default."""

import json

import pytest
import torch
from pydantic import ValidationError
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.sample import Sampler, apply_repetition_penalty
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
from freetoken.server.generation import resolve_sampling
from freetoken.server.openai_api import _resolve_sampling
from freetoken.utils.hf import load_generation_sampling

VOCAB = 97


@pytest.mark.parametrize("penalty", [1.05, 1.3, 0.8])
def test_penalty_matches_transformers_processor(penalty):
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(3, VOCAB, generator=gen) * 4
    # Rows of different lengths, with repeated ids, as prompt + generated history.
    hist = [torch.randint(0, VOCAB, (n,), generator=gen) for n in (1, 40, 300)]
    ours = logits.clone()
    apply_repetition_penalty(ours, [(i, h.to(torch.int32), penalty) for i, h in enumerate(hist)])
    proc = RepetitionPenaltyLogitsProcessor(penalty=penalty)
    for i, h in enumerate(hist):
        ref = proc(h.unsqueeze(0), logits[i : i + 1].clone())
        torch.testing.assert_close(ours[i], ref[0], rtol=0, atol=0)


def _req(cls, table_idx, n, penalty, **kw):
    return cls(
        input_ids=torch.arange(n, dtype=torch.int32), table_idx=table_idx, cached_len=0,
        output_len=4, uid=table_idx, cache_handle=None,
        sampling_params=SamplingParams(repetition_penalty=penalty, **kw),
    )


def test_sampler_reads_history_from_token_pool_and_skips_off_and_chunked_rows():
    pool = torch.zeros(4, 16, dtype=torch.int32)
    pool[1, :5] = torch.tensor([7, 7, 3, 9, 2], dtype=torch.int32)
    reqs = [_req(Req, 0, 3, 1.0), _req(Req, 1, 5, 1.5), _req(ChunkedReq, 2, 6, 1.5)]
    sampler = Sampler(torch.device("cpu"), VOCAB)
    args = sampler.prepare(Batch(reqs=reqs, phase="decode"), pool)
    assert args.temperatures is None  # all greedy
    assert [(row, ids.tolist(), p) for row, ids, p in args.repetition] == [(1, [7, 7, 3, 9, 2], 1.5)]
    # Greedy picks the top logit unless the penalty pushes it below the runner-up.
    logits = torch.zeros(3, VOCAB)
    logits[:, 7] = 3.0
    logits[:, 11] = 2.5
    tokens = sampler.sample(logits, args)
    assert tokens.tolist() == [7, 11, 7]
    assert logits[1, 7] == 3.0  # the caller's logits stay raw (first-step logprobs read them)
    assert sampler.prepare(Batch(reqs=reqs[:1], phase="decode"), pool).repetition == []
    assert sampler.prepare(Batch(reqs=reqs, phase="decode")).repetition == []


def test_model_default_applies_and_request_overrides():
    model = {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05}
    sp = resolve_sampling(
        temperature=None, top_k=None, top_p=None, max_tokens=8, ignore_eos=False,
        model_sampling=model,
    )
    assert (sp.temperature, sp.top_k, sp.repetition_penalty) == (0.1, 50, 1.05)
    req = ChatCompletionRequest(model="m", messages=[], repetition_penalty=1.0)
    assert _resolve_sampling(req, model).repetition_penalty == 1.0
    req = CompletionRequest(model="m", prompt="x")
    assert _resolve_sampling(req, model).repetition_penalty == 1.05
    assert _resolve_sampling(req, {}).repetition_penalty == 1.0


def test_request_rejects_non_positive_penalty():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[], repetition_penalty=0)
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest(model="m", messages=[], max_tokens=1, repetition_penalty=-1)
    assert AnthropicMessagesRequest(
        model="m", messages=[], max_tokens=1, repetition_penalty=1.05
    ).repetition_penalty == 1.05


@pytest.mark.parametrize(
    "cfg, expected",
    [
        (
            {"do_sample": True, "temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05},
            {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05},
        ),
        ({"do_sample": False, "repetition_penalty": 1.1}, {"temperature": 0.0, "repetition_penalty": 1.1}),
        ({"do_sample": True, "temperature": 0.7}, {"temperature": 0.7}),
    ],
)
def test_generation_config_loads_repetition_penalty(tmp_path, cfg, expected):
    (tmp_path / "generation_config.json").write_text(json.dumps(cfg))
    got = load_generation_sampling(str(tmp_path))
    assert {k: got[k] for k in expected} == pytest.approx(expected)
    if "repetition_penalty" not in expected:
        assert "repetition_penalty" not in got
