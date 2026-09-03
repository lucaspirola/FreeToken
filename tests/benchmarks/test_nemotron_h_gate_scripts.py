"""GPU-free unit tests for the Nemotron-H gate scripts.

Everything here runs on the CPU without a checkpoint or a server: argument parsing,
the SSE reassembly that every serving gate depends on, and the comparison metrics the
layer-parity gate accepts or rejects on.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import gate_nemotron_h_serving as gate  # noqa: E402
import parity_nemotron_h_layers as parity  # noqa: E402


# --------------------------------------------------------------- serving gates


def _sse(chunks: list[dict]) -> list[bytes]:
    lines: list[bytes] = []
    for chunk in chunks:
        lines.append(b"data: " + json.dumps(chunk).encode())
        lines.append(b"")
    lines.append(b"data: [DONE]")
    return lines


def _content_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}]}


def test_parse_args_puts_shared_options_before_the_subcommand():
    args = gate.parse_args(["--port", "9001", "--max-tokens", "8", "batch-invariance"])
    assert args.gate == "batch-invariance"
    assert gate.origin_of(args) == "http://127.0.0.1:9001"
    assert args.max_tokens == 8
    assert args.stream is True
    assert args.concurrency == 16


def test_base_url_overrides_host_and_port_and_strips_trailing_slash():
    args = gate.parse_args(["--base-url", "http://example:1/", "tool-call"])
    assert gate.origin_of(args) == "http://example:1"


def test_parse_stages_rejects_empty_and_non_positive():
    assert gate.parse_stages("1,6,16,1") == [1, 6, 16, 1]
    with pytest.raises(ValueError):
        gate.parse_stages("")
    with pytest.raises(ValueError):
        gate.parse_stages("1,0")


def test_collect_stream_concatenates_content_split_across_events():
    """A single token can straddle two ``data:`` events; grepping raw SSE lies."""
    chunks = [_content_chunk(piece) for piece in ("The pass", "code is 56", "63623.")]
    chunks.append({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]})
    chunks.append({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3}})
    result = gate.collect_stream(gate.iter_sse_payloads(_sse(chunks)))
    assert result["content"] == "The passcode is 5663623."
    assert result["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 11
    assert "5663623" not in "".join(chunk["choices"][0]["delta"].get("content", "")
                                   for chunk in chunks[:1])


def test_iter_sse_payloads_stops_at_done_and_skips_non_data_lines():
    lines = [
        b": keep-alive",
        b"",
        b"data: " + json.dumps(_content_chunk("a")).encode(),
        b"data: [DONE]",
        b"data: " + json.dumps(_content_chunk("never")).encode(),
    ]
    payloads = list(gate.iter_sse_payloads(lines))
    assert len(payloads) == 1
    assert gate.collect_stream(payloads)["content"] == "a"


def test_collect_stream_joins_reasoning_and_tool_argument_fragments():
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "thin"}, "index": 0}]},
        {"choices": [{"delta": {"reasoning": "king"}, "index": 0}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "get_current_weather", "arguments": '{"ci'},
                            }
                        ]
                    },
                    "index": 0,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Boston"}'}}]},
                    "index": 0,
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    result = gate.collect_stream(gate.iter_sse_payloads(_sse(chunks)))
    assert result["reasoning"] == "thinking"
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"] == [
        {"id": "call_1", "name": "get_current_weather", "arguments": '{"city": "Boston"}'}
    ]
    assert gate.decode_tool_arguments(result["tool_calls"][0]) == {"city": "Boston"}


def test_collect_response_matches_the_streaming_shape():
    body = {
        "choices": [
            {
                "message": {
                    "content": "hi",
                    "tool_calls": [
                        {"id": "c", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4}},
    }
    result = gate.collect_response(body)
    assert result["content"] == "hi"
    assert result["tool_calls"] == [{"id": "c", "name": "f", "arguments": "{}"}]
    assert gate.cached_tokens(result["usage"]) == 4
    assert gate.prompt_tokens(result["usage"]) == 5
    assert gate.cached_tokens(None) == 0


def test_first_divergence_reports_the_split_point():
    assert gate.first_divergence("abc", "abc") is None
    assert gate.first_divergence("abc", "abd") == 2
    assert gate.first_divergence("abc", "ab") == 2


def test_batch_invariance_failures_flag_divergent_and_missing_answers():
    solo = {"p0": "391", "p1": "Canberra"}
    assert gate.batch_invariance_failures(solo, [("p0", "391"), ("p1", "Canberra")]) == []
    failures = gate.batch_invariance_failures(solo, [("p0", "392")])
    assert any("p0" in failure and "diverged at char 2" in failure for failure in failures)
    assert any("p1: no concurrent answer recorded" == failure for failure in failures)


def test_prefix_cache_failures_require_equality_and_a_real_cache_hit():
    cold = {"content": "42", "usage": {"prompt_tokens": 4100, "prompt_tokens_details": {}}}
    warm = {
        "content": "42",
        "usage": {"prompt_tokens": 4100, "prompt_tokens_details": {"cached_tokens": 4000}},
    }
    assert prefix_ok(cold, warm)
    cold_miss = dict(warm)
    assert gate.prefix_cache_failures(
        cold, dict(warm, content="43"), expect_cache_report=False
    )
    no_hit = {"content": "42", "usage": {"prompt_tokens": 4100}}
    failures = gate.prefix_cache_failures(cold, no_hit, expect_cache_report=True)
    assert any("did not exceed" in failure for failure in failures)
    assert any("below 50%" in failure for failure in failures)
    # With cache reporting off the counters are ignored entirely.
    assert gate.prefix_cache_failures(cold, no_hit, expect_cache_report=False) == []
    assert cold_miss["content"] == "42"


def prefix_ok(cold, warm) -> bool:
    return gate.prefix_cache_failures(cold, warm, expect_cache_report=True) == []


def test_elastic_ramp_failures_catch_errors_truncation_and_empty_answers():
    stages = [
        {"concurrency": 1, "errors": [], "completed": 1, "empty": 0},
        {"concurrency": 6, "errors": ["s1r2: HTTP 500"], "completed": 5, "empty": 0},
        {"concurrency": 16, "errors": [], "completed": 16, "empty": 2},
    ]
    failures = gate.elastic_ramp_failures(stages)
    assert "concurrency 6: s1r2: HTTP 500" in failures
    assert "concurrency 6: 5 of 6 requests completed" in failures
    assert "concurrency 16: 2 empty answers" in failures
    assert len(failures) == 3


def test_tool_call_failures_check_finish_reason_name_and_arguments():
    good = {
        "finish_reason": "tool_calls",
        "tool_calls": [{"name": "get_current_weather", "arguments": '{"city": "Boston"}'}],
    }
    assert gate.tool_call_failures(good, expected_name="get_current_weather") == []
    assert gate.tool_call_failures(
        {"finish_reason": "stop", "tool_calls": []}, expected_name="get_current_weather"
    ) == [
        "finish_reason is 'stop', expected 'tool_calls'",
        "no tool call was returned",
    ]
    broken = {
        "finish_reason": "tool_calls",
        "tool_calls": [{"name": "get_current_weather", "arguments": '{"city":'}],
    }
    failures = gate.tool_call_failures(broken, expected_name="get_current_weather")
    assert any("not a JSON object" in failure for failure in failures)
    missing = {
        "finish_reason": "tool_calls",
        "tool_calls": [{"name": "get_current_weather", "arguments": '{"unit": "celsius"}'}],
    }
    assert any(
        "missing 'city'" in failure
        for failure in gate.tool_call_failures(missing, expected_name="get_current_weather")
    )


def test_render_markdown_marks_failures_and_write_appends(tmp_path):
    report = {
        "date": "2026-09-04",
        "base_url": "http://127.0.0.1:8000",
        "model": "nemotron",
        "stream": True,
        "thinking": False,
        "gates": [
            {"name": "tool-call", "passed": True, "failures": [], "summary": "ok"},
            {
                "name": "prefix-cache",
                "passed": False,
                "failures": ["warm cached_tokens 0 did not exceed cold 0"],
                "summary": "no hit",
            },
        ],
    }
    text = gate.render_markdown(report)
    assert "| tool-call | PASS | ok |" in text
    assert "| prefix-cache | **FAIL** | no hit |" in text
    assert "- `prefix-cache`: warm cached_tokens 0 did not exceed cold 0" in text
    path = tmp_path / "results" / "gates.md"
    gate.write_result_markdown(str(path), text)
    gate.write_result_markdown(str(path), text)
    assert path.read_text().count("| tool-call | PASS | ok |") == 2


def test_shared_prefix_is_unique_per_run_and_long_enough():
    first = gate.build_shared_prefix(4096, "aaaa")
    second = gate.build_shared_prefix(4096, "bbbb")
    assert first != second
    assert first.count("Note 0:") == 1
    assert len(first) > 4096


def test_chat_body_is_greedy_and_disables_thinking_by_default():
    args = gate.parse_args(["--max-tokens", "16", "tool-call"])
    body = gate.chat_body(args, "m", [{"role": "user", "content": "hi"}], tool_choice="auto")
    assert body["temperature"] == 0.0 and body["top_p"] == 1.0
    assert body["max_completion_tokens"] == 16
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["tool_choice"] == "auto"
    thinking = gate.parse_args(["--thinking", "tool-call"])
    assert "chat_template_kwargs" not in gate.chat_body(thinking, "m", [])


# ----------------------------------------------------------------- layer parity


def _parity_args(**overrides):
    values = {
        "min_cosine": 0.999,
        "min_expert_match": 0.995,
        "max_scaled_err": 0.05,
    }
    values.update(overrides)
    return type("Args", (), values)


def test_parse_layers_and_args_defaults():
    args = parity.parse_args(["--model", "/tmp/model"])
    assert parity.parse_layers(args.layers) == [0, 1, 5]
    assert args.tokens == 512 and args.device == "cuda" and args.seed == 1234
    with pytest.raises(ValueError):
        parity.parse_layers("0,-1")


def test_comparison_metrics_are_exact_for_identical_tensors():
    tensor = torch.randn(64, 32)
    metrics = parity.comparison_metrics(tensor, tensor.clone())
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["max_abs_err"] == 0.0
    assert metrics["scaled_max_abs_err"] == 0.0
    assert metrics["rel_l2"] == 0.0


def test_comparison_metrics_detect_a_perturbation():
    torch.manual_seed(0)
    reference = torch.randn(128, 16)
    actual = reference + torch.randn(128, 16) * 0.5
    metrics = parity.comparison_metrics(reference, actual)
    assert metrics["cosine"] < 0.99
    assert metrics["scaled_max_abs_err"] > 0.0
    assert math.isfinite(metrics["rel_l2"])
    with pytest.raises(ValueError):
        parity.comparison_metrics(reference, actual[:4])


def test_expert_id_agreement_ignores_within_token_order():
    reference = torch.tensor([[3, 1, 7], [0, 5, 9]])
    permuted = torch.tensor([[7, 3, 1], [9, 0, 5]])
    assert parity.expert_id_agreement(reference, permuted) == 1.0
    partial = torch.tensor([[7, 3, 2], [9, 0, 5]])
    assert parity.expert_id_agreement(reference, partial) == pytest.approx(5 / 6)


def test_gate_failures_apply_each_threshold():
    args = _parity_args()
    passing = {
        "layer": 1,
        "kind": "moe",
        "cosine": 0.9995,
        "scaled_max_abs_err": 0.01,
        "expert_match": 0.999,
    }
    assert parity.gate_failures(passing, args) == []
    failing = dict(passing, cosine=0.99, scaled_max_abs_err=0.9, expert_match=0.9)
    failures = parity.gate_failures(failing, args)
    assert len(failures) == 3
    assert all(failure.startswith("layer 1 (moe)") for failure in failures)
    nan = dict(passing, cosine=float("nan"))
    assert parity.gate_failures(nan, args)
    # Layers without routing report no expert-id metric and must not be penalised.
    assert parity.gate_failures(
        {"layer": 0, "kind": "mamba", "cosine": 1.0, "scaled_max_abs_err": 0.0}, args
    ) == []


def test_render_table_and_markdown_cover_routed_and_unrouted_layers():
    rows = [
        {
            "layer": 0,
            "kind": "mamba",
            "cosine": 0.99999,
            "max_abs_err": 1e-3,
            "scaled_max_abs_err": 1e-4,
            "rel_l2": 2e-4,
        },
        {
            "layer": 1,
            "kind": "moe",
            "cosine": 0.9998,
            "max_abs_err": 2e-3,
            "scaled_max_abs_err": 3e-4,
            "rel_l2": 4e-4,
            "expert_match": 0.9993,
        },
    ]
    table = parity.render_table(rows)
    assert "mamba" in table and "moe" in table and "0.99999" in table
    text = parity.render_markdown(
        {
            "date": "2026-09-04",
            "model": "/m",
            "tokens": 512,
            "seed": 1234,
            "device": "cuda",
            "layers": rows,
            "failures": ["layer 1 (moe): cosine 0.5 is below 0.999"],
            "accepted": False,
        }
    )
    assert "| 0 | mamba |" in text
    assert "| 1 | moe |" in text
    assert "—" in text  # the mamba row has no expert-id column value
    assert "**FAIL**" in text


class _StubModule:
    """Minimal BaseOP stand-in: reflective state_dict plus a strict load."""

    def __init__(self, buffers: dict[str, torch.Tensor]):
        self.buffers = dict(buffers)
        self.loaded: dict[str, torch.Tensor] | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.buffers)

    def load_state_dict(self, payload: dict[str, torch.Tensor]) -> None:
        self.loaded = payload


def test_load_module_broadcasts_per_tensor_scales_and_renames_the_nvfp4_global():
    module = _StubModule(
        {
            "weight": torch.empty(4, 2, dtype=torch.uint8),
            "weight_scale": torch.empty(4, dtype=torch.float32),
            "weight_global": torch.empty(4, dtype=torch.float16),
        }
    )
    parity.load_module(
        module,
        {
            "weight": torch.ones(4, 2, dtype=torch.uint8),
            "weight_scale": torch.tensor(0.5),
            "weight_scale_2": torch.tensor([0.25]),
            "input_scale": torch.tensor(2.0),
        },
        torch.device("cpu"),
    )
    loaded = module.loaded
    assert loaded is not None
    assert loaded["weight_scale"].shape == (4,)
    assert torch.equal(loaded["weight_scale"], torch.full((4,), 0.5))
    assert loaded["weight_global"].dtype == torch.float16
    assert torch.equal(loaded["weight_global"], torch.full((4,), 0.25, dtype=torch.float16))
    # The optional FP8 activation scale is forwarded even though it is not a buffer.
    assert float(loaded["input_scale"]) == 2.0


def test_exact_attention_backend_matches_a_manual_causal_gqa_reference():
    torch.manual_seed(3)
    tokens, num_q, num_kv, head_dim = 7, 4, 2, 8
    backend = parity.ExactAttnBackend(num_q, num_kv, head_dim)
    q = torch.randn(tokens, num_q, head_dim)
    k = torch.randn(tokens, num_kv * head_dim)
    v = torch.randn(tokens, num_kv * head_dim)
    out = backend.forward(q, k, v, 0, None)
    assert out.shape == (tokens, num_q, head_dim)

    key = k.view(tokens, num_kv, head_dim).repeat_interleave(num_q // num_kv, dim=1)
    value = v.view(tokens, num_kv, head_dim).repeat_interleave(num_q // num_kv, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        is_causal=True,
    )[0].transpose(0, 1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_prefill_batch_shape_matches_what_the_mamba_scan_reads():
    batch = parity.prefill_batch(512, slot=3)
    assert batch.padded_reqs is batch.reqs
    req = batch.padded_reqs[0]
    assert (req.extend_len, req.cached_len, req.linear_slot_idx) == (512, 0, 3)
    assert req.mamba_ping_pong is None and req.mamba_last_track_seqlen is None
    assert not batch.is_decode
