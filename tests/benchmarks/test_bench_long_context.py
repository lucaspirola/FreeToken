from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import bench_long_context as bench  # noqa: E402


def test_synthetic_needle_sample_is_unambiguous_and_long():
    question, expected = bench.synthetic_needle_sample()
    assert expected == "5663623"
    assert question.count(expected) == 1
    assert question.endswith("What is the secret passcode? State the digits clearly.")
    assert len(question) > 5_000_000


def test_prefill_rates_reads_last_scheduler_sample(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "input throughput (token/s): 100.00 instant, 80.00 average\n"
        "unrelated\n"
        "input throughput (token/s): 70.50 instant, 75.25 average\n"
    )
    assert bench.prefill_rates(str(log)) == {
        "prefill_samples": 2,
        "prefill_instant_tok_s": 70.5,
        "prefill_average_tok_s": 75.25,
    }


def _gate_args(**overrides):
    values = {
        "baseline_json": None,
        "max_prefill_regression_pct": 3.0,
        "max_instant_prefill_regression_pct": 5.0,
        "max_decode_regression_pct": 3.0,
        "min_prefill_tok_s": None,
        "min_instant_prefill_tok_s": None,
        "min_decode_tok_s": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_acceptance_failures_enforces_absolute_and_relative_gates(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        '{"prefill_tok_s": 100, "prefill_instant_tok_s": 80, '
        '"decode_tok_s": 50}\n'
    )
    row = {
        "prefill_tok_s": 96.0,
        "prefill_instant_tok_s": 77.0,
        "decode_tok_s": 49.0,
    }
    failures = bench.acceptance_failures(
        row,
        _gate_args(
            baseline_json=str(baseline),
            min_decode_tok_s=49.5,
        ),
    )
    assert any("average prefill regressed 4.00%" in failure for failure in failures)
    assert any("decode 49.0 is below 49.50" in failure for failure in failures)
    assert not any("instant prefill regressed" in failure for failure in failures)


def test_acceptance_rejects_mismatched_baseline_identity(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"model": "q4.gguf", "prompt_tokens": 65536, '
        '"prefill_tok_s": 100, "prefill_instant_tok_s": 80, '
        '"decode_tok_s": 50}\n'
    )
    row = {
        "model": "q6.gguf",
        "prompt_tokens": 65536,
        "prefill_average_tok_s": 100.0,
        "prefill_instant_tok_s": 80.0,
        "decode_tok_s": 50.0,
    }
    failures = bench.acceptance_failures(
        row, _gate_args(baseline_json=str(baseline))
    )
    assert failures == [
        "incompatible performance baseline: model='q6.gguf' vs 'q4.gguf'"
    ]


class _CharTokenizer:
    """Round-trip-exact stand-in: one token per character, so token offsets == char offsets."""

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002 - HF signature
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002 - HF signature
        return "".join(chr(token) for token in ids)


def _needle_document(before: int, after: int, expected: str = "5663623") -> str:
    return "head. " * 200 + "a" * before + f" {expected} " + "b" * after + " tail?" * 400


def _depth(text: str, expected: str) -> float:
    return text.index(expected) / len(text)


def test_trim_filler_preserves_the_needle_depth():
    """The needle must keep its relative depth, or the long-context sweep stops being a
    needle-in-the-middle test: draining the largest gap first removes all the filler BEFORE
    the needle and pins it near token 0 at every target, so the sweep only grows the text
    after it (a retention test at an ever-larger retrieval distance, not retrieval at depth).
    The residual drift is the fixed protected head/needle/tail windows, which do not shrink
    with the target."""
    tokenizer = _CharTokenizer()
    expected = "5663623"
    text = _needle_document(300_000, 300_000, expected)
    source_depth = _depth(text, expected)

    for target, tolerance in ((16_384, 0.05), (65_536, 0.02), (262_144, 0.01)):
        trimmed, seen_original, actual = bench.trim_filler(tokenizer, text, expected, target)
        assert seen_original == len(text)
        assert actual == target
        assert trimmed.count(expected) == 1
        assert trimmed.endswith(text[-512:])
        assert abs(_depth(trimmed, expected) - source_depth) < tolerance


def test_trim_filler_keeps_an_off_center_needle_off_center():
    tokenizer = _CharTokenizer()
    expected = "5663623"
    text = _needle_document(450_000, 150_000, expected)
    source_depth = _depth(text, expected)
    assert source_depth > 0.7

    trimmed, _original, actual = bench.trim_filler(tokenizer, text, expected, 65_536)
    assert actual == 65_536
    assert abs(_depth(trimmed, expected) - source_depth) < 0.05


def test_trim_filler_rejects_a_target_that_cannot_hold_the_protected_regions():
    tokenizer = _CharTokenizer()
    with pytest.raises(ValueError):
        bench.trim_filler(tokenizer, _needle_document(3_000, 3_000), "5663623", 1_024)


def test_stream_completion_asks_the_question_through_the_chat_endpoint(monkeypatch):
    """The needle gate must stay anchored to a question.

    A raw /v1/completions continuation of the haystack is decided by its first sampled
    token, which legitimately differs between prefill chunkings for every Mamba-2
    implementation -- that turned the 131,072-token needle into a coin flip. Pin the
    endpoint, the chat wrapper, and the thinking-off toggle so it cannot drift back.
    """
    captured = {}
    events = [
        b'data: {"choices":[{"delta":{"content":"The secret "}}]}',
        b"",
        b'data: {"choices":[{"delta":{"content":"passcode is 5663623."}}]}',
        b'data: {"usage":{"prompt_tokens":11,"completion_tokens":2},"choices":[]}',
        b"data: [DONE]",
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter(events)

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr(bench.urllib.request, "urlopen", fake_urlopen)
    result = bench.stream_completion("http://x", "m", "prompt", 8)

    assert captured["url"].endswith("/v1/chat/completions")
    body = captured["body"]
    assert body["messages"] == [{"role": "user", "content": "prompt"}]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_completion_tokens"] == 8 and body["ignore_eos"] is True
    assert body["temperature"] == 0.0 and "prompt" not in body
    # Pieces are joined before anything greps them: a token split across two events
    # must not read as a miss.
    assert result["text"] == "The secret passcode is 5663623."
    assert len(result["stamps"]) == 2
    assert result["usage"]["prompt_tokens"] == 11
