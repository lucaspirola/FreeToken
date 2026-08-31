from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import bench_long_context as bench  # noqa: E402


class _IdentityTokenizer:
    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        _ = skip_special_tokens
        return "".join(chr(value) for value in ids)


def test_synthetic_needle_sample_is_unambiguous_and_long():
    question, expected = bench.synthetic_needle_sample()
    assert expected == "5663623"
    assert question.count(expected) == 1
    assert question.endswith("What is the secret passcode? State the digits clearly.")
    assert len(question) > 5_000_000


def test_trim_filler_supports_compact_multi_agent_protection():
    tokenizer = _IdentityTokenizer()
    expected = "7319041"
    text = "start " + "a" * 2000 + expected + "b" * 2000 + " final question"
    trimmed, original, actual = bench.trim_filler(
        tokenizer,
        text,
        expected,
        1024,
        protected_prefix_tokens=128,
        protected_needle_context_tokens=128,
        protected_tail_tokens=256,
    )

    assert original == len(text)
    assert actual == 1024
    assert expected in trimmed
    assert trimmed.endswith(" final question")


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
