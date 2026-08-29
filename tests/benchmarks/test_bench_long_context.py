from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import bench_long_context as bench  # noqa: E402


def test_synthetic_needle_sample_is_unambiguous_and_long():
    question, expected = bench.synthetic_needle_sample()
    assert expected == "5663623"
    assert question.count(expected) == 1
    assert question.endswith("What is the secret passcode? State the digits clearly.")
    assert len(question) > 5_000_000
