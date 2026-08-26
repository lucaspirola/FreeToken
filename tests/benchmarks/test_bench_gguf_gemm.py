"""GPU-free case-building checks for the GGUF GEMM benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import bench_gguf_gemm as bench


def test_default_cases_cover_both_ornith_moe_projections():
    args = bench.parse_args([])
    cases = bench.build_cases(args)
    moe = [case for case in cases if case["op"] == "moe"]

    assert {case["projection"] for case in moe} == {"gate_up", "down"}
    assert len(moe) == len(args.moe_tokens) * 2


def test_projection_filter_and_dense_cross_product():
    args = bench.parse_args(
        [
            "--dense-rows",
            "8",
            "16",
            "--dense-types",
            "q4_k",
            "q6_k",
            "--moe-tokens",
            "272",
            "--moe-projections",
            "down",
        ]
    )
    cases = bench.build_cases(args)
    dense = [case for case in cases if case["op"] == "dense"]
    moe = [case for case in cases if case["op"] == "moe"]

    assert len(dense) == 4
    assert len(moe) == 1
    assert moe[0]["projection"] == "down"


def test_default_activation_scale_is_overflow_safe():
    assert bench.parse_args([]).activation_scale == 1e-3
