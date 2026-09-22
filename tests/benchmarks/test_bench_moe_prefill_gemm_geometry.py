"""CPU-only coverage for bench_moe_prefill_gemm's selectable geometry.

No GPU and no server: geometry resolution, bank shapes (built with plain torch tensors,
no CUDA), and the config filename/round-trip contract against the real
`freetoken.moe.fused_nvfp4` helpers.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import bench_moe_prefill_gemm as bench  # noqa: E402

from freetoken.moe.fused_nvfp4 import (  # noqa: E402
    _load_nvfp4_moe_configs,
    nvfp4_config_filename,
    nvfp4_moe_config,
)

DEVICE_NAME = "NVIDIA_GeForce_RTX_5080"


# --- geometry resolution -------------------------------------------------------------


def test_no_flags_reproduces_nemotron_exactly():
    args = bench.parse_args([])
    geo = bench.resolve_geometry(args)
    assert geo == dict(hidden=2688, intermediate=1856, experts=128, top_k=6,
                        activation="relu2", gated=False)


def test_model_ornith_sets_the_whole_group():
    args = bench.parse_args(["--model", "ornith"])
    geo = bench.resolve_geometry(args)
    assert geo == dict(hidden=2048, intermediate=512, experts=256, top_k=8,
                        activation="silu", gated=True)


def test_explicit_flag_overrides_model_preset():
    args = bench.parse_args(["--model", "ornith", "--top-k", "4"])
    geo = bench.resolve_geometry(args)
    assert geo["top_k"] == 4
    assert geo["hidden"] == 2048  # rest of the ornith preset still applies


def test_explicit_flags_alone_build_a_custom_geometry():
    args = bench.parse_args([
        "--hidden", "2048", "--intermediate", "512", "--experts", "256",
        "--top-k", "8", "--activation", "silu", "--gated",
    ])
    geo = bench.resolve_geometry(args)
    assert geo == dict(hidden=2048, intermediate=512, experts=256, top_k=8,
                        activation="silu", gated=True)


# --- GEMM shapes and config filenames, exactly as specified ---------------------------


def _apply(model: str):
    """Resolve --model and stamp the module globals main() would, without touching CUDA."""
    args = bench.parse_args(["--model", model])
    geo = bench.resolve_geometry(args)
    bench.H, bench.I, bench.E, bench.TOP_K, bench.ACTIVATION, bench.GATED = (
        geo["hidden"], geo["intermediate"], geo["experts"], geo["top_k"],
        geo["activation"], geo["gated"],
    )
    return geo


@pytest.fixture(autouse=True)
def _restore_globals():
    saved = (bench.H, bench.I, bench.E, bench.TOP_K, bench.ACTIVATION, bench.GATED)
    yield
    bench.H, bench.I, bench.E, bench.TOP_K, bench.ACTIVATION, bench.GATED = saved


def test_nemotron_gemm_shapes_and_filenames():
    geo = _apply("nemotron")
    gate_up_n = bench._gate_up_n()
    assert (geo["experts"], gate_up_n, geo["hidden"]) == (128, 1856, 2688)
    assert (geo["experts"], geo["hidden"], geo["intermediate"]) == (128, 2688, 1856)

    name1 = nvfp4_config_filename(geo["experts"], gate_up_n, geo["hidden"], DEVICE_NAME)
    name2 = nvfp4_config_filename(geo["experts"], geo["hidden"], geo["intermediate"], DEVICE_NAME)
    assert name1 == f"nvfp4,E=128,N=1856,K=2688,device_name={DEVICE_NAME}.json"
    assert name2 == f"nvfp4,E=128,N=2688,K=1856,device_name={DEVICE_NAME}.json"


def test_ornith_gemm_shapes_and_filenames():
    geo = _apply("ornith")
    gate_up_n = bench._gate_up_n()
    assert gate_up_n == 2 * geo["intermediate"] == 1024
    assert geo["hidden"] == 2048

    name1 = nvfp4_config_filename(geo["experts"], gate_up_n, geo["hidden"], DEVICE_NAME)
    name2 = nvfp4_config_filename(geo["experts"], geo["hidden"], geo["intermediate"], DEVICE_NAME)
    assert name1 == f"nvfp4,E=256,N=1024,K=2048,device_name={DEVICE_NAME}.json"
    assert name2 == f"nvfp4,E=256,N=2048,K=512,device_name={DEVICE_NAME}.json"


# --- bank construction (CPU tensors, no CUDA) -----------------------------------------


def test_banks_ungated_shapes_byte_identical_to_nemotron():
    _apply("nemotron")
    gup, gup_s, gup_g, down, down_s, down_g = bench._banks(128, 0, "cpu")
    H, I = bench.H, bench.I
    assert gup.shape == (128, I, H // 2)
    assert gup_s.shape == (128, I, H // 16)
    assert gup_g.shape == (128, I)
    assert down.shape == (128, H, I // 2)
    assert down_s.shape == (128, H, I // 16)
    assert down_g.shape == (128, H)


def test_banks_gated_doubles_gate_up_n_only():
    _apply("ornith")
    gup, gup_s, gup_g, down, down_s, down_g = bench._banks(256, 0, "cpu")
    H, I = bench.H, bench.I
    assert gup.shape == (256, 2 * I, H // 2)
    assert gup_s.shape == (256, 2 * I, H // 16)
    assert gup_g.shape == (256, 2 * I)
    # down bank unaffected by gating
    assert down.shape == (256, H, I // 2)
    assert down_s.shape == (256, H, I // 16)
    assert down_g.shape == (256, H)


def test_banks_ungated_rng_sequence_unchanged():
    """Same seed, ungated path -> identical tensors to a hand-built reference using the
    original (pre-refactor) call sequence, proving the no-flags path is byte-identical."""
    import torch

    _apply("nemotron")
    H, I = bench.H, bench.I
    got = bench._banks(128, 0, "cpu")

    g = torch.Generator().manual_seed(0)
    want = (
        torch.randint(0, 256, (128, I, H // 2), dtype=torch.uint8, generator=g),
        (torch.rand(128, I, H // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn),
        torch.full((128, I), 0.5, dtype=torch.float16),
        torch.randint(0, 256, (128, H, I // 2), dtype=torch.uint8, generator=g),
        (torch.rand(128, H, I // 16, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn),
        torch.full((128, H), 0.75, dtype=torch.float16),
    )
    for g_out, w_out in zip(got, want):
        assert torch.equal(g_out.float(), w_out.float())


# --- FLOPs / weight-bytes formulas match the ungated baseline exactly ------------------


def test_pair_flops_ungated_matches_original_formula():
    _apply("nemotron")
    m = 8192
    assert bench._pair_flops(m) == 2.0 * m * bench.TOP_K * (bench.H * bench.I) * 2.0


def test_pair_weight_bytes_ungated_matches_original_formula():
    _apply("nemotron")
    H, I = bench.H, bench.I
    expected = (I * (H // 2) + I * (H // 16)) + (H * (I // 2) + H * (I // 16))
    assert bench._pair_weight_bytes(1) == expected


# --- JSON round-trip: what the tuner would write is what the server reads back --------


def test_json_round_trip_selects_bucket(tmp_path, monkeypatch):
    geo = _apply("ornith")
    gate_up_n = bench._gate_up_n()
    import triton

    name = nvfp4_config_filename(geo["experts"], gate_up_n, geo["hidden"], DEVICE_NAME)
    version_dir = tmp_path / f"triton_{triton.__version__.replace('.', '_')}"
    version_dir.mkdir()
    cfg = {
        "256": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=64,
                    GROUP_SIZE_M=8, num_warps=4, num_stages=3),
        "8192": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32,
                     GROUP_SIZE_M=1, num_warps=8, num_stages=4),
    }
    (version_dir / name).write_text(json.dumps(cfg))

    monkeypatch.setenv("FREETOKEN_MOE_CONFIG_DIR", str(tmp_path))
    _load_nvfp4_moe_configs.cache_clear()
    try:
        got = nvfp4_moe_config(300, gate_up_n, geo["hidden"], DEVICE_NAME, geo["experts"])
        # 300 is nearer to 256 than 8192 -> selects the 256 bucket verbatim.
        assert got == cfg["256"]
    finally:
        _load_nvfp4_moe_configs.cache_clear()
