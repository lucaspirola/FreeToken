"""CPU-only coverage for tune_nvfp4_moe's selectable geometry and, critically, its
``--write`` path: the tool that actually WRITES the tuned JSON the server loads.

No GPU, no server, no measurement loop: `_tune_prefill`'s Triton/torch.cuda timing is
never called. The write path (`_write_configs`) is exercised directly against a
hand-built `tables` dict shaped exactly like what `_tune_prefill` would hand it, with
`torch.cuda.get_device_name` monkeypatched so no CUDA query happens at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import tune_nvfp4_moe as tuner  # noqa: E402

from freetoken.moe.fused_nvfp4 import (  # noqa: E402
    PREFILL_CONFIG_KEYS,
    _load_nvfp4_moe_configs,
    nvfp4_config_filename,
    nvfp4_moe_config,
)

DEVICE_NAME_SPACED = "NVIDIA GeForce RTX 5080"
DEVICE_NAME = "NVIDIA_GeForce_RTX_5080"


@pytest.fixture(autouse=True)
def _restore_globals():
    saved = (tuner.H, tuner.I, tuner.E, tuner.TOP_K, tuner.ACTIVATION, tuner.GATED)
    yield
    tuner.H, tuner.I, tuner.E, tuner.TOP_K, tuner.ACTIVATION, tuner.GATED = saved


def _apply(model: str) -> dict:
    args = tuner.parse_args(["--model", model])
    geo = tuner.resolve_geometry(args)
    tuner.H, tuner.I, tuner.E, tuner.TOP_K, tuner.ACTIVATION, tuner.GATED = (
        geo["hidden"], geo["intermediate"], geo["experts"], geo["top_k"],
        geo["activation"], geo["gated"],
    )
    args.experts = geo["experts"]
    return geo


# --- geometry resolution shared with bench_moe_prefill_gemm.py (same source module) ---


def test_no_flags_reproduces_nemotron_exactly():
    args = tuner.parse_args([])
    geo = tuner.resolve_geometry(args)
    assert geo == dict(hidden=2688, intermediate=1856, experts=128, top_k=6,
                        activation="relu2", gated=False)


def test_model_ornith_sets_the_whole_group():
    args = tuner.parse_args(["--model", "ornith"])
    geo = tuner.resolve_geometry(args)
    assert geo == dict(hidden=2048, intermediate=512, experts=256, top_k=8,
                        activation="silu", gated=True)


def test_shares_the_exact_geometry_table_with_bench_script():
    """The two scripts must read one table, not two copies that can drift apart."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import bench_moe_prefill_gemm as bench

    assert tuner.MODEL_GEOMETRIES is bench.MODEL_GEOMETRIES


# --- bank / gemm-tuple shapes (no CUDA calls; only shapes and Python objects) ----------


def test_banks_gated_doubles_gate_up_n_only():
    _apply("ornith")
    (gup, gup_s, gup_g), (down, down_s, down_g) = tuner._banks(256, "cpu", 0)
    H, I = tuner.H, tuner.I
    assert gup.shape == (256, 2 * I, H // 2)
    assert gup_s.shape == (256, 2 * I, H // 16)
    assert gup_g.shape == (256, 2 * I)
    assert down.shape == (256, H, I // 2)
    assert down_s.shape == (256, H, I // 16)
    assert down_g.shape == (256, H)


def test_banks_ungated_shapes_byte_identical_to_nemotron():
    _apply("nemotron")
    (gup, gup_s, gup_g), (down, down_s, down_g) = tuner._banks(128, "cpu", 0)
    H, I = tuner.H, tuner.I
    assert gup.shape == (128, I, H // 2)
    assert down.shape == (128, H, I // 2)


def test_tune_prefill_table_keys_ornith_are_n_k_not_swapped():
    """The (N, K) tuple `_tune_prefill` would key `tables` on, without running it."""
    _apply("ornith")
    n = tuner._shared_gate_up_n(tuner.I, tuner.GATED)
    assert (n, tuner.H) == (1024, 2048)  # gate_up
    assert (tuner.H, tuner.I) == (2048, 512)  # down


# --- the critical part: --write's exact filenames, structure and association ----------


def _fake_tables(n_gate_up: int, h: int, i: int) -> dict:
    """A `tables` dict shaped exactly like `_tune_prefill`'s return, with a distinct
    (fake) config per GEMM per M-bucket so a swap between the two files is detectable."""
    def cfg(block_m, tag):
        return {
            "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": tag, "BLOCK_SIZE_KB": 32,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
        }

    return {
        (n_gate_up, h): {256: cfg(64, tag=111), 8192: cfg(128, tag=222)},
        (h, i): {256: cfg(64, tag=333), 8192: cfg(128, tag=444)},
    }


def test_write_configs_ornith_exact_filenames_and_association(tmp_path, monkeypatch):
    geo = _apply("ornith")
    n = tuner._shared_gate_up_n(geo["intermediate"], geo["gated"])
    assert n == 1024 and geo["hidden"] == 2048 and geo["intermediate"] == 512

    tables = _fake_tables(n, geo["hidden"], geo["intermediate"])

    import torch
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: DEVICE_NAME_SPACED)

    args = tuner.parse_args(["--model", "ornith"])
    args.experts = geo["experts"]
    args.config_dir = str(tmp_path)
    args.merge = False

    written = tuner._write_configs(tables, args, device=0)
    written_names = {p.name for p in written}

    gate_up_name = f"nvfp4,E=256,N=1024,K=2048,device_name={DEVICE_NAME}.json"
    down_name = f"nvfp4,E=256,N=2048,K=512,device_name={DEVICE_NAME}.json"
    assert written_names == {gate_up_name, down_name}
    # Exactly what nvfp4_config_filename() itself produces -- not a near-miss.
    assert gate_up_name == nvfp4_config_filename(256, 1024, 2048, DEVICE_NAME)
    assert down_name == nvfp4_config_filename(256, 2048, 512, DEVICE_NAME)

    import triton
    version_dir = tmp_path / f"triton_{triton.__version__.replace('.', '_')}"
    gate_up_path = version_dir / gate_up_name
    down_path = version_dir / down_name
    assert gate_up_path.exists() and down_path.exists()

    gate_up_payload = json.loads(gate_up_path.read_text())
    down_payload = json.loads(down_path.read_text())

    # Association: gate_up's tiles (tag 111/222) must land in the N=1024,K=2048 file,
    # down's (tag 333/444) in the N=2048,K=512 file -- not swapped.
    assert gate_up_payload["256"]["BLOCK_SIZE_N"] == 111
    assert gate_up_payload["8192"]["BLOCK_SIZE_N"] == 222
    assert down_payload["256"]["BLOCK_SIZE_N"] == 333
    assert down_payload["8192"]["BLOCK_SIZE_N"] == 444

    # Every written entry has exactly PREFILL_CONFIG_KEYS, nothing else -- what
    # `nvfp4_moe_config` / `_load_nvfp4_moe_configs` expect back.
    for payload in (gate_up_payload, down_payload):
        for bucket_cfg in payload.values():
            assert set(bucket_cfg) == set(PREFILL_CONFIG_KEYS)

    # Round-trip: nvfp4_moe_config() reads exactly what was written, from the right file.
    monkeypatch.setenv("FREETOKEN_MOE_CONFIG_DIR", str(tmp_path))
    _load_nvfp4_moe_configs.cache_clear()
    try:
        got_gate_up = nvfp4_moe_config(300, 1024, 2048, DEVICE_NAME, 256)
        got_down = nvfp4_moe_config(300, 2048, 512, DEVICE_NAME, 256)
        assert got_gate_up["BLOCK_SIZE_N"] == 111  # nearer to the 256 bucket
        assert got_down["BLOCK_SIZE_N"] == 333
        # And the two shapes are NOT interchangeable: asking for the down shape must
        # never return a gate_up-tuned tile.
        assert got_gate_up != got_down
    finally:
        _load_nvfp4_moe_configs.cache_clear()


def test_write_configs_nemotron_no_flags_shape_unchanged(tmp_path, monkeypatch):
    """No-flags geometry still produces the historical Nemotron filenames."""
    _apply("nemotron")
    tables = _fake_tables(tuner.I, tuner.H, tuner.I)  # ungated: gate_up N == I

    import torch
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: DEVICE_NAME_SPACED)

    args = tuner.parse_args([])
    args.experts = tuner.E
    args.config_dir = str(tmp_path)
    args.merge = False

    written = tuner._write_configs(tables, args, device=0)
    names = {p.name for p in written}
    assert names == {
        f"nvfp4,E=128,N=1856,K=2688,device_name={DEVICE_NAME}.json",
        f"nvfp4,E=128,N=2688,K=1856,device_name={DEVICE_NAME}.json",
    }
