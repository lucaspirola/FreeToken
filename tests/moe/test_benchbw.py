"""CPU-only geometry tests for ``ft bench bw`` (no CUDA, no kernels launched).

Covers the ungated (up+down only) expert layout added for Nemotron-3.5-Lightning:
the gate_up-side banks are ``[I, H]``, not SwiGLU's ``[2I, H]``.
"""

import pytest
import torch

from freetoken.moe.benchbw import (
    WORKLOADS,
    Workload,
    _expert_bytes,
    _offload_bank_specs,
)

# Nemotron-3.5-Lightning routed-expert geometry.
hidden, inter = 2688, 1856
LIGHTNING_MOE_LAYERS = 23


def test_ungated_nvfp4_expert_bytes_matches_hand_count():
    # per expert: up [I, H] + down [H, I], each e2m1-packed (2 weights/byte) with
    # one-byte per-16 block scales and a 2-byte fp16 row global.
    up = inter * (hidden // 2) + inter * (hidden // 16) + inter * 2
    down = hidden * (inter // 2) + hidden * (inter // 16) + hidden * 2
    assert up == 2_809_984 and down == 2_811_648
    assert _expert_bytes("nvfp4", hidden, inter, gated=False) == up + down == 5_621_632


def test_ungated_nvfp4_is_gated_minus_the_gate_half():
    gated = _expert_bytes("nvfp4", hidden, inter, gated=True)
    ungated = _expert_bytes("nvfp4", hidden, inter, gated=False)
    gate_half = inter * (hidden // 2) + inter * (hidden // 16) + inter * 2
    assert gated - ungated == gate_half
    assert gated == 8_431_616


def test_lightning_preset_total_bank_size_is_15_4_gib():
    wl = WORKLOADS["nemotron3.5-lightning"]
    assert (wl.hidden, wl.inter, wl.experts, wl.top_k) == (hidden, inter, 128, 6)
    assert wl.gated is False and wl.activation == "relu2" and wl.formats == ("nvfp4",)
    total = _expert_bytes("nvfp4", wl.hidden, wl.inter, wl.gated) * wl.experts * LIGHTNING_MOE_LAYERS
    assert total / 2**30 == pytest.approx(15.4, abs=0.05)


def test_gated_default_unchanged_for_existing_presets():
    for name in ("qwen3.6-moe", "glm4.7-nvfp4", "minimax-m2.5", "gpt-oss-120b", "dsv4"):
        wl = WORKLOADS[name]
        assert wl.gated is True
        for fmt in wl.formats:
            # the default argument and the preset's flag agree, and both equal the
            # historical 2I-rowed sizing
            assert (_expert_bytes(fmt, wl.hidden, wl.inter)
                    == _expert_bytes(fmt, wl.hidden, wl.inter, wl.gated))
    # bf16 gated sizing is exactly 3 * I * H elements (gate|up + down)
    assert _expert_bytes("bf16", 2048, 768, True) == 3 * 768 * 2048 * 2


def test_offload_bank_specs_element_counts_gated_and_ungated():
    u8, f16 = torch.uint8, torch.float16
    gated = _offload_bank_specs("nvfp4", hidden, inter, gated=True)
    ungated = _offload_bank_specs("nvfp4", hidden, inter, gated=False)
    assert set(gated) == set(ungated)
    assert gated == {
        "gate_up_packed": (2 * inter * (hidden // 2), u8),
        "gate_up_scale": (2 * inter * (hidden // 16), u8),
        "gate_up_global": (2 * inter, f16), "down_packed": (hidden * (inter // 2), u8),
        "down_scale": (hidden * (inter // 16), u8), "down_global": (hidden, f16),
    }
    assert ungated == {
        "gate_up_packed": (inter * (hidden // 2), u8),
        "gate_up_scale": (inter * (hidden // 16), u8),
        "gate_up_global": (inter, f16), "down_packed": (hidden * (inter // 2), u8),
        "down_scale": (hidden * (inter // 16), u8), "down_global": (hidden, f16),
    }
    # down-side banks are identical either way; every gate_up-side bank halves
    for name in ("down_packed", "down_scale", "down_global"):
        assert gated[name] == ungated[name]
    for name in ("gate_up_packed", "gate_up_scale", "gate_up_global"):
        assert gated[name][0] == 2 * ungated[name][0]


def test_ungated_bf16_specs_drop_the_gate_half():
    specs = _offload_bank_specs("bf16", hidden, inter, gated=False)
    bf16 = torch.bfloat16
    assert specs == {"gate_up": (inter * hidden, bf16), "down": (hidden * inter, bf16)}
    assert _expert_bytes("bf16", hidden, inter, gated=False) == 2 * inter * hidden * 2


@pytest.mark.parametrize("fmt", ["fp8_block", "mxfp4_triton", "ds_fp4"])
def test_ungated_refused_for_formats_without_a_real_layout(fmt):
    # Only bf16/nvfp4 have an ungated layout in the runtime; the others must refuse
    # rather than silently report a made-up (wrong) byte count.
    with pytest.raises(NotImplementedError):
        _offload_bank_specs(fmt, hidden, inter, gated=False)
    _offload_bank_specs(fmt, hidden, inter, gated=True)  # gated still works


def test_report_and_json_survive_a_missing_cpu_moe_result():
    import json

    from freetoken.moe.benchbw import _note, _print_kernels, _print_report

    entry = {
        "expert_bytes": _expert_bytes("nvfp4", hidden, inter, gated=False), "synth_experts": 128,
        "cpu_moe_gbs": None, "cpu_moe_isa": None, "isa_sweep": None,
        "pcie_gather_gbs": 21.5, "cpu_moe_overlap_gbs": None,
        "pcie_gather_overlap_gbs": None, "ratio": None, "recommended": "offload", "note": None,
    }
    _note(entry, "cpu moe unavailable (no avx512)")
    assert entry["note"] == "cpu moe unavailable (no avx512)"
    _print_kernels({"nvfp4": entry}, 0)
    wl = WORKLOADS["nemotron3.5-lightning"]
    report = {
        "host": "h", "gpu": {"index": 0, "name": "gpu"},
        "cpu": {"physical_cores": 8, "threads_used": 8}, "threshold": 2.0,
        "ceilings": {"cpu_stream_read_gbs": 30.0, "pcie_linear_h2d_gbs": 25.0,
                     "pcie_linear_d2h_gbs": 25.0},
        "dtypes": {}, "dtype_kernels": {},
        "workloads": {wl.name: {
            "model": {"name": wl.name, "hidden": wl.hidden, "inter": wl.inter,
                      "experts": wl.experts, "top_k": wl.top_k},
            "kernels": {"nvfp4": entry},
            "recommended_moe_backend": {"nvfp4": "offload"},
        }},
        "out_path": "/dev/null",
    }
    _print_report(report)
    assert json.loads(json.dumps(report))["workloads"][wl.name]["kernels"]["nvfp4"]["ratio"] is None


def test_workload_gated_field_defaults_true():
    assert Workload("x", 64, 32, 4, 2, ("bf16",)).gated is True
