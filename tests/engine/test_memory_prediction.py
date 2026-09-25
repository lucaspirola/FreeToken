"""engine/memory_prediction.py: the config-based startup prediction, its calibration
against the measured prefill transients, the WARN band and the runtime reserve a
fixed-size start takes out of the ratio-1.00 budget. CPU only."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest

from freetoken.engine import memory_prediction as mp
from freetoken.engine.cache_budget import net_cache_budget_bytes
from freetoken.engine.engine import _startup_kv_budget

GiB = 1024**3


def _nemotron_35_lightning():
    """NVIDIA-Nemotron-3.5-Lightning-30B-A3B geometry as parse_config builds it."""
    return SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        hidden_size=2688, hidden_act="relu2", num_experts=128, num_experts_per_tok=6,
        moe_intermediate_size=1856, intermediate_size=1856,
        shared_expert_intermediate_size=3712, first_k_dense_replace=0,
        num_qo_heads=32, num_qo_heads_per_layer=None, num_kv_heads=2, head_dim=128,
        attention_groups=(
            SimpleNamespace(kind="linear_gated_delta", state_layout="mamba2",
                            num_key_heads=8, num_value_heads=64, key_head_dim=64,
                            value_head_dim=128, track_chunk_size=128),
            SimpleNamespace(kind="full", num_kv_heads=2, head_dim=128, mla=False),
        ),
    )


def _ornith_15_35b():
    """Ornith-1.5-35B-A3B (Qwen3.5-MoE text tower) geometry; NVFP4 and EXL3 share it."""
    return SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        hidden_size=2048, hidden_act="silu", num_experts=256, num_experts_per_tok=8,
        moe_intermediate_size=512, intermediate_size=0,
        shared_expert_intermediate_size=512, first_k_dense_replace=0,
        num_qo_heads=16, num_qo_heads_per_layer=None, num_kv_heads=2, head_dim=256,
        attention_groups=(
            SimpleNamespace(kind="linear_gated_delta", state_layout="kv",
                            num_key_heads=16, num_value_heads=32, key_head_dim=128,
                            value_head_dim=128, track_chunk_size=64),
            SimpleNamespace(kind="full", num_kv_heads=2, head_dim=256, mla=False),
        ),
    )


# Measured "Prefill headroom: transient" values (8192-token chunk) the WARN band is
# chosen from: model, host, measured GiB, journal.
CALIBRATION = [
    (_nemotron_35_lightning, "mamba2", "ft-ck / ft-g5 / owner 5080", 0.65,
     "results/ck4-box/*-journal.txt (70 starts, all 0.65)"),
    (_ornith_15_35b, "gdn", "rented 5080 native Linux", 0.98,
     "results/ornith-headroom-box/ornith-*-r100-journal.txt"),
    (_ornith_15_35b, "gdn", "owner 5080 WSL2 (reserved 1.07, allocated 0.97)", 1.07,
     "results/ornith-headroom-local/ornith-hr-*-journal.txt"),
    (_ornith_15_35b, "gdn", "EXL3 Ornith 5.0bpw, owner 5080", 1.00,
     "exl3 tasks/ornith-exl3/results/step4-prefill-v2-2026-09-24/*.log"),
]


@pytest.mark.parametrize("make, peak, host, measured_gib, source", CALIBRATION)
def test_calibration_within_band(make, peak, host, measured_gib, source):
    pred = mp.predict_prefill_transient(make(), 8192)
    assert pred.peak_layer == peak
    assert not pred.coarse
    rel, outside = mp.compare(pred.bytes, int(measured_gib * GiB))
    # every calibration point is inside the WARN band, with room: the worst is -18%
    assert abs(rel) <= 0.20, (host, rel)
    assert not outside


def test_calibration_values_pinned():
    """The closed forms themselves (so a term change shows up as a diff here)."""
    nem = mp.predict_prefill_transient(_nemotron_35_lightning(), 8192)
    orn = mp.predict_prefill_transient(_ornith_15_35b(), 8192)
    assert round(nem.bytes / GiB, 3) == 0.677
    assert round(orn.bytes / GiB, 3) == 0.876
    # MoE is the runner-up on Nemotron, as the 598/128 MiB OOM sites suggested
    assert nem.layers["moe"] < nem.layers["mamba2"]
    assert nem.layers["full_attention"] < nem.layers["moe"]


def test_transient_scales_with_chunk():
    small = mp.predict_prefill_transient(_nemotron_35_lightning(), 4096).bytes
    big = mp.predict_prefill_transient(_nemotron_35_lightning(), 8192).bytes
    assert 1.9 < big / small < 2.1


def test_unknown_group_kind_is_coarse():
    mc = _ornith_15_35b()
    mc.attention_groups = mc.attention_groups + (SimpleNamespace(kind="dsv4"),)
    pred = mp.predict_prefill_transient(mc, 8192)
    assert pred.coarse and pred.unknown_kinds == ("dsv4",)


def test_family_hook_prices_its_kind(monkeypatch):
    mc = _ornith_15_35b()
    mc.attention_groups = mc.attention_groups + (SimpleNamespace(kind="dsv4"),)
    spec = SimpleNamespace(prefill_transient="fake.module:terms")
    monkeypatch.setattr("freetoken.models.register.get_model_spec", lambda arch: spec)
    monkeypatch.setattr(
        "freetoken.models.register._load_attr",
        lambda module, attr: (lambda cfg, T, a: {"dsv4": 3 * GiB}),
    )
    pred = mp.predict_prefill_transient(mc, 8192)
    assert not pred.coarse
    assert pred.peak_layer == "dsv4"
    assert pred.bytes == pred.base + 3 * GiB


def _prediction(transient_bytes, coarse=False):
    t = mp.TransientPrediction(
        chunk_tokens=8192, bytes=transient_bytes, peak_layer="x", layers={},
        base=0, coarse=coarse,
    )
    return mp.StartupPrediction(
        transient=t, linear_state_bytes=0, kv_floor_bytes=0, kv_floor_tokens=0,
        non_expert_weight_bytes=None, graph_pool_bytes=32 * 1024**2, graph_max_bs=1,
    )


def test_runtime_reserve_growable_is_zero():
    assert mp.runtime_reserve_bytes(_prediction(GiB), growable=True, baseline_free=15 * GiB) == 0


def test_runtime_reserve_fixed_size():
    from freetoken.engine.growable_kv import PREFILL_HEADROOM_MARGIN_BYTES, VMM_COMMIT_CUSHION_BYTES

    r = mp.runtime_reserve_bytes(_prediction(GiB), growable=False, baseline_free=15 * GiB)
    assert r == GiB + 32 * 1024**2 + PREFILL_HEADROOM_MARGIN_BYTES
    # a tiny transient still reserves the VMM commit cushion
    r = mp.runtime_reserve_bytes(_prediction(1), growable=False, baseline_free=15 * GiB)
    assert r == VMM_COMMIT_CUSHION_BYTES + 32 * 1024**2 + PREFILL_HEADROOM_MARGIN_BYTES


def test_runtime_reserve_coarse_keeps_old_ratio_floor():
    r = mp.runtime_reserve_bytes(_prediction(GiB, coarse=True), growable=False, baseline_free=40 * GiB)
    assert r == 4 * GiB


def test_compare_band():
    assert mp.compare(125, 100) == (0.25, False)
    assert mp.compare(126, 100)[1]
    assert mp.compare(74, 100)[1]
    assert mp.compare(5, 0) == (0.0, False)


def test_budgets_subtract_reserve():
    assert net_cache_budget_bytes(1.0, 1000, 100, 50) == 850
    assert net_cache_budget_bytes(1.0, 1000, 100, 50, 200) == 650
    # 1.00 of free minus what the model took minus the reserve
    assert _startup_kv_budget(1.0, 1000, 700, 100) == 600
    assert _startup_kv_budget(1.0, 1000, 700) == 700


def test_default_ratio_is_one():
    from freetoken.engine.config import EngineConfig

    assert EngineConfig.__dataclass_fields__["memory_ratio"].default == 1.0
    assert EngineConfig.__dataclass_fields__["runtime_reserve_bytes"].default == 0


def _write_shard(path, tensors):
    header = {}
    off = 0
    for name, nbytes in tensors.items():
        header[name] = {"dtype": "U8", "shape": [nbytes], "data_offsets": [off, off + nbytes]}
        off += nbytes
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"\0" * off)


def test_non_expert_weights_from_headers(tmp_path):
    _write_shard(tmp_path / "model-00001.safetensors", {
        "model.layers.0.mlp.experts.0.gate_proj.weight": 1000,
        "model.layers.0.mlp.shared_expert.gate_proj.weight": 30,
        "model.layers.0.self_attn.q_proj.weight": 20,
        "model.visual.blocks.0.attn.qkv.weight": 500,
    })
    _write_shard(tmp_path / "model-00002.safetensors", {"lm_head.weight": 7})
    assert mp.predict_non_expert_weight_bytes(str(tmp_path)) == 57
    assert mp.predict_non_expert_weight_bytes(str(tmp_path / "missing")) is None
