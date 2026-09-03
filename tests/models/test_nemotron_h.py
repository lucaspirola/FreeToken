"""Nemotron-H (Nemotron-3-Super latent MoE, Nemotron-3.5-Lightning flat MoE).

Fixtures are built through the REAL ``transformers.NemotronHConfig`` rather than a
SimpleNamespace: transformers >= 5.8 rewrites ``layers_block_type`` on construction
(``mamba`` -> ``linear_attention``, ``attention`` -> ``full_attention``), so a hand-rolled
namespace with the legacy spellings cannot catch a parser that only knows the old names.

The ``needs_weights`` tests read the real checkpoint pointed at by
``FREETOKEN_NEMOTRON_LIGHTNING_PATH``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

from freetoken.distributed import get_tp_info, set_tp_info
from freetoken.models.nemotron_h.config import parse_config
from freetoken.models.nemotron_h.model import NemotronHForCausalLM
from freetoken.models.register import get_model_spec
from freetoken.moe.expert_banks import bank_bytes_estimate
from freetoken.moe.fused_nvfp4 import _run_act
from freetoken.utils import torch_dtype

_LIGHTNING_SLICE = ["mamba", "moe", "mamba", "moe", "mamba", "attention", "moe"]
_NVFP4 = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
_FP8 = {"quant_algo": "FP8"}


def _quantized_layers(layer_types, *, latent: bool, attn_fp8: bool) -> dict:
    """The checkpoint's per-module quant map (modelopt MIXED_PRECISION shape)."""
    out: dict[str, dict] = {}
    for i, kind in enumerate(layer_types):
        prefix = f"backbone.layers.{i}.mixer"
        if kind == "mamba":
            out[f"{prefix}.in_proj"] = dict(_FP8)
            out[f"{prefix}.out_proj"] = dict(_FP8)
        elif kind == "moe":
            out[f"{prefix}.shared_experts.up_proj"] = dict(_NVFP4)
            out[f"{prefix}.shared_experts.down_proj"] = dict(_NVFP4)
            for expert in range(2):
                out[f"{prefix}.experts.{expert}.up_proj"] = dict(_NVFP4)
                out[f"{prefix}.experts.{expert}.down_proj"] = dict(_NVFP4)
            if latent:
                out[f"{prefix}.fc1_latent_proj"] = dict(_FP8)
                out[f"{prefix}.fc2_latent_proj"] = dict(_FP8)
        elif kind == "attention" and attn_fp8:
            out[f"{prefix}.o_proj"] = dict(_FP8)
    out["lm_head"] = dict(_NVFP4)
    return out


def _hf_config(layer_types=None, *, latent=None, attn_fp8=False, **overrides):
    """Nemotron-3.5-Lightning geometry (flat MoE) unless ``latent`` is given."""
    layer_types = list(layer_types or _LIGHTNING_SLICE)
    quantized = _quantized_layers(layer_types, latent=latent is not None, attn_fp8=attn_fp8)
    kwargs = dict(
        layers_block_type=layer_types,
        quantization_config={
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt",
            "quantized_layers": quantized,
        },
        head_dim=128,
        max_position_embeddings=1048576,
        rope_theta=10000.0,
        n_groups=8,
        mamba_num_heads=64,
        mamba_head_dim=64,
        ssm_state_size=128,
        conv_kernel=4,
        chunk_size=128,
        time_step_min=0.001,
        mlp_hidden_act="relu2",
        mamba_ssm_cache_dtype="float32",
        moe_latent_size=latent,
        moe_shared_expert_intermediate_size=3712,
        n_shared_experts=1,
        num_key_value_heads=2,
        num_attention_heads=32,
        hidden_size=2688,
        vocab_size=131072,
        intermediate_size=1856,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        n_routed_experts=128,
        num_experts_per_tok=6,
        moe_intermediate_size=1856,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        architectures=["NemotronHForCausalLM"],
    )
    kwargs.update(overrides)
    return NemotronHConfig(**kwargs)


def _super_config(layer_types=None, **overrides):
    """Nemotron-3-Super geometry: latent MoE, wider Mamba state, FP8 o_proj."""
    kwargs = dict(
        latent=1024,
        attn_fp8=True,
        hidden_size=4096,
        mamba_num_heads=128,
        moe_shared_expert_intermediate_size=5376,
        n_routed_experts=512,
        num_experts_per_tok=22,
        moe_intermediate_size=2688,
        intermediate_size=2688,
        max_position_embeddings=262144,
        routed_scaling_factor=5.0,
    )
    kwargs.update(overrides)
    return _hf_config(layer_types, **kwargs)


def _meta_model(config):
    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(0, 1)
    object.__setattr__(config, "moe_backend", "offload")
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return NemotronHForCausalLM(config)


# ---------------------------------------------------------------- layer types


def test_transformers_remaps_layer_names_and_parser_normalizes_them():
    hf = _hf_config()
    # The remap is what the parser actually sees; assert it here so the fixture is
    # provably exercising it rather than the legacy spellings.
    assert hf.layers_block_type[0] == "linear_attention"
    assert hf.layers_block_type[5] == "full_attention"

    config = parse_config(hf)
    args = config.nemotron_h_args
    assert args.layer_types == ("mamba", "moe", "mamba", "moe", "mamba", "attention", "moe")
    assert config.attention_groups[0].name == "mamba"
    assert config.attention_groups[0].layer_ids == (0, 2, 4)
    assert config.attention_groups[1].layer_ids == (5,)
    assert config.moe_layer_ids == (1, 3, 6)
    assert config.num_moe_layers == 3


def test_legacy_layer_names_still_parse():
    # A config object that was never through transformers' remap (e.g. a raw shim).
    hf = _hf_config()
    raw = SimpleNamespace(**hf.to_dict())
    raw.layers_block_type = list(_LIGHTNING_SLICE)
    raw.num_hidden_layers = hf.num_hidden_layers
    assert parse_config(raw).nemotron_h_args.layer_types[0] == "mamba"


def test_mlp_block_type_is_rejected():
    hf = _hf_config(["mamba", "mlp", "attention"])
    assert hf.layers_block_type[1] == "mlp"
    with pytest.raises(ValueError, match="unsupported Nemotron-H block type"):
        parse_config(hf)


# ------------------------------------------------------------- MoE geometry


def test_lightning_has_no_moe_latent_projection():
    config = parse_config(_hf_config())
    assert config.nemotron_h_args.moe_latent_size is None
    assert config.expert_hidden_size == 2688 == config.hidden_size
    assert not config.expert_gated
    assert config.hidden_act == "relu2"

    state = _meta_model(config).state_dict()
    assert not any("latent_proj" in key for key in state)
    assert not any(".experts." in key for key in state)


def test_super_keeps_its_latent_projections():
    config = parse_config(_super_config())
    assert config.expert_hidden_size == 1024
    state = _meta_model(config).state_dict()
    assert state["backbone.layers.1.mixer.fc1_latent_proj.weight"].shape == (1024, 4096)
    assert state["backbone.layers.1.mixer.fc2_latent_proj.weight"].shape == (4096, 1024)


def test_ungated_nvfp4_bank_estimate():
    config = parse_config(_hf_config())
    hidden, inner = 2688, 1856
    per_expert = inner * (hidden // 2 + hidden // 16 + 2) + hidden * (
        inner // 2 + inner // 16 + 2
    )
    assert bank_bytes_estimate(config) == 3 * 128 * per_expert


def test_relu2_expert_activation_is_ungated():
    x = torch.tensor([[-2.0, 0.5, 3.0]])
    out = torch.empty_like(x)
    _run_act("relu2", x, out, 1.702, 7.0)
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.25, 9.0]]))


# ------------------------------------------------------------------- quant


def test_quant_axes_lightning():
    config = parse_config(_hf_config())
    args = config.nemotron_h_args
    # Only the Mamba in/out projections are FP8 here: attention is bf16 end to end, so
    # "any FP8 module in the checkpoint" must NOT set attn_quant.
    assert config.attn_quant == "none"
    assert config.expert_quant == "nvfp4"
    assert config.dense_quant == "nvfp4"
    assert config.lm_head_quant == "nvfp4"
    assert args.module_quant("backbone.layers.0.mixer.in_proj") == "fp8_pertensor"
    assert args.module_quant("backbone.layers.0.mixer.out_proj") == "fp8_pertensor"
    assert args.module_quant("backbone.layers.1.mixer.shared_experts.up_proj") == "nvfp4"
    assert args.module_quant("backbone.layers.1.mixer.shared_experts.down_proj") == "nvfp4"
    assert args.module_quant("lm_head") == "nvfp4"
    assert args.module_quant("backbone.layers.5.mixer.o_proj") == "none"
    # Routed experts never become resident modules.
    assert args.module_quant("backbone.layers.1.mixer.experts.0.up_proj") == "none"


def test_quant_axes_super_keeps_fp8_attention():
    config = parse_config(_super_config())
    assert config.attn_quant == "fp8_pertensor"
    args = config.nemotron_h_args
    assert args.module_quant("backbone.layers.5.mixer.o_proj") == "fp8_pertensor"
    assert args.module_quant("backbone.layers.1.mixer.fc1_latent_proj") == "fp8_pertensor"


def test_dense_dequant_escape_hatch(monkeypatch):
    monkeypatch.setenv("FREETOKEN_NEMOTRON_DENSE_DEQUANT", "1")
    config = parse_config(_hf_config())
    args = config.nemotron_h_args
    assert config.dense_quant == "none" and config.lm_head_quant == "none"
    assert args.module_quant("backbone.layers.1.mixer.shared_experts.up_proj") == "dequant_bf16"
    assert args.module_quant("lm_head") == "dequant_bf16"

    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear, Nvfp4LMHead

    model = _meta_model(config)
    shared = model.backbone.layers.op_list[1].mixer.shared_experts
    assert not isinstance(shared.up_proj, Nvfp4DenseLinear)
    assert not isinstance(model.lm_head, Nvfp4LMHead)
    state = model.state_dict()
    assert state["backbone.layers.1.mixer.shared_experts.up_proj.weight"].dtype == torch.bfloat16
    assert state["lm_head.weight"].dtype == torch.bfloat16


def test_config_asserts():
    with pytest.raises(ValueError, match="relu2"):
        parse_config(_hf_config(mlp_hidden_act="silu"))
    with pytest.raises(ValueError, match="group-limited routing"):
        parse_config(_hf_config(n_group=2))
    with pytest.raises(ValueError, match="group-limited routing"):
        parse_config(_hf_config(topk_group=4))
    with pytest.raises(ValueError, match="one shared expert"):
        parse_config(_hf_config(n_shared_experts=2))


def test_non_fp32_ssm_cache_dtype_warns(capsys):
    # FreeToken's logger writes to its own stdout handler, not through propagation.
    parse_config(_hf_config(mamba_ssm_cache_dtype="bfloat16"))
    captured = capsys.readouterr().out
    assert "mamba_ssm_cache_dtype=bfloat16" in captured
    capsys.readouterr()
    parse_config(_hf_config())
    assert "mamba_ssm_cache_dtype" not in capsys.readouterr().out


# ------------------------------------------------------- single-stream gate


def _full(layer_types):
    return layer_types


def test_single_stream_only_tracks_state_bytes(monkeypatch):
    # Lightning: 23 mamba layers x (conv 6144x3 bf16 + 64x64x128 fp32) = ~47 MiB/slot.
    lightning = parse_config(_hf_config(["mamba", "moe"] * 23 + ["attention"] * 6))
    assert not lightning.single_stream_only

    # Super: 128 mamba heads doubles the recurrent state -> ~4.06 MiB per mamba layer.
    super_cfg = parse_config(_super_config(["mamba", "moe"] * 30))
    assert super_cfg.single_stream_only

    monkeypatch.setenv("FREETOKEN_NEMOTRON_MULTI_STREAM", "1")
    assert not parse_config(_super_config(["mamba", "moe"] * 30)).single_stream_only


def test_state_bytes_per_slot_matches_the_pool_arithmetic():
    from freetoken.models.nemotron_h.config import _state_bytes_per_slot

    config = parse_config(_hf_config(["mamba", "moe"] * 23 + ["attention"] * 6))
    group = config.attention_groups[0]
    conv_dim = 2 * 8 * 128 + 64 * 64
    expected = 23 * (conv_dim * 3 * 2 + 64 * 128 * 64 * 4)
    assert _state_bytes_per_slot(group) == expected
    assert expected < 96 * 1024 * 1024


# ------------------------------------------------------------- built model


def test_offload_model_has_no_resident_expert_tensors():
    config = parse_config(_hf_config())
    state = _meta_model(config).state_dict()
    assert not any(".experts." in key for key in state)
    assert state["backbone.layers.0.mixer.in_proj.weight"].dtype == torch.float8_e4m3fn
    assert state["backbone.layers.1.mixer.gate.weight"].shape == (128, 2688)


def test_dense_nvfp4_modules_and_lm_head_are_native():
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear, Nvfp4LMHead

    config = parse_config(_hf_config())
    model = _meta_model(config)
    shared = model.backbone.layers.op_list[1].mixer.shared_experts
    assert isinstance(shared.up_proj, Nvfp4DenseLinear)
    assert isinstance(shared.down_proj, Nvfp4DenseLinear)
    assert isinstance(model.lm_head, Nvfp4LMHead)

    state = model.state_dict()
    up = "backbone.layers.1.mixer.shared_experts.up_proj"
    down = "backbone.layers.1.mixer.shared_experts.down_proj"
    assert state[f"{up}.weight"].shape == (3712, 2688 // 2)
    assert state[f"{up}.weight"].dtype == torch.uint8
    assert state[f"{up}.weight_scale"].shape == (3712, 2688 // 16)
    assert state[f"{up}.weight_scale"].dtype == torch.float8_e4m3fn
    assert state[f"{up}.weight_global"].shape == (3712,)
    assert state[f"{up}.weight_global"].dtype == torch.float16
    assert state[f"{down}.weight"].shape == (2688, 3712 // 2)
    assert state["lm_head.weight"].shape == (131072, 2688 // 2)
    assert state["lm_head.weight_scale"].shape == (131072, 2688 // 16)
    assert state["lm_head.weight_global"].shape == (131072,)


def test_mamba_mixer_clamps_dt_on_prefill_only():
    config = parse_config(_hf_config())
    mixer = _meta_model(config).backbone.layers.op_list[0].mixer
    assert mixer.dt_limit == (0.001, float("inf"))
    # The single-step update must stay unclamped (HF reference + flashinfer parity).
    import inspect

    src = inspect.getsource(type(mixer)._decode_scan)
    assert "dt_limit" not in src


def test_registry_entry():
    spec = get_model_spec("NemotronHForCausalLM")
    assert spec.module == "freetoken.models.nemotron_h"


# ------------------------------------------------------------ weight loader


def _write_synthetic_checkpoint(tmp_path: Path) -> Path:
    """A miniature Nemotron-3.5-shaped checkpoint: one mamba, one moe, one attention
    layer, plus the tensors the loader must drop (routed experts, mtp, KV scales)."""
    from safetensors.torch import save_file

    H, VOCAB, SHARED, INNER = 32, 64, 64, 32
    layers = ["mamba", "moe", "attention"]
    hf = _hf_config(
        layers,
        hidden_size=H,
        vocab_size=VOCAB,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=16,
        moe_shared_expert_intermediate_size=SHARED,
        moe_intermediate_size=INNER,
        intermediate_size=INNER,
        n_routed_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=1024,
    )
    (tmp_path / "config.json").write_text(json.dumps(hf.to_dict()))

    fp8 = torch.float8_e4m3fn
    conv_dim = 4 * 8 + 2 * 2 * 16
    in_dim = 4 * 8 + conv_dim + 4

    def fp4(out: int, inn: int):
        return {
            ".weight": torch.randint(0, 255, (out, inn // 2), dtype=torch.uint8),
            ".weight_scale": torch.ones(out, inn // 16).to(fp8),
            ".weight_scale_2": torch.tensor([0.03]),
        }

    tensors: dict[str, torch.Tensor] = {
        "backbone.embeddings.weight": torch.randn(VOCAB, H, dtype=torch.bfloat16),
        "backbone.norm_f.weight": torch.randn(H, dtype=torch.bfloat16),
        "backbone.layers.0.norm.weight": torch.randn(H, dtype=torch.bfloat16),
        "backbone.layers.0.mixer.in_proj.weight": torch.randn(in_dim, H).to(fp8),
        "backbone.layers.0.mixer.in_proj.weight_scale": torch.tensor([0.5]),
        "backbone.layers.0.mixer.in_proj.input_scale": torch.tensor([0.25]),
        "backbone.layers.0.mixer.out_proj.weight": torch.randn(H, 32).to(fp8),
        "backbone.layers.0.mixer.out_proj.weight_scale": torch.tensor([0.5]),
        "backbone.layers.0.mixer.conv1d.weight": torch.randn(conv_dim, 1, 4),
        "backbone.layers.0.mixer.conv1d.bias": torch.randn(conv_dim),
        "backbone.layers.0.mixer.dt_bias": torch.randn(4),
        "backbone.layers.0.mixer.A_log": torch.randn(4),
        "backbone.layers.0.mixer.D": torch.randn(4),
        "backbone.layers.0.mixer.norm.weight": torch.randn(32),
        "backbone.layers.1.mixer.gate.weight": torch.randn(4, H),
        "backbone.layers.1.mixer.gate.e_score_correction_bias": torch.randn(4),
        "backbone.layers.2.mixer.q_proj.weight": torch.randn(32, H, dtype=torch.bfloat16),
        "backbone.layers.2.mixer.k_proj.weight": torch.randn(16, H, dtype=torch.bfloat16),
        "backbone.layers.2.mixer.v_proj.weight": torch.randn(16, H, dtype=torch.bfloat16),
        "backbone.layers.2.mixer.o_proj.weight": torch.randn(H, 32, dtype=torch.bfloat16),
        "backbone.layers.2.mixer.k_proj.k_scale": torch.tensor([1.0]),
        "backbone.layers.2.mixer.v_proj.v_scale": torch.tensor([1.0]),
        "mtp.layers.0.mixer.o_proj.weight": torch.randn(H, 32, dtype=torch.bfloat16),
    }
    for suffix, tensor in fp4(SHARED, H).items():
        tensors[f"backbone.layers.1.mixer.shared_experts.up_proj{suffix}"] = tensor
    for suffix, tensor in fp4(H, SHARED).items():
        tensors[f"backbone.layers.1.mixer.shared_experts.down_proj{suffix}"] = tensor
    for suffix, tensor in fp4(INNER, H).items():
        tensors[f"backbone.layers.1.mixer.experts.0.up_proj{suffix}"] = tensor
    for suffix, tensor in fp4(VOCAB, H).items():
        tensors[f"lm_head{suffix}"] = tensor

    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def test_iter_weights_keeps_dense_nvfp4_native(tmp_path):
    from freetoken.models.nemotron_h.weight import iter_weights

    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(0, 1)
    path = _write_synthetic_checkpoint(tmp_path)
    emitted = dict(
        iter_weights(str(path), torch.device("cpu"), include_moe_experts=False,
                     include_non_moe=True)
    )

    # Routed experts, MTP and the (unused) FP8 KV calibration scales never reach the model.
    assert not any(".experts." in key for key in emitted)
    assert not any(key.startswith("mtp.") for key in emitted)
    assert not any(key.endswith((".k_scale", ".v_scale")) for key in emitted)

    up = "backbone.layers.1.mixer.shared_experts.up_proj"
    assert emitted[f"{up}.weight"].dtype == torch.uint8
    assert emitted[f"{up}.weight"].shape == (64, 16)
    assert emitted[f"{up}.weight_scale"].dtype == torch.float8_e4m3fn
    assert emitted[f"{up}.weight_scale"].shape == (64, 2)
    assert emitted[f"{up}.weight_global"].dtype == torch.float16
    assert emitted[f"{up}.weight_global"].shape == (64,)
    torch.testing.assert_close(
        emitted[f"{up}.weight_global"],
        torch.full((64,), 0.03, dtype=torch.float16),
        rtol=1e-3, atol=1e-3,
    )
    assert emitted["backbone.layers.1.mixer.shared_experts.down_proj.weight"].shape == (32, 32)
    assert emitted["lm_head.weight"].dtype == torch.uint8
    assert emitted["lm_head.weight_global"].shape == (64,)

    # FP8 Mamba projections keep their checkpoint dtype with a per-row scale vector.
    assert emitted["backbone.layers.0.mixer.in_proj.weight"].dtype == torch.float8_e4m3fn
    assert emitted["backbone.layers.0.mixer.in_proj.weight_scale"].shape == (
        emitted["backbone.layers.0.mixer.in_proj.weight"].shape[0],
    )
    assert emitted["backbone.layers.0.mixer.in_proj.input_scale"].shape == ()

    # q/k/v are still fused into one qkv_proj.
    assert emitted["backbone.layers.2.mixer.qkv_proj.weight"].shape == (64, 32)
    assert not any(".q_proj." in key for key in emitted)


def test_iter_weights_dequantizes_dense_nvfp4_under_the_escape_hatch(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("the NVFP4 dequant kernel is CUDA-only")
    from freetoken.models.nemotron_h.weight import iter_weights

    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(0, 1)
    monkeypatch.setenv("FREETOKEN_NEMOTRON_DENSE_DEQUANT", "1")
    path = _write_synthetic_checkpoint(tmp_path)
    emitted = dict(
        iter_weights(str(path), torch.device("cpu"), include_moe_experts=False,
                     include_non_moe=True)
    )
    up = "backbone.layers.1.mixer.shared_experts.up_proj"
    assert emitted[f"{up}.weight"].dtype == torch.bfloat16
    assert emitted[f"{up}.weight"].shape == (64, 32)
    assert f"{up}.weight_global" not in emitted
    assert emitted["lm_head.weight"].dtype == torch.bfloat16


# ------------------------------------------------- real checkpoint (gated)

_REAL_PATH_ENV = "FREETOKEN_NEMOTRON_LIGHTNING_PATH"


def _real_path() -> Path:
    raw = os.environ.get(_REAL_PATH_ENV)
    if not raw or not Path(raw).is_dir():
        pytest.skip(f"set {_REAL_PATH_ENV} to a local Nemotron-3.5-Lightning checkpoint")
    return Path(raw)


@pytest.mark.needs_weights
def test_real_lightning_config_parses_and_builds_on_meta():
    from transformers import AutoConfig

    path = _real_path()
    config = parse_config(AutoConfig.from_pretrained(str(path)))
    assert config.num_layers == 52
    assert config.hidden_size == 2688 == config.expert_hidden_size
    assert config.num_experts == 128 and config.num_experts_per_tok == 6
    assert config.moe_intermediate_size == 1856
    assert config.shared_expert_intermediate_size == 3712
    assert len(config.attention_groups[0].layer_ids) == 23
    assert config.attention_groups[1].layer_ids == (5, 12, 19, 26, 33, 42)
    assert len(config.moe_layer_ids) == 23
    assert config.attn_quant == "none"
    assert config.dense_quant == "nvfp4" and config.lm_head_quant == "nvfp4"
    assert not config.single_stream_only
    assert config.nemotron_h_args.time_step_min == 0.001

    state = _meta_model(config).state_dict()
    assert not any(".experts." in key for key in state)
    assert not any("latent_proj" in key for key in state)
    assert state["lm_head.weight"].shape == (131072, 1344)


@pytest.mark.needs_weights
def test_real_dense_nvfp4_matches_the_dequant_reference():
    if not torch.cuda.is_available():
        pytest.skip("NVFP4 W4A16 kernels are CUDA-only")
    import safetensors
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear, Nvfp4LMHead
    from freetoken.models.qwen3_5_moe.weight import _dequant_nvfp4_weight, _nvfp4_parts

    path = _real_path()
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 1024**3:
        pytest.skip("needs ~6 GiB free VRAM for the bf16 lm_head reference")

    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    shared_base = "backbone.layers.1.mixer.shared_experts.up_proj"

    def cosine(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return float(torch.dot(a, b) / (a.norm() * b.norm()))

    # --- shared expert (dense NVFP4 -> Nvfp4DenseLinear) ---
    with safetensors.safe_open(
        str(path / index[shared_base + ".weight"]), framework="pt", device="cuda"
    ) as f:
        w, scale, glob = _nvfp4_parts(f, shared_base)
        scale_2 = f.get_tensor(shared_base + ".weight_scale_2")
    out_features, half_in = w.shape
    linear = Nvfp4DenseLinear(half_in * 2, out_features)
    linear.load_state_dict(
        {"weight": w, "weight_scale": scale, "weight_global": glob}
    )
    ref_w = _dequant_nvfp4_weight(w, scale, scale_2)
    x = torch.randn(8, half_in * 2, dtype=torch.bfloat16, device="cuda") * 0.05
    got = linear.forward(x)
    ref = (x.float() @ ref_w.float().t()).to(torch.bfloat16)
    assert got.shape == ref.shape
    assert cosine(got, ref) > 0.999
    del ref_w, linear, w, scale, glob

    # --- lm_head (dense NVFP4 -> Nvfp4LMHead) ---
    with safetensors.safe_open(
        str(path / index["lm_head.weight"]), framework="pt", device="cuda"
    ) as f:
        w, scale, glob = _nvfp4_parts(f, "lm_head")
        scale_2 = f.get_tensor("lm_head.weight_scale_2")
    head = Nvfp4LMHead(w.shape[0], w.shape[1] * 2)
    head.load_state_dict({"weight": w, "weight_scale": scale, "weight_global": glob})
    ref_w = _dequant_nvfp4_weight(w, scale, scale_2)
    x = torch.randn(4, w.shape[1] * 2, dtype=torch.bfloat16, device="cuda") * 0.05

    import freetoken.core as core

    ctx = SimpleNamespace(batch=SimpleNamespace(is_prefill=False))
    prev = core.get_global_ctx
    core.get_global_ctx = lambda: ctx
    try:
        logits = head.forward(x)
    finally:
        core.get_global_ctx = prev
    ref = (x.float() @ ref_w.float().t()).to(torch.bfloat16)
    assert logits.shape == ref.shape
    assert cosine(logits, ref) > 0.999
    assert torch.equal(logits.argmax(-1), ref.argmax(-1))
