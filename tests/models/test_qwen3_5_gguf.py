from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest


def _shim():
    metadata = {
        "qwen35moe.block_count": 41,
        "qwen35moe.nextn_predict_layers": 1,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.context_length": 262144,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.rope.dimension_count": 64,
        "qwen35moe.rope.freq_base": 10_000_000.0,
        "qwen35moe.ssm.state_size": 128,
        "qwen35moe.ssm.inner_size": 4096,
        "qwen35moe.ssm.group_count": 16,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.expert_feed_forward_length": 512,
        "qwen35moe.expert_shared_feed_forward_length": 512,
    }
    return SimpleNamespace(
        metadata=metadata,
        model_path="ornith.gguf",
        vocab_size=248320,
        tie_word_embeddings=False,
        architectures=["Qwen3_5MoeGGUFForConditionalGeneration"],
    )


def test_qwen35moe_gguf_config_excludes_mtp_and_builds_hybrid_groups(monkeypatch):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen3_5_moe import gguf

    monkeypatch.setattr(reader, "gguf_tensor_type", lambda path, name: 12)
    monkeypatch.setattr(gguf, "_expert_types", lambda shim: ((12, 14),) * 40)
    config = gguf.parse_gguf_config(_shim())

    assert config.num_layers == 40
    assert config.rotary_config.max_position == 262144
    assert config.rotary_config.rotary_dim == 64
    assert config.num_experts == 256
    assert config.num_experts_per_tok == 8
    assert config.moe_intermediate_size == 512
    assert config.shared_expert_intermediate_size == 512
    assert config.expert_quant == "gguf"
    assert config.gguf_embed_quant == 12
    assert len(config.gguf_expert_types) == 40

    linear = config.linear_attention_group()
    assert linear is not None
    assert len(linear.layer_ids) == 30
    assert linear.num_key_heads == 16
    assert linear.num_value_heads == 32
    assert linear.key_head_dim == linear.value_head_dim == 128
    full = next(group for group in config.attention_groups if group.name == "full")
    assert tuple(full.layer_ids) == tuple(range(3, 40, 4))


def test_engine_yarn_override_extends_ornith_rope_and_full_group(monkeypatch):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.models.gguf import reader
    from freetoken.models.qwen3_5_moe import gguf

    monkeypatch.setattr("freetoken.engine.config.cached_load_hf_config", lambda path: _shim())
    monkeypatch.setattr(reader, "gguf_tensor_type", lambda path, name: 12)
    monkeypatch.setattr(gguf, "_expert_types", lambda shim: ((12, 14),) * 40)
    engine = EngineConfig(
        model_path="ornith.gguf",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        rope_yarn_factor=2.0,
        max_seq_len_override=524288,
    )
    config = engine.model_config
    assert config.rotary_config.max_position == 524288
    assert config.rotary_config.scaling == {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 262144,
    }
    full = next(group for group in config.attention_groups if group.name == "full")
    assert full.rotary_config is config.rotary_config
    assert engine.max_seq_len == 524288


def test_qwen35moe_gguf_registry_mapping():
    from freetoken.models import register
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY

    key = "Qwen3_5MoeGGUFForConditionalGeneration"
    assert GGUF_ARCH_TO_REGISTRY["qwen35moe"] == key
    assert key in register._MODEL_REGISTRY

    from freetoken.models.gguf.tokenizer import _TOKENIZER_ARCH

    assert _TOKENIZER_ARCH["qwen35moe"] == "qwen3_moe"


def test_inverse_v_permutation_restores_grouped_head_order(monkeypatch):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen3_5_moe import gguf

    monkeypatch.setattr(reader, "gguf_tensor_type", lambda path, name: 12)
    monkeypatch.setattr(gguf, "_expert_types", lambda shim: ((12, 14),) * 40)
    config = gguf.parse_gguf_config(_shim())
    grouped = torch.arange(32 * 3).reshape(32 * 3, 1)
    # llama.cpp stores [G0v0, G1v0, ..., G0v1, G1v1, ...].
    tiled = grouped.reshape(16, 2, 3).permute(1, 0, 2).reshape(32 * 3, 1)
    assert torch.equal(gguf._undo_v_rows(tiled, config, 3), grouped)


def test_split_linear_packs_adjacent_equal_quant_types(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q6_K, row_bytes
    from freetoken.models.qwen3_5_moe.gguf import GGUFSplitLinear

    linear = GGUFSplitLinear(256, (("qkv", 2), ("z", 3), ("a", 1)))
    linear.qkv.materialize(GGML_Q4_K)
    linear.z.materialize(GGML_Q4_K)
    linear.a.materialize(GGML_Q6_K)
    q4_row = row_bytes(256, GGML_Q4_K)
    q6_row = row_bytes(256, GGML_Q6_K)
    state = {
        "qkv.qweight": torch.full((2, q4_row), 11, dtype=torch.uint8),
        "z.qweight": torch.full((3, q4_row), 22, dtype=torch.uint8),
        "a.qweight": torch.full((1, q6_row), 33, dtype=torch.uint8),
    }
    linear.load_state_dict(state)

    assert state == {}
    assert linear._fused_groups is not None
    assert [(qt, tuple(w.shape)) for qt, w in linear._fused_groups] == [
        (GGML_Q4_K, (5, q4_row)),
        (GGML_Q6_K, (1, q6_row)),
    ]
    q4_packed = linear._fused_groups[0][1]
    assert q4_packed.untyped_storage().data_ptr() == linear.qkv.qweight.untyped_storage().data_ptr()
    assert q4_packed.untyped_storage().data_ptr() == linear.z.qweight.untyped_storage().data_ptr()
    assert torch.all(linear.qkv.qweight == 11)
    assert torch.all(linear.z.qweight == 22)

    calls = []

    def fake_mul(x, weight, quant_type):
        calls.append((quant_type, weight.shape[0]))
        return torch.full((x.shape[0], weight.shape[0]), quant_type, dtype=x.dtype)

    monkeypatch.setattr("freetoken.layers.gguf.fused_mul_mat_gguf", fake_mul)
    out = linear.forward(torch.zeros((2, 256), dtype=torch.float32))
    assert calls == [(GGML_Q4_K, 5), (GGML_Q6_K, 1)]
    assert out.shape == (2, 6)


def test_permuted_input_gguf_linear_uses_tiled_v_layout(monkeypatch):
    from freetoken.models.gguf.dequant import GGML_Q6_K
    from freetoken.models.qwen3_5_moe.gguf import PermutedInputGGUFLinear

    linear = PermutedInputGGUFLinear(
        256, 2, num_key_heads=2, num_value_heads=4, head_dim=64
    )
    linear.materialize(GGML_Q6_K)
    captured = []

    def fake_mul(x, weight, quant_type):
        captured.append(x.clone())
        return torch.zeros((x.shape[0], weight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr("freetoken.layers.gguf.fused_mul_mat_gguf", fake_mul)
    x = torch.arange(512, dtype=torch.float32).reshape(2, 256)
    linear.forward(x)

    expected = x.reshape(2, 2, 2, 64).permute(0, 2, 1, 3).reshape(2, 256)
    assert torch.equal(captured[0], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GGUF MMVQ needs CUDA")
def test_fused_q6_v_permutation_matches_explicit_layout():
    from freetoken.kernel.gguf import (
        ggml_mul_mat_vec_a8,
        ggml_mul_mat_vec_q6_permuted_a8,
    )
    from freetoken.models.gguf.dequant import GGML_Q6_K, row_bytes

    torch.manual_seed(7)
    num_key_heads, values_per_key, head_dim = 2, 2, 64
    width = num_key_heads * values_per_key * head_dim
    rows = 32
    weight = torch.randint(
        0, 256, (rows, row_bytes(width, GGML_Q6_K)), device="cuda", dtype=torch.uint8
    )
    x = torch.randn(2, width, device="cuda", dtype=torch.bfloat16)
    tiled = (
        x.reshape(2, num_key_heads, values_per_key, head_dim)
        .permute(0, 2, 1, 3)
        .reshape(2, width)
        .contiguous()
    )

    expected = ggml_mul_mat_vec_a8(weight, tiled, GGML_Q6_K, rows)
    actual = ggml_mul_mat_vec_q6_permuted_a8(
        weight, x, rows, num_key_heads, values_per_key, head_dim
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton shared-expert epilogue needs CUDA")
def test_fused_shared_expert_epilogue_matches_torch():
    from freetoken.kernel.triton.shared_expert import fused_shared_expert_add_

    torch.manual_seed(0)
    routed = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn_like(routed)
    gate = torch.randn(8, 1, device="cuda", dtype=torch.bfloat16)
    expected = routed.float() + shared.float() * torch.sigmoid(gate.float())

    actual = fused_shared_expert_add_(routed.clone(), shared, gate)

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
