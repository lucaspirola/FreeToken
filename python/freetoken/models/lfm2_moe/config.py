"""LiquidAI LFM2 MoE (``Lfm2MoeForCausalLM``, model_type ``lfm2_moe``): LFM2.5-24B-A2B and siblings.

Layers are either a gated short convolution (``conv``) or GQA attention with per-head
q/k RMSNorm (``full_attention``). The first ``num_dense_layers`` feed-forwards are dense
SwiGLU; the rest are sigmoid-routed MoE with a selection-only expert bias.
"""

from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import QuantConfig
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_expert_quant,
)

_LAYER_KINDS = ("conv", "full_attention")


def _get(hf_config: Any, key: str, default: Any = None) -> Any:
    return getattr(hf_config, key, default)


def layer_types(hf_config: Any) -> tuple[str, ...]:
    """``layer_types`` from the config, or exllamav3's fallback (every 4th layer is attention)."""
    n = int(hf_config.num_hidden_layers)
    kinds = _get(hf_config, "layer_types")
    if kinds is None:
        return tuple("full_attention" if (i + 1) % 4 == 0 else "conv" for i in range(n))
    kinds = tuple(str(k) for k in kinds)
    if len(kinds) != n:
        raise ValueError(f"lfm2_moe: {len(kinds)} layer_types for {n} layers")
    unknown = sorted(set(kinds) - set(_LAYER_KINDS))
    if unknown:
        raise ValueError(f"lfm2_moe: unknown layer types {unknown} (known: {list(_LAYER_KINDS)})")
    return kinds


def _rope(hf_config: Any) -> tuple[float, dict | None]:
    params = _get(hf_config, "rope_parameters") or {}
    theta = params.get("rope_theta", _get(hf_config, "rope_theta", 1000000.0))
    rope_type = params.get("rope_type", "default")
    scaling = None if rope_type in (None, "default") else {
        k: v for k, v in params.items() if not isinstance(v, (list, dict))
    }
    return float(theta), scaling


def _has_quantized_lm_head(hf_config: Any) -> bool:
    """True when the checkpoint stores its own (quantized) ``lm_head``: exllamav3 writes one
    unless ``head_bits`` is 16, even though the HF config ties the embeddings."""
    if _get(hf_config, "quantization_config") is None:
        return False
    return QuantConfig.from_hf(hf_config).scheme_for_name("lm_head") is not None


def parse_config(hf_config: Any) -> ModelConfig:
    kinds = layer_types(hf_config)
    conv_ids = tuple(i for i, k in enumerate(kinds) if k == "conv")
    attn_ids = tuple(i for i, k in enumerate(kinds) if k == "full_attention")

    hidden = int(hf_config.hidden_size)
    num_heads = int(hf_config.num_attention_heads)
    num_kv_heads = int(_get(hf_config, "num_key_value_heads", num_heads))
    head_dim = int(_get(hf_config, "head_dim", None) or hidden // num_heads)
    if hidden % num_heads:
        raise ValueError(f"lfm2_moe: hidden {hidden} is not a multiple of {num_heads} heads")
    if bool(_get(hf_config, "conv_bias", False)):
        raise NotImplementedError("lfm2_moe: conv_bias=true is not supported (LFM2.5 ships none)")
    hidden_act = str(_get(hf_config, "hidden_act", "silu"))
    if hidden_act != "silu":
        raise NotImplementedError(f"lfm2_moe: hidden_act {hidden_act!r} is not supported")
    if not bool(_get(hf_config, "use_expert_bias", True)):
        raise NotImplementedError("lfm2_moe: use_expert_bias=false is not supported")

    theta, scaling = _rope(hf_config)
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=int(hf_config.max_position_embeddings),
        base=theta,
        scaling=scaling,
    )
    conv_group = LinearGatedDeltaGroupConfig(
        name="conv",
        layer_ids=conv_ids,
        # state_layout="short_conv": the slot holds only the conv's left context, width
        # num_value_heads * value_head_dim = hidden; the head split only serves TP sharding.
        num_key_heads=num_heads,
        num_value_heads=num_heads,
        key_head_dim=0,
        value_head_dim=hidden // num_heads,
        conv_kernel_dim=int(_get(hf_config, "conv_L_cache", 3)),
        output_gate="none",
        state_layout="short_conv",
    )
    attn_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=attn_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
    )
    groups = tuple(sorted((conv_group, attn_group), key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30))

    num_dense = int(_get(hf_config, "num_dense_layers", 0))
    tie = bool(_get(hf_config, "tie_word_embeddings", False)) and not _has_quantized_lm_head(hf_config)
    return ModelConfig(
        num_layers=int(hf_config.num_hidden_layers),
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(hf_config.vocab_size),
        intermediate_size=int(hf_config.intermediate_size),
        rms_norm_eps=float(_get(hf_config, "norm_eps", 1e-5)),
        rotary_config=rotary,
        hidden_act=hidden_act,
        tie_word_embeddings=tie,
        num_experts=int(hf_config.num_experts),
        num_experts_per_tok=int(hf_config.num_experts_per_tok),
        moe_intermediate_size=int(hf_config.moe_intermediate_size),
        norm_topk_prob=bool(_get(hf_config, "norm_topk_prob", True)),
        model_type=str(_get(hf_config, "model_type", "lfm2_moe")),
        architectures=list(_get(hf_config, "architectures", None) or ["Lfm2MoeForCausalLM"]),
        moe_enabled=True,
        expert_quant=detect_expert_quant(hf_config),
        use_qk_norm=True,
        first_k_dense_replace=num_dense,
        routed_scaling_factor=float(_get(hf_config, "routed_scaling_factor", 1.0)),
        attention_groups=groups,
    )


__all__ = ["layer_types", "parse_config"]
