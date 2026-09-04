from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

# transformers >= 5.8 renames the Nemotron-H block kinds on load
# (``remap_legacy_layer_types``: mamba/conv -> linear_attention, attention ->
# full_attention). Checkpoints on disk still carry the legacy spellings, and
# ``parse_config`` may be handed either, so both map onto FreeToken's names here.
_LAYER_KIND = {
    "mamba": "mamba",
    "conv": "mamba",
    "linear_attention": "mamba",
    "attention": "attention",
    "full_attention": "attention",
    "moe": "moe",
}

# Attention projections whose quantization decides ``ModelConfig.attn_quant``. The
# checkpoint's per-module map is mixed precision: Nemotron-3-Super quantizes o_proj to
# FP8 while Nemotron-3.5-Lightning leaves all four bf16 and only the *Mamba* in/out
# projections are FP8, so "any FP8 module in the checkpoint" is not the attention answer.
_ATTN_PROJ_RE = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj)$")

# One live Mamba-2 state (conv + fp32 SSM, all mamba layers) above this size makes
# multi-slot serving cost more VRAM than the 16 GB class of card can spare, so the
# engine is pinned to one sequence at a time. Nemotron-3.5-Lightning is ~47 MiB/slot
# (23 layers x 64 heads x 64 x 128 fp32) and runs concurrent; Nemotron-3-Super is
# ~160 MiB/slot and stays single-stream.
_SINGLE_STREAM_STATE_BYTES = 96 * 1024 * 1024

_TRUE = {"1", "true", "yes", "on"}


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE


def dense_dequant_enabled() -> bool:
    """``FREETOKEN_NEMOTRON_DENSE_DEQUANT=1`` restores the legacy path that dequantizes
    the dense NVFP4 matrices (shared experts + lm_head) to bf16 at load instead of
    serving them native W4A16. Escape hatch for numerics debugging; costs ~1.6 GiB."""
    return _env_true("FREETOKEN_NEMOTRON_DENSE_DEQUANT")


def multi_stream_forced() -> bool:
    """``FREETOKEN_NEMOTRON_MULTI_STREAM=1`` clears ``single_stream_only`` regardless of
    the per-slot Mamba state size (measurement escape hatch)."""
    return _env_true("FREETOKEN_NEMOTRON_MULTI_STREAM")


@dataclass(frozen=True)
class NemotronHArgs:
    layer_types: tuple[str, ...]
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    n_groups: int
    conv_kernel: int
    chunk_size: int
    mamba_intermediate_size: int
    # Nemotron-3-Super projects the residual stream down around its experts;
    # Nemotron-3.5-Lightning has no latent MoE (``moe_latent_size: null``).
    moe_latent_size: int | None
    shared_intermediate_size: int
    # Lower clamp on the discretized Mamba-2 timestep (HF ``time_step_min``).
    time_step_min: float
    fp8_modules: frozenset[str]
    nvfp4_dense_modules: frozenset[str]
    nvfp4_lm_head_modules: frozenset[str]
    # Resolved quant mode of the dense NVFP4 matrices: "nvfp4" (native W4A16) or
    # "dequant_bf16" (loader expands them, FREETOKEN_NEMOTRON_DENSE_DEQUANT=1).
    dense_mode: str = "nvfp4"

    def module_quant(self, name: str) -> str:
        if name in self.fp8_modules:
            return "fp8_pertensor"
        if name in self.nvfp4_dense_modules or name in self.nvfp4_lm_head_modules:
            return self.dense_mode
        return "none"


def _quantized_modules(hf_config: Any) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Split the checkpoint's per-module quant map into (fp8, dense NVFP4, NVFP4 lm_head).

    Routed experts are excluded: they never become resident modules (they live in the
    NVFP4 offload banks). ``lm_head`` is kept separate from the other dense NVFP4
    matrices so ``dense_quant`` and ``lm_head_quant`` can be set independently."""
    quant = getattr(hf_config, "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else lambda k, d=None: getattr(quant, k, d)
    layers = get("quantized_layers", {}) or {}
    fp8: set[str] = set()
    nvfp4: set[str] = set()
    lm_head: set[str] = set()
    for name, spec in layers.items():
        if ".experts." in name:
            continue
        algo = str((spec or {}).get("quant_algo", "")).lower()
        if algo == "fp8":
            fp8.add(name)
        elif "fp4" in algo:
            (lm_head if name == "lm_head" or name.endswith(".lm_head") else nvfp4).add(name)
    return frozenset(fp8), frozenset(nvfp4), frozenset(lm_head)


def _layer_types(hf_config: Any) -> tuple[str, ...]:
    kinds = []
    for kind in hf_config.layers_block_type:
        mapped = _LAYER_KIND.get(str(kind))
        if mapped is None:
            raise ValueError(
                f"unsupported Nemotron-H block type {kind!r} "
                f"(known: {sorted(set(_LAYER_KIND))})"
            )
        kinds.append(mapped)
    return tuple(kinds)


def _state_bytes_per_slot(group: LinearGatedDeltaGroupConfig) -> int:
    """Mamba-2 state bytes for one request across all mamba layers (conv in model dtype,
    recurrent in the SSM dtype) -- the same arithmetic the state pool allocates with."""
    import torch

    from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req

    return linear_state_bytes_per_req(group, 1, torch.bfloat16)


def parse_config(hf_config: Any) -> ModelConfig:
    layer_types = _layer_types(hf_config)
    mamba_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "mamba")
    attention_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "attention")
    moe_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "moe")
    fp8_modules, nvfp4_dense, nvfp4_lm_head = _quantized_modules(hf_config)

    hidden_act = str(getattr(hf_config, "mlp_hidden_act", "relu2"))
    if hidden_act != "relu2":
        raise ValueError(
            f"Nemotron-H expects ungated ReLU^2 experts (mlp_hidden_act='relu2'), "
            f"got {hidden_act!r}"
        )
    n_group = int(getattr(hf_config, "n_group", 1))
    topk_group = int(getattr(hf_config, "topk_group", 1))
    if n_group != 1 or topk_group != 1:
        raise ValueError(
            f"Nemotron-H group-limited routing is unimplemented "
            f"(n_group={n_group}, topk_group={topk_group}; both must be 1)"
        )
    n_shared = int(getattr(hf_config, "n_shared_experts", 1))
    if n_shared != 1:
        raise ValueError(f"Nemotron-H expects exactly one shared expert, got {n_shared}")
    cache_dtype = str(getattr(hf_config, "mamba_ssm_cache_dtype", "float32"))
    if cache_dtype != "float32":
        from freetoken.utils import init_logger

        init_logger(__name__).warning(
            "Nemotron-H checkpoint asks for mamba_ssm_cache_dtype=%s; FreeToken sizes the "
            "recurrent state from FREETOKEN_MAMBA_SSM_DTYPE (default float32) instead.",
            cache_dtype,
        )

    latent = getattr(hf_config, "moe_latent_size", None)
    moe_latent_size = None if latent is None else int(latent)
    hidden_size = int(hf_config.hidden_size)
    expert_hidden_size = moe_latent_size if moe_latent_size is not None else hidden_size

    head_dim = int(getattr(hf_config, "head_dim", 128))
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=int(hf_config.max_position_embeddings),
        base=float(getattr(hf_config, "rope_theta", 10000.0)),
        scaling=None,
    )
    mamba_group = LinearGatedDeltaGroupConfig(
        name="mamba",
        layer_ids=mamba_ids,
        # state_layout="mamba2": the pool slot is the native SSD / flashinfer
        # [H, P, N] block, so no scan input or output is ever transposed. The generic
        # axis names map to Mamba-2 as num_value_heads=H (ssm heads),
        # key_head_dim=P (head dim), value_head_dim=N (d_state), num_key_heads=G
        # (B/C groups) -- which also makes _linear_local_dims' mamba2 conv width
        # H*P + 2*G*N = 6144 come out right.
        num_key_heads=int(hf_config.n_groups),
        num_value_heads=int(hf_config.mamba_num_heads),
        key_head_dim=int(hf_config.mamba_head_dim),
        value_head_dim=int(hf_config.ssm_state_size),
        conv_kernel_dim=int(hf_config.conv_kernel),
        output_gate=True,
        state_layout="mamba2",
        # Radix state snapshots land on multiples of the SSD chunk (128), not the
        # FLA/GDN 64: the chunk scan only materialises a state at its own boundaries.
        track_chunk_size=int(hf_config.chunk_size),
    )
    groups = (
        mamba_group,
        FullAttentionGroupConfig(
            name="full",
            layer_ids=attention_ids,
            num_kv_heads=int(hf_config.num_key_value_heads),
            head_dim=head_dim,
            rotary_config=rotary,
        ),
    )
    dense_mode = "dequant_bf16" if dense_dequant_enabled() else "nvfp4"
    args = NemotronHArgs(
        layer_types=layer_types,
        mamba_num_heads=int(hf_config.mamba_num_heads),
        mamba_head_dim=int(hf_config.mamba_head_dim),
        ssm_state_size=int(hf_config.ssm_state_size),
        n_groups=int(hf_config.n_groups),
        conv_kernel=int(hf_config.conv_kernel),
        chunk_size=int(hf_config.chunk_size),
        mamba_intermediate_size=int(hf_config.mamba_num_heads * hf_config.mamba_head_dim),
        moe_latent_size=moe_latent_size,
        shared_intermediate_size=int(hf_config.moe_shared_expert_intermediate_size),
        time_step_min=float(getattr(hf_config, "time_step_min", 0.0)),
        fp8_modules=fp8_modules,
        nvfp4_dense_modules=nvfp4_dense,
        nvfp4_lm_head_modules=nvfp4_lm_head,
        dense_mode=dense_mode,
    )
    attn_fp8 = any(_ATTN_PROJ_RE.search(name) for name in fp8_modules)
    state_bytes = _state_bytes_per_slot(mamba_group)
    return ModelConfig(
        num_layers=int(hf_config.num_hidden_layers),
        num_qo_heads=int(hf_config.num_attention_heads),
        num_kv_heads=int(hf_config.num_key_value_heads),
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=int(hf_config.vocab_size),
        intermediate_size=int(hf_config.intermediate_size),
        rms_norm_eps=float(hf_config.layer_norm_epsilon),
        rotary_config=rotary,
        hidden_act=hidden_act,
        tie_word_embeddings=bool(hf_config.tie_word_embeddings),
        num_experts=int(hf_config.n_routed_experts),
        num_experts_per_tok=int(hf_config.num_experts_per_tok),
        moe_intermediate_size=int(hf_config.moe_intermediate_size),
        norm_topk_prob=bool(hf_config.norm_topk_prob),
        model_type=str(hf_config.model_type),
        architectures=list(hf_config.architectures),
        moe_enabled=True,
        expert_quant="nvfp4",
        attn_quant="fp8_pertensor" if attn_fp8 else "none",
        dense_quant="nvfp4" if (nvfp4_dense and dense_mode == "nvfp4") else "none",
        lm_head_quant="nvfp4" if (nvfp4_lm_head and dense_mode == "nvfp4") else "none",
        shared_expert_intermediate_size=int(hf_config.moe_shared_expert_intermediate_size),
        n_shared_experts=n_shared,
        routed_scaling_factor=float(hf_config.routed_scaling_factor),
        n_group=n_group,
        topk_group=topk_group,
        attention_groups=groups,
        moe_layer_ids=moe_ids,
        expert_hidden_size=expert_hidden_size,
        expert_gated=False,
        nemotron_h_args=args,
        single_stream_only=(
            state_bytes > _SINGLE_STREAM_STATE_BYTES and not multi_stream_forced()
        ),
    )


__all__ = [
    "NemotronHArgs",
    "dense_dequant_enabled",
    "multi_stream_forced",
    "parse_config",
]
