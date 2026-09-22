from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        extra_attrs = None
        if config.gguf_expert_types is not None:
            assert layer_id is not None
            gu_t, dn_t = config.gguf_expert_types[layer_id]
            extra_attrs = {
                "gguf_gate_up_type": gu_t,
                "gguf_down_type": dn_t,
                "gguf_gate_up_rows": 2 * config.moe_intermediate_size,
                "gguf_down_rows": config.hidden_size,
            }
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
            extra_attrs=extra_attrs,
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        router_logits = self.gate.forward(hidden_states)
        shared_gate = self.shared_expert_gate.forward(hidden_states)
        # Decode-only GGUF fast path: the shared expert has the same Q4_K/Q6_K
        # projection types as the routed banks. Read its resident packed rows
        # through the grouped MMVQ launches without duplicating them in the
        # scarce expert cache (especially valuable for the larger Q6 model).
        from freetoken.core import get_global_ctx
        from freetoken.layers.moe import OffloadMoELayer
        from freetoken.moe.fused import fused_topk

        ctx = get_global_ctx()
        gate_up = self.shared_expert.gate_up_proj
        down = self.shared_expert.down_proj
        cache = self.experts.offload_cache if isinstance(self.experts, OffloadMoELayer) else None
        can_fuse_shared = (
            ctx.batch.is_decode
            and cache is not None
            and cache.quant_format == "gguf"
            and cache.decode_target == "gpu"
            and not cache.is_cpu_layer(self.experts.layer_id)
            and getattr(gate_up, "qweight", None) is not None
            and getattr(down, "qweight", None) is not None
            and getattr(gate_up, "_quant_type", None) == self.experts.gguf_gate_up_type
            and getattr(down, "_quant_type", None) == self.experts.gguf_down_type
        )
        if can_fuse_shared:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.experts.top_k,
                renormalize=self.experts.renormalize,
            )
            routed = self.experts.routed_forward_with_shared_gguf(
                hidden_states,
                topk_weights,
                topk_ids,
                gate_up.qweight,
                down.qweight,
                shared_gate,
            )
            return routed.view(num_tokens, hidden_dim)

        shared = self.shared_expert.forward(hidden_states)
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        from freetoken.kernel.triton.shared_expert import fused_shared_expert_add_

        return fused_shared_expert_add_(routed, shared, shared_gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
