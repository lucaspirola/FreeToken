from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

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

TopK = Tuple[torch.Tensor, torch.Tensor]


class Lfm2MoeMLP(BaseOP):
    """Dense SwiGLU of the first ``num_dense_layers`` layers: ``w2(silu(w1 x) * w3 x)``,
    with ``w1|w3`` fused into one GEMM (``w13``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        inter = config.intermediate_size
        self.w13 = LinearColParallelMerged(
            config.hidden_size, [inter, inter], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.w13",
        )
        self.w2 = LinearRowParallel(
            inter, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.w2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward(silu_and_mul(self.w13.forward(x)))


class Lfm2MoeSparseBlock(BaseOP):
    """Routed experts, no shared expert. Routing matches HF ``Lfm2MoeTopKRouter``: sigmoid
    scores, ``expert_bias`` added for the selection only, the unbiased scores of the chosen
    experts renormalized (sum + 1e-6), then scaled by ``routed_scaling_factor``."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        # routers stay unquantized whatever the checkpoint stores
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        # fp32: rounding the bias can flip which experts are selected
        self.expert_bias = torch.empty(config.num_experts, dtype=torch.float32)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )

    def route(self, hidden_states: torch.Tensor) -> TopK:
        logits = torch.nn.functional.linear(hidden_states.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        _, topk_ids = torch.topk(scores + self.expert_bias, self.top_k, dim=-1)
        topk_weights = scores.gather(-1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = self.route(hidden_states)
        return self.experts.routed_forward(hidden_states, topk_weights, topk_ids)


__all__ = ["Lfm2MoeMLP", "Lfm2MoeSparseBlock"]
