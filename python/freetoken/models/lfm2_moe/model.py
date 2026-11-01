from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.qwen3.attention import Qwen3Attention
from freetoken.utils import nvtx_annotate

from .moe import Lfm2MoeMLP, Lfm2MoeSparseBlock
from .short_conv import Lfm2ShortConv

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Lfm2MoeDecoderLayer(BaseOP):
    """Pre-norm block: ``x += mixer(operator_norm(x)); x += feed_forward(ffn_norm(x))``.

    The attention mixer is Qwen3's QK-norm GQA attention; the weight reader renames the
    checkpoint's ``out_proj`` / ``q_layernorm`` / ``k_layernorm`` onto its attributes."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_conv = config.is_linear_layer(layer_id)
        if self._is_conv:
            group = config.linear_attention_group()
            assert group is not None
            self.conv = Lfm2ShortConv(
                config.hidden_size, group.conv_kernel_dim, layer_id,
                quant_config=config.quant, prefix=f"{prefix}.conv",
            )
        else:
            self.self_attn = Qwen3Attention(config, layer_id, has_qk_norm=True, prefix=f"{prefix}.self_attn")
        self.feed_forward = (
            Lfm2MoeMLP(config, prefix=f"{prefix}.feed_forward")
            if layer_id < config.first_k_dense_replace
            else Lfm2MoeSparseBlock(config, layer_id, prefix=f"{prefix}.feed_forward")
        )
        self.operator_norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.operator_norm.forward(x, residual)
        x = self.conv.forward(x) if self._is_conv else self.self_attn.forward(x)
        x, residual = self.ffn_norm.forward(x, residual)
        x = self.feed_forward.forward(x)
        return x, residual


class Lfm2MoeModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Lfm2MoeDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.embedding_norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.embedding_norm.forward(x, residual)[0]


class Lfm2MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Lfm2MoeModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Lfm2MoeForCausalLM"]
