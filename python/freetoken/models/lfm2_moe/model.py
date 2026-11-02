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


def add_rmsnorm_fp32(
    norm: RMSNormFused, x: torch.Tensor, residual: torch.Tensor | None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``residual += x; x = norm(residual)`` with the residual stream kept in fp32 and the normed
    output in ``x.dtype``. ``residual=None`` starts the stream from ``x``."""
    if residual is None:
        residual = x.float()
        out = torch.empty_like(x)
    else:
        out = x
    if x.is_cuda:
        from freetoken.kernel.triton.norm import fused_add_rmsnorm, rmsnorm

        # the Triton kernels store each tensor in its own dtype; flashinfer needs one dtype for both
        if out is x:
            fused_add_rmsnorm(x, residual, norm.weight, norm.eps)
        else:
            rmsnorm(residual, norm.weight, norm.eps, out=out)
        return out, residual
    if out is x:
        residual += x.float()
    inv = torch.rsqrt(residual.pow(2).mean(-1, keepdim=True) + norm.eps)
    out.copy_(residual * inv * norm.weight.float())
    return out, residual


class Lfm2MoeDecoderLayer(BaseOP):
    """Pre-norm block: ``x += mixer(operator_norm(x)); x += feed_forward(ffn_norm(x))``.

    The residual stream is fp32, as in exllamav3: it lowered the mean KL against exllamav3 from
    0.170 to 0.154 with bf16 activations and from 0.116 to 0.106 with fp16 activations.

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
        x, residual = add_rmsnorm_fp32(self.operator_norm, x, residual)
        x = self.conv.forward(x) if self._is_conv else self.self_attn.forward(x)
        x, residual = add_rmsnorm_fp32(self.ffn_norm, x, residual)
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
        return add_rmsnorm_fp32(self.embedding_norm, x, residual)[0]


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


__all__ = ["Lfm2MoeForCausalLM", "add_rmsnorm_fp32"]
