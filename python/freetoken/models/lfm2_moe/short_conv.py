"""LFM2 double-gated short convolution.

    B, C, x = split(in_proj(h), 3)
    y = causal_depthwise_conv(B * x)        # kernel conv_L_cache, no bias, no activation
    out = out_proj(C * y)

The per-request conv state (the last ``kernel - 1`` values of ``B * x``) lives in
``ctx.linear_state_pool`` under ``state_layout="short_conv"``; there is no recurrent state.
"""

from __future__ import annotations

import torch
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.kernel.triton.transpose import transpose
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.layers.quantization import QuantConfig


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[hidden, 1, K]`` (checkpoint key ``conv.conv.weight``)."""

    def __init__(self, dim: int, kernel: int):
        self.weight = torch.empty(dim, 1, kernel)


class Lfm2ShortConv(BaseOP):
    def __init__(
        self, hidden_size: int, conv_kernel_size: int, layer_id: int, *,
        quant_config: QuantConfig | None = None, prefix: str = "",
    ):
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.in_proj = LinearReplicated(
            hidden_size, 3 * hidden_size, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.in_proj",
        )
        self.conv = _DepthwiseConv1d(hidden_size, conv_kernel_size)
        self.out_proj = LinearReplicated(
            hidden_size, hidden_size, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.out_proj",
        )

    def _conv_weight(self) -> torch.Tensor:
        return self.conv.weight.squeeze(1)

    def _write_track_snapshot(self, pool, li: int, bx: torch.Tensor, fla) -> None:
        """Hybrid-radix snapshot: the conv state at the track boundary is the ``kernel - 1``
        values of ``B * x`` just before it, gathered from this forward's rows."""
        cv = pool.conv_states[li]
        window = bx[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, hidden, K-1]
        cv.index_copy_(0, fla.track_dst, window.to(cv.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        h = self.hidden_size
        bcx = self.in_proj.forward(hidden_states)
        b, c, x = torch.split(bcx, [h, h, h], dim=-1)
        bx = b * x
        li = pool.local_index(self.layer_id)
        if batch.is_decode:
            y = causal_conv1d_decode(
                bx, pool.conv_states[li], self._conv_weight(), fla.cache_indices, activation=False
            )
        else:
            if fla.track_dst is not None:
                self._write_track_snapshot(pool, li, bx, fla)
            y = causal_conv1d_varlen(
                transpose(bx), self._conv_weight(), pool.conv_states[li],
                fla.cu_seqlens, fla.cache_indices, fla.has_initial_state, activation=False,
            )
            y = transpose(y)
        return self.out_proj.forward(c * y)


def short_conv_reference(
    hidden_states: torch.Tensor, in_proj_weight: torch.Tensor, conv_weight: torch.Tensor,
    out_proj_weight: torch.Tensor, conv_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence in plain torch (the HF forward), for tests: returns ``(out, new_state)``.

    ``conv_state`` is the ``[hidden, K-1]`` left context (zeros when None)."""
    h = hidden_states.shape[-1]
    k = conv_weight.shape[-1]
    b, c, x = (hidden_states @ in_proj_weight.T).split(h, dim=-1)
    bx = (b * x).T  # [hidden, T]
    left = conv_state if conv_state is not None else bx.new_zeros(h, k - 1)
    full = torch.cat([left.to(bx.dtype), bx], dim=-1)
    y = torch.nn.functional.conv1d(full.unsqueeze(0), conv_weight.reshape(h, 1, k), groups=h)[0]
    out = (c * y.T) @ out_proj_weight.T
    return out, full[:, -(k - 1):]


__all__ = ["Lfm2ShortConv", "short_conv_reference"]
