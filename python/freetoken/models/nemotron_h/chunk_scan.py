"""Memory-efficient rewrite of transformers' pure-Torch Mamba-2 chunked scan.

``transformers.models.nemotron_h.modeling_nemotron_h.mamba2_chunk_scan`` is the
reference fallback used when ``mamba_ssm`` is absent. It is numerically right but
writes every contraction as a broadcast multiply followed by ``.sum(dim)``, which
materializes six-dimensional temporaries:

    G             = (C[..., None, :, :] * B[..., None, :, :]).sum(-1)   -> [b, c, l, s, h, n]
    Y_diag        = (M[..., None] * x[:, :, None]).sum(3)               -> [b, c, l, s, h, p]
    states        = (B_decay[..., None, :] * x[..., None]).sum(2)       -> [b, c, s, h, p, n]
    C_times_states= (C[..., None, :] * states[:, :, None]).sum(-1)      -> [b, c, l, h, p, n]

Each of those is O(chunk_size * head_dim * state_size) *per token*: measured at
4.19 MiB/token/layer on Lightning's geometry (H=64, P=64, N=128, chunk=128), i.e.
17 GiB for one layer at a 4096-token prefill chunk, which OOMs a 16 GB card on any
prompt past ~1K tokens.

This module keeps the algorithm and the operation order identical and expresses the
same four contractions as ``einsum`` (batched GEMMs), so no tensor above rank 5 is
ever allocated. Peak transient drops to ~0.15 MiB/token/layer -- ~28x less -- and it
is faster, since the reductions become cuBLAS calls instead of elementwise kernels.

Everything else (softplus/dt clamp, fp32 accumulation, chunk padding, D residual,
initial-state threading, returned final state) is byte-for-byte the reference's.
Phase 2 replaces this entirely with the vendored Triton SSD kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.nemotron_h.modeling_nemotron_h import (
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
)

__all__ = ["mamba2_chunk_scan"]


def mamba2_chunk_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
):
    """Drop-in for the transformers reference; see the module docstring."""
    batch_size, sequence_length, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])

    hidden_states = hidden_states.float()
    repeat = num_heads // num_groups
    B = B.float().repeat_interleave(repeat, dim=2, output_size=num_heads)
    C = C.float().repeat_interleave(repeat, dim=2, output_size=num_heads)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    D_residual = None
    if D is not None:
        D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None].float()
    A = A.to(hidden_states.dtype) * dt.float()

    # Rearrange into blocks/chunks: x/B/C [b, c, l, h, *], A [b, c, l, h]
    hidden_states, A, B, C = (
        reshape_into_chunks(tensor, pad_size, chunk_size)
        for tensor in (hidden_states, A, B, C)
    )

    A = A.permute(0, 3, 1, 2)  # [b, h, c, l]
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Intra-chunk (diagonal blocks): the causal-mask analogue.
    L = torch.exp(segment_sum(A))  # [b, h, c, l, s]
    # M[b,c,h,l,s] = L * sum_n C[b,c,l,h,n] B[b,c,s,h,n]
    M = torch.einsum("bclhn,bcshn->bchls", C, B) * L.transpose(1, 2)
    Y_diag = torch.einsum("bchls,bcshp->bclhp", M, hidden_states)
    del M, L

    # 2. Per-chunk state (right term of the off-diagonal low-rank factorization).
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, 2, 3, 1)[..., None]
    states = torch.einsum("bcshn,bcshp->bchpn", B_decay, hidden_states)
    del B_decay

    # 3. Inter-chunk recurrence (middle term): exact states at chunk boundaries.
    previous_states = (
        initial_states[:, None].to(dtype=states.dtype, device=states.device)
        if initial_states is not None
        else torch.zeros_like(states[:, :1])
    )
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    new_states = torch.einsum("bijh,bihpn->bjhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. State -> output per chunk (left term).
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn->bclhp", C, states) * state_decay_out.permute(
        0, 2, 3, 1
    )[..., None]

    output = Y_diag + Y_off
    output = output.reshape(batch_size, -1, num_heads, head_dim)

    if D_residual is not None:
        output = output + D_residual

    if pad_size > 0:
        output = output[:, :sequence_length]

    if return_final_states:
        return output, final_state
    return output
