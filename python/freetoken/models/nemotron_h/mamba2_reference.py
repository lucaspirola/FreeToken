"""Pure-PyTorch Mamba-2 reference path for the Nemotron-H mixer (A/B only).

This is the scan that shipped in Phase 1, lifted out of ``model.py`` when the Triton
SSD kernels landed (task 2A4). It is numerically the reference the kernels are checked
against and is *much* slower -- a Python loop over requests for prefill, and HF's
single-step ``mamba2_selective_state_update`` for decode -- so nothing on the serving
path reaches it unless ``FREETOKEN_MAMBA2_REF=1`` is set:

    FREETOKEN_MAMBA2_REF=1 ft serve ...      # A/B against the kernel path

The recurrent pool is the ``state_layout="mamba2"`` block ``[slots, H, P, N]``, the
native SSD layout, so unlike the Phase 1 code this module transposes nothing: the
chunk scan and the HF state update both already speak ``[H, P, N]``.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx

__all__ = [
    "reference_decode_scan",
    "reference_enabled",
    "reference_gated_rmsnorm",
    "reference_prefill_scan",
]


def reference_enabled() -> bool:
    """True when FREETOKEN_MAMBA2_REF selects the pure-PyTorch path."""
    return os.environ.get("FREETOKEN_MAMBA2_REF", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def reference_gated_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """``norm(x * silu(gate))`` per contiguous group of ``group_size`` channels."""
    dtype = x.dtype
    x = x.float() * F.silu(gate.float())
    shape = x.shape
    grouped = x.view(*shape[:-1], shape[-1] // group_size, group_size)
    grouped = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    return grouped.view(shape).to(dtype) * weight


def _scan(mixer, x, dt, B, C, initial):
    """The reference recurrence (transformers' pure-Torch chunk fallback, used when
    mamba_ssm is absent), with its four broadcast-and-sum contractions written as
    einsums. Same math to fp32 roundoff; 4.2 MiB/token of rank-6 temporaries becomes
    0.22 MiB/token, which is what makes a >1K-token prefill chunk fit in 16 GB.
    See models/nemotron_h/chunk_scan.py."""
    from .chunk_scan import mamba2_chunk_scan

    return mamba2_chunk_scan(
        x.unsqueeze(0),
        dt.unsqueeze(0),
        mixer.A,
        B.unsqueeze(0),
        C.unsqueeze(0),
        chunk_size=mixer.chunk_size,
        D=mixer.D,
        dt_bias=mixer.dt_bias,
        initial_states=initial.unsqueeze(0),
        dt_softplus=True,
        dt_limit=mixer.dt_limit,
        return_final_states=True,
    )


def reference_prefill_scan(mixer, x, dt, B, C, fla, pool) -> torch.Tensor:
    """Per-request chunk scan, one Python iteration per sequence in the batch."""
    li = pool.local_index(mixer.layer_id)
    if fla.fresh_state_indices is not None:
        pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
    outputs = []
    offset = 0
    for req in get_global_ctx().batch.padded_reqs:
        length = req.extend_len
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        initial = pool.recurrent_states[li, slot].float()
        sx, sdt, sB, sC = (v[offset : offset + length] for v in (x, dt, B, C))

        # Hybrid-radix asks for at most one mid-chunk snapshot per request. Split the
        # reference scan at that boundary so the donated state is exact.
        boundary = None
        if req.mamba_last_track_seqlen is not None:
            candidate = req.mamba_last_track_seqlen - req.cached_len
            if 0 < candidate < length:
                boundary = candidate
        if boundary is not None:
            out1, state1 = _scan(
                mixer, sx[:boundary], sdt[:boundary], sB[:boundary], sC[:boundary], initial
            )
            assert req.mamba_ping_pong is not None
            dst = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
            pool.recurrent_states[li, dst].copy_(state1[0])
            out2, final = _scan(
                mixer, sx[boundary:], sdt[boundary:], sB[boundary:], sC[boundary:], state1[0]
            )
            out = torch.cat((out1[0], out2[0]), dim=0)
        else:
            scanned, final = _scan(mixer, sx, sdt, sB, sC, initial)
            out = scanned[0]
        pool.recurrent_states[li, slot].copy_(final[0])
        outputs.append(out)
        offset += length
    return torch.cat(outputs, dim=0)


def reference_decode_scan(mixer, x, dt, B, C, fla, pool) -> torch.Tensor:
    """HF's single-token selective state update, gather/scatter around the pool."""
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        mamba2_selective_state_update,
    )

    li = pool.local_index(mixer.layer_id)
    indices = fla.cache_indices.long()
    state = pool.recurrent_states[li].index_select(0, indices).contiguous()
    A = mixer.A[:, None, None].expand(-1, mixer.head_dim, mixer.state_size)
    out = mamba2_selective_state_update(
        state,
        x,
        dt[:, :, None].expand(-1, -1, mixer.head_dim),
        A,
        B,
        C,
        mixer.D[:, None].expand(-1, mixer.head_dim),
        dt_bias=mixer.dt_bias[:, None].expand(-1, mixer.head_dim),
        dt_softplus=True,
    )
    pool.recurrent_states[li].index_copy_(0, indices, state)
    return out
