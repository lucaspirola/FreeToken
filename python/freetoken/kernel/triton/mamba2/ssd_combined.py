# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Vendored from vLLM `vllm/model_executor/layers/mamba/ops/ssd_combined.py`
# (the `mamba_chunk_scan_combined_varlen` entry point), which itself adapts
# https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py
#
# FreeToken changes: einops `rearrange` replaced with `reshape`/`view` (the two
# tensors involved are always contiguous), the `vllm.platforms` CPU-dispatch tail
# dropped, `packaging` version probe dropped.

# ruff: noqa: E501

from __future__ import annotations

import torch

from freetoken.kernel.triton.mamba2.ssd_bmm import _bmm_chunk_fwd
from freetoken.kernel.triton.mamba2.ssd_chunk_scan import _chunk_scan_fwd
from freetoken.kernel.triton.mamba2.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
)
from freetoken.kernel.triton.mamba2.ssd_state_passing import _state_passing_fwd


def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    return_intermediate_states=False,
    seq_idx=None,
    cu_seqlens=None,
    cu_chunk_seqlens=None,
    last_chunk_indices=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (cu_chunk_seqlens.shape[0] - 1,)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if (
        x.stride(-1) != 1 and x.stride(0) != 1
    ):  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(0) != 1
    ):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    assert cu_seqlens is not None, "Assuming varlen input - must supply cu_seqlens"

    batch = cu_seqlens.shape[0] - 1
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.

    # 1. Compute chunked cumsum of A * dt (dt may go through a softplus first)
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states = _chunk_state_fwd(
        B, x, dt, dA_cumsum, cu_chunk_seqlens, states_in_fp32=True
    )

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at
    # chunk boundaries, parallelized across sequences via last_chunk_indices.
    nchunks = states.shape[0]
    states = _state_passing_fwd(
        states.reshape(nchunks, nheads, headdim * dstate),
        dA_cumsum,  # (nheads, nchunks, chunk_size)
        last_chunk_indices,
        initial_states=(
            initial_states.reshape(batch, nheads, headdim * dstate)
            if initial_states is not None
            else None
        ),
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
        # `states` is dead after this call, so let the recurrence overwrite it
        # rather than doubling the chunk-state footprint.
        inplace=True,
    )
    states = states.view(nchunks, nheads, headdim, dstate)

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    CB = _bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=torch.float32)

    # 5. Scan and compute the diagonal blocks, taking the carried state into
    # account. Chunks never straddle a sequence boundary, so a chunk whose
    # seq_idx differs from its predecessor's starts from initial_states (or from
    # zeros when there are none).
    _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        cu_chunk_seqlens,
        out,  # in-place update
        seq_idx,
        D=D,
        z=z,
        initial_states=initial_states,
    )

    if return_intermediate_states:
        return states
    return states[last_chunk_indices]


def mamba_chunk_scan_combined_varlen(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_seqlens,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_intermediate_states=False,
    state_dtype=None,
):
    """
    Argument:
        x: (seqlen, nheads, headdim)
        dt: (seqlen, nheads)
        A: (nheads)
        B: (seqlen, ngroups, dstate)
        C: (seqlen, ngroups, dstate)
        chunk_size: int
        cu_seqlens: (batch + 1,)
        cu_chunk_seqlens: (nchunks + 1,)
        last_chunk_indices: (batch,)
        seq_idx: (nchunks,)
        out: (seqlen, nheads, headdim) preallocated output tensor
        D: (nheads, headdim) or (nheads,)
        z: (seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        dt_softplus: Whether to apply softplus to dt
        state_dtype: The data type of the ssm state
    Return:
        (batch, nheads, headdim, dstate) final states, or, when
        `return_intermediate_states`, all (nchunks, nheads, headdim, dstate)
        chunk-boundary states.
    """
    assert cu_seqlens is not None, "cu_seqlens must be provided assuming varlen input"
    assert seq_idx is not None

    return _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        return_intermediate_states=return_intermediate_states,
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        state_dtype=state_dtype,
    )
