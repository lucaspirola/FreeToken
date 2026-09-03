"""Triton Mamba-2 SSD (state-space duality) prefill kernels.

Vendored from vLLM's ``vllm/model_executor/layers/mamba/ops/ssd_*.py`` -- the
sequence-aligned varlen chunk variant that takes ``cu_seqlens`` /
``cu_chunk_seqlens`` / ``last_chunk_indices`` / ``seq_idx`` and a dense
per-sequence ``initial_states`` block -- which in turn adapts Tri Dao and
Albert Gu's reference implementation in ``state-spaces/mamba`` v2.2.4. Both are
Apache-2.0; the per-file headers carry the attribution.

Layout contract (Nemotron-3.5 Lightning: H=64, P=64, N=128, G=8, chunk=128):

* ``x``   ``[total_tokens, H, P]``   bf16
* ``dt``  ``[total_tokens, H]``      bf16 or fp32
* ``B``/``C`` ``[total_tokens, G, N]`` bf16
* ``A``/``D``/``dt_bias`` ``[H]``    fp32 (``A`` already negative, i.e. ``-exp(A_log)``)
* recurrent state pool ``[slots, H, P, N]`` fp32 -- the native SSD / flashinfer
  layout, no transposes anywhere in the hot path.

FreeToken changes vs vLLM: no ``vllm.*`` or ``einops`` imports; autotune config
lists pruned to <= 8 entries that fit the ~100 KB shared memory an sm_120
(GB203 / RTX 5080) block can hold; ``autotune_cache_kwargs`` so the winning
config survives process restarts; ``do_not_specialize`` on every stride that
scales with the chunk count so a new prompt length never recompiles;
fp32 ``dA_cumsum`` / states / ``CB`` accumulation with bf16 x/B/C loads.

The kernels themselves only write freshly allocated outputs -- the recurrent
pool scatter lives in :func:`mamba2_prefill`, which keeps them autotune-safe
(an autotuner re-runs a kernel many times; an in-place state update would be
applied many times).
"""

from __future__ import annotations

import torch

from freetoken.kernel.triton.mamba2.metadata import (
    Mamba2Metadata,
    build_mamba2_metadata,
)
from freetoken.kernel.triton.mamba2.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)

__all__ = [
    "Mamba2Metadata",
    "build_mamba2_metadata",
    "mamba2_prefill",
    "mamba_chunk_scan_combined_varlen",
]


def mamba2_prefill(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    A: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    meta: Mamba2Metadata,
    cu_seqlens: torch.Tensor,
    state_source: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
    return_intermediate_states: bool = False,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked SSD scan over a varlen prefill batch, with pool gather/scatter.

    Args:
        x: ``[total_tokens, H, P]`` post-conv SSM input.
        dt: ``[total_tokens, H]`` raw (pre-bias, pre-softplus) timestep.
        B: ``[total_tokens, G, N]``.
        C: ``[total_tokens, G, N]``.
        A: ``[H]`` fp32, already ``-exp(A_log)``.
        D: ``[H]`` (or ``[H, P]``) skip connection, or None.
        dt_bias: ``[H]`` fp32, or None.
        meta: chunk plan from :func:`build_mamba2_metadata`, built for the same
            ``cu_seqlens``.
        cu_seqlens: ``[S + 1]`` cumulative token counts (int32 or int64).
        state_source: ``[slots, H, P, N]`` fp32 recurrent pool for this layer.
            Read for the carried state and written back in place.
        indices: ``[S]`` pool slot of each sequence.
        has_initial_state: ``[S]`` bool. Rows that are False start from zeros
            regardless of what the pool holds. ``None`` means "trust the pool"
            -- the caller already zeroed fresh slots (what
            ``FLAMetadata.fresh_state_indices`` does today).
        return_intermediate_states: also return the state at *every* chunk
            boundary, ``[num_chunks, H, P, N]`` fp32, addressed as
            ``meta.chunk_offsets[i] + c`` for chunk ``c`` of sequence ``i``.
        dt_softplus: apply softplus to ``dt + dt_bias`` (threshold 20).
        dt_limit: post-softplus clamp.
        out: optional preallocated ``[total_tokens, H, P]`` output.

    Returns:
        ``(out, intermediate_states)`` where ``out`` is ``[total_tokens, H, P]``
        in ``x``'s dtype and ``intermediate_states`` is None unless requested.
        The per-sequence final states are scattered into ``state_source`` at
        ``indices``; slots not named by ``indices`` are left untouched.
    """
    total_tokens, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    num_seqs = int(indices.shape[0])
    assert cu_seqlens.shape[0] == num_seqs + 1, (
        f"cu_seqlens has {cu_seqlens.shape[0]} entries for {num_seqs} sequences"
    )
    assert meta.num_seqs == num_seqs, (
        f"metadata was built for {meta.num_seqs} sequences, got {num_seqs}"
    )
    assert state_source.shape[1:] == (nheads, headdim, dstate), (
        f"state pool is {tuple(state_source.shape)}, expected "
        f"[slots, {nheads}, {headdim}, {dstate}]"
    )
    assert dt.shape == (total_tokens, nheads)
    assert C.shape == B.shape == (total_tokens, ngroups, dstate)

    index = indices.to(torch.long)

    # Gather the carried state into the dense [S, H, P, N] block the kernels
    # index by sequence id. index_select always allocates, so the mask below is
    # safe to apply in place and the pool is never mutated before the scan.
    initial_states = state_source.index_select(0, index).to(torch.float32)
    if has_initial_state is not None:
        assert has_initial_state.shape == (num_seqs,)
        initial_states.mul_(
            has_initial_state.to(initial_states.dtype).view(-1, 1, 1, 1)
        )
    initial_states = initial_states.contiguous()

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape

    # Always ask for the full chunk-boundary state block: it is the same tensor
    # the kernel allocates either way, so the per-sequence final states are a
    # gather off it rather than a second allocation.
    all_states = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        meta.chunk_size,
        cu_seqlens,
        meta.cu_chunk_seqlens,
        meta.last_chunk_indices,
        meta.seq_idx,
        out,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        return_intermediate_states=True,
        state_dtype=torch.float32,
    )

    final_states = all_states.index_select(0, meta.last_chunk_indices.to(torch.long))
    state_source.index_copy_(0, index, final_states.to(state_source.dtype))

    return out, (all_states if return_intermediate_states else None)
