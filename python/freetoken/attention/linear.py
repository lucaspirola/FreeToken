from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.kernel.triton.mamba2 import Mamba2Metadata


@dataclass
class FLAMetadata:
    """Per-forward GatedDeltaNet (flash-linear-attention) metadata, built once per
    forward and shared by every GDN layer -- mirrors ``BaseAttnMetadata``. Replaces the
    per-layer rebuilds the GDN op used to do (``cu_seqlens`` arange, per-request
    ``cache_indices``/``has_initial_state``), which were pageable, synchronous H2D copies
    issued in each of the 30 GDN layers.

    Fields:
      cu_seqlens          query indptr; decode = arange(bs+1) (1 token/req), prefill =
                          cumsum of extend_len. int32 on device.
      cache_indices       per-request recurrent/conv state slot (= Req.table_idx). int32.
      has_initial_state   prefill only: whether each request continues a cached prefix
                          (cached_len > 0). None for decode (state always present).
      fresh_state_indices prefill only: the state-pool slots whose sequence is fresh
                          (cached_len == 0) and must be zeroed before the chunk kernel
                          reads them in place. None if there are none / for decode.
    """

    cu_seqlens: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor | None = None
    fresh_state_indices: torch.Tensor | None = None

    # Mamba-2 (state_layout="mamba2") prefill chunk plan for the SSD kernels: chunk_size,
    # cu_chunk_seqlens, last_chunk_indices, seq_idx, chunk_offsets, num_chunks. None for
    # GDN models and for decode (one token per request needs no chunk cut).
    mamba2: "Mamba2Metadata | None" = None

    # --- hybrid-radix track-checkpoint (extra_buffer) fields; all None when not caching ---
    # For each request crossing a track_chunk_size-aligned boundary this forward, snapshot its
    # recurrent + conv state into a donatable pool slot, written on the forward stream by the
    # mixer (Qwen3_5GatedDeltaNet._write_track_snapshot / NemotronHMamba2Mixer._prefill_scan).
    track_dst: torch.Tensor | None = None        # [nt] int64 dst pool slot per tracked req
    # [nt] int64 row into the per-chunk state block. GDN numbers rows by the state BEFORE a
    # chunk and Mamba-2 by the state AFTER it, so the row for the same boundary differs by one
    # -- see _build_track_metadata.
    track_h_row: torch.Tensor | None = None
    track_conv_src: torch.Tensor | None = None   # [nt, kernel-1] int64 conv-input token positions


def build_fla_metadata(batch: "Batch", device: torch.device) -> FLAMetadata:
    """Build the per-forward GDN metadata. Uses pinned host staging + non_blocking H2D
    (the input_ids/attn-metadata pattern), so the copies overlap the forward instead of
    stalling it.

    Decode is one token per request, so ``cu_seqlens`` is a plain ``arange(bs+1)`` and
    ``cache_indices`` is ``batch.linear_table_idx`` (already int32) -- reused as-is. Under
    CUDA graph the decode ``FLAMetadata`` is instead built directly in
    ``GraphCaptureBuffer.set_batch`` against the persistent buffers (stable addresses); this
    builder serves the eager scheduler path and direct-op test callers.
    """
    reqs = batch.padded_reqs
    pin = {"device": "cpu", "pin_memory": True}

    # GDN state slot per request: the hybrid-radix live slot (decoupled from table_idx) when
    # allocated, else table_idx (naive / force-naive GDN models keep the old keying).
    def gdn_slot(r):
        return r.linear_slot_idx if r.linear_slot_idx is not None else r.table_idx

    if batch.is_decode:
        bs = len(reqs)
        cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device=device)
        # the scheduler stages linear_table_idx from gdn_slot (decode), reused as-is here
        assert batch.linear_table_idx is not None
        return FLAMetadata(cu_seqlens=cu_seqlens, cache_indices=batch.linear_table_idx)

    # prefill: cumsum of query (extend) lengths, per-request slot + continuation flags.
    lens = [r.extend_len for r in reqs]
    cu_host = torch.tensor([0, *lens], dtype=torch.int64, **pin).cumsum_(0)
    idx_host = torch.tensor([gdn_slot(r) for r in reqs], dtype=torch.int32, **pin)
    has_init_host = torch.tensor([r.cached_len > 0 for r in reqs], dtype=torch.bool, **pin)
    fresh = [gdn_slot(r) for r in reqs if r.cached_len == 0]
    fresh_host = torch.tensor(fresh, dtype=torch.int64, **pin) if fresh else None

    group = _linear_group()
    mamba2 = None
    if group is not None and group.state_layout == "mamba2":
        from freetoken.kernel.triton.mamba2 import build_mamba2_metadata

        # build_mamba2_metadata rejects a zero-length sequence (it would have no chunk to
        # carry its state, making last_chunk_indices ambiguous). A prefill batch never
        # carries one -- PrefillAdder only admits an extend of >= 1 token -- so assert
        # rather than silently dropping the row out of the scan.
        assert all(length > 0 for length in lens), (
            f"Mamba-2 prefill batch carries an empty extend: {lens}"
        )
        mamba2 = build_mamba2_metadata(
            cu_host.tolist(), chunk_size=group.track_chunk_size, device=device
        )

    track_dst, track_h_row, track_conv_src = _build_track_metadata(
        reqs, lens, device, pin, group
    )

    return FLAMetadata(
        cu_seqlens=cu_host.to(device, non_blocking=True),
        cache_indices=idx_host.to(device, non_blocking=True),
        has_initial_state=has_init_host.to(device, non_blocking=True),
        fresh_state_indices=(
            fresh_host.to(device, non_blocking=True) if fresh_host is not None else None
        ),
        mamba2=mamba2,
        track_dst=track_dst, track_h_row=track_h_row, track_conv_src=track_conv_src,
    )


def _linear_group():
    """The running model's recurrent-state group, or None outside an engine (direct-op
    tests that drive the metadata builder without a global context)."""
    from freetoken.core import get_global_ctx

    try:
        ctx = get_global_ctx()
    except (AssertionError, RuntimeError):
        return None
    pool = None if ctx is None else getattr(ctx, "linear_state_pool", None)
    return None if pool is None else pool.group


def _build_track_metadata(reqs, lens, device, pin, group):
    """Hybrid-radix (extra_buffer): for each request that crosses a x``track_chunk_size``
    boundary this prefill forward, snapshot its recurrent state at the deepest mid-chunk
    boundary into its current ping-pong slot. Returns (track_dst, track_h_row,
    track_conv_src) device int64 tensors, or (None, None, None) when no request tracks
    (non-hybrid, or all extends <= chunk).

    ``track_h_row`` indexes the per-chunk state block the scan returns. The two families
    number those rows differently, so the layout decides the offset:

    ``kv`` (FLA/GDN)  ``h[boh_i + c]`` is the state BEFORE chunk ``c`` (= after ``c``
                      chunks), so the row for a boundary of ``c`` chunks is ``boh_i + c``.
    ``mamba2`` (SSD)  ``states[chunk_offsets_i + c]`` is the state AFTER chunk ``c``
                      (= after ``c + 1`` chunks), so the same boundary is ``boh_i + c - 1``.
    """
    if not any(r.mamba_ping_pong is not None for r in reqs):
        return None, None, None
    from freetoken.core import get_global_ctx
    from freetoken.kernel.fla.chunk import CHUNK_SIZE

    chunk = CHUNK_SIZE if group is None else group.track_chunk_size
    # -1 for mamba2: its intermediate row c holds the state after chunk c, not before it.
    row_bias = -1 if (group is not None and group.state_layout == "mamba2") else 0

    km1 = get_global_ctx().linear_state_pool.conv_states.shape[-1]  # conv_kernel_dim - 1
    # boh[i] = first chunk row of sequence i = sum of ceil(len_j / chunk) for j < i. The
    # same cut both kernel families use (FLA prepare_chunk_offsets / Mamba2Metadata
    # chunk_offsets), computed once on the host so no device read is needed here.
    boh, total, off = [], 0, 0
    offsets = []
    for length in lens:
        boh.append(total)
        offsets.append(off)
        total += -(-length // chunk)
        off += length
    dst, h_row, conv_src = [], [], []
    for i, r in enumerate(reqs):
        if r.mamba_ping_pong is None:
            continue
        # deepest mid-chunk boundary strictly inside the extend (the per-chunk states have
        # it; the exact extend-end state lives in the live slot -> finish-donate).
        c = (r.extend_len - 1) // chunk
        if c < 1:
            continue
        boundary = r.cached_len + c * chunk
        dst.append(r.mamba_ping_pong[r.mamba_next_track_idx])
        h_row.append(boh[i] + c + row_bias)
        conv_src.append([offsets[i] + c * chunk - km1 + j for j in range(km1)])
        r.mamba_last_track_seqlen = boundary
        r.mamba_next_track_idx = 1 - r.mamba_next_track_idx
    if not dst:
        return None, None, None
    to = lambda xs, **kw: torch.tensor(xs, **pin, **kw).to(device, non_blocking=True)
    return (to(dst, dtype=torch.int64), to(h_row, dtype=torch.int64),
            to(conv_src, dtype=torch.int64))


__all__ = ["FLAMetadata", "build_fla_metadata"]
