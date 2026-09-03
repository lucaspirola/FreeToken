"""Host-side chunk metadata for the Mamba-2 SSD prefill kernels.

The SSD chunk scan wants the prefill token stream cut into *logical chunks* that
(a) never straddle a sequence boundary and (b) are at most ``chunk_size`` tokens
long. Everything the kernels need to walk that cut is precomputed here on the
host and shipped to the device as one pinned, non-blocking transfer.

Alignment note (deviation from vLLM): vLLM's ``compute_varlen_chunk_metadata``
aligns chunk boundaries to the *global* token offset (``chunk_size - pos %
chunk_size``), so a sequence that does not start on a multiple of ``chunk_size``
gets a short leading chunk. FreeToken aligns to each *sequence's own* start
instead, so sequence ``i`` has exactly ``ceil(len_i / chunk_size)`` chunks and
chunk ``c`` of that sequence always covers its tokens
``[c*chunk_size, (c+1)*chunk_size)``. That is what makes the hybrid-radix
snapshot contract (``chunk_offsets[i] + c`` indexes the state after token
``(c+1)*chunk_size``) hold; both cuts are numerically equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Mamba2Metadata:
    """Per-batch chunk plan consumed by ``mamba2_prefill``.

    All tensors are int32 and live on the model device; they are contiguous
    slices of a single packed buffer, so building them costs one H2D copy.
    """

    chunk_size: int
    #: [num_chunks + 1] exclusive prefix sum of logical-chunk lengths.
    cu_chunk_seqlens: torch.Tensor
    #: [num_seqs] index of each sequence's last logical chunk.
    last_chunk_indices: torch.Tensor
    #: [num_chunks] owning sequence index of each logical chunk.
    seq_idx: torch.Tensor
    #: [num_seqs] index of each sequence's first logical chunk. The
    #: intermediate state after chunk ``c`` of sequence ``i`` is row
    #: ``chunk_offsets[i] + c`` of the returned intermediate-state tensor.
    chunk_offsets: torch.Tensor
    #: Total number of logical chunks in the batch.
    num_chunks: int

    @property
    def num_seqs(self) -> int:
        return int(self.last_chunk_indices.shape[0])


def build_mamba2_metadata(
    cu_seqlens_host: list[int] | torch.Tensor,
    chunk_size: int = 128,
    *,
    device: torch.device | str,
) -> Mamba2Metadata:
    """Build the chunk plan for a prefill batch.

    Args:
        cu_seqlens_host: ``[num_seqs + 1]`` cumulative token counts, starting at
            0. A python list is preferred; a tensor is accepted and read on the
            host (pass a CPU tensor -- a CUDA one forces a device sync).
        chunk_size: physical chunk size, a power of two (128 for Nemotron-3.5).
        device: destination device for the metadata tensors.

    Every sequence must be non-empty: an empty extend has no chunk to carry its
    state and would make ``last_chunk_indices`` ambiguous.
    """
    if isinstance(cu_seqlens_host, torch.Tensor):
        cu = [int(v) for v in cu_seqlens_host.tolist()]
    else:
        cu = [int(v) for v in cu_seqlens_host]
    assert len(cu) >= 1 and cu[0] == 0, "cu_seqlens must start at 0"
    assert chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0, (
        f"chunk_size must be a power of two, got {chunk_size}"
    )

    num_seqs = len(cu) - 1
    chunk_lens: list[int] = []
    seq_idx: list[int] = []
    last_chunk_indices: list[int] = []
    chunk_offsets: list[int] = []

    for i in range(num_seqs):
        length = cu[i + 1] - cu[i]
        assert length > 0, f"sequence {i} is empty (cu_seqlens={cu})"
        chunk_offsets.append(len(chunk_lens))
        pos = 0
        while pos < length:
            take = min(chunk_size, length - pos)
            chunk_lens.append(take)
            seq_idx.append(i)
            pos += take
        last_chunk_indices.append(len(chunk_lens) - 1)

    num_chunks = len(chunk_lens)
    cu_chunk_seqlens = [0]
    for n in chunk_lens:
        cu_chunk_seqlens.append(cu_chunk_seqlens[-1] + n)
    assert cu_chunk_seqlens[-1] == cu[-1]

    # One pinned staging buffer -> one non_blocking H2D -> four contiguous views.
    packed_host = torch.tensor(
        cu_chunk_seqlens + last_chunk_indices + seq_idx + chunk_offsets,
        dtype=torch.int32,
        device="cpu",
        # pinning only pays off (and only initialises CUDA) for a device copy
        pin_memory=torch.device(device).type == "cuda",
    )
    packed = packed_host.to(device, non_blocking=True)

    off = 0
    cu_chunk_t = packed.narrow(0, off, num_chunks + 1)
    off += num_chunks + 1
    last_chunk_t = packed.narrow(0, off, num_seqs)
    off += num_seqs
    seq_idx_t = packed.narrow(0, off, num_chunks)
    off += num_chunks
    chunk_offsets_t = packed.narrow(0, off, num_seqs)

    return Mamba2Metadata(
        chunk_size=chunk_size,
        cu_chunk_seqlens=cu_chunk_t,
        last_chunk_indices=last_chunk_t,
        seq_idx=seq_idx_t,
        chunk_offsets=chunk_offsets_t,
        num_chunks=num_chunks,
    )
