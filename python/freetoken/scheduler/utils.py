from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams
    from freetoken.hidden_states import HiddenStateSpec

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    session_id: str | None = None
    session_ttl_seconds: float | None = None
    # Switchyard prefill-probe export; see freetoken/hidden_states.py.
    hidden_states: HiddenStateSpec | None = None
    # Match against the empty prefix (see Req.no_prefix_cache).
    no_prefix_cache: bool = False
    # This prompt exceeds the KV pool's maximum size, so it can never be admitted and the
    # admission loop skips it on every pass. Latches the one warning it is worth.
    oversize_warned: bool = False
    # Memoised prefix match: the ``CacheManager.prefix_fingerprint()`` under which
    # ``match_cached_len`` was measured. While the fingerprint still holds, the admission
    # arithmetic can be run WITHOUT re-walking the radix tree -- a refused 118K-token prompt
    # would otherwise be matched from scratch on every pass, which is the entire cost of a
    # stalled scheduler (soak report S5).
    match_fp: tuple | None = None
    match_cached_len: int = 0
    # Consecutive passes this fresh prompt has been refused admission. Drives the aging rule
    # in PrefillManager.schedule_next_batch: a young refusal lets its queue-mates overtake,
    # an old one reserves the pool until it is served.
    refused_passes: int = 0

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
