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
