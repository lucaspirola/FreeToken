from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from freetoken.core import SamplingParams
from freetoken.hidden_states import HiddenStateSpec

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Optional precomputed multimodal soft-token embeddings (GPU tensor). Only used by
    # the in-process offline path; remains None for the (serialized) online path.
    mm_embeds: torch.Tensor | None = None
    session_id: str | None = None
    session_ttl_seconds: float | None = None
    session_reclaimable: bool = False
    # Switchyard prefill-probe export (freetoken/hidden_states.py). Already resolved
    # against --hidden-states-dir by the frontend; None on every ordinary request.
    hidden_states: HiddenStateSpec | None = None
    # Force a full recompute of the prompt (see Req.no_prefix_cache).
    no_prefix_cache: bool = False


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
    session_id: str | None = None


@dataclass
class CloseSessionBackendMsg(BaseBackendMsg):
    session_id: str
    request_id: str


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)
