from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from freetoken.core import Batch


class AttnType(str, Enum):
    """Attention-type taxonomy, one value per KV-pool family. A model's attention
    groups map onto these (KVCacheGroupSpec.attn_type); each backend declares the
    set it can serve (BackendInfo.supported_types)."""

    FULL = "full"  # uniform causal MHA/GQA -> MHAKVCache
    SWA = "swa"  # sliding-window hybrid (window/sinks) -> HybridSWAKVCache
    MLA = "mla"  # plain latent-KV MLA -> MLAKVCache
    DSA = "dsa"  # latent-KV MLA + DSA sparse indexer -> DSAKVCache
    DSV4 = "dsv4"  # DSV4 window+compressed sparse -> DSV4PagedKVCache
    LINEAR = "linear"  # GDN/mamba state layers -> LinearStatePool
    # GQA block-sparse (MiniMax-M3): paged GQA K/V + a per-sparse-layer index-key
    # slab; the indexer picks top-k 128-token blocks per query -> BSAKVCache
    BSA = "bsa"
    # QSA compressed-block sparse (Qwen3.8-Flash-Next): paged GQA K/V + a compressed
    # index-key slab (one row per index_ratio tokens, row = slot // index_ratio) +
    # a per-request pending ring for unclosed groups -> QSAKVCache
    QSA = "qsa"

    @property
    def backend_driven(self) -> bool:
        # LINEAR layers reach their kernels directly (fla ops + batch.fla_metadata),
        # not through an attention backend, so they never constrain backend choice.
        return self is not AttnType.LINEAR


@dataclass
class AttentionSpec:
    sliding_window: int | None = None
    sm_scale: float | None = None
    sinks: torch.Tensor | None = None
    # rows of a multimodal span (batch.mm_block_ends) also attend to the span's later keys
    bidirectional_mm_blocks: bool = False


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...

    def reset_capture(self) -> None:
        """Drop CUDA-graph capture scratch so ``init_capture_graph`` can re-run after a
        runtime cache rebuild. The default clears the common capture state (guarded by
        ``hasattr`` so backends that hold only a subset, e.g. dsv4, are safe). Backends
        with extra per-bs graph state (FlashInfer ``graph_wrappers``) override this."""
        if hasattr(self, "capture"):
            self.capture = None
        if hasattr(self, "capture_bs"):
            self.capture_bs = []
        if hasattr(self, "max_graph_bs"):
            self.max_graph_bs = 0


class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch, attn_spec=attn_spec)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)

    def reset_capture(self) -> None:
        # Only the decode backend is ever captured (see init_capture_graph above).
        self.decode_backend.reset_capture()


class DecodeGatedBackend:
    """Attention backend proxy that calls ``gate(layer_id)`` before every DECODE
    attention call, on the stream the call is issued on, then delegates.

    Model-agnostic plumbing for the bounded expert mirror's write-back
    scheduling (``MirrorResidency.attention_gate``): the gate records an event
    in front of each decode attention kernel (inside the CUDA graph when
    captured), and the host-issued write-back DMA waits for those events so it
    overlaps attention -- which uses no PCIe -- rather than the in-graph expert
    fetches. Every attribute other than the attention entry points is the inner
    backend's, read and written through (no isinstance check on the backend
    exists; ``getattr(backend, "dsa_enabled")`` reads through).
    """

    _GATED = ("forward", "mla_forward", "bsa_forward", "qsa_forward")

    def __init__(self, inner, gate) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_gate", gate)
        import inspect

        for name in self._GATED:
            fn = getattr(inner, name, None)
            if fn is None:
                continue
            sig = inspect.signature(fn)
            params = list(sig.parameters)
            li, bi = params.index("layer_id"), params.index("batch")

            def gated(*args, _fn=fn, _li=li, _bi=bi, **kwargs):
                batch = kwargs["batch"] if "batch" in kwargs else args[_bi]
                if not batch.is_prefill:
                    layer_id = kwargs["layer_id"] if "layer_id" in kwargs else args[_li]
                    self._gate(int(layer_id))
                return _fn(*args, **kwargs)

            object.__setattr__(self, name, gated)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value) -> None:
        setattr(self._inner, name, value)
