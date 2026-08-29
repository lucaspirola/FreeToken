from __future__ import annotations

import math
from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool
from .quant import NONE, KVQuantSpec
from .quant_storage import QuantizedKVStorageMixin


class MHAKVCache(QuantizedKVStorageMixin, BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        quant: KVQuantSpec = NONE,
        grow_step_tokens: int = 0,
    ) -> None:
        self._quant = quant
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._compute_dtype = dtype
        storage_head_dim = head_dim // self._quant.elements_per_byte
        self._logical_num_pages = num_pages
        self._grow_step_pages = (
            max(1, grow_step_tokens // page_size) if grow_step_tokens else 0
        )
        if grow_step_tokens and grow_step_tokens % page_size:
            raise ValueError(
                f"kv_grow_step_tokens={grow_step_tokens} must be divisible by page_size={page_size}"
            )
        self._committed_pages = (
            min(num_pages - 1, self._grow_step_pages) if self._grow_step_pages else num_pages - 1
        )
        self._kv_vmm = None
        self._scale_vmm = None
        buffer_dtype = self._buffer_dtype(dtype)
        if self._grow_step_pages:
            self._kv_buffer, self._kv_vmm, self._kv_vmm_geometry = self._allocate_growable(
                slabs=2 * num_storage_layers,
                logical_pages=num_pages,
                committed_pages=self._committed_pages,
                page_row_elements=page_size * local_kv_heads * storage_head_dim,
                dtype=buffer_dtype,
                device=device,
                trailing_shape=(page_size, local_kv_heads, storage_head_dim),
                leading_shape=(2, num_storage_layers),
            )
        else:
            kv_shape = (
                2, num_storage_layers, num_pages, page_size, local_kv_heads, storage_head_dim
            )
            self._kv_buffer = torch.empty(kv_shape, device=device, dtype=buffer_dtype)
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        # Scales key off the LOGICAL head_dim: extent D // BLOCK regardless of packing.
        log_shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
        if self._grow_step_pages and self._quant.enabled:
            scale_dim = self._quant.scale_shape(log_shape)[-1]
            self._scale_buffer, self._scale_vmm, self._scale_vmm_geometry = self._allocate_growable(
                slabs=2 * num_storage_layers,
                logical_pages=num_pages,
                committed_pages=self._committed_pages,
                page_row_elements=page_size * local_kv_heads * scale_dim,
                dtype=torch.float16,
                device=device,
                trailing_shape=(page_size, local_kv_heads, scale_dim),
                leading_shape=(2, num_storage_layers),
            )
        else:
            self._scale_buffer = self._alloc_scales(log_shape, device)
        self._k_scale = self._scale_buffer[0] if self._scale_buffer is not None else None
        self._v_scale = self._scale_buffer[1] if self._scale_buffer is not None else None
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, storage_head_dim)

    @staticmethod
    def _allocate_growable(
        *, slabs: int, logical_pages: int, committed_pages: int,
        page_row_elements: int, dtype: torch.dtype, device: torch.device,
        trailing_shape: tuple[int, ...], leading_shape: tuple[int, ...],
    ):
        """Reserve a stable multi-slab VA and map page prefixes in every K/V-layer slab.

        Page zero is the permanent CUDA-graph dummy page; usable page ids begin at one.
        Padding the page axis makes every slab boundary VMM-granularity aligned.
        """
        from freetoken.kernel.vmm import VMMTensor, allocation_granularity

        granularity = allocation_granularity(device)
        row_bytes = page_row_elements * torch.empty((), dtype=dtype).element_size()
        pages_per_granule = granularity // math.gcd(granularity, row_bytes)
        padded_pages = (
            (logical_pages + pages_per_granule - 1) // pages_per_granule * pages_per_granule
        )
        slab_bytes = padded_pages * row_bytes
        mapped_per_slab = (
            (committed_pages + 1) * row_bytes + granularity - 1
        ) // granularity * granularity
        initial_ranges = [(i * slab_bytes, mapped_per_slab) for i in range(slabs)]
        shape = (*leading_shape, padded_pages, *trailing_shape)
        allocation = VMMTensor(
            shape, dtype=dtype, device=device, initial_ranges=initial_ranges
        )
        geometry = (slabs, row_bytes, slab_bytes, mapped_per_slab)
        return allocation.tensor, allocation, geometry

    @property
    def growable(self) -> bool:
        return self._grow_step_pages > 0

    @property
    def committed_pages(self) -> int:
        """Physically committed usable pages (dummy page zero excluded)."""
        return self._committed_pages

    def _mapped_bytes_at(self, usable_pages: int, geometry) -> int:
        slabs, row_bytes, _slab_bytes, _mapped = geometry
        granularity = self._kv_vmm.granularity
        per_slab = ((usable_pages + 1) * row_bytes + granularity - 1) // granularity * granularity
        return slabs * per_slab

    def mapped_bytes_for_pages(self, usable_pages: int) -> int:
        if not self.growable:
            kv, _ = self.unit_bytes()
            return usable_pages * self._kv_buffer.shape[3] * kv
        total = self._mapped_bytes_at(usable_pages, self._kv_vmm_geometry)
        if self._scale_vmm is not None:
            total += self._mapped_bytes_at(usable_pages, self._scale_vmm_geometry)
        return total

    def commit_pages(self, usable_pages: int) -> None:
        """Commit all layer stripes through ``usable_pages`` without moving their pointers."""
        if not self.growable:
            raise RuntimeError("KV pool is not growable")
        if usable_pages <= self._committed_pages:
            return
        if usable_pages > self._logical_num_pages - 1:
            raise ValueError(
                f"cannot commit {usable_pages} pages; maximum is {self._logical_num_pages - 1}"
            )
        for allocation, geometry in (
            (self._kv_vmm, self._kv_vmm_geometry),
            (self._scale_vmm, getattr(self, "_scale_vmm_geometry", None)),
        ):
            if allocation is None:
                continue
            slabs, row_bytes, slab_bytes, old_mapped = geometry
            granularity = allocation.granularity
            new_mapped = (
                (usable_pages + 1) * row_bytes + granularity - 1
            ) // granularity * granularity
            if new_mapped > old_mapped:
                allocation.commit_ranges([
                    (i * slab_bytes + old_mapped, new_mapped - old_mapped)
                    for i in range(slabs)
                ])
                geometry = (slabs, row_bytes, slab_bytes, new_mapped)
                if allocation is self._kv_vmm:
                    self._kv_vmm_geometry = geometry
                else:
                    self._scale_vmm_geometry = geometry
        self._committed_pages = usable_pages

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        if self.growable:
            raise RuntimeError("growable KV pages are committed in place, not rebuilt")
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, storage_head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        # Drop the scale slab too before reallocating, for the same reason the KV slab is
        # dropped: holding the old one alive can OOM a rebuild the target size would fit.
        self._k_scale = None
        self._v_scale = None
        self._scale_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        kv_shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, storage_head_dim)
        self._kv_buffer = torch.empty(kv_shape, device=device, dtype=dtype)
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        log_shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, storage_head_dim * self._quant.elements_per_byte)
        self._scale_buffer = self._alloc_scales(log_shape, device)
        self._k_scale = self._scale_buffer[0] if self._scale_buffer is not None else None
        self._v_scale = self._scale_buffer[1] if self._scale_buffer is not None else None
        self._storage_shape = (num_pages * page_size, local_kv_heads, storage_head_dim)
        self._logical_num_pages = num_pages
        self._committed_pages = num_pages - 1

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    @classmethod
    def growable_mapped_bytes_for_config(
        cls, config, usable_pages: int, granularity: int
    ) -> int:
        """Exact VMM physical bytes at a growth boundary, including slab rounding."""
        spec = next(
            s for s in config.model_config.kv_cache_group_specs()
            if s.num_layers > 0 and not s.is_swa
        )
        local_heads = div_even(
            spec.num_kv_heads, config.tp_info.size, allow_replicate=True
        )
        quant = config.kv_quant
        storage_dim = spec.head_dim // quant.elements_per_byte
        payload_dtype = quant.storage_dtype if quant.enabled else config.dtype
        rows = [
            config.page_size * local_heads * storage_dim
            * torch.empty((), dtype=payload_dtype).element_size()
        ]
        if quant.enabled:
            rows.append(
                config.page_size * local_heads * (spec.head_dim // 32)
                * torch.empty((), dtype=torch.float16).element_size()
            )
        slabs = 2 * spec.num_layers
        return sum(
            slabs
            * (((usable_pages + 1) * row + granularity - 1) // granularity * granularity)
            for row in rows
        )

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = self._logical_num_pages * int(buf.shape[3])
        total = (
            2 * int(buf.shape[1]) * self._logical_num_pages
            * int(buf.shape[3]) * int(buf.shape[4]) * int(buf.shape[5]) * buf.element_size()
        )
        if self._scale_buffer is not None:
            scale = self._scale_buffer
            total += (
                2 * int(scale.shape[1]) * self._logical_num_pages
                * int(scale.shape[3]) * int(scale.shape[4]) * int(scale.shape[5])
                * scale.element_size()
            )
        return int(total) // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index), : self._logical_num_pages]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index), : self._logical_num_pages]

    def k_scale(self, index: int) -> torch.Tensor | None:
        return None if self._k_scale is None else self._k_scale[self._dense(index), : self._logical_num_pages]

    def v_scale(self, index: int) -> torch.Tensor | None:
        return None if self._v_scale is None else self._v_scale[self._dense(index), : self._logical_num_pages]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        scale_shape = (self._storage_shape[0], self._storage_shape[1], -1)
        self._store_kv_into(
            self._k_buffer[dense, : self._logical_num_pages].view(self._storage_shape),
            self._v_buffer[dense, : self._logical_num_pages].view(self._storage_shape),
            None if self._k_scale is None else self._k_scale[dense, : self._logical_num_pages].view(scale_shape),
            None if self._v_scale is None else self._v_scale[dense, : self._logical_num_pages].view(scale_shape),
            out_loc,
            k,
            v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._compute_dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
