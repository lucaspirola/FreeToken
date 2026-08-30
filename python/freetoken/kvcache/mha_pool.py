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
        quant_k: KVQuantSpec | None = None,
        quant_v: KVQuantSpec | None = None,
        grow_step_tokens: int = 0,
    ) -> None:
        self._quant_k = quant if quant_k is None else quant_k
        self._quant_v = quant if quant_v is None else quant_v
        self._quant = self._quant_k  # legacy readers; new paths use quant_k/quant_v
        self._asymmetric = self._quant_k != self._quant_v
        if self._asymmetric and not (self._quant_k.enabled and self._quant_v.enabled):
            raise ValueError("asymmetric K/V storage currently requires two quantized formats")
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
        k_storage_head_dim = self._quant_k.storage_dim(head_dim) if self._quant_k.enabled else head_dim
        v_storage_head_dim = self._quant_v.storage_dim(head_dim) if self._quant_v.enabled else head_dim
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
        self._initial_committed_pages = self._committed_pages
        # Each entry is one logical growth boundary plus the exact CUDA VMM mappings created
        # for it. CUDA can only unmap a complete physical mapping, so retaining these records
        # makes shrink an exact reverse of growth even when slab rounding differs by KV dtype.
        self._growth_segments: list[tuple[int, list[tuple[list, list[tuple[int, int]], int]]]] = []
        self._vmm_entries: list[list] = []
        self._kv_vmm = None
        self._scale_vmm = None
        if self._asymmetric and self._grow_step_pages:
            self._k_buffer, k_vmm, k_geometry = self._allocate_growable(
                slabs=num_storage_layers,
                logical_pages=num_pages,
                committed_pages=self._committed_pages,
                page_row_elements=page_size * local_kv_heads * k_storage_head_dim,
                dtype=self._quant_k.storage_dtype,
                device=device,
                trailing_shape=(page_size, local_kv_heads, k_storage_head_dim),
                leading_shape=(num_storage_layers,),
            )
            self._v_buffer, v_vmm, v_geometry = self._allocate_growable(
                slabs=num_storage_layers,
                logical_pages=num_pages,
                committed_pages=self._committed_pages,
                page_row_elements=page_size * local_kv_heads * v_storage_head_dim,
                dtype=self._quant_v.storage_dtype,
                device=device,
                trailing_shape=(page_size, local_kv_heads, v_storage_head_dim),
                leading_shape=(num_storage_layers,),
            )
            self._vmm_entries.extend([[k_vmm, k_geometry], [v_vmm, v_geometry]])
            self._kv_buffer = None
            self._kv_vmm = k_vmm
        elif self._asymmetric:
            self._k_buffer = torch.empty(
                (num_storage_layers, num_pages, page_size, local_kv_heads, k_storage_head_dim),
                device=device,
                dtype=self._quant_k.storage_dtype,
            )
            self._v_buffer = torch.empty(
                (num_storage_layers, num_pages, page_size, local_kv_heads, v_storage_head_dim),
                device=device,
                dtype=self._quant_v.storage_dtype,
            )
            self._kv_buffer = None
        elif self._grow_step_pages:
            buffer_dtype = self._buffer_dtype(dtype)
            self._kv_buffer, self._kv_vmm, self._kv_vmm_geometry = self._allocate_growable(
                slabs=2 * num_storage_layers,
                logical_pages=num_pages,
                committed_pages=self._committed_pages,
                page_row_elements=page_size * local_kv_heads * k_storage_head_dim,
                dtype=buffer_dtype,
                device=device,
                trailing_shape=(page_size, local_kv_heads, k_storage_head_dim),
                leading_shape=(2, num_storage_layers),
            )
            self._vmm_entries.append([self._kv_vmm, self._kv_vmm_geometry])
        else:
            buffer_dtype = self._buffer_dtype(dtype)
            kv_shape = (
                2, num_storage_layers, num_pages, page_size, local_kv_heads, k_storage_head_dim
            )
            self._kv_buffer = torch.empty(kv_shape, device=device, dtype=buffer_dtype)
        if not self._asymmetric:
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
        # Scales key off the LOGICAL head_dim: extent D // BLOCK regardless of packing.
        log_shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
        if self._grow_step_pages and self._quant_k.enabled:
            scale_dim = self._quant_k.scale_shape(log_shape)[-1]
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
            self._vmm_entries.append([self._scale_vmm, self._scale_vmm_geometry])
        else:
            self._scale_buffer = self._alloc_scales(log_shape, device)
        self._k_scale = self._scale_buffer[0] if self._scale_buffer is not None else None
        self._v_scale = self._scale_buffer[1] if self._scale_buffer is not None else None
        self._device = device
        self._storage_shape_k = (num_pages * page_size, local_kv_heads, k_storage_head_dim)
        self._storage_shape_v = (num_pages * page_size, local_kv_heads, v_storage_head_dim)
        self._storage_shape = self._storage_shape_k

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

    def _mapped_bytes_at(self, usable_pages: int, entry) -> int:
        allocation, geometry = entry
        slabs, row_bytes, _slab_bytes, _mapped = geometry
        granularity = allocation.granularity
        per_slab = ((usable_pages + 1) * row_bytes + granularity - 1) // granularity * granularity
        return slabs * per_slab

    def mapped_bytes_for_pages(self, usable_pages: int) -> int:
        if not self.growable:
            kv, _ = self.unit_bytes()
            return usable_pages * self._k_buffer.shape[2] * kv
        return sum(self._mapped_bytes_at(usable_pages, entry) for entry in self._vmm_entries)

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
        while self._committed_pages < usable_pages:
            next_boundary = min(
                usable_pages,
                ((self._committed_pages // self._grow_step_pages) + 1)
                * self._grow_step_pages,
            )
            committed: list[tuple[list, list[tuple[int, int]], int]] = []
            try:
                for entry in self._vmm_entries:
                    allocation, geometry = entry
                    slabs, row_bytes, slab_bytes, old_mapped = geometry
                    granularity = allocation.granularity
                    new_mapped = (
                        (next_boundary + 1) * row_bytes + granularity - 1
                    ) // granularity * granularity
                    ranges = (
                        [
                            (i * slab_bytes + old_mapped, new_mapped - old_mapped)
                            for i in range(slabs)
                        ]
                        if new_mapped > old_mapped
                        else []
                    )
                    if ranges:
                        allocation.commit_ranges(ranges)
                    committed.append((entry, ranges, old_mapped))
            except Exception:
                for entry, ranges, _old_mapped in reversed(committed):
                    if ranges:
                        entry[0].uncommit_ranges(ranges)
                raise

            for entry, _ranges, _old_mapped in committed:
                allocation, geometry = entry
                slabs, row_bytes, slab_bytes, _ = geometry
                new_mapped = self._mapped_per_slab(next_boundary, allocation, row_bytes)
                entry[1] = (slabs, row_bytes, slab_bytes, new_mapped)
            self._growth_segments.append((next_boundary, committed))
            self._committed_pages = next_boundary

    @staticmethod
    def _mapped_per_slab(usable_pages: int, allocation, row_bytes: int) -> int:
        granularity = allocation.granularity
        return (
            (usable_pages + 1) * row_bytes + granularity - 1
        ) // granularity * granularity

    def decommit_pages(self, usable_pages: int) -> None:
        """Release complete growth segments above ``usable_pages`` without moving pointers."""
        if not self.growable:
            raise RuntimeError("KV pool is not growable")
        if not self._initial_committed_pages <= usable_pages <= self._committed_pages:
            raise ValueError(
                f"cannot decommit to {usable_pages} pages; valid range is "
                f"[{self._initial_committed_pages}, {self._committed_pages}]"
            )
        valid_boundaries = {
            self._initial_committed_pages,
            *(boundary for boundary, _entries in self._growth_segments),
        }
        if usable_pages not in valid_boundaries:
            raise ValueError(
                f"decommit target {usable_pages} is not a committed growth boundary"
            )

        while self._growth_segments and self._growth_segments[-1][0] > usable_pages:
            _boundary, committed = self._growth_segments.pop()
            for entry, ranges, _old_mapped in reversed(committed):
                if ranges:
                    entry[0].uncommit_ranges(ranges)
            for entry, _ranges, old_mapped in committed:
                slabs, row_bytes, slab_bytes, _ = entry[1]
                entry[1] = (slabs, row_bytes, slab_bytes, old_mapped)
        self._committed_pages = usable_pages

    def copy_pages(self, source_pages: torch.Tensor, destination_pages: torch.Tensor) -> None:
        """Copy complete physical pages for scheduler compaction.

        Page zero is the graph dummy page and is never supplied here. The copy is enqueued on
        the caller's current stream; the scheduler performs compaction only at a no-forward-
        in-flight boundary and rewrites page-table references after these copies.
        """
        if source_pages.numel() != destination_pages.numel():
            raise ValueError("source and destination page counts differ")
        if source_pages.numel() == 0:
            return
        src = source_pages.to(device=self._device, dtype=torch.long)
        dst = destination_pages.to(device=self._device, dtype=torch.long)
        if self._asymmetric:
            self._k_buffer.index_copy_(1, dst, self._k_buffer.index_select(1, src))
            self._v_buffer.index_copy_(1, dst, self._v_buffer.index_select(1, src))
        else:
            self._kv_buffer.index_copy_(2, dst, self._kv_buffer.index_select(2, src))
        if self._scale_buffer is not None:
            self._scale_buffer.index_copy_(
                2, dst, self._scale_buffer.index_select(2, src)
            )

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        if self.growable:
            raise RuntimeError("growable KV pages are committed in place, not rebuilt")
        if self._asymmetric:
            num_storage_layers, _old_pages, page_size, local_kv_heads, k_dim = self._k_buffer.shape
            v_dim = self._v_buffer.shape[-1]
            device = self._device
            self._k_buffer = torch.empty(
                (num_storage_layers, num_pages, page_size, local_kv_heads, k_dim),
                device=device, dtype=self._quant_k.storage_dtype,
            )
            self._v_buffer = torch.empty(
                (num_storage_layers, num_pages, page_size, local_kv_heads, v_dim),
                device=device, dtype=self._quant_v.storage_dtype,
            )
            scale_dim = self._quant_k.logical_dim(k_dim) // 32
            self._scale_buffer = torch.empty(
                (2, num_storage_layers, num_pages, page_size, local_kv_heads, scale_dim),
                device=device, dtype=torch.float16,
            )
            self._k_scale, self._v_scale = self._scale_buffer[0], self._scale_buffer[1]
            self._storage_shape_k = (num_pages * page_size, local_kv_heads, k_dim)
            self._storage_shape_v = (num_pages * page_size, local_kv_heads, v_dim)
            self._storage_shape = self._storage_shape_k
            self._logical_num_pages = num_pages
            self._committed_pages = num_pages - 1
            return
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
        self._storage_shape_k = self._storage_shape
        self._storage_shape_v = self._storage_shape
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
        quant_k = getattr(config, "kv_quant_k", config.kv_quant)
        quant_v = getattr(config, "kv_quant_v", config.kv_quant)
        total = 0
        for quant in (quant_k, quant_v):
            storage_dim = quant.storage_dim(spec.head_dim) if quant.enabled else spec.head_dim
            payload_dtype = quant.storage_dtype if quant.enabled else config.dtype
            row = config.page_size * local_heads * storage_dim * torch.empty(
                (), dtype=payload_dtype
            ).element_size()
            total += spec.num_layers * (
                ((usable_pages + 1) * row + granularity - 1) // granularity * granularity
            )
            if quant.enabled:
                scale_row = config.page_size * local_heads * (spec.head_dim // 32) * 2
                total += spec.num_layers * (
                    ((usable_pages + 1) * scale_row + granularity - 1)
                    // granularity * granularity
                )
        return total

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        tokens = self._logical_num_pages * int(self._k_buffer.shape[2])
        if self._asymmetric:
            total = sum(
                int(buf.shape[0]) * self._logical_num_pages * int(buf.shape[2])
                * int(buf.shape[3]) * int(buf.shape[4]) * buf.element_size()
                for buf in (self._k_buffer, self._v_buffer)
            )
        else:
            buf = self._kv_buffer
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
            self._k_buffer[dense, : self._logical_num_pages].view(self._storage_shape_k),
            self._v_buffer[dense, : self._logical_num_pages].view(self._storage_shape_v),
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
        return self._k_buffer.dtype

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._compute_dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
