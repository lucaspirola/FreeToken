"""Bounded exclusive host residency for immutable native NVFP4 expert rows.

GPU victims are restored from the checkpoint, never copied device-to-host. Only
ownership metadata crosses back from CUDA. This experimental pool is eager-only
and single-caller; a completed upload relinquishes its host slot before reuse.
"""
from __future__ import annotations

import json
import mmap
import os
import struct

import ctypes
import torch

_BLK = 4096
_TENSOR_MAP = (
    ("up_proj.weight", "gate_up_packed", False),
    ("up_proj.weight_scale", "gate_up_scale", False),
    ("up_proj.weight_scale_2", "gate_up_global", True),
    ("down_proj.weight", "down_packed", False),
    ("down_proj.weight_scale", "down_scale", False),
    ("down_proj.weight_scale_2", "down_global", True),
)


def nvfp4_bank_shapes(hidden_size: int, intermediate_size: int) -> dict:
    """Runtime bank shapes for one native, non-gated NVFP4 expert row."""
    h, i = hidden_size, intermediate_size
    if h <= 0 or i <= 0 or h % 16 or i % 16:
        raise ValueError("native NVFP4 dimensions must be positive multiples of 16")
    return {
        "gate_up_packed": ((i, h // 2), torch.uint8),
        "gate_up_scale": ((i, h // 16), torch.float8_e4m3fn),
        "gate_up_global": ((i,), torch.float16),
        "down_packed": ((h, i // 2), torch.uint8),
        "down_scale": ((h, i // 16), torch.float8_e4m3fn),
        "down_global": ((h,), torch.float16),
    }


class ExclusiveExpertPool:
    """Fixed-size pinned LRU with transactional disk reads and explicit ownership."""

    def __init__(self, model_path: str, num_layers: int, num_experts: int,
                 capacity: int, *, hidden_size: int, intermediate_size: int):
        if capacity <= 0 or num_layers <= 0 or num_experts <= 0:
            raise ValueError("exclusive pool capacity and model dimensions must be positive")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.capacity = capacity
        self.shapes = nvfp4_bank_shapes(hidden_size, intermediate_size)
        self.schema_order = tuple(self.shapes)
        self.sources = {
            name: [torch.empty((num_experts, *tail), dtype=dtype, device="meta")
                   for _ in range(num_layers)]
            for name, (tail, dtype) in self.shapes.items()
        }
        self._shard_fds: dict[str, int] = {}
        self._fd_size: dict[int, int] = {}
        self._records: dict[int, tuple[int, list[tuple[int, int]]]] = {}
        self._scratch = None
        self._sview = None
        self._closed = False
        self.banks: dict[str, torch.Tensor] = {}
        self.host_slot_of_id = [-1] * (num_layers * num_experts)
        self.id_of_host_slot = [-1] * capacity
        self.gpu_ids: set[int] = set()
        self._last_use = [0] * capacity
        self._clock = 0
        self.pool_hits = 0
        self.pool_misses = 0
        self.refill_bytes = 0
        self._events: list[torch.cuda.Event | None] = [None] * capacity
        self._stream: torch.cuda.Stream | None = None
        try:
            self._scan_checkpoint(model_path)
            scratch_bytes = max(sum(end - start for start, end in _regions_of(pieces))
                                for _fd, pieces in self._records.values())
            self._scratch = mmap.mmap(-1, scratch_bytes)
            self._sview = memoryview(self._scratch)
            for name, (tail, dtype) in self.shapes.items():
                self.banks[name] = torch.empty((capacity, *tail), dtype=dtype, pin_memory=True)
            self.pool_bytes = capacity * sum(_row_bytes(tail, dtype)
                                            for tail, dtype in self.shapes.values())
        except BaseException:
            self.close()
            raise

    def _scan_checkpoint(self, model_path: str) -> None:
        with open(os.path.join(model_path, "config.json")) as f:
            layer_types = json.load(f)["layers_block_type"]
        layer_ids = [i for i, kind in enumerate(layer_types) if kind == "moe"]
        if len(layer_ids) != self.num_layers:
            raise ValueError("checkpoint MoE layer count does not match exclusive pool")
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        headers: dict[str, tuple[int, dict]] = {}
        for layer, backbone_layer in enumerate(layer_ids):
            for expert in range(self.num_experts):
                keys = [f"backbone.layers.{backbone_layer}.mixer.experts.{expert}.{suffix}"
                        for suffix, _, _ in _TENSOR_MAP]
                shards = {weight_map[key] for key in keys}
                if len(shards) != 1:
                    raise ValueError("exclusive pool requires each expert's tensors in one shard")
                shard = next(iter(shards))
                if shard not in headers:
                    path = os.path.join(model_path, shard)
                    with open(path, "rb") as f:
                        n = struct.unpack("<Q", f.read(8))[0]
                        header = json.loads(f.read(n))
                    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                    self._shard_fds[shard] = fd
                    self._fd_size[fd] = os.fstat(fd).st_size
                    headers[shard] = (8 + n, header)
                data_start, header = headers[shard]
                fd = self._shard_fds[shard]
                pieces = []
                for key, (_, name, scalar) in zip(keys, _TENSOR_MAP):
                    entry = header[key]
                    off, end = entry["data_offsets"]
                    expected = 4 if scalar else _row_bytes(*self.shapes[name])
                    dtype = "F32" if scalar else ("U8" if name.endswith("packed") else "F8_E4M3")
                    expected_shape = list(self.shapes[name][0])
                    if (entry["dtype"] != dtype or end - off != expected or off < 0
                            or data_start + end > self._fd_size[fd]
                            or (not scalar and entry["shape"] != expected_shape)):
                        raise ValueError(f"unsupported or corrupt native NVFP4 tensor: {key}")
                    pieces.append((data_start + off, end - off))
                self._records[layer * self.num_experts + expert] = (fd, pieces)

    def _flat_id(self, layer: int, expert: int) -> int:
        if self._closed:
            raise RuntimeError("exclusive pool is closed")
        if not 0 <= layer < self.num_layers or not 0 <= expert < self.num_experts:
            raise IndexError((layer, expert))
        return layer * self.num_experts + expert

    def host_slot(self, layer: int, expert: int) -> int:
        return self.host_slot_of_id[self._flat_id(layer, expert)]

    def release(self, slot: int) -> None:
        """Wait for CPU-visible DMA completion, then relinquish the host row."""
        event = self._events[slot]
        if event is not None:
            event.synchronize()
            self._events[slot] = None
        flat = self.id_of_host_slot[slot]
        if flat >= 0:
            self.host_slot_of_id[flat] = -1
            self.id_of_host_slot[slot] = -1

    def refill(self, layer: int, expert: int) -> int:
        """Read all bytes before replacing an LRU owner; failed reads are retryable."""
        flat = self._flat_id(layer, expert)
        if flat in self.gpu_ids:
            raise RuntimeError(f"cannot duplicate GPU-resident expert {flat} in host pool")
        slot = self.host_slot_of_id[flat]
        if slot >= 0:
            # Served from the bounded host pool: no checkpoint read. This counter
            # pair is the ONLY runtime signal that separates an exclusive server
            # from the whole-model-in-RAM profile, where no row is ever re-read.
            self.pool_hits += 1
            self._clock += 1
            self._last_use[slot] = self._clock
            return slot
        self.pool_misses += 1
        fd, pieces = self._records[flat]
        regions = _regions_of(pieces)
        cursors = []
        cursor = 0
        for start, end in regions:
            # Partial direct reads occur before EOF too. Round retries back to
            # a block boundary so both the file offset and mmap address remain
            # aligned, re-reading any trailing partial block in place.
            length = end - start
            required = min(end, self._fd_size[fd]) - start
            done = 0
            while done < required:
                retry = done // _BLK * _BLK
                view = self._sview[cursor + retry:cursor + length]
                try:
                    got = os.preadv(fd, [view], start + retry)
                finally:
                    view.release()
                advanced = retry + got
                if advanced <= done:
                    raise OSError(f"O_DIRECT checkpoint read stalled at {start + done}")
                done = advanced
            cursors.append(cursor)
            cursor += length
        slot = next((s for s, owner in enumerate(self.id_of_host_slot) if owner < 0), -1)
        if slot < 0:
            slot = min(range(self.capacity), key=self._last_use.__getitem__)
        self.release(slot)
        # Slot is unpublished while CPU conversion/copy is in progress. A failure
        # here leaves a free slot, never an apparently valid partial expert.
        for (_, name, scalar), (off, length) in zip(_TENSOR_MAP, pieces):
            region = next(i for i, (start, end) in enumerate(regions)
                          if start <= off and off + length <= end)
            start = cursors[region] + off - regions[region][0]
            raw = self._sview[start:start + length]
            try:
                dst = self.banks[name][slot]
                if scalar:
                    # Match the loader's F32 -> F16 conversion. struct.unpack on
                    # the raw mmap bytes: torch.frombuffer on an unaligned 4-byte
                    # memoryview slice is UB-adjacent, and fill_(python float)
                    # raises on float32 values that overflow fp16 (the loader's
                    # tensor conversion overflows to inf instead).
                    f32 = struct.unpack("<f", bytes(raw))[0]
                    dst.copy_(torch.tensor(f32, dtype=torch.float16))
                else:
                    # Plain memcpy into the pinned row: a torch copy_ into
                    # pinned memory costs milliseconds (dispatcher), a memmove
                    # is sub-0.1 ms for these row sizes. ctypes accepts integer
                    # addresses only -- c_void_p rejects memoryview slices -- so
                    # take the mmap address through a zero-copy tensor view.
                    src = torch.frombuffer(raw, dtype=torch.uint8)
                    try:
                        ctypes.memmove(
                            dst.view(torch.uint8).data_ptr(), src.data_ptr(), length
                        )
                    finally:
                        del src
            finally:
                raw.release()
        self._clock += 1
        self._last_use[slot] = self._clock
        self.id_of_host_slot[slot] = flat
        self.host_slot_of_id[flat] = slot
        self.refill_bytes += cursor
        return slot

    def upload_async(self, layer: int, expert: int, dst: dict[str, torch.Tensor],
                     dst_index: int) -> int:
        """Stage an H2D transfer; wait_upload commits GPU ownership and frees RAM."""
        slot = self.refill(layer, expert)
        device = dst[self.schema_order[0]].device
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        elif self._stream.device != device:
            raise ValueError("exclusive pool cannot upload across CUDA devices")
        event = self._events[slot]
        if event is not None:
            event.synchronize()
        # A previous wait/release clears the event: create it again, not just when
        # the stream is first allocated. Wait for previous GEMMs before overwrite.
        event = torch.cuda.Event()
        self._stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(self._stream):
                for name in self.schema_order:
                    dst[name][dst_index].copy_(self.banks[name][slot], non_blocking=True)
                event.record(self._stream)
        except BaseException:
            self._stream.synchronize()
            raise
        self._events[slot] = event
        return slot

    def wait_upload(self, slot: int) -> None:
        """Complete an upload before reusing its RAM; stream waits alone are unsafe."""
        if self._events[slot] is None:
            raise RuntimeError("no upload pending for host slot")
        flat = self.id_of_host_slot[slot]
        self.release(slot)
        self.gpu_ids.add(flat)

    def synchronize_all(self) -> None:
        for event in self._events:
            if event is not None:
                event.synchronize()

    def close(self) -> None:
        if self._closed:
            return
        self.synchronize_all()
        if self._sview is not None:
            self._sview.release()
            self._sview = None
        if self._scratch is not None:
            self._scratch.close()
            self._scratch = None
        for fd in self._shard_fds.values():
            os.close(fd)
        self._shard_fds.clear()
        self.banks.clear()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _row_bytes(tail: tuple, dtype: torch.dtype) -> int:
    count = torch.empty((), dtype=dtype).element_size()
    for dim in tail:
        count *= dim
    return count


def _regions_of(pieces):
    aligned = sorted((off // _BLK * _BLK, -(-(off + length) // _BLK) * _BLK)
                     for off, length in pieces)
    regions = []
    for start, end in aligned:
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(regions[-1][1], end))
        else:
            regions.append((start, end))
    return regions
