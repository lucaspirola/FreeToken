"""Exclusive host residency for offloaded MoE experts (experimental).

Opt-in via ``FREETOKEN_EXCLUSIVE_EXPERT_RAM=1`` (engine wiring). The default offload
path keeps EVERY expert row resident in pinned host RAM as the DMA source for GPU
misses. This module instead owns a bounded pool of ``capacity`` host rows backed by
the immutable checkpoint on disk: when an expert is needed and not host-resident, a
victim host slot is refilled by reading the expert's six tensors straight from the
native safetensors shards with O_DIRECT. Displaced GPU experts never travel
GPU -> host; nothing is ever written back (the checkpoint is the canonical backing
store, and expert weights are immutable at inference time).

NVFP4 native layout per expert (matches ``models/nemotron_h/weight.py`` and the
verified probe this module replaces): ``up_proj.weight`` [I, H//2] U8, ``weight_scale``
[I, H//16] F8, ``weight_scale_2`` scalar F32 (expanded to one fp16 per output row),
and the down_proj triple. Runtime bank rows follow ``_BANK_SCHEMAS["nvfp4"]``.
"""

from __future__ import annotations

import json
import mmap
import os
import struct

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_BLK = 4096

# (checkpoint suffix, runtime bank name, scalar-global?)
_TENSOR_MAP = (
    ("up_proj.weight", "gate_up_packed", False),
    ("up_proj.weight_scale", "gate_up_scale", False),
    ("up_proj.weight_scale_2", "gate_up_global", True),
    ("down_proj.weight", "down_packed", False),
    ("down_proj.weight_scale", "down_scale", False),
    ("down_proj.weight_scale_2", "down_global", True),
)


def nvfp4_bank_shapes(hidden_size: int, intermediate_size: int) -> dict:
    """Runtime bank shapes for ONE expert row, in ``_BANK_SCHEMAS["nvfp4"]`` order."""
    h, i = hidden_size, intermediate_size
    return {
        "gate_up_packed": ((i, h // 2), torch.uint8),
        "gate_up_scale": ((i, h // 16), torch.float8_e4m3fn),
        "gate_up_global": ((i,), torch.float16),
        "down_packed": ((h, i // 2), torch.uint8),
        "down_scale": ((h, i // 16), torch.float8_e4m3fn),
        "down_global": ((h,), torch.float16),
    }


class ExclusiveExpertPool:
    """Bounded pinned host pool of expert rows, refilled O_DIRECT from the checkpoint.

    Occupancy is plain LRU over ``capacity`` slots. One CUDA event per slot serializes
    "upload in flight" against "slot bytes overwritten"; a caller must
    :meth:`wait_upload` before trusting / overwriting a slot it previously launched
    an upload from.
    """

    def __init__(
        self,
        model_path: str,
        num_layers: int,
        num_experts: int,
        capacity: int,
        *,
        hidden_size: int,
        intermediate_size: int,
    ):
        if capacity < num_experts:
            raise ValueError(
                f"exclusive pool capacity {capacity} < num_experts {num_experts}: "
                "prefill materializes a whole layer into the pool"
            )
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.capacity = capacity
        self.shapes = nvfp4_bank_shapes(hidden_size, intermediate_size)
        self.schema_order = tuple(self.shapes)  # == _BANK_SCHEMAS["nvfp4"]

        self._shard_fds: dict[str, int] = {}
        self._records: dict[int, tuple[int, list[tuple[int, int]]]] = {}
        self._scan_checkpoint(model_path)

        self.banks: dict[str, torch.Tensor] = {}
        pool_bytes = 0
        for name, (tail, dtype) in self.shapes.items():
            self.banks[name] = torch.empty((capacity, *tail), dtype=dtype, pin_memory=True)
            pool_bytes += capacity * _row_bytes(tail, dtype)
        self.pool_bytes = pool_bytes

        # scratch must hold ALL aligned regions of one expert at once (cursors
        # accumulate). Region totals vary per expert by alignment phase (each
        # tensor's 4 KiB window), so take the MAX across records, not record[0].
        scratch_bytes = max(
            (
                sum(end - start for start, end in _regions_of(pieces))
                for _fd, pieces in self._records.values()
            ),
            default=_BLK,
        )
        self._scratch = mmap.mmap(-1, scratch_bytes)
        self._sview = memoryview(self._scratch)

        # (layer, expert) -> host slot; -1 = not host-resident
        self.host_slot_of_id = [-1] * (num_layers * num_experts)
        self.id_of_host_slot = [-1] * capacity
        self._last_use = [0] * capacity
        self._clock = 0
        self._events: list[torch.cuda.Event | None] = [None] * capacity
        self._stream: torch.cuda.Stream | None = None

    # ---------- checkpoint scan ----------

    def _scan_checkpoint(self, model_path: str) -> None:
        from freetoken.models.nemotron_h.weight import _EXPERT_KEY_RE
        with open(os.path.join(model_path, "config.json")) as f:
            layer_types = json.load(f)["layers_block_type"]
        layer_ids = tuple(
            i for i, kind in enumerate(layer_types) if kind == "moe"
        )
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"no safetensors index under {model_path}")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        by_layer: dict[int, dict[int, str]] = {}
        for key, shard in weight_map.items():
            m = _EXPERT_KEY_RE.match(key)
            if not m:
                continue
            backbone_layer = int(m.group("layer"))
            try:
                bank_layer = layer_ids.index(backbone_layer)
            except ValueError:
                continue
            by_layer.setdefault(bank_layer, {})[int(m.group("expert"))] = shard
        if not by_layer:
            raise ValueError(f"no NVFP4 expert tensors under {model_path}")

        headers: dict[str, tuple[int, dict]] = {}
        for bank_layer in range(self.num_layers):
            experts = by_layer.get(bank_layer)
            if not experts or len(experts) != self.num_experts:
                raise ValueError(
                    f"bank layer {bank_layer}: found "
                    f"{0 if not experts else len(experts)} of {self.num_experts} experts"
                )
            for expert in range(self.num_experts):
                shard = experts[expert]
                if shard not in self._shard_fds:
                    path = os.path.join(model_path, shard)
                    with open(path, "rb") as f:
                        n = struct.unpack("<Q", f.read(8))[0]
                        header = json.loads(f.read(n))
                    self._shard_fds[shard] = os.open(path, os.O_RDONLY | os.O_DIRECT)
                    headers[shard] = (n, header)
                n, header = headers[shard]
                pieces = []
                for suffix, _bank, _scalar in _TENSOR_MAP:
                    key = (
                        f"backbone.layers.{layer_ids[bank_layer]}.mixer.experts."
                        f"{expert}.{suffix}"
                    )
                    entry = header[key]
                    off, end = entry["data_offsets"]
                    pieces.append((8 + n + off, end - off))
                self._records[bank_layer * self.num_experts + expert] = (
                    self._shard_fds[shard],
                    pieces,
                )
        self._largest_region = max(
            end - start
            for _fd, pieces in self._records.values()
            for start, end in _regions_of(pieces)
        )

    # ---------- host residency ----------

    def host_slot(self, layer: int, expert: int) -> int:
        return self.host_slot_of_id[layer * self.num_experts + expert]

    def _take_slot(self, layer: int, expert: int) -> int:
        self._clock += 1
        for slot in range(self.capacity):
            if self.id_of_host_slot[slot] < 0:
                self._install(slot, layer, expert)
                return slot
        slot = min(range(self.capacity), key=lambda s: self._last_use[s])
        event = self._events[slot]
        if event is not None:
            event.synchronize()  # the pending H2D finished reading these bytes
            self._events[slot] = None
        self.host_slot_of_id[self.id_of_host_slot[slot]] = -1
        self._install(slot, layer, expert)
        return slot

    def _install(self, slot: int, layer: int, expert: int) -> None:
        flat = layer * self.num_experts + expert
        self.id_of_host_slot[slot] = flat
        self.host_slot_of_id[flat] = slot
        self._last_use[slot] = self._clock

    # ---------- refill ----------

    def refill(self, layer: int, expert: int) -> int:
        """Load one expert's six rows into the pool (O_DIRECT); returns its host slot."""
        slot = self.host_slot(layer, expert)
        if slot >= 0:
            self._last_use[slot] = self._clock = self._clock + 1
            return slot
        slot = self._take_slot(layer, expert)
        flat = layer * self.num_experts + expert
        fd, pieces = self._records[flat]
        # Group the six tensors into their minimal enclosing 4 KiB-aligned regions,
        # each read to its OWN cursor in the scratch buffer (regions overlap nothing;
        # cursors are cumulative so every region keeps its bytes until extraction).
        aligned = sorted(
            (off // _BLK * _BLK, -(-(off + length) // _BLK) * _BLK)
            for off, length in pieces
        )
        regions: list[list[int]] = []
        for start, end in aligned:
            if regions and start <= regions[-1][1]:
                regions[-1][1] = max(regions[-1][1], end)
            else:
                regions.append([start, end])
        cursors: list[int] = []
        cursor = 0
        for region_at, (start, end) in enumerate(regions):
            need = end - start
            # POSIX allows O_DIRECT reads to return short; drain to completion.
            # Aligned request => partials are whole blocks, so cursor stays aligned.
            while need:
                got = os.preadv(fd, [self._sview[cursor : cursor + need]], start)
                if got <= 0:
                    raise OSError(f"O_DIRECT read stalled: {cursor=} {start=} {need=}")
                assert got % _BLK == 0 and start % _BLK == 0 and cursor % _BLK == 0
                start += got
                cursor += got
                need -= got
            cursors.append(cursor - (end - regions[region_at][0]))
        for (suffix, bank_name, is_global), (off, length) in zip(_TENSOR_MAP, pieces):
            region_at = next(
                (i for i, (s, e) in enumerate(regions) if s <= off and off + length <= e)
            )
            start, _end = regions[region_at]
            raw = self._sview[
                cursors[region_at] + (off - start) : cursors[region_at] + (off - start) + length
            ]
            tensor = self.banks[bank_name][slot]
            if is_global:
                # exactly the loader's conversion: F32 scalar -> fp16, then broadcast
                scalar = torch.tensor(struct.unpack("<f", raw)[0]).to(torch.float16)
                tensor.fill_(scalar)
            else:
                assert length == _row_bytes(*self.shapes[bank_name]), (
                    bank_name,
                    length,
                    self.shapes[bank_name],
                )
                tensor.view(torch.uint8).flatten()[:] = torch.frombuffer(
                    bytearray(raw), dtype=torch.uint8
                )
        return slot

    # ---------- upload ----------

    def upload_async(self, layer: int, expert: int, dst: dict[str, torch.Tensor], dst_index: int) -> int:
        """Schedule every bank row H2D into ``dst[bank][dst_index]`` on a copy stream.

        Returns the host slot; the slot stays BUSY (its bytes belong to this upload)
        until :meth:`wait_upload` or a later refill of the same slot.
        """
        slot = self.refill(layer, expert)
        if self._stream is None:
            self._stream = torch.cuda.Stream()
            self._events = [torch.cuda.Event() for _ in range(self.capacity)]
        event = self._events[slot]
        if event is not None:
            event.synchronize()  # serialize against the slot's previous upload
        with torch.cuda.stream(self._stream):
            for name in self.schema_order:
                dst[name][dst_index].copy_(self.banks[name][slot], non_blocking=True)
            self._events[slot].record()
        return slot

    def wait_upload(self, slot: int) -> None:
        event = self._events[slot]
        if event is not None:
            torch.cuda.current_stream().wait_event(event)

    def synchronize_all(self) -> None:
        for event in self._events:
            if event is not None:
                event.synchronize()


def _row_bytes(tail: tuple, dtype: torch.dtype) -> int:
    size = torch.empty((), dtype=dtype).element_size()
    count = 1
    for dim in tail:
        count *= dim
    return count * size if size else count // 2  # float8 has element_size 1


def _regions_of(pieces):
    aligned = sorted(
        (off // _BLK * _BLK, -(-(off + length) // _BLK) * _BLK)
        for off, length in pieces
    )
    regions: list[tuple[int, int]] = []
    for start, end in aligned:
        if regions and start <= regions[-1][1]:
            last = regions[-1]
            regions[-1] = (last[0], max(last[1], end))
        else:
            regions.append((start, end))
    return regions
