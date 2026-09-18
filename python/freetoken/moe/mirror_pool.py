"""Bounded host mirror of the GPU expert cache (FREETOKEN_MIRROR_EXPERT_RAM=1).

The baseline profile keeps every expert row pinned in host RAM (15.41 GiB for
Nemotron-3.5-Lightning) and fetches a GPU miss straight from that whole-model
copy. This pool keeps only ``capacity`` rows -- enough to hold the complement of
the GPU cache plus whatever duplicates still fit -- and preserves the property
that makes the baseline fast: **every GPU miss is served from host RAM, never
from disk**.

Residency invariant
-------------------
Each expert row is ``on_gpu``, ``in_pool``, or both. The union must cover all
``L*E`` rows at all times::

    on_gpu(id) or in_pool(id)   for every id            (coverage)

Coverage is what lets a GPU miss be a pure host->device row copy. It is
maintained by the swap: when expert ``new`` is admitted into GPU slot ``s``,
displacing victim ``v``,

* ``new`` is copied pool -> GPU (H2D), and
* if ``v`` has no pool row (it was GPU-only), the *pool row just vacated by*
  ``new`` receives ``v`` (D2H).

Because ``new`` leaves the pool exactly when ``v`` needs a pool row, the pool
never allocates: a swap is a permutation of pool rows. When ``v`` already has a
pool row (a duplicate), the D2H is skipped entirely and eviction is free -- this
is the common case below the KV ceiling, where ``capacity + gpu_slots > L*E``
leaves room for duplicates.

Duplicates are seeded for the GPU's *coldest* rows, since those are the ones LFU
will evict first; see ``seed_duplicates``.

Disk is touched only by ``load_initial`` at startup. The hot path is pure PCIe,
so CUDA graphs and prefill overlap stay enabled exactly as in the baseline.
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


def plan_capacity(num_layers: int, num_experts: int, final_gpu_slots: int,
                  reserve: int | None = None) -> int:
    """Rows the mirror must hold so coverage survives the KV ceiling.

    The GPU cache shrinks as the growable KV arena takes its slots back; it is
    smallest (``final_gpu_slots``) at the KV ceiling, which is exactly when the
    host side must be largest. Sizing for that worst case up front is deliberate:
    growing a pinned pool later costs ~762 ms/GiB (measured), a stall no request
    should ever pay, and the rows are needed eventually anyway.

    ``reserve`` rows on top of the coverage requirement stay available so a
    writeback always has somewhere to land that the same step's upload is not
    reading (see mirror_kernels), and so a prefill materialize can stage a whole
    layer plus the experts it displaces. Both effects peak at one layer each, so
    the default is ``2 * num_experts``.
    """
    if num_layers <= 0 or num_experts <= 0 or final_gpu_slots < 0:
        raise ValueError("mirror capacity needs positive geometry")
    if reserve is None:
        # One layer for writeback landing rows, one for a prefill materialize,
        # plus one more so a decode burst cannot drain the stack between the
        # per-step pushes (measured: 4401 starved writebacks with 2 layers on a
        # 21K-token request).
        reserve = 3 * num_experts
    total = num_layers * num_experts
    return min(max(total - final_gpu_slots, 0) + reserve, total)


class MirrorExpertPool:
    """Fixed-size pinned host mirror; swaps rows with the GPU cache, never disk."""

    def __init__(self, model_path: str, num_layers: int, num_experts: int,
                 capacity: int, *, hidden_size: int, intermediate_size: int,
                 device: torch.device | None = None):
        total = num_layers * num_experts
        if capacity <= 0 or num_layers <= 0 or num_experts <= 0:
            raise ValueError("mirror pool capacity and model dimensions must be positive")
        if capacity > total:
            raise ValueError("mirror capacity above the model's expert count is waste")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.capacity = capacity
        self.total = total
        self.device = device
        self.shapes = nvfp4_bank_shapes(hidden_size, intermediate_size)
        self.schema_order = tuple(self.shapes)
        # Meta sources keep the engine's bank-shape budgeting working without
        # ever holding a byte; the real rows live in `banks`.
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
        # Residency maps. `pool_row_of_id[id]` is the pool row holding `id`, or
        # -1. `id_of_pool_row[row]` is its inverse. Both are host-side ints; the
        # device-side mirror used by the copy kernels is built by the cache.
        self.pool_row_of_id = [-1] * total
        self.id_of_pool_row = [-1] * capacity
        self.swaps = 0
        self.free_evictions = 0
        self.d2h_rows = 0
        try:
            self._scan_checkpoint(model_path)
            scratch_bytes = max(sum(end - start for start, end in _regions_of(pieces))
                                for _fd, pieces in self._records.values())
            self._scratch = mmap.mmap(-1, scratch_bytes)
            self._sview = memoryview(self._scratch)
            for name, (tail, dtype) in self.shapes.items():
                self.banks[name] = torch.empty(
                    (capacity, *tail), dtype=dtype, pin_memory=True
                )
            self.row_bytes = {
                name: _row_bytes(tail, dtype) for name, (tail, dtype) in self.shapes.items()
            }
            self.pool_bytes = capacity * sum(self.row_bytes.values())
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------
    # Checkpoint scan + startup fill (the only disk contact)
    # ------------------------------------------------------------------

    def _scan_checkpoint(self, model_path: str) -> None:
        with open(os.path.join(model_path, "config.json")) as f:
            layer_types = json.load(f)["layers_block_type"]
        layer_ids = [i for i, kind in enumerate(layer_types) if kind == "moe"]
        if len(layer_ids) != self.num_layers:
            raise ValueError("checkpoint MoE layer count does not match mirror pool")
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        headers: dict[str, tuple[int, dict]] = {}
        for layer, backbone_layer in enumerate(layer_ids):
            for expert in range(self.num_experts):
                keys = [f"backbone.layers.{backbone_layer}.mixer.experts.{expert}.{suffix}"
                        for suffix, _, _ in _TENSOR_MAP]
                shards = {weight_map[key] for key in keys}
                if len(shards) != 1:
                    raise ValueError("mirror pool requires each expert's tensors in one shard")
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

    def _read_row(self, flat: int, row: int) -> None:
        """Read expert ``flat`` from the checkpoint into pool row ``row``."""
        fd, pieces = self._records[flat]
        regions = _regions_of(pieces)
        size = self._fd_size[fd]
        cursors, cursor = [], 0
        for start, end in regions:
            want = min(end, -(-size // _BLK) * _BLK) - start
            got = 0
            while got < want:
                chunk = os.preadv(fd, [self._sview[cursor + got:cursor + want]], start + got)
                if chunk == 0:
                    if start + got >= size:
                        break  # ragged final block: EOF inside the aligned tail
                    raise OSError(f"short O_DIRECT read for expert {flat}")
                got += chunk
            cursors.append(cursor)
            cursor += end - start
        for (_, name, scalar), (off, length) in zip(_TENSOR_MAP, pieces):
            region = next(i for i, (start, end) in enumerate(regions)
                          if start <= off and off + length <= end)
            start = cursors[region] + off - regions[region][0]
            raw = self._sview[start:start + length]
            try:
                dst = self.banks[name][row]
                if scalar:
                    # Match the loader's F32 -> F16 tensor conversion (overflow to
                    # inf, not fill_()'s error). struct.unpack avoids
                    # torch.frombuffer on a possibly-unaligned 4-byte slice.
                    f32 = struct.unpack("<f", bytes(raw))[0]
                    dst.copy_(torch.tensor(f32, dtype=torch.float16))
                else:
                    # memmove, not Tensor.copy_: the torch dispatcher costs ~5.7 ms
                    # per row here, a memmove 0.077 ms (measured).
                    src = torch.frombuffer(raw, dtype=torch.uint8)
                    try:
                        ctypes.memmove(
                            dst.view(torch.uint8).data_ptr(), src.data_ptr(), length
                        )
                    finally:
                        del src
            finally:
                raw.release()

    def load_initial(self, gpu_ids) -> None:
        """Fill the pool so coverage holds, then spend the slack on duplicates.

        ``gpu_ids`` is the set of expert ids the GPU cache already holds (it is
        filled from the same checkpoint by the regular loader). Rows not on the
        GPU are mandatory -- they exist nowhere else. Remaining rows mirror GPU
        residents so their eviction can skip the D2H.
        """
        gpu_ids = set(gpu_ids)
        missing = [flat for flat in range(self.total) if flat not in gpu_ids]
        if len(missing) > self.capacity:
            raise RuntimeError(
                f"mirror pool of {self.capacity} rows cannot cover {len(missing)} "
                f"experts absent from the GPU cache"
            )
        row = 0
        for flat in missing:
            self._read_row(flat, row)
            self.pool_row_of_id[flat] = row
            self.id_of_pool_row[row] = flat
            row += 1
        return row

    def seed_duplicates(self, cold_first, reserve: int | None = None) -> int:
        """Mirror GPU-resident experts into the pool's leftover rows.

        ``cold_first`` lists GPU-resident ids in eviction order (coldest first),
        which is the order in which the eviction policy will want them: a
        duplicate only pays off if that row is evicted before the duplicate is
        displaced.

        ``reserve`` rows are left unowned so a writeback always has a landing
        spot and a prefill materialize can stage a layer plus its victims;
        defaults to two layers, matching ``plan_capacity``. Returns the number
        seeded.
        """
        if reserve is None:
            reserve = 2 * self.num_experts
        free_rows = [r for r in range(self.capacity) if self.id_of_pool_row[r] < 0]
        if reserve:
            free_rows = free_rows[:-reserve] if reserve < len(free_rows) else []
        seeded = 0
        for flat, row in zip(cold_first, free_rows):
            if self.pool_row_of_id[flat] >= 0:
                continue
            self._read_row(flat, row)
            self.pool_row_of_id[flat] = row
            self.id_of_pool_row[row] = flat
            seeded += 1
        return seeded

    # ------------------------------------------------------------------
    # Hot path bookkeeping (no disk, no allocation)
    # ------------------------------------------------------------------

    def plan_swap(self, new_id: int, victim_id: int) -> tuple[int, int]:
        """Book a swap and return ``(src_row, d2h_row)``.

        ``src_row`` is the pool row holding ``new_id`` (the H2D source).
        ``d2h_row`` is where ``victim_id`` must be written back, or -1 when the
        victim already has a pool row (the free case).

        Ownership is updated here, so the caller must issue both copies.
        """
        src_row = self.pool_row_of_id[new_id]
        if src_row < 0:
            raise RuntimeError(
                f"coverage violated: expert {new_id} is neither on GPU nor in the pool"
            )
        self.swaps += 1
        if victim_id < 0:
            # Admission into a never-used slot: nothing leaves the GPU. The row
            # stays in the pool as a duplicate of the now-GPU-resident expert.
            return src_row, -1
        if self.pool_row_of_id[victim_id] >= 0:
            self.free_evictions += 1
            return src_row, -1
        # The vacated row takes the victim: a permutation, never an allocation.
        self.pool_row_of_id[new_id] = -1
        self.pool_row_of_id[victim_id] = src_row
        self.id_of_pool_row[src_row] = victim_id
        self.d2h_rows += 1
        return src_row, src_row

    def covers(self, flat: int, on_gpu: bool) -> bool:
        return on_gpu or self.pool_row_of_id[flat] >= 0

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self._sview is not None:
            self._sview.release()
            self._sview = None
        if self._scratch is not None:
            self._scratch.close()
            self._scratch = None
        for fd in getattr(self, "_shard_fds", {}).values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._shard_fds = {}

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
