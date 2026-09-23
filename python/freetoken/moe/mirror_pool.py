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

# Which bank each expert projection feeds, and the safetensors dtype string each
# tensor kind must carry. Roles come from the model's Nvfp4ExpertSourceSpec
# (proj_to_role), so nothing here names a projection: Nemotron-H calls its
# ungated projection up_proj, Qwen3.5/Ornith has gate_proj + up_proj, MiniMax
# calls them w1/w3/w2, and all three land in the same six banks.
_ROLE_BANKS = {
    "gate": ("gate_up_packed", "gate_up_scale", "gate_up_global"),
    "up": ("gate_up_packed", "gate_up_scale", "gate_up_global"),
    "down": ("down_packed", "down_scale", "down_global"),
}
_KIND_BANK_INDEX = {"weight": 0, "weight_scale": 1, "weight_scale_2": 2}
_KIND_DTYPE = {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32"}


def nvfp4_bank_shapes(hidden_size: int, intermediate_size: int,
                      *, gated: bool = False) -> dict:
    """Runtime bank shapes for one native NVFP4 expert row.

    Mirrors ``models.nvfp4_banks._alloc_nvfp4_host_banks`` exactly, including its
    gate|up fusion on the output-row axis: a gated expert's gate_up banks hold
    ``2 * intermediate_size`` rows (gate first, then up), an ungated one holds
    ``intermediate_size``. The mirror's rows must be byte-identical to what the
    regular loader produces, because a GPU miss is served from either.
    """
    h, i = hidden_size, intermediate_size
    g = 2 if gated else 1
    if h <= 0 or i <= 0 or h % 16 or i % 16:
        raise ValueError("native NVFP4 dimensions must be positive multiples of 16")
    return {
        "gate_up_packed": ((g * i, h // 2), torch.uint8),
        "gate_up_scale": ((g * i, h // 16), torch.float8_e4m3fn),
        "gate_up_global": ((g * i,), torch.float16),
        "down_packed": ((h, i // 2), torch.uint8),
        "down_scale": ((h, i // 16), torch.float8_e4m3fn),
        "down_global": ((h,), torch.float16),
    }


def default_reserve_rows(num_experts: int) -> int:
    """Rows the pool always holds back, never counted towards coverage.

    One layer for writeback landing rows, one for a prefill materialize, and
    one so a decode burst cannot drain the stack between the per-step pushes
    (measured: 4401 starved writebacks with only two layers on a 21K-token
    request). The planner and the pool must agree on this number: the pool's
    arena floor is derived from it, and a floor computed against a smaller
    reserve than the pool actually withholds starves writebacks instead of
    breaking coverage outright, which is far harder to see.
    """
    return 3 * num_experts


_RESERVE_ROWS_ENV = "FREETOKEN_MIRROR_RESERVE_ROWS"


def resolve_reserve_rows(num_experts: int, *, env: dict | None = None) -> int:
    """Resolve the pool's reserve-row count from ``FREETOKEN_MIRROR_RESERVE_ROWS``.

    This is the ONE place that env var is read. Call it once and pass the
    same return value to both ``plan_capacity(reserve=...)`` and
    ``MirrorExpertPool(reserve_rows=...)`` -- those two are required to agree
    (see both docstrings), and resolving the env var independently for each
    is exactly how they would drift apart.

    Unset or empty -> ``default_reserve_rows(num_experts)``, i.e. today's
    behaviour, byte-identical. Any other value must parse as a positive
    integer row count; anything else (zero, negative, non-integer) raises
    ``ValueError`` naming the variable -- a typo must fail loudly rather than
    silently fall back to the default and produce a valid-looking arm sized
    against the wrong reserve.
    """
    env = os.environ if env is None else env
    raw = env.get(_RESERVE_ROWS_ENV)
    if raw is None or not raw.strip():
        return default_reserve_rows(num_experts)
    stripped = raw.strip()
    try:
        value = int(stripped)
    except ValueError:
        raise ValueError(
            f"{_RESERVE_ROWS_ENV}={raw!r} is not a valid integer row count "
            "(must be a positive integer, e.g. \"128\")"
        ) from None
    if value <= 0:
        raise ValueError(
            f"{_RESERVE_ROWS_ENV}={raw!r} must be a positive integer row "
            "count (0 or negative leaves the pool unable to cover anything)"
        )
    return value


def prefill_buffer_slots(num_experts: int) -> int:
    """GPU slots the prefill double buffer physically occupies at the head of the cache.

    ``_init_prefill_overlap_buffers`` views the first ``2 * num_experts`` slots
    as two whole-layer buffers. A decode resident may sit in this region too
    (there is no separate reservation any more -- an earlier design fenced
    decode out of it entirely with a victim floor, which cost 256 of 2173
    arena slots on Nemotron for no coverage benefit): the region's only
    special property is that ``offload_cache._invalidate_prefill_buffer``
    writes a slot's occupant back to the pool before every fill here
    overwrites it, so the coverage invariant survives the overwrite exactly
    as it does for a decode eviction anywhere else in the cache.

    The return value is still needed by the coverage math that does NOT
    change: ``_prefetch_split_mirror`` treats any expert whose slot falls in
    this region as a miss (its bytes there are volatile within the chunk),
    so a mirror pool must be sized as if these slots held no resident for the
    purpose of coverage sizing, even though decode may in fact be using them
    between prefills.
    """
    return 2 * num_experts


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
    reading (see mirror_kernels). The default is ``default_reserve_rows`` --
    THREE layers, not the two this docstring used to claim; the third is decode
    burst slack, and the planner and the pool must agree on the number or the
    arena floor is priced against a reserve the pool does not withhold.
    """
    if num_layers <= 0 or num_experts <= 0 or final_gpu_slots < 0:
        raise ValueError("mirror capacity needs positive geometry")
    if reserve is None:
        reserve = default_reserve_rows(num_experts)
    total = num_layers * num_experts
    return min(max(total - final_gpu_slots, 0) + reserve, total)


def resolve_cache_schema(pool, bank_schema: tuple, layout) -> dict:
    """The {cache_bank_name: pool_bank_name} mapping this pool can serve ``bank_schema``
    under, or a ``ValueError`` naming why it cannot.

    The common case -- a checkpoint's own loader and this pool agree on names,
    because both come from the fixed ``nvfp4``/``gguf`` bank-name tuples in
    ``offload_cache._BANK_SCHEMAS`` -- needs nothing more than identity and is
    returned first, unconditionally, so a model on that path (Nemotron-H,
    every GGUF family) is byte-for-byte unaffected by anything below.

    A kernel-method model (``layout`` is the method's own ``BankSpec`` dict,
    ``offload_cache.OffloadMoeCache.layout``) may bind its NVFP4 banks to
    different NAMES for the exact same raw ModelOpt row bytes -- the Triton
    inline-dequant kernel calls its packed-weight bank ``"gate_up"``,
    ``nvfp4_bank_shapes`` (what this pool reads off disk) calls the same bytes
    ``"gate_up_packed"``. Rather than special-case that one kernel by name
    (which would fail the next kernel-method model that reuses the same raw
    layout under yet another name, or silently "match" one that does not),
    this generalizes on the one thing that actually decides whether a GPU miss
    can be served from this pool's rows: position for position, is the cache's
    non-resident bank the same ``(shape, dtype)`` as the pool's own raw NVFP4
    row for that role? The exact byte content is not re-derived here -- it is
    proven once per gating/shard combination this repo ships, in
    tests/moe/test_mirror_pool.py, against the real kernel's own ``pack()`` --
    but the shape/dtype match is the runtime-checkable proxy for it, and any
    layout that fails it (Marlin, b12x: pre-tiled for their GEMM, globals
    folded into a GPU-resident alpha instead of a bank, so the bank *count*
    already differs) is refused by name instead of silently mismatched.
    """
    own = tuple(pool.schema_order)
    target = tuple(bank_schema)
    if own == target:
        return dict(zip(own, own))
    if pool.quant_format != "nvfp4":
        raise ValueError(
            f"mirror pool geometry/schema does not match cache: this "
            f"{pool.quant_format!r} pool serves banks {own}, the cache "
            f"expects {target}"
        )
    if layout is None:
        raise ValueError(
            "mirror pool geometry/schema does not match cache: this NVFP4 "
            f"pool serves banks {own}, the cache expects {target} with no "
            "kernel layout to check them against (a kernel-method model must "
            "pass its method's layout() into the cache)"
        )
    if len(own) != len(target):
        raise ValueError(
            f"mirror pool cannot serve this cache's {len(target)}-bank "
            f"layout {target} with its {len(own)} raw NVFP4 banks {own} -- "
            "a different bank count is not a raw checkpoint-row layout (e.g. "
            "Marlin/b12x pre-tile the weights and fold the global scale into "
            "a GPU-resident alpha instead of a bank)"
        )
    mapping = {}
    for pool_name, cache_name in zip(own, target):
        want_tail, want_dtype = pool.shapes[pool_name]
        spec = layout.get(cache_name)
        if spec is None or spec.resident:
            raise ValueError(
                f"mirror pool cannot serve cache bank {cache_name!r}: it is "
                "not a non-resident row bank in this layout"
            )
        if not getattr(spec, "raw_row", False):
            raise ValueError(
                f"mirror pool cannot serve cache bank {cache_name!r}: the kernel "
                "does not declare it a raw checkpoint row (BankSpec.raw_row), and "
                "a matching shape alone does not prove the bytes are the same"
            )
        if tuple(spec.shape) != tuple(want_tail) or spec.dtype != want_dtype:
            raise ValueError(
                f"mirror pool cannot serve cache bank {cache_name!r}: its "
                f"layout gives shape {tuple(spec.shape)} dtype {spec.dtype}, "
                f"but the raw NVFP4 checkpoint row for this role is "
                f"{tuple(want_tail)} {want_dtype} -- {cache_name!r} is not a "
                "raw checkpoint row (pre-tiled kernel banks, e.g. Marlin, "
                "cannot be served by the mirror pool)"
            )
        mapping[cache_name] = pool_name
    return mapping


class MirrorExpertPool:
    """Fixed-size pinned host mirror; swaps rows with the GPU cache, never disk."""

    def __init__(self, model_path: str, num_layers: int, num_experts: int,
                 capacity: int, *, hidden_size: int, intermediate_size: int,
                 spec=None, config=None,
                 device: torch.device | None = None,
                 reserve_rows: int | None = None,
                 source=None):
        """``source=None`` (default): the native NVFP4 path, byte-identical to
        before this parameter existed -- every line below that reads ``spec``
        and calls ``nvfp4_bank_shapes``/``self._scan_checkpoint`` is unchanged.

        ``source`` (S12c): an alternative checkpoint format that has already
        indexed its own expert rows (e.g. a GGUF file -- see
        ``models/qwen3_5_moe/gguf.gguf_expert_row_extents``), bypassing
        ``_scan_checkpoint`` entirely. It carries three things this
        constructor needs and nothing ``_read_row``/``load_initial``/
        ``seed_duplicates`` do not already handle generically:

        * ``quant_format`` -- the ``ExpertBanks`` format name the placeholder
          banks and the engine's bank schema are keyed on ("gguf").
        * ``shapes`` -- ``{bank_name: (tail_shape, dtype)}``, one size class
          for every bank (a mirror pool has one pointer/stride per bank; a
          source with more than one row size per bank must refuse before
          reaching here -- see ``gguf_mirror_bank_shapes``).
        * ``records``/``shard_fds``/``fd_size`` -- exactly ``_scan_checkpoint``'s
          own output shape, so ``_read_row`` (which only reads
          ``self._records``/``self._shard_fds``/``self._fd_size``) needs no
          change to serve either format.
        """
        total = num_layers * num_experts
        if capacity <= 0 or num_layers <= 0 or num_experts <= 0:
            raise ValueError("mirror pool capacity and model dimensions must be positive")
        if capacity > total:
            raise ValueError("mirror capacity above the model's expert count is waste")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.capacity = capacity
        self.total = total
        self.reserve_rows = (default_reserve_rows(num_experts)
                             if reserve_rows is None else reserve_rows)
        if capacity < total and capacity <= self.reserve_rows:
            raise ValueError(
                f"mirror capacity {capacity} does not exceed the "
                f"{self.reserve_rows}-row writeback/staging reserve: the pool "
                f"could cover no expert at all (raise --moe-mirror-host-rows)"
            )
        self.device = device
        self._spec = spec
        self._config = config
        self._source = source
        self.gated = bool(getattr(spec, "gated", False))
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        if source is None:
            self.quant_format = "nvfp4"
            self.shapes = nvfp4_bank_shapes(hidden_size, intermediate_size,
                                            gated=self.gated)
        else:
            self.quant_format = source.quant_format
            self.shapes = source.shapes
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
        # flat id -> ((fd, pieces), ...): one read group per shard the expert spans.
        self._records: dict[int, tuple[tuple[int, list[tuple]], ...]] = {}
        self._scratch = None
        self._sview = None
        self._closed = False
        self.banks: dict[str, torch.Tensor] = {}
        # Residency maps. `pool_row_of_id[id]` is the pool row holding `id`, or
        # -1. `id_of_pool_row[row]` is its inverse. Both are host-side ints; the
        # device-side mirror used by the copy kernels is built by the cache.
        self.pool_row_of_id = [-1] * total
        self.id_of_pool_row = [-1] * capacity
        try:
            if source is None:
                self._scan_checkpoint(model_path)
            else:
                self._records, self._shard_fds, self._fd_size = (
                    source.records, source.shard_fds, source.fd_size
                )
            scratch_bytes = max(
                sum(end - start for start, end in
                    _regions_of([(off, length) for off, length, *_rest in pieces]))
                for groups in self._records.values() for _fd, pieces in groups
            )
            self._scratch = mmap.mmap(-1, scratch_bytes)
            self._sview = memoryview(self._scratch)
            # NOT pin_memory=True. torch's caching host allocator rounds every
            # allocation to the next power of two
            # (ATen/core/CachingHostAllocator.h: `roundSize =
            # PowerOf2Ceil(size)`, then allocate_host_memory(roundSize)), and
            # this pool asks it for two banks of several GiB each. Measured on
            # Nemotron, host RAM actually pinned against the pool it was asked
            # for:
            #
            #     1700 rows  pool  8.90 GiB -> 9.11 GiB pinned   (banks 3.95 -> 4)
            #     2100 rows  pool 10.99 GiB -> 18.11 GiB pinned  (banks 4.88 -> 8)
            #     2500 rows  pool 13.08 GiB -> 18.12 GiB pinned  (banks 5.81 -> 8)
            #     2944 rows  pool 15.41 GiB -> 18.12 GiB pinned  (banks 6.85 -> 8)
            #
            # i.e. up to 2x the pool, and a capacity knob whose RAM cost does
            # not move at all across most of its range -- which is exactly how
            # the first sweeps read. The baseline is nearly exact (15.41 ->
            # 15.49) only because it allocates per layer, where the rounding
            # has little room to bite.
            #
            # Allocating pageable and pinning the exact byte range with
            # cudaHostRegister (the same call the baseline's banks go through,
            # freetoken.kernel.pinned.host_register) costs the allocator's
            # reuse, which this pool does not want: the banks are allocated
            # once at startup and live for the process.
            from freetoken.kernel.pinned import host_register

            self._registered = []
            for name, (tail, dtype) in self.shapes.items():
                bank = torch.empty((capacity, *tail), dtype=dtype)
                if source is not None:
                    # A GGUF row's raw bytes cover only its unaligned tensor
                    # width; the bank's per-row stride is padded to 64B (the
                    # loader's own alignment, models/qwen3_5_moe/gguf.py
                    # _expert_layer_geometry). _read_row never writes those
                    # padding columns (its pieces are exactly the source's
                    # extents), so zero them once here instead of leaving
                    # torch.empty garbage a kernel could read past a row's
                    # real width. The NVFP4 path never takes this branch --
                    # every one of its bytes is written by _read_row.
                    bank.zero_()
                nbytes = bank.numel() * bank.element_size()
                host_register(bank.data_ptr(), nbytes)
                self._registered.append(bank.data_ptr())
                self.banks[name] = bank
            self.row_bytes = {
                name: _row_bytes(tail, dtype) for name, (tail, dtype) in self.shapes.items()
            }
            self.pool_bytes = capacity * sum(self.row_bytes.values())
        except BaseException:
            self.close()
            raise

        # The expert-arena floor this pool implies. Coverage needs every
        # expert the GPU does not hold to own a pool row outside the reserve:
        #
        #     total - gpu_slots <= capacity - reserve_rows
        #     =>  gpu_slots >= total - capacity + reserve_rows
        #
        # so a BIGGER pool LOWERS the floor and leaves more room for KV, which
        # is the entire point of the RAM knob. Derived from this pool's own
        # geometry: no model constant, and --moe-mirror-host-rows moves it.
        #
        # A saturated pool (capacity == total) is exempt, and not by rounding:
        # every expert is mirrored at all times, so an eviction never writes
        # back and a materialize never stages -- the reserve those rows exist
        # for is unreachable. Folding the additive reserve in anyway would
        # demand gpu_slots >= reserve_rows from a pool that already holds a
        # copy of everything, which is how the toy geometries in the repro
        # scripts (total == 3 * num_experts) end up "impossible".
        self.min_gpu_slots = (
            0 if self.capacity >= self.total
            else max(self.total - self.capacity + self.reserve_rows, 0)
        )

    def adopt_cache_schema(self, bank_schema, layout) -> None:
        """Rename this pool's banks to the attaching cache's names, if it can serve them.

        Called once, from ``OffloadMoeCache.attach_residency``, with the cache's
        own final ``bank_schema``/``layout``. ``resolve_cache_schema`` decides
        whether this pool's raw NVFP4 rows are what ``bank_schema`` names (it
        raises ``ValueError`` naming the exact mismatch otherwise); a positive
        answer is metadata-only here -- ``self.banks``' tensors, and every byte
        already read into them, move to their new dict key unchanged, so a
        rename can never be the reason a served row differs from what
        ``load_initial``/``_read_row`` wrote.
        """
        mapping = resolve_cache_schema(self, bank_schema, layout)
        if tuple(bank_schema) == tuple(self.schema_order):
            return  # already named exactly as the cache expects
        self.banks = {name: self.banks[mapping[name]] for name in bank_schema}
        self.sources = {name: self.sources[mapping[name]] for name in bank_schema}
        self.shapes = {name: self.shapes[mapping[name]] for name in bank_schema}
        self.row_bytes = {name: self.row_bytes[mapping[name]] for name in bank_schema}
        self.schema_order = tuple(bank_schema)

    # ------------------------------------------------------------------
    # Checkpoint scan + startup fill (the only disk contact)
    # ------------------------------------------------------------------

    def _row_layout(self, role: str, kind: str):
        """Where one checkpoint tensor lands inside this pool's row, and its shape.

        Returns ``(bank, dst_byte_offset, broadcast_entries, expected_shape)``.
        ``broadcast_entries`` is nonzero only for the per-tensor global scale,
        which the banks hold expanded to one FP16 per output row (matching the
        loader, which fills ``gate_up_global[expert, :I]`` from a scalar).
        """
        h, i = self.hidden_size, self.intermediate_size
        bank = _ROLE_BANKS[role][_KIND_BANK_INDEX[kind]]
        # gate occupies the first I output rows, up the next I -- but only when
        # the model is gated; an ungated model's single projection starts at 0.
        row_off = i if (role == "up" and self.gated) else 0
        if role == "down":
            row_off = 0
        if kind == "weight":
            width = (h // 2) if role != "down" else (i // 2)
            rows = i if role != "down" else h
            return bank, row_off * width, 0, [rows, width]
        if kind == "weight_scale":
            width = (h // 16) if role != "down" else (i // 16)
            rows = i if role != "down" else h
            return bank, row_off * width, 0, [rows, width]
        entries = i if role != "down" else h
        return bank, row_off * 2, entries, []

    def _scan_checkpoint(self, model_path: str) -> None:
        """Index every expert row's bytes in the checkpoint, via the model's spec.

        No key format, layer-type list or projection name appears here: the
        model's ``Nvfp4ExpertSourceSpec`` supplies the key pattern, the
        projection->role map and the layer->bank mapping, so Nemotron-H
        (ungated, ``backbone.layers.N.mixer``) and Qwen3.5/Ornith (gated,
        ``model.language_model.layers.N.mlp``) index through the same code.
        """
        spec = self._spec
        if spec is None:
            raise ValueError(
                "the bounded expert mirror needs this model's NVFP4 expert "
                "source spec; models.nvfp4_banks.expert_source_spec returned "
                "None, so its checkpoint layout has never been verified here"
            )
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        # (bank_layer, expert) -> [(key, role, kind)], filtered to MoE layers.
        wanted: dict[tuple[int, int], list[tuple[str, str, str]]] = {}
        for key in weight_map:
            match = spec.key_pattern.match(key)
            if match is None:
                continue
            layer = int(match.group("layer"))
            bank_layer = spec.layer_to_bank(layer, self._config)
            if bank_layer is None:
                continue
            if not 0 <= bank_layer < self.num_layers:
                raise ValueError(
                    f"{spec.desc}: bank layer {bank_layer} outside "
                    f"[0, {self.num_layers})"
                )
            expert = int(match.group("expert"))
            if not 0 <= expert < self.num_experts:
                raise ValueError(f"{spec.desc}: expert {expert} outside the layer")
            role = spec.proj_to_role.get(match.group("proj"))
            if role is None:
                raise ValueError(
                    f"{spec.desc}: unknown projection {match.group('proj')!r}"
                )
            kind = match.group("kind")
            if kind not in _KIND_BANK_INDEX:
                raise ValueError(f"{spec.desc}: unknown tensor kind {kind!r}")
            wanted.setdefault((bank_layer, expert), []).append((key, role, kind))

        expected_roles = ({"gate", "up", "down"} if self.gated else {"up", "down"})
        expected_tensors = 3 * len(expected_roles)
        if len(wanted) != self.total:
            raise ValueError(
                f"{spec.desc}: checkpoint has {len(wanted)} expert rows, "
                f"the pool was sized for {self.total}"
            )

        headers: dict[str, tuple[int, dict]] = {}
        for (bank_layer, expert), entries in wanted.items():
            if len(entries) != expected_tensors:
                raise ValueError(
                    f"{spec.desc}: layer {bank_layer} expert {expert} has "
                    f"{len(entries)} tensors, expected {expected_tensors}"
                )
            if {role for _k, role, _kind in entries} != expected_roles:
                raise ValueError(
                    f"{spec.desc}: layer {bank_layer} expert {expert} does not "
                    f"carry exactly {sorted(expected_roles)}"
                )
            # One read group per shard the expert's tensors live in. HF's size-based
            # sharder cuts wherever the byte budget runs out, so an expert can
            # straddle two files (Ornith ships two such experts); each group is
            # read on its own and lands in its own bank bytes, so the row is the
            # same whichever file each tensor came from.
            groups = []
            for shard in sorted({weight_map[key] for key, _r, _k in entries}):
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
                # Sorted so a row's reads walk the file forward, and so the record is
                # deterministic regardless of weight_map iteration order.
                for key, role, kind in sorted(entries, key=lambda e: (e[1], e[2])):
                    if weight_map[key] != shard:
                        continue
                    entry = header[key]
                    off, end_off = entry["data_offsets"]
                    bank, dst, broadcast, shape = self._row_layout(role, kind)
                    expected = 4 if broadcast else shape[0] * shape[1]
                    if (entry["dtype"] != _KIND_DTYPE[kind]
                            or end_off - off != expected or off < 0
                            or data_start + end_off > self._fd_size[fd]
                            or (not broadcast and entry["shape"] != shape)):
                        raise ValueError(f"unsupported or corrupt native NVFP4 tensor: {key}")
                    pieces.append((data_start + off, end_off - off, bank, dst, broadcast))
                groups.append((fd, pieces))
            self._records[bank_layer * self.num_experts + expert] = tuple(groups)

    def _read_row(self, flat: int, row: int) -> None:
        """Read expert ``flat`` from the checkpoint into pool row ``row``."""
        for fd, pieces in self._records[flat]:
            self._read_group(flat, row, fd, pieces)

    def _read_group(self, flat: int, row: int, fd: int, pieces) -> None:
        """One shard's share of expert ``flat``: read its extents, scatter to ``row``."""
        regions = _regions_of([(off, length) for off, length, _b, _d, _bc in pieces])
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
        for off, length, bank, dst_off, broadcast in pieces:
            region = next(idx for idx, (start, end) in enumerate(regions)
                          if start <= off and off + length <= end)
            start = cursors[region] + off - regions[region][0]
            raw = self._sview[start:start + length]
            try:
                dst = self.banks[bank][row]
                if broadcast:
                    # Match the loader's F32 -> F16 tensor conversion (overflow to
                    # inf, not fill_()'s error). struct.unpack avoids
                    # torch.frombuffer on a possibly-unaligned 4-byte slice.
                    f32 = struct.unpack("<f", bytes(raw))[0]
                    begin = dst_off // 2
                    dst[begin:begin + broadcast].copy_(
                        torch.tensor(f32, dtype=torch.float16)
                    )
                else:
                    # memmove, not Tensor.copy_: the torch dispatcher costs ~5.7 ms
                    # per row here, a memmove 0.077 ms (measured).
                    src = torch.frombuffer(raw, dtype=torch.uint8)
                    try:
                        ctypes.memmove(
                            dst.view(torch.uint8).data_ptr() + dst_off,
                            src.data_ptr(), length,
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
        defaults to this pool's own ``reserve_rows``, which is what the arena
        floor was priced against. Returns the number seeded.
        """
        if reserve is None:
            reserve = self.reserve_rows
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

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        # Unpin before the tensors are collected: a cudaHostRegister pin
        # outliving its allocation is a dangling registration.
        for addr in getattr(self, "_registered", ()):  # noqa: B007
            try:
                from freetoken.kernel.pinned import host_unregister

                host_unregister(addr)
            except Exception:                      # teardown must not raise
                pass
        self._registered = []
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