from __future__ import annotations

import collections
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from freetoken.utils import download_hf_weight
from tqdm import tqdm

LayerToBank = Callable[[int, object], int | None]
DropPageCache = Callable[[str], None]


@dataclass(frozen=True)
class Nvfp4ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str
    # Conventional MoEs concatenate gate|up (2I rows). Nemotron-H has one ungated
    # up projection (I rows) followed by ReLU^2.
    gated: bool = True
    # Optional ModelConfig attribute holding the expert input/output width.  The
    # residual hidden_size remains unchanged for the rest of the model.
    hidden_size_attr: str | None = None
    kind_map: dict[str, str] | None = None
    # The checkpoint stores the QUANT-side global scale (local fp8 scales were
    # multiplied by it before the cast); the banks keep its reciprocal.
    global_reciprocal: bool = False


def _canon_kind(spec: "Nvfp4ExpertSourceSpec", kind: str) -> str:
    return spec.kind_map.get(kind, kind) if spec.kind_map else kind


def _ingest_global(spec: "Nvfp4ExpertSourceSpec", tensor: torch.Tensor) -> torch.Tensor:
    if spec.global_reciprocal:
        tensor = 1.0 / tensor.float()
    return tensor.to(torch.float16)


def _kind_suffix(kind: str) -> str:
    return {"weight": "", "weight_scale": "_scale", "weight_scale_2": "_global"}[kind]


def expert_source_spec(config) -> Nvfp4ExpertSourceSpec | None:
    """This model's NVFP4 expert source spec, or None if it does not publish one.

    The spec is what lets a consumer read expert tensors out of a checkpoint
    without hardcoding one model's key layout -- the difference between
    ``backbone.layers.N.mixer.experts.E.up_proj`` (Nemotron-H, ungated) and
    ``model.language_model.layers.N.mlp.experts.E.gate_proj`` (Qwen3.5/Ornith,
    gated), which are otherwise the same six banks.

    Only models that export ``NVFP4_EXPERT_SOURCE_SPEC`` are resolved. A model
    whose spec has never been exercised through this path returns None, so the
    caller refuses explicitly instead of reading a layout nobody verified.
    """
    import importlib

    model_type = getattr(config, "model_type", "") or ""
    if not model_type.isidentifier():
        return None
    try:
        module = importlib.import_module(f"freetoken.models.{model_type}.weight")
    except ModuleNotFoundError:
        return None
    spec = getattr(module, "NVFP4_EXPERT_SOURCE_SPEC", None)
    return spec if isinstance(spec, Nvfp4ExpertSourceSpec) else None


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def _bank_layer(spec: Nvfp4ExpertSourceSpec, layer: int, config) -> int | None:
    bank_layer = spec.layer_to_bank(layer, config)
    if bank_layer is None:
        return None
    num_layers = _num_moe_layers(config)
    if bank_layer < 0 or bank_layer >= num_layers:
        raise ValueError(
            f"{spec.desc}: bank layer {bank_layer} for checkpoint layer {layer} "
            f"is outside [0, {num_layers})"
        )
    return bank_layer


# Canonical (role, kind) tensors of one NVFP4 expert. ``role`` is "gate" / "up" /
# "down" (what models.Nvfp4ExpertSourceSpec.proj_to_role maps checkpoint projection
# names onto); ``kind`` is the modelopt-native tensor kind -- the packed FP4 codes,
# the per-block FP8 scale, and the scalar FP32 global scale (``weight_scale_2``).
# A checkpoint using different on-disk names for the same three kinds (e.g.
# compressed-tensors' ``weight_packed`` / ``weight_scale`` / ``weight_global_scale``)
# is meant to be normalised to these three by the caller before it looks anything
# up here -- see ``kind_map`` below.
_CANONICAL_KINDS = ("weight", "weight_scale", "weight_scale_2")


@dataclass(frozen=True)
class Nvfp4RowTensor:
    """Where one checkpoint tensor for one ``(role, kind)`` lands in a host row.

    ``bank`` names one of the 6 host banks (see :func:`nvfp4_expert_row_layout`).
    ``byte_offset`` is this tensor's offset, in bytes, within ONE row of that
    bank (i.e. within ``bank_shapes[bank][0]`` flattened row-major) -- add the
    bank's own per-row byte stride times the expert/layer index to get an
    absolute offset into a real ``[E, ...]`` bank tensor.

    ``checkpoint_shape`` and ``checkpoint_dtype`` describe the ON-DISK
    safetensors tensor this slice is filled from -- what a header check (S8)
    validates against -- which is not always the bank's own storage dtype:
    ``weight_scale_2`` is a scalar FP32 on disk, broadcast on the host to one
    FP16 per output row (the dequant multiply wants it per-row; see
    ``load_nvfp4_expert_source_banks``), so its bank dtype (float16) differs
    from its checkpoint dtype (float32) and its bank shape (one FP16 per
    output row) differs from its checkpoint shape (``()``, a scalar).
    """

    bank: str
    byte_offset: int
    checkpoint_shape: tuple[int, ...]
    checkpoint_dtype: torch.dtype


@dataclass(frozen=True)
class Nvfp4RowLayout:
    """The single source of truth for one NVFP4 expert row's byte layout.

    ``bank_shapes``: the 6 banks' per-row ``(shape, dtype)`` -- what
    ``_alloc_nvfp4_host_banks`` allocates (prefixed with ``[num_layers]`` and
    ``[E]``) and what the bounded host mirror (``moe/mirror_pool.py``) will
    allocate too, once it consumes this function (deferred to after the
    upstream merge, S5a part 2 -- see the module docstring below).

    ``tensors``: ``(role, kind) -> Nvfp4RowTensor`` for every canonical tensor
    of one expert (6 entries: gate/up/down x weight/weight_scale, plus the two
    global scales, fewer for an ungated model, which has no "gate" role).

    ``row_bytes``: total HOST bytes for one full expert row, summed across all
    6 banks (post-conversion storage, i.e. the global scale counted as its
    broadcast FP16 array, not its on-disk scalar) -- what pin-budget sizing
    needs per expert (replaces ``offload_cache._BANK_BYTES_PER_EXPERT["nvfp4"]``,
    deferred to after the merge, see the module docstring below).
    """

    bank_shapes: dict[str, tuple[tuple[int, ...], torch.dtype]]
    tensors: dict[tuple[str, str], Nvfp4RowTensor]
    row_bytes: int


def nvfp4_expert_row_layout(
    H: int, I: int, *, gated: bool, kind_map: dict[str, str] | None = None
) -> Nvfp4RowLayout:
    """Compute the NVFP4 expert-row byte layout once, for every consumer to share.

    This was four independent copies before this function existed --
    ``models/nvfp4_banks.py`` (this module, gated-aware), ``moe/mirror_pool.py``
    (``nvfp4_bank_shapes`` + ``_row_layout`` + ``_KIND_DTYPE``, gated-aware but
    unaware a checkpoint might name its tensor kinds differently),
    ``moe/offload_cache.py`` (``_BANK_BYTES_PER_EXPERT["nvfp4"]``, which assumed
    gated ``2*I`` unconditionally), and ``engine/cache_budget.py`` (derived from
    real tensors, so it agreed by construction). Change a model and all four had
    to be found by hand; miss one and the result is silent corruption, not an
    error. See ``tasks/exclusive-expert-ram/reviews/2026-09-22-refactor-plan-final.md``
    section 3.4.

    Layout, gate|up fused on the output-row axis (matches the checkpoint: a
    gated model's ``gate_proj``/``up_proj`` become one ``[2*I, ...]`` tensor
    pair, gate first then up; an ungated model has only "up", occupying the
    whole ``[I, ...]``)::

        gate_up_packed  [(2 if gated else 1) * I, H // 2]   uint8   (packed FP4 codes)
        gate_up_scale   [(2 if gated else 1) * I, H // 16]  fp8_e4m3 (per-block scale)
        gate_up_global  [(2 if gated else 1) * I]           float16 (per-row global scale)
        down_packed     [H, I // 2]                         uint8
        down_scale      [H, I // 16]                        fp8_e4m3
        down_global     [H]                                 float16

    ``kind_map``: an upstream ``Nvfp4ExpertSourceSpec`` field that does not
    exist in this tree yet (it arrives with the S0 merge -- section 3.4,
    step S5a). It will let a checkpoint spell these three canonical kinds
    differently on disk (a compressed-tensors NVFP4 checkpoint uses
    ``weight_packed`` / ``weight_scale`` / ``weight_global_scale`` in place of
    modelopt's ``weight`` / ``weight_scale`` / ``weight_scale_2``) by mapping
    the on-disk name to the canonical one before a caller looks it up in
    ``tensors``. Accepted here so the signature upstream expects is already
    correct; not consumed by this function (the layout below is computed
    purely from ``H``, ``I`` and ``gated`` -- it does not depend on what a
    checkpoint calls anything) and **untested against a real compressed-tensors
    expert checkpoint until the merge lands** (see the CPU test in
    ``tests/moe/test_nvfp4_row_layout.py``, which documents exactly this gap:
    ``load_nvfp4_expert_source_banks`` itself still refuses compressed-tensors
    kind names today, because *its* kind handling is hardcoded to modelopt's
    three names -- the same four-copies problem this function exists to end,
    just not finished yet).

    Remaining consumers to wire after the merge (S5a part 2, not done by this
    function): ``moe/mirror_pool.py``'s ``nvfp4_bank_shapes`` / ``_row_layout``
    / ``_KIND_DTYPE`` should call this instead of re-deriving the layout, and
    ``moe/offload_cache.py``'s ``_BANK_BYTES_PER_EXPERT["nvfp4"]`` should call
    ``row_bytes`` instead of its own (currently gated-only) formula. Neither
    file is touched here: both are owned by a different lane in this phase.
    """
    if H <= 0 or I <= 0 or H % 16 or I % 16:
        raise ValueError("native NVFP4 dimensions must be positive multiples of 16")

    fp8 = torch.float8_e4m3fn
    g = 2 if gated else 1
    bank_shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "gate_up_packed": ((g * I, H // 2), torch.uint8),
        "gate_up_scale": ((g * I, H // 16), fp8),
        "gate_up_global": ((g * I,), torch.float16),
        "down_packed": ((H, I // 2), torch.uint8),
        "down_scale": ((H, I // 16), fp8),
        "down_global": ((H,), torch.float16),
    }

    def _itemsize(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    roles = ("gate", "up", "down") if gated else ("up", "down")
    tensors: dict[tuple[str, str], Nvfp4RowTensor] = {}
    for role in roles:
        packed_bank, scale_bank, global_bank = _ROLE_BANKS_OF[role]
        # gate occupies output rows [0, I), up occupies [I, 2I) when gated; an
        # ungated model's single "up" role starts at 0 (the whole I); "down"
        # always starts at 0 (it has its own banks, not shared with gate/up).
        row_off = I if (role == "up" and gated) else 0
        rows = H if role == "down" else I
        packed_width = (I // 2) if role == "down" else (H // 2)
        scale_width = (I // 16) if role == "down" else (H // 16)

        tensors[(role, "weight")] = Nvfp4RowTensor(
            bank=packed_bank,
            byte_offset=row_off * packed_width,
            checkpoint_shape=(rows, packed_width),
            checkpoint_dtype=torch.uint8,
        )
        tensors[(role, "weight_scale")] = Nvfp4RowTensor(
            bank=scale_bank,
            byte_offset=row_off * scale_width,
            checkpoint_shape=(rows, scale_width),
            checkpoint_dtype=fp8,
        )
        tensors[(role, "weight_scale_2")] = Nvfp4RowTensor(
            bank=global_bank,
            byte_offset=row_off * _itemsize(torch.float16),
            checkpoint_shape=(),
            checkpoint_dtype=torch.float32,
        )

    row_bytes = sum(
        _itemsize(dtype) * (1 if not shape else _prod(shape))
        for shape, dtype in bank_shapes.values()
    )
    return Nvfp4RowLayout(bank_shapes=bank_shapes, tensors=tensors, row_bytes=row_bytes)


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# role -> (packed bank, scale bank, global-scale bank). "gate" and "up" share
# their three banks (the checkpoint fuses them on the output-row axis); "down"
# has its own three.
_ROLE_BANKS_OF = {
    "gate": ("gate_up_packed", "gate_up_scale", "gate_up_global"),
    "up": ("gate_up_packed", "gate_up_scale", "gate_up_global"),
    "down": ("down_packed", "down_scale", "down_global"),
}


def _alloc_nvfp4_host_banks(num_layers: int, E: int, H: int, I: int, *, gated: bool = True):
    """6 NVFP4 source banks, one ``[E, ...]`` tensor per layer (independent allocations),
    unpinned (pin-after-fill): register only after fill to skip cudaHostAlloc's slow
    commit. Caller fills each layer's ``.tensor`` then pins it (per-layer, via
    ``PinPipeline``, as its writes complete).

    Shapes come from :func:`nvfp4_expert_row_layout`, the single source of truth
    for this layout -- this function no longer carries its own copy."""
    from freetoken.moe.host_banks import alloc_layer_banks

    layout = nvfp4_expert_row_layout(H, I, gated=gated)
    specs = {
        name: ((E, *shape), dtype) for name, (shape, dtype) in layout.bank_shapes.items()
    }
    return alloc_layer_banks(specs, num_layers)


def load_nvfp4_expert_source_banks(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Build the 6 native NVFP4 source banks by streaming checkpoint shards (serial per-shard read).

    ModelOpt row layout: gate/up fused on the output-row axis, down separate; the per-tensor
    global scale (weight_scale_2) is kept as a separate per-output-row FP16 bank (``*_global``),
    so dequant is ``fp4 * block_scale * global``. Each bank is one ``[E, ...]`` tensor per
    layer, indexed by ``[bank_layer][expert]``. (The marlin/b12x backends repack these and
    fold the global into per-expert alphas; see moe/nvfp4_backends.py.)

    ``layer_sink=None`` (serving): pin each bank layer as its writes complete, via an
    internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter; for
    marlin/b12x the provider wraps it in a per-layer repacking sink first): the
    completion tracker fires into it instead -- nothing here is pinned, and the sink
    may release banks it has written out, so the returned tensors are only valid
    until then (the caller owns that tradeoff).
    """
    folder = download_hf_weight(model_path)
    index_path = os.path.join(folder, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    E = config.num_experts
    H = getattr(config, spec.hidden_size_attr) if spec.hidden_size_attr else config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    for shard in sorted(set(weight_map.values())):
        drop_page_cache(os.path.join(folder, shard))

    weight_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    global_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        bank_layer = _bank_layer(spec, layer, config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = _canon_kind(spec, match.group("kind"))
        if kind == "weight_scale_2":
            global_shards[shard].append((name, match, bank_layer))
        elif kind in {"weight", "weight_scale"}:
            weight_shards[shard].append((name, match, bank_layer))
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for shard in sorted(global_shards):
        path = os.path.join(folder, shard)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name, match, _bank_layer_id in global_shards[shard]:
                key = (
                    int(match.group("layer")),
                    int(match.group("expert")),
                    match.group("proj"),
                )
                globals_map[key] = _ingest_global(spec, f.get_tensor(name))
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I, gated=spec.gated)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    def _load(sink) -> int:
        tensors_per_expert = 6 if spec.gated else 4
        tracker = LayerCompletionTracker(E * tensors_per_expert, _hb, sink)
        placed = 0
        for shard in tqdm(sorted(weight_shards), desc=f"Loading {spec.desc}", disable=not primary):
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, match, bank_layer_id in weight_shards[shard]:
                    layer = int(match.group("layer"))
                    expert = int(match.group("expert"))
                    proj = match.group("proj")
                    role = spec.proj_to_role[proj]
                    kind = _canon_kind(spec, match.group("kind"))
                    tensor = f.get_tensor(name)
                    if kind == "weight":
                        if role == "gate":
                            gate_up_packed[bank_layer_id][expert, :I] = tensor
                        elif role == "up":
                            if spec.gated:
                                gate_up_packed[bank_layer_id][expert, I:] = tensor
                            else:
                                gate_up_packed[bank_layer_id][expert] = tensor
                        elif role == "down":
                            down_packed[bank_layer_id][expert] = tensor
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    else:
                        global_scale = globals_map[(layer, expert, proj)]
                        if role == "gate":
                            gate_up_scale[bank_layer_id][expert, :I] = tensor
                            gate_up_global[bank_layer_id][expert, :I] = global_scale
                        elif role == "up":
                            if spec.gated:
                                gate_up_scale[bank_layer_id][expert, I:] = tensor
                                gate_up_global[bank_layer_id][expert, I:] = global_scale
                            else:
                                gate_up_scale[bank_layer_id][expert] = tensor
                                gate_up_global[bank_layer_id][expert] = global_scale
                        elif role == "down":
                            down_scale[bank_layer_id][expert] = tensor
                            down_global[bank_layer_id][expert] = global_scale
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    tracker.note(bank_layer_id)
                    placed += 1
            drop_page_cache(path)
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = num_layers * E * (6 if spec.gated else 4)
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


def load_nvfp4_expert_source_banks_parallel(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """parallel counterpart of :func:`load_nvfp4_expert_source_banks`, byte-for-byte same
    placement. bulk weight/weight_scale read via chunked multi-threaded O_DIRECT reader
    (iter_expert_tensors_parallel); tiny globals (``weight_scale_2``) stay serial (negligible
    bytes). ``layer_sink``: see :func:`load_nvfp4_expert_source_banks`."""
    from freetoken.models.weight import iter_expert_tensors_parallel

    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    E = config.num_experts
    H = getattr(config, spec.hidden_size_attr) if spec.hidden_size_attr else config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    weight_info: dict[str, tuple[re.Match[str], int]] = {}  # name -> (match, bank_layer)
    global_names_by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        kind = _canon_kind(spec, match.group("kind"))
        if kind == "weight_scale_2":
            global_names_by_shard[shard].append(name)
        elif kind in {"weight", "weight_scale"}:
            weight_info[name] = (match, bank_layer)
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    # Pass 1: tiny per-tensor global scales (serial; data is scalar-per-expert).
    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for shard in sorted(global_names_by_shard):
        path = os.path.join(folder, shard)
        drop_page_cache(path)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name in global_names_by_shard[shard]:
                m = spec.key_pattern.match(name)
                globals_map[(int(m.group("layer")), int(m.group("expert")), m.group("proj"))] = (
                    _ingest_global(spec, f.get_tensor(name))
                )
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I, gated=spec.gated)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    # Pass 2: bulk weight/weight_scale via the common parallel reader; place by name.
    def _load(sink) -> int:
        tensors_per_expert = 6 if spec.gated else 4
        tracker = LayerCompletionTracker(E * tensors_per_expert, _hb, sink)
        placed = 0
        for name, tensor in iter_expert_tensors_parallel(
            folder, lambda n: n in weight_info, workers=workers, chunk=chunk
        ):
            match, bank_layer_id = weight_info[name]
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            proj = match.group("proj")
            role = spec.proj_to_role[proj]
            kind = _canon_kind(spec, match.group("kind"))
            if kind == "weight":
                if role == "gate":
                    gate_up_packed[bank_layer_id][expert, :I] = tensor
                elif role == "up":
                    if spec.gated:
                        gate_up_packed[bank_layer_id][expert, I:] = tensor
                    else:
                        gate_up_packed[bank_layer_id][expert] = tensor
                else:
                    down_packed[bank_layer_id][expert] = tensor
            else:
                g = globals_map[(layer, expert, proj)]
                if role == "gate":
                    gate_up_scale[bank_layer_id][expert, :I] = tensor
                    gate_up_global[bank_layer_id][expert, :I] = g
                elif role == "up":
                    if spec.gated:
                        gate_up_scale[bank_layer_id][expert, I:] = tensor
                        gate_up_global[bank_layer_id][expert, I:] = g
                    else:
                        gate_up_scale[bank_layer_id][expert] = tensor
                        gate_up_global[bank_layer_id][expert] = g
                else:
                    down_scale[bank_layer_id][expert] = tensor
                    down_global[bank_layer_id][expert] = g
            tracker.note(bank_layer_id)
            placed += 1
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = num_layers * E * (6 if spec.gated else 4)
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


def iter_nvfp4_expert_pieces(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    drop_page_cache: DropPageCache | None = None,
    primary: bool = True,
):
    """One piece per routed expert: ``gate`` / ``up`` / ``down`` codes plus their ``_scale``
    (fp8 block scales) and ``_global`` (the per-tensor scale, reciprocal for quant-side dialects,
    fp16) companions, straight from the safetensors shards.

    Serial reads walk the shards in order; ``parallel`` uses the chunked O_DIRECT reader. Either
    way tensors of one expert may span shards, so they are grouped by (layer, expert) as they land.
    """
    from freetoken.models.loader import drop_page_cache as _drop
    from freetoken.models.loader import safetensors_weight_map
    from freetoken.moe.expert_pieces import per_expert_pieces

    drop = drop_page_cache or _drop
    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)

    wanted: dict[str, tuple[int, int, str]] = {}
    for name in weight_map:
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = _canon_kind(spec, match.group("kind"))
        if kind not in ("weight", "weight_scale", "weight_scale_2"):
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")
        wanted[name] = (bank_layer, int(match.group("expert")), spec.proj_to_role[proj] + _kind_suffix(kind))
    expected = _num_moe_layers(config) * config.num_experts * 9
    if len(wanted) != expected:
        raise ValueError(f"{spec.desc}: found {len(wanted)} expert tensors, expected {expected}")

    def _serial():
        by_shard: dict[str, list[str]] = collections.defaultdict(list)
        for name, shard in weight_map.items():
            if name in wanted:
                by_shard[shard].append(name)
        for shard in tqdm(sorted(by_shard), desc=f"Loading {spec.desc}", disable=not primary):
            path = os.path.join(folder, shard)
            drop(path)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name in by_shard[shard]:
                    tensor = f.get_tensor(name)
                    if wanted[name][2].endswith("_global"):
                        tensor = _ingest_global(spec, tensor)
                    yield name, tensor
            drop(path)

    def _parallel():
        from freetoken.models.weight import iter_expert_tensors_parallel

        for name, tensor in iter_expert_tensors_parallel(folder, lambda n: n in wanted, workers=workers, chunk=chunk):
            if wanted[name][2].endswith("_global"):
                tensor = _ingest_global(spec, tensor)
            yield name, tensor

    return per_expert_pieces(_parallel() if parallel else _serial(), wanted.get, tensors_per_expert=9)


__all__ = [
    "Nvfp4ExpertSourceSpec",
    "iter_nvfp4_expert_pieces",
    "load_nvfp4_expert_source_banks",
    "load_nvfp4_expert_source_banks_parallel",
]
