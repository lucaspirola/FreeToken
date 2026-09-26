"""EXL3 routed experts: where each expert's bytes live in the checkpoint, and the two readers.

One index of file extents (``Exl3ExpertIndex``) feeds both readers, so they cannot disagree:

* ``iter_exl3_expert_pieces`` -- the whole-model path (``build_expert_banks`` -> the kernel's
  ``pack``), one piece per batch of experts in the checkpoint's own form
  (``{gate,up,down}_{trellis,suh,svh}``).
* ``exl3_expert_row_extents`` -- the RAM saver's pool ``source`` (``MirrorExpertPool``,
  the protocol S12c gave GGUF): per expert, the file extents and where each lands in the bank row.

The bank row is the checkpoint's bytes, unchanged (``BankSpec.raw_row``)::

    gate_up_trellis  int16 [2, H/16, I/16, 16*bits]   gate trellis, then up trellis
    gate_up_suh      fp16  [2, H]                     gate suh, up suh
    gate_up_svh      fp16  [2, I]                     gate svh, up svh
    down_trellis     int16 [I/16, H/16, 16*bits]
    down_suh         fp16  [I]
    down_svh         fp16  [H]

A model opts in by exporting ``exl3_expert_spec(config) -> Exl3ExpertSpec`` from its package.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch

BANK_NAMES = ("gate_up_trellis", "gate_up_suh", "gate_up_svh", "down_trellis", "down_suh", "down_svh")
_ROLES = ("gate", "up", "down")
_KINDS = ("trellis", "suh", "svh")
_ST_DTYPES = {"I16": (torch.int16, 2), "F16": (torch.float16, 2), "I32": (torch.int32, 4)}
_WANT_DTYPE = {"trellis": "I16", "suh": "F16", "svh": "F16"}
_FLAG_VALUE = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}


@dataclass(frozen=True)
class Exl3ExpertSpec:
    """A model's routed-expert naming. ``key_template`` takes ``layer``, ``expert``, ``proj`` and
    ``kind`` (``trellis``/``suh``/``svh``/``mul1``/``mcg``); ``layer`` is the checkpoint layer of
    bank layer ``bank_layer`` = ``layer - first_layer``."""

    key_template: str
    proj_names: dict[str, str] = field(default_factory=lambda: {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"})
    first_layer: int = 0
    desc: str = "EXL3 experts"

    def key(self, layer: int, expert: int, role: str, kind: str) -> str:
        return self.key_template.format(layer=layer + self.first_layer, expert=expert, proj=self.proj_names[role], kind=kind)

    def module_template(self) -> str:
        """The per-projection module name template (``Exl3Config.expert_facts`` input)."""
        return self.key_template.replace(".{kind}", "")


def exl3_bank_shapes(hidden: int, inter: int, bits: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    h, i = hidden, inter
    if h % 128 or i % 128:
        raise ValueError(f"EXL3 experts need 128-aligned hidden/intermediate, got {h}/{i}")
    return {
        "gate_up_trellis": ((2, h // 16, i // 16, 16 * bits), torch.int16),
        "gate_up_suh": ((2, h), torch.float16),
        "gate_up_svh": ((2, i), torch.float16),
        "down_trellis": ((i // 16, h // 16, 16 * bits), torch.int16),
        "down_suh": ((i,), torch.float16),
        "down_svh": ((h,), torch.float16),
    }


def _placement(hidden: int, inter: int, bits: int) -> dict[tuple[str, str], tuple[str, int, tuple[int, ...]]]:
    """(role, kind) -> (bank, byte offset in that bank's row, the checkpoint tensor's shape)."""
    h, i = hidden, inter
    gu_trellis = (h // 16, i // 16, 16 * bits)
    gu_bytes = h // 16 * (i // 16) * 16 * bits * 2
    return {
        ("gate", "trellis"): ("gate_up_trellis", 0, gu_trellis),
        ("up", "trellis"): ("gate_up_trellis", gu_bytes, gu_trellis),
        ("gate", "suh"): ("gate_up_suh", 0, (h,)),
        ("up", "suh"): ("gate_up_suh", 2 * h, (h,)),
        ("gate", "svh"): ("gate_up_svh", 0, (i,)),
        ("up", "svh"): ("gate_up_svh", 2 * i, (i,)),
        ("down", "trellis"): ("down_trellis", 0, (i // 16, h // 16, 16 * bits)),
        ("down", "suh"): ("down_suh", 0, (i,)),
        ("down", "svh"): ("down_svh", 0, (h,)),
    }


def _read_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def _weight_map(model_path: str) -> dict[str, str]:
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    single = os.path.join(model_path, "model.safetensors")
    _, header = _read_header(single)
    return {k: "model.safetensors" for k in header if k != "__metadata__"}


class Exl3ExpertIndex:
    """Every served expert's tensors as ``(shard, file offset, nbytes)``, validated against the
    bank geometry; the codebook flag tensors are located (``flags``) but not stored."""

    def __init__(self, model_path: str, spec: Exl3ExpertSpec, *, num_layers: int, num_experts: int,
                 hidden: int, inter: int, bits: int, codebook: str):
        self.model_path = model_path
        self.spec = spec
        self.num_layers, self.num_experts = num_layers, num_experts
        self.hidden, self.inter, self.bits, self.codebook = hidden, inter, bits, codebook
        self.shapes = exl3_bank_shapes(hidden, inter, bits)
        self.placement = _placement(hidden, inter, bits)
        flag = {"mul1": "mul1", "mcg": "mcg"}.get(codebook)
        weight_map = _weight_map(model_path)
        headers: dict[str, tuple[int, dict]] = {}
        # flat id -> [(shard, file offset, nbytes, bank, dst byte offset)], and the flag extents
        self.extents: dict[int, list[tuple[str, int, int, str, int]]] = {}
        self.flags: dict[int, list[tuple[str, int]]] = {}
        for layer in range(num_layers):
            for expert in range(num_experts):
                pieces, flags = [], []
                for role in _ROLES:
                    for kind in _KINDS + ((flag,) if flag else ()):
                        key = spec.key(layer, expert, role, kind)
                        shard = weight_map.get(key)
                        if shard is None:
                            raise ValueError(f"{spec.desc}: {key} is missing from the checkpoint")
                        if shard not in headers:
                            headers[shard] = _read_header(os.path.join(model_path, shard))
                        data_start, header = headers[shard]
                        entry = header[key]
                        off, end = entry["data_offsets"]
                        if kind == flag:
                            if entry["dtype"] != "I32" or end - off != 4:
                                raise ValueError(f"{spec.desc}: {key} is not an int32 scalar")
                            flags.append((shard, data_start + off))
                            continue
                        bank, dst, shape = self.placement[(role, kind)]
                        if entry["dtype"] != _WANT_DTYPE[kind] or tuple(entry["shape"]) != shape:
                            raise ValueError(
                                f"{spec.desc}: {key} is {entry['dtype']} {entry['shape']}, expected "
                                f"{_WANT_DTYPE[kind]} {list(shape)} for {bits}-bit experts of {hidden}x{inter}"
                            )
                        pieces.append((shard, data_start + off, end - off, bank, dst))
                self.extents[layer * num_experts + expert] = sorted(pieces, key=lambda p: (p[0], p[1]))
                self.flags[layer * num_experts + expert] = flags
        # every row byte is written by exactly one piece
        row_bytes = {name: int(np.prod(tail)) * torch.empty((), dtype=dt).element_size() for name, (tail, dt) in self.shapes.items()}
        covered = {name: 0 for name in self.shapes}
        for _shard, _off, length, bank, _dst in self.extents[0]:
            covered[bank] += length
        if covered != row_bytes:
            raise AssertionError(f"EXL3 expert pieces do not tile the bank row: {covered} vs {row_bytes}")
        self.row_bytes = row_bytes

    def check_flags(self, flats, read) -> None:
        """Assert the codebook flag of every expert in ``flats`` holds the kernel's multiplier; ``read(shard, off, n)`` returns bytes."""
        if self.codebook not in _FLAG_VALUE:
            return
        want = _FLAG_VALUE[self.codebook]
        for flat in flats:
            for shard, off in self.flags[flat]:
                value = struct.unpack("<I", read(shard, off, 4))[0]
                if value != want:
                    layer, expert = divmod(flat, self.num_experts)
                    raise ValueError(f"{self.spec.desc}: layer {layer} expert {expert} {self.codebook} multiplier {value:#x} != {want:#x}")


# ---------------------------------------------------------------------------
# Model hooks and geometry
# ---------------------------------------------------------------------------


def expert_spec_for(config) -> Exl3ExpertSpec:
    from freetoken.models.register import _load_attr, get_model_spec

    spec = get_model_spec(config.architectures[0])
    try:
        hook = _load_attr(spec.module, "exl3_expert_spec")
    except AttributeError:
        raise NotImplementedError(f"{spec.module} provides no exl3_expert_spec: its EXL3 expert layout was never verified") from None
    return hook(config)


def expert_facts_for(config) -> tuple[int, str]:
    """``(bits, codebook)`` of every routed expert (one value; the dialect refuses a mix)."""
    from freetoken.layers.quantization import get_quant_config

    quant = get_quant_config()
    if quant is None or getattr(quant, "dialect", None) != "exl3":
        raise ValueError("EXL3 experts need the checkpoint's Exl3Config installed")
    spec = expert_spec_for(config)
    return quant.expert_facts(spec.module_template(), config.num_moe_layers, config.num_experts, first_layer=spec.first_layer)


def expert_index_for(model_path: str, config) -> Exl3ExpertIndex:
    bits, codebook = expert_facts_for(config)
    return Exl3ExpertIndex(
        model_path, expert_spec_for(config), num_layers=config.num_moe_layers, num_experts=config.num_experts,
        hidden=config.hidden_size, inter=config.moe_intermediate_size, bits=bits, codebook=codebook,
    )


def _pread_file(model_path: str):
    fds: dict[str, int] = {}

    def read(shard: str, off: int, n: int) -> bytes:
        if shard not in fds:
            fds[shard] = os.open(os.path.join(model_path, shard), os.O_RDONLY)
        return os.pread(fds[shard], n, off)

    def close():
        for fd in fds.values():
            os.close(fd)

    return read, close


# ---------------------------------------------------------------------------
# Whole-model reader
# ---------------------------------------------------------------------------


def iter_exl3_expert_pieces(model_path: str, config, *, batch: int = 16) -> Iterator[tuple[int, int, int, dict[str, torch.Tensor]]]:
    """Pieces ``(bank_layer, e0, e1, {gate_trellis, gate_suh, ..., down_svh})`` read from the index's extents."""
    index = expert_index_for(model_path, config)
    return _iter_pieces(index, batch=batch)


def _iter_pieces(index: Exl3ExpertIndex, *, batch: int = 16):
    read, close = _pread_file(index.model_path)
    try:
        for layer in range(index.num_layers):
            for e0 in range(0, index.num_experts, batch):
                e1 = min(e0 + batch, index.num_experts)
                piece = empty_piece(index, e1 - e0)
                flats = [layer * index.num_experts + e for e in range(e0, e1)]
                index.check_flags(flats, read)
                for j, flat in enumerate(flats):
                    read_expert(index, flat, read, piece, j)
                yield layer, e0, e1, piece
    finally:
        close()


def empty_piece(index: Exl3ExpertIndex, n: int) -> dict[str, torch.Tensor]:
    dtype_of = {"trellis": torch.int16, "suh": torch.float16, "svh": torch.float16}
    return {
        f"{role}_{kind}": torch.empty((n, *index.placement[(role, kind)][2]), dtype=dtype_of[kind])
        for role in _ROLES for kind in _KINDS
    }


def read_expert(index: Exl3ExpertIndex, flat: int, read, piece: dict[str, torch.Tensor], j: int) -> None:
    """Fill row ``j`` of ``piece`` with expert ``flat``'s tensors: one read per contiguous run of
    its extents (the whole expert, for exllamav3's writer)."""
    runs: list[list[tuple]] = []
    for p in index.extents[flat]:
        if runs and runs[-1][-1][0] == p[0] and runs[-1][-1][1] + runs[-1][-1][2] == p[1]:
            runs[-1].append(p)
        else:
            runs.append([p])
    for run in runs:
        start = run[0][1]
        raw = read(run[0][0], start, run[-1][1] + run[-1][2] - start)
        for _shard, off, length, bank, dst in run:
            role, kind = _role_kind_of(index.placement, bank, dst)
            piece[f"{role}_{kind}"][j].view(torch.uint8).reshape(-1).copy_(
                torch.frombuffer(bytearray(raw[off - start: off - start + length]), dtype=torch.uint8)
            )


def _role_kind_of(placement, bank: str, dst: int) -> tuple[str, str]:
    for (role, kind), (b, d, _shape) in placement.items():
        if b == bank and d == dst:
            return role, kind
    raise KeyError((bank, dst))


# ---------------------------------------------------------------------------
# RAM saver (mirror pool) source
# ---------------------------------------------------------------------------


class Exl3MirrorSource:
    """The mirror pool's ``source`` protocol (see ``MirrorExpertPool.__init__``)."""

    quant_format = "exl3"

    def __init__(self, *, shapes, records, shard_fds, fd_size):
        self.shapes = shapes
        self.records = records
        self.shard_fds = shard_fds
        self.fd_size = fd_size


def exl3_mirror_bank_shapes(config) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Pool bank shapes from config + the installed Exl3Config (no file I/O beyond its table)."""
    bits, _codebook = expert_facts_for(config)
    return exl3_bank_shapes(config.hidden_size, config.moe_intermediate_size, bits)


def exl3_expert_row_extents(model_path: str, config) -> Exl3MirrorSource:
    """Pool source over the checkpoint's routed experts: one read group per expert per shard,
    pieces landing at the same byte offsets ``pack`` writes (both come from ``Exl3ExpertIndex``)."""
    index = expert_index_for(model_path, config)
    # the codebook flag of each layer's first and last expert: a spot check that costs no bulk read
    read, close = _pread_file(model_path)
    try:
        index.check_flags([l * index.num_experts + e for l in range(index.num_layers) for e in (0, index.num_experts - 1)], read)
    finally:
        close()
    shard_fds: dict[str, int] = {}
    fd_size: dict[int, int] = {}
    records = {}
    try:
        for flat, extents in index.extents.items():
            groups: dict[str, list] = {}
            for shard, off, length, bank, dst in extents:
                if shard not in shard_fds:
                    fd = os.open(os.path.join(model_path, shard), os.O_RDONLY | os.O_DIRECT)
                    shard_fds[shard] = fd
                    fd_size[fd] = os.fstat(fd).st_size
                if off + length > fd_size[shard_fds[shard]]:
                    raise ValueError(f"{index.spec.desc}: expert {flat} extent runs past the end of {shard}")
                groups.setdefault(shard, []).append((off, length, bank, dst, 0))
            records[flat] = tuple((shard_fds[shard], pieces) for shard, pieces in sorted(groups.items()))
    except BaseException:
        for fd in shard_fds.values():
            os.close(fd)
        raise
    return Exl3MirrorSource(shapes=index.shapes, records=records, shard_fds=shard_fds, fd_size=fd_size)


def exl3_mirror_hooks(config):
    """``(row_extents_fn, bank_shapes_fn)`` when the model declares its EXL3 expert layout, else None."""
    try:
        expert_spec_for(config)
    except NotImplementedError:
        return None
    return exl3_expert_row_extents, exl3_mirror_bank_shapes


__all__ = [
    "BANK_NAMES", "Exl3ExpertIndex", "Exl3ExpertSpec", "Exl3MirrorSource",
    "exl3_bank_shapes", "exl3_expert_row_extents", "exl3_mirror_bank_shapes", "exl3_mirror_hooks",
    "expert_facts_for", "expert_index_for", "expert_spec_for", "iter_exl3_expert_pieces",
]
