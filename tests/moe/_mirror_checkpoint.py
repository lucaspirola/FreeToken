"""A minimal NVFP4 expert checkpoint, laid out the way Nemotron-H's are.

The mirror tests need a real checkpoint on disk: the pool locates rows through
the model's shipped ``Nvfp4ExpertSourceSpec``, so a test that invented its own
key format would only prove the test agrees with itself. Three test modules
wanted the same writer, so it lives here once rather than in three drifting
copies.

Row bytes are a deterministic function of the flat expert id, which is what
makes byte-exactness assertions possible: every served row can be checked
against what the checkpoint says it should be.
"""
from __future__ import annotations

import json
import os
import struct

from freetoken.moe.mirror_pool import nvfp4_bank_shapes

# Bank -> (key suffix, safetensors dtype) for an UNGATED Nemotron-H expert.
_SUFFIX_OF = {
    "gate_up_packed": ("up_proj.weight", "U8"),
    "gate_up_scale": ("up_proj.weight_scale", "F8_E4M3"),
    "gate_up_global": ("up_proj.weight_scale_2", "F32"),
    "down_packed": ("down_proj.weight", "U8"),
    "down_scale": ("down_proj.weight_scale", "F8_E4M3"),
    "down_global": ("down_proj.weight_scale_2", "F32"),
}


def row_byte(flat: int, k: int) -> int:
    """The byte this writer puts at offset ``k`` of expert ``flat``'s rows."""
    return ((flat * 13 + k) % 251) + 1


def write_nvfp4_checkpoint(root: str, layers: int, experts: int,
                           hidden: int, intermediate: int) -> None:
    shapes = nvfp4_bank_shapes(hidden, intermediate)
    header: dict[str, dict] = {}
    blob = bytearray()
    off = 0
    for layer in range(layers):
        for e in range(experts):
            flat = layer * experts + e
            for name, (tail, _dt) in shapes.items():
                suffix, dt = _SUFFIX_OF[name]
                key = f"backbone.layers.{layer}.mixer.experts.{e}.{suffix}"
                if dt == "F32":
                    payload, shape = struct.pack("<f", float(flat + 1)), []
                else:
                    n = 1
                    for d in tail:
                        n *= d
                    payload = bytes(row_byte(flat, k) for k in range(n))
                    shape = list(tail)
                header[key] = {"dtype": dt, "shape": shape,
                               "data_offsets": [off, off + len(payload)]}
                blob += payload
                off += len(payload)
    head = json.dumps(header).encode()
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        f.write(blob)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in header}}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"layers_block_type": ["moe"] * layers}, f)


# ---------------------------------------------------------------------------
# General writer: gated or not, modelopt or compressed-tensors tensor-kind
# names. Used by tests/moe/test_nvfp4_row_layout.py to exercise all three of
# S5a's checkpoint variants against one writer instead of three copies.
# ---------------------------------------------------------------------------

_PROJ_OF_ROLE = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}

# naming convention -> canonical kind -> on-disk suffix. "modelopt" is the
# convention every current model spec's key_pattern matches; "compressed_tensors"
# is llm-compressor's own NVFP4 naming (real elsewhere in this tree only for
# DENSE layers -- see models/loader.py -- never yet for MoE experts, which is
# exactly the gap this checkpoint variant demonstrates).
_KIND_SUFFIX_OF_NAMING = {
    "modelopt": {
        "weight": "weight", "weight_scale": "weight_scale", "weight_scale_2": "weight_scale_2",
    },
    "compressed_tensors": {
        "weight": "weight_packed", "weight_scale": "weight_scale",
        "weight_scale_2": "weight_global_scale",
    },
}
_CHECKPOINT_DTYPE_STR = {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32"}


def row_tensor_byte(flat: int, role: str, kind: str, k: int) -> int:
    """The deterministic on-disk byte at offset ``k`` of expert ``flat``'s
    ``(role, kind)`` tensor, before any host-side conversion. Salted by role and
    kind so gate/up/down and weight/weight_scale never collide by coincidence."""
    salt = {"gate": 1, "up": 2, "down": 3}[role] * 1000 + {"weight": 11, "weight_scale": 23}[kind]
    return ((flat * 13 + salt * 7 + k) % 251) + 1


def row_tensor_global(flat: int, role: str) -> float:
    """The deterministic on-disk FP32 scalar global-scale for expert ``flat``'s role."""
    return float(flat * 3 + {"gate": 1, "up": 2, "down": 3}[role])


def write_generic_nvfp4_checkpoint(
    root: str, layers: int, experts: int, hidden: int, intermediate: int,
    *, gated: bool, naming: str = "modelopt",
) -> None:
    """A synthetic NVFP4 expert checkpoint with a chosen gating and tensor-kind
    naming convention, keyed ``backbone.layers.N.mixer.experts.E.{proj}.{kind}``
    (Nemotron-H's key shape; the projection and kind names vary with ``gated``
    and ``naming``, everything else about the key format does not, since this
    writer exists to isolate those two variables).

    Tensor shapes come from ``models.nvfp4_banks.nvfp4_expert_row_layout``, not
    a copy: this writer is only a valid test of that function's byte layout if
    it does not re-derive the layout itself.
    """
    from freetoken.models.nvfp4_banks import nvfp4_expert_row_layout

    roles = ("gate", "up", "down") if gated else ("up", "down")
    kind_suffix = _KIND_SUFFIX_OF_NAMING[naming]
    layout = nvfp4_expert_row_layout(hidden, intermediate, gated=gated)

    header: dict[str, dict] = {}
    blob = bytearray()
    off = 0
    for layer in range(layers):
        for e in range(experts):
            flat = layer * experts + e
            for role in roles:
                proj = _PROJ_OF_ROLE[role]
                for kind in ("weight", "weight_scale", "weight_scale_2"):
                    suffix = kind_suffix[kind]
                    key = f"backbone.layers.{layer}.mixer.experts.{e}.{proj}.{suffix}"
                    dt = _CHECKPOINT_DTYPE_STR[kind]
                    if kind == "weight_scale_2":
                        payload = struct.pack("<f", row_tensor_global(flat, role))
                        shape: list[int] = []
                    else:
                        rows, width = layout.tensors[(role, kind)].checkpoint_shape
                        payload = bytes(
                            row_tensor_byte(flat, role, kind, k) for k in range(rows * width)
                        )
                        shape = [rows, width]
                    header[key] = {"dtype": dt, "shape": shape,
                                   "data_offsets": [off, off + len(payload)]}
                    blob += payload
                    off += len(payload)
    head = json.dumps(header).encode()
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        f.write(blob)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in header}}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"layers_block_type": ["moe"] * layers}, f)
