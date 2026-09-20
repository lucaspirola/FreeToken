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
