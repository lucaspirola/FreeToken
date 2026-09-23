"""S8's model conformance check, parametrised over every NVFP4 expert source
spec this tree exports.

The spec list is ENUMERATED FROM CODE (walk ``freetoken.models``'s
subpackages, import each family's ``weight`` module, collect every attribute
that is an ``Nvfp4ExpertSourceSpec`` instance, dedup by identity) rather than
hardcoded, so a family added later is covered the moment its module imports
cleanly -- the plan's list of families (nemotron_h, qwen3_5_moe, minimax_m2,
minimax_m3, glm4_moe, gemma4) undercounts what is actually in the tree after
the upstream merge: qwen4_exp has one too, and glm5_next exports two (a
ModelOpt spec and a compressed-tensors one), for 9 specs across 8 modules.

Each case:
  1. builds a small synthetic NVFP4 checkpoint whose keys come from the
     spec's own ``key_template`` (never a hardcoded key shape), for a tiny
     synthetic config that satisfies that spec's ``layer_to_bank``;
  2. asserts every generated key matches the spec's own ``key_pattern``
     (the regex and the template must describe the same keys);
  3. runs ``check_experts.check_expert_tensors`` -- the S8 conformance
     core -- against it and asserts it accepts a conformant checkpoint and
     refuses one with a renamed tensor kind, naming the kind.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import pkgutil
import struct
from types import SimpleNamespace

import pytest

import freetoken.models as _models_pkg
from freetoken.models.check_experts import CheckFailed, check_expert_tensors
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec, nvfp4_expert_row_layout

LAYERS, EXPERTS, H, I = 2, 3, 32, 32

_CHECKPOINT_DTYPE_STR = {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32"}


def _iter_nvfp4_specs() -> list[tuple[str, Nvfp4ExpertSourceSpec]]:
    seen_ids: set[int] = set()
    found: list[tuple[str, Nvfp4ExpertSourceSpec]] = []
    for info in pkgutil.iter_modules(_models_pkg.__path__):
        if not info.ispkg:
            continue
        try:
            mod = importlib.import_module(f"freetoken.models.{info.name}.weight")
        except ModuleNotFoundError:
            continue
        for attr_name in sorted(vars(mod)):
            value = vars(mod)[attr_name]
            if isinstance(value, Nvfp4ExpertSourceSpec) and id(value) not in seen_ids:
                seen_ids.add(id(value))
                found.append((f"{info.name}.{attr_name}", value))
    return found


_SPECS = _iter_nvfp4_specs()
# Every family known to export a spec at plan-writing time, plus what the upstream
# merge actually left behind (qwen4_exp, glm5_next's second spec) -- see module
# docstring. This is a floor, not a ceiling: a new spec is picked up automatically.
_EXPECTED_MODULES = {
    "nemotron_h", "qwen3_5_moe", "qwen4_exp",
    "minimax_m2", "minimax_m3", "glm4_moe", "gemma4", "glm5_next",
}


def test_enumeration_covers_every_known_family():
    found_modules = {name.split(".")[0] for name, _ in _SPECS}
    missing = _EXPECTED_MODULES - found_modules
    assert not missing, f"no NVFP4 expert source spec found for {missing}"
    assert len(_SPECS) >= 9, _SPECS  # nemotron_h, qwen3_5_moe, qwen4_exp, minimax_m2,
    # minimax_m3, glm4_moe, gemma4, glm5_next x2


def _synthetic_config(spec: Nvfp4ExpertSourceSpec) -> SimpleNamespace:
    """The minimum a spec's ``layer_to_bank`` / ``hidden_size_attr`` need, without
    a real family's full HF config.json (out of scope for this test -- see
    ``check_experts.check_expert_tensors``'s docstring)."""
    return SimpleNamespace(
        expert_gated=spec.gated,
        hidden_size=H,
        expert_hidden_size=H,
        moe_intermediate_size=I,
        num_experts=EXPERTS,
        num_layers=LAYERS,
        first_k_dense_replace=0,
        moe_layer_ids=tuple(range(LAYERS)),
        num_moe_layers=LAYERS,
    )


def _write_checkpoint(
    root: str,
    spec: Nvfp4ExpertSourceSpec,
    config,
    *,
    split_experts: frozenset[tuple[int, int]] = frozenset(),
) -> None:
    """A synthetic checkpoint keyed exactly by ``spec.key_template``, shaped by
    ``nvfp4_expert_row_layout`` (the S5a single source of truth -- not a copy).

    ``split_experts``: (layer, expert) pairs whose ONE "down"-role tensor set
    is written to a second shard while the rest of that expert's tensors stay
    in the first -- a synthetic version of the routine case where HF's sharder
    cuts a shard boundary in the middle of an expert (real in
    Ornith-1.5-35B-A3B-NVFP4's checkpoint, see docs/models.md). Every other
    expert's tensors stay in one shard.
    """
    layout = nvfp4_expert_row_layout(H, I, gated=spec.gated, kind_map=spec.kind_map)
    role_to_proj = {role: proj for proj, role in spec.proj_to_role.items()}
    canon_to_disk = {canon: disk for disk, canon in (spec.kind_map or {}).items()}
    roles = ("gate", "up", "down") if spec.gated else ("up", "down")

    header: dict[str, dict] = {}
    shard_of_key: dict[str, str] = {}
    blobs: dict[str, bytearray] = {"model.safetensors": bytearray(), "model-2.safetensors": bytearray()}
    offsets: dict[str, int] = {"model.safetensors": 0, "model-2.safetensors": 0}
    for layer in range(LAYERS):
        bank_layer = spec.layer_to_bank(layer, config)
        assert bank_layer == layer, (spec.desc, layer, bank_layer)
        for expert in range(EXPERTS):
            for role in roles:
                proj = role_to_proj[role]
                shard = (
                    "model-2.safetensors"
                    if (role == "down" and (layer, expert) in split_experts)
                    else "model.safetensors"
                )
                for canon_kind in ("weight", "weight_scale", "weight_scale_2"):
                    on_disk_kind = canon_to_disk.get(canon_kind, canon_kind)
                    key = spec.key_template.format(
                        layer=layer, expert=expert, proj=proj, kind=on_disk_kind
                    )
                    assert spec.key_pattern.match(key) is not None, (
                        f"{spec.desc}: key_template produced {key!r}, which "
                        f"key_pattern {spec.key_pattern.pattern!r} does not match"
                    )
                    tensor = layout.tensors[(role, canon_kind)]
                    if canon_kind == "weight_scale_2":
                        payload = struct.pack("<f", float(layer * EXPERTS + expert + 1))
                        shape: list[int] = []
                    else:
                        rows, width = tensor.checkpoint_shape
                        n = rows * width
                        salt = layer * 97 + expert * 13 + ord(role[0])
                        payload = bytes(((salt + k) % 251) + 1 for k in range(n))
                        shape = [rows, width]
                    off = offsets[shard]
                    header[key] = {
                        "dtype": _CHECKPOINT_DTYPE_STR[canon_kind],
                        "shape": shape,
                        "data_offsets": [off, off + len(payload)],
                    }
                    shard_of_key[key] = shard
                    blobs[shard] += payload
                    offsets[shard] += len(payload)

    for shard, blob in blobs.items():
        if not blob and shard != "model.safetensors":
            continue
        shard_header = {k: v for k, v in header.items() if shard_of_key[k] == shard}
        head = json.dumps(shard_header).encode()
        with open(os.path.join(root, shard), "wb") as f:
            f.write(struct.pack("<Q", len(head)))
            f.write(head)
            f.write(blob)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": shard_of_key}, f)


@pytest.mark.parametrize("name,spec", _SPECS, ids=[n for n, _ in _SPECS])
def test_conformant_checkpoint_passes(tmp_path, name, spec):
    config = _synthetic_config(spec)
    _write_checkpoint(str(tmp_path), spec, config)

    layout, per_expert, num_moe_layers, expected_experts, spanning = check_expert_tensors(
        str(tmp_path), spec, config, mechanism=name
    )
    assert num_moe_layers == LAYERS
    assert expected_experts == LAYERS * EXPERTS
    assert len(per_expert) == LAYERS * EXPERTS
    assert spanning == []


@pytest.mark.parametrize("name,spec", _SPECS, ids=[n for n, _ in _SPECS])
def test_cross_shard_expert_is_reported_not_refused(tmp_path, name, spec):
    """An expert split across two shards (routine wherever HF's sharder cuts --
    real in Ornith-1.5-35B-A3B-NVFP4) must NOT refuse the checkpoint: the
    loaders that actually read it (load_nvfp4_expert_source_banks* and
    iter_nvfp4_expert_pieces) group tensors by (layer, expert) as they stream
    every shard and tolerate this, and so does the mirror pool (one read group
    per shard), so this is reported, not raised."""
    config = _synthetic_config(spec)
    split = frozenset({(0, 1)})
    _write_checkpoint(str(tmp_path), spec, config, split_experts=split)

    layout, per_expert, num_moe_layers, expected_experts, spanning = check_expert_tensors(
        str(tmp_path), spec, config, mechanism=name
    )
    assert expected_experts == LAYERS * EXPERTS
    assert len(per_expert) == LAYERS * EXPERTS  # every expert's tensors were still all found
    assert spanning == [(0, 1)]


@pytest.mark.parametrize("name,spec", _SPECS, ids=[n for n, _ in _SPECS])
def test_renamed_kind_is_refused_naming_the_kind(tmp_path, name, spec):
    config = _synthetic_config(spec)
    # Write a perfectly conformant checkpoint under the ORIGINAL spec (real,
    # correctly-named on-disk keys)...
    _write_checkpoint(str(tmp_path), spec, config)

    # ...then check it against a spec whose kind_map renames one on-disk kind
    # (weight_scale_2's, whatever this spec calls it) onto a bogus canonical
    # kind the row layout does not have -- exactly what a checkpoint using a
    # tensor-kind convention the code does not know about would look like.
    canon_to_disk = {canon: disk for disk, canon in (spec.kind_map or {}).items()}
    g2_on_disk = canon_to_disk.get("weight_scale_2", "weight_scale_2")
    corrupt_kind_map = dict(spec.kind_map or {})
    corrupt_kind_map[g2_on_disk] = "renamed_kind"
    corrupt_spec = dataclasses.replace(spec, kind_map=corrupt_kind_map)

    with pytest.raises(CheckFailed, match="renamed_kind"):
        check_expert_tensors(str(tmp_path), corrupt_spec, config, mechanism=name)
