"""exllamav3 EXL3 checkpoints: per-tensor trellis bit widths, read from ``quantization_config.json``.

``config.json`` carries only the summary (``quant_method: exl3``, average bits, codebook); the
per-module facts -- which modules are quantized, each one's bit width and codebook -- live in the
``tensor_storage`` table of the ``quantization_config.json`` exllamav3 writes next to it. Each
quantized module stores ``trellis`` / ``suh`` / ``svh`` plus a codebook flag tensor (``mul1`` or
``mcg``; neither means 3INST), whose value must be the multiplier the kernels hardcode.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any, ClassVar

import torch

from ..registry import register_dialect
from ..scheme import QuantKind, QuantScheme, exl3_scheme
from .base import QuantConfig, Stored, cfg_get

_FLAGS = ("mul1", "mcg")
_MULTIPLIER = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}


@functools.lru_cache(maxsize=4)
def _tensor_storage(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        q = json.load(f)
    storage = q.get("tensor_storage")
    if not isinstance(storage, dict):
        raise ValueError(f"{path} has no tensor_storage table; not an exllamav3 EXL3 export")
    return storage


def _module_facts(name: str, entry: dict) -> tuple[int, str] | None:
    """``(bits, codebook)`` of one tensor_storage entry, or None if it is stored unquantized."""
    if entry.get("quant_format") != "exl3":
        return None
    stored = {key.rsplit(".", 1)[1]: meta for key, meta in entry.get("stored_tensors", {}).items()}
    if "trellis" not in stored or "suh" not in stored or "svh" not in stored:
        missing = sorted({"trellis", "suh", "svh"} - set(stored))
        raise NotImplementedError(
            f"{name}: EXL3 module stores {sorted(stored)}; missing {missing} "
            "(packed su/sv sign fields from older exports are not supported)"
        )
    flags = [f for f in _FLAGS if f in stored]
    if len(flags) > 1:
        raise ValueError(f"{name}: EXL3 module flags both codebooks {flags}")
    codebook = flags[0] if flags else "3inst"
    width = stored["trellis"]["shape"][-1]
    if width % 16:
        raise NotImplementedError(f"{name}: half-integer EXL3 bitrate (tile width {width}) is not supported")
    bits = width // 16
    declared = entry.get("bits_per_weight")
    if declared is not None and int(declared) != bits:
        raise ValueError(f"{name}: bits_per_weight {declared} disagrees with the trellis tile width {width}")
    return bits, codebook


@register_dialect
class Exl3Config(QuantConfig):
    """exllamav3 exports (``quant_method: exl3``)."""

    dialect = "exl3"
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {
        QuantKind.EXL3: {"trellis": "trellis", "suh": "suh", "svh": "svh", "mul1": "mul1", "mcg": "mcg"},
    }

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        root = cfg_get(hf_config, "_name_or_path") or cfg_get(hf_config, "name_or_path")
        path = os.path.join(str(root), "quantization_config.json") if root else None
        if not path or not os.path.isfile(path):
            raise NotImplementedError(
                "EXL3 checkpoint: the per-tensor quantization_config.json exllamav3 writes next to "
                f"config.json was not found (looked for {path!r})"
            )
        self.path = path
        self.storage_table = _tensor_storage(path)
        self._facts: dict[str, tuple[int, str] | None] = {}

    def facts(self, name: str) -> tuple[int, str] | None:
        """``(bits, codebook)`` of the module ``name`` (checkpoint naming), or None if unquantized / absent.

        A routed-expert container (``...mlp.experts``, which the table does not list: it stores
        ``experts.<e>.<proj>``) gets the one value all its projections share (``_container_facts``)."""
        if name not in self._facts:
            entry = self.storage_table.get(name)
            if entry is not None:
                self._facts[name] = _module_facts(name, entry)
            else:
                self._facts[name] = self._container_facts(name)
        return self._facts[name]

    def _container_facts(self, name: str) -> tuple[int, str] | None:
        prefix = name + "."
        children = {key: entry for key, entry in self.storage_table.items() if key.startswith(prefix)}
        if not children:
            return None
        seen = {_module_facts(key, entry) for key, entry in children.items()}
        if len(seen) != 1:
            raise NotImplementedError(f"{name}: EXL3 children mix bit widths / codebooks / quantized and not {sorted(seen, key=str)}")
        return seen.pop()

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        facts = self.facts(name)
        if facts is None:
            return None
        bits, codebook = facts
        scheme = exl3_scheme(bits, codebook)
        flag = {"mul1": "mul1", "mcg": "mcg"}.get(codebook)
        if flag is None:
            return scheme
        return QuantScheme(scheme.kind, scheme.weight, scheme.roles | {flag})

    def fuse_parts(self, target: str, scheme: QuantScheme, parts: list[dict[str, torch.Tensor]]):
        """One layer's EXL3 tensors from its checkpoint parts (``_DenseReader``'s hook).

        ``trellis`` becomes part-major and flat (each part's ``[K/16, N_j/16, 16*bits]`` tiles
        back to back), ``suh`` is stacked ``[parts, K]`` (each part rotates its input with its own
        signs), ``svh`` concatenated ``[N]``. The codebook flag is checked and dropped: the kernel
        hardcodes the multiplier."""
        if scheme.kind is not QuantKind.EXL3:
            return None
        widths = {tuple(p["trellis"].shape[::2]) for p in parts}
        if len(widths) != 1:
            raise NotImplementedError(f"{target}: fused EXL3 parts differ in K or bit width: {sorted(widths)}")
        for p in parts:
            for flag in _FLAGS:
                if flag in p and int(p[flag].reshape(-1)[0].item()) & 0xFFFFFFFF != _MULTIPLIER[flag]:
                    raise ValueError(f"{target}: {flag} multiplier {int(p[flag].item()):#x} is not {_MULTIPLIER[flag]:#x}")
        return {
            "trellis": torch.cat([p["trellis"].contiguous().reshape(-1) for p in parts]),
            "suh": torch.stack([p["suh"].reshape(-1) for p in parts]),
            "svh": torch.cat([p["svh"].reshape(-1) for p in parts]),
        }

    def expert_facts(self, key_template: str, num_layers: int, num_experts: int, *, first_layer: int = 0) -> tuple[int, str]:
        """The single ``(bits, codebook)`` every routed expert projection shares; refuses a mix
        (the expert banks hold one row size for every layer)."""
        seen: set[tuple[int, str]] = set()
        for layer in range(first_layer, first_layer + num_layers):
            for expert in range(num_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    name = key_template.format(layer=layer, expert=expert, proj=proj)
                    facts = self.facts(name)
                    if facts is None:
                        raise ValueError(f"EXL3 routed expert {name} is not EXL3-quantized in {self.path}")
                    seen.add(facts)
        if len(seen) != 1:
            raise NotImplementedError(f"EXL3 routed experts mix bit widths / codebooks {sorted(seen)}; one bank row size is required")
        return seen.pop()
