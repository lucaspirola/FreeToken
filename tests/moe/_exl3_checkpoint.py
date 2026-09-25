"""A minimal exllamav3 EXL3 checkpoint laid out the way the Ornith (Qwen3.5-MoE) export is.

Per routed-expert projection: ``trellis`` int16 ``[K/16, N/16, 16*bits]``, ``suh`` fp16 ``[K]``,
``svh`` fp16 ``[N]`` and the ``mul1`` int32 scalar; a ``quantization_config.json`` with the
``tensor_storage`` table next to ``config.json``, split over two shards. safetensors orders the
data by dtype, then name, so unlike exllamav3's writer (one contiguous run per expert) an expert's
tensors are scattered here: the readers' general case. Values are seeded random, so byte-exactness
assertions compare against real content.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

ARCH = "Qwen3_5MoeForConditionalGeneration"
MUL1 = 0x83DCD12D
KEY = "model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"


def _as_int32(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def expert_tensors(layers: int, experts: int, hidden: int, inter: int, bits: int, *, seed: int = 0,
                   flag: int = MUL1, bits_of=None) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for layer in range(layers):
        for e in range(experts):
            for proj, (k, n) in (("gate_proj", (hidden, inter)), ("up_proj", (hidden, inter)), ("down_proj", (inter, hidden))):
                b = bits if bits_of is None else bits_of(layer, e, proj)
                base = KEY.format(layer=layer, expert=e, proj=proj)
                out[f"{base}.trellis"] = torch.randint(-(1 << 15), 1 << 15, (k // 16, n // 16, 16 * b), generator=g, dtype=torch.int32).to(torch.int16)
                out[f"{base}.suh"] = (torch.randn(k, generator=g) * 0.5).to(torch.float16)
                out[f"{base}.svh"] = (torch.randn(n, generator=g) * 0.01).to(torch.float16)
                out[f"{base}.mul1"] = torch.tensor(_as_int32(flag), dtype=torch.int32)
    return out


def write_exl3_checkpoint(root: str, layers: int = 2, experts: int = 3, hidden: int = 256, inter: int = 128,
                          bits: int = 5, **kw) -> dict[str, torch.Tensor]:
    os.makedirs(root, exist_ok=True)
    tensors = expert_tensors(layers, experts, hidden, inter, bits, **kw)
    tensors["model.language_model.norm.weight"] = torch.ones(hidden, dtype=torch.float16)
    keys = list(tensors)
    half = len(keys) // 2
    shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
    weight_map = {}
    for shard, names in shards.items():
        save_file({n: tensors[n].contiguous() for n in names}, os.path.join(root, shard))
        weight_map.update({n: shard for n in names})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    storage = {}
    for name, t in tensors.items():
        module, kind = name.rsplit(".", 1)
        if ".experts." not in module:
            continue
        entry = storage.setdefault(module, {"quant_format": "exl3", "stored_tensors": {}})
        entry["stored_tensors"][name] = {"shape": list(t.shape), "torch_dtype": str(t.dtype).split(".")[1]}
        if kind == "trellis":
            entry["bits_per_weight"] = t.shape[-1] // 16
    with open(os.path.join(root, "quantization_config.json"), "w") as f:
        json.dump({"tensor_storage": storage}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"architectures": [ARCH], "quantization_config": {"quant_method": "exl3", "bits": float(bits)}}, f)
    return tensors


def model_config(layers: int = 2, experts: int = 3, hidden: int = 256, inter: int = 128) -> SimpleNamespace:
    """The ModelConfig fields models/exl3_banks.py reads."""
    return SimpleNamespace(
        architectures=[ARCH], model_type="qwen3_5_moe", num_layers=layers, num_moe_layers=layers,
        num_experts=experts, hidden_size=hidden, moe_intermediate_size=inter,
    )


def install_quant(root: str):
    from freetoken.layers.quantization import set_quant_config
    from freetoken.layers.quantization.configs.exl3 import Exl3Config

    quant = Exl3Config({"quant_method": "exl3"}, {"_name_or_path": root})
    set_quant_config(quant)
    return quant
