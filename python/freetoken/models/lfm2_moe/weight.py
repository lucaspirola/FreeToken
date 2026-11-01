"""LFM2 MoE checkpoint reader (bf16 dense tensors, EXL3 exports from exllamav3).

Dense tensors go through the shared ``_DenseReader`` (each Linear read under its scheme in
the checkpoint's QuantConfig, fused parts combined). Routed experts are read by the offload
cache from the EXL3 banks (``exl3_expert_spec``), never from here.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import get_quant_config
from freetoken.models.loader import iter_weight_files
from freetoken.models.qwen3_5_moe.weight import _DenseReader
from freetoken.models.register import get_model_spec
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_RE = re.compile(r"\.feed_forward\.experts\.")
# checkpoint leaf -> attribute of the reused Qwen3Attention
_RENAMES = (
    (".self_attn.out_proj.", ".self_attn.o_proj."),
    (".self_attn.q_layernorm.", ".self_attn.q_norm."),
    (".self_attn.k_layernorm.", ".self_attn.k_norm."),
)
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3", "gate", "lm_head",
})


def rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale")):
        return None
    for old, new in _RENAMES:
        raw_name = raw_name.replace(old, new)
    return raw_name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("lfm2_moe weight loading supports TP=1 only")
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    if include_moe_experts:
        raise NotImplementedError(
            f"lfm2_moe serves its routed experts from the offload cache only (expert_quant "
            f"{config.expert_quant!r}); use an EXL3 export and the offload MoE strategy"
        )
    if not include_non_moe:
        return
    reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0]), _LINEAR_LEAVES)
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not get_tp_info().is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if _EXPERT_RE.search(raw_name):
                    continue
                name = rename(raw_name)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = reader.add(name, tensor)
                if emitted is None:
                    yield name, tensor
                else:
                    yield from emitted
    if reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


def exl3_expert_spec(config):
    """The per-expert EXL3 tensor names (``models.exl3_banks``); experts start after the dense layers."""
    from freetoken.models.exl3_banks import Exl3ExpertSpec

    return Exl3ExpertSpec(
        key_template="model.layers.{layer}.feed_forward.experts.{expert}.{proj}.{kind}",
        proj_names={"gate": "w1", "up": "w3", "down": "w2"},
        first_layer=config.first_k_dense_replace,
        desc="LFM2 MoE EXL3 experts",
    )


__all__ = ["exl3_expert_spec", "iter_weights", "rename"]
