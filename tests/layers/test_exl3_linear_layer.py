"""An EXL3 fused projection end to end through the layer machinery: the dialect resolves the
scheme from ``quantization_config.json``, ``Exl3LinearMethod.create_weights`` declares the
tensors ``Exl3Config.fuse_parts`` produces from the checkpoint parts, and the selected kernel's
forward matches the per-part torch reference."""
from __future__ import annotations

import json

import pytest
import torch

import freetoken.distributed.info as di
from freetoken.kernel.triton.exl3 import linear_reference
from freetoken.layers.linear import LinearQKVMerged
from freetoken.layers.quantization.configs.exl3 import Exl3Config
from freetoken.layers.quantization.names import NameMap

PREFIX = "model.layers.0.self_attn"
HIDDEN, HEAD_DIM, QO, KV = 256, 128, 2, 1
MUL1 = 0x83DCD12D - (1 << 32)


@pytest.fixture(autouse=True)
def _tp1():
    try:
        di.get_tp_info()
    except RuntimeError:
        di.set_tp_info(0, 1)


def _parts(bits_of, seed=0):
    g = torch.Generator().manual_seed(seed)
    parts = {}
    for proj, n in (("q_proj", QO * HEAD_DIM), ("k_proj", KV * HEAD_DIM), ("v_proj", KV * HEAD_DIM)):
        bits = bits_of[proj]
        parts[proj] = {
            "trellis": torch.randint(-(1 << 15), 1 << 15, (HIDDEN // 16, n // 16, 16 * bits), generator=g, dtype=torch.int32).to(torch.int16),
            "suh": (torch.randn(HIDDEN, generator=g) * 0.5).half(),
            "svh": (torch.randn(n, generator=g) * 0.02).half(),
            "mul1": torch.tensor(MUL1, dtype=torch.int32),
        }
    return parts


def _quant(tmp_path, parts):
    storage = {
        f"{PREFIX}.{proj}": {
            "quant_format": "exl3", "bits_per_weight": p["trellis"].shape[-1] // 16,
            "stored_tensors": {f"{PREFIX}.{proj}.{k}": {"shape": list(t.shape)} for k, t in p.items()},
        }
        for proj, p in parts.items()
    }
    (tmp_path / "quantization_config.json").write_text(json.dumps({"tensor_storage": storage}))
    return Exl3Config({"quant_method": "exl3"}, {"_name_or_path": str(tmp_path)},
                      name_map=NameMap(packed=(("qkv_proj", ("q_proj", "k_proj", "v_proj")),)))


def _layer(quant):
    return LinearQKVMerged(HIDDEN, HEAD_DIM, QO, KV, False, quant_config=quant, prefix=f"{PREFIX}.qkv_proj")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton EXL3 kernels need CUDA")
def test_fused_qkv_layer_matches_the_reference(tmp_path):
    parts = _parts({"q_proj": 5, "k_proj": 5, "v_proj": 5})
    quant = _quant(tmp_path, parts)
    layer = _layer(quant)
    assert layer.quant_method.kernel.name == "triton"
    fused = quant.fuse_parts(f"{PREFIX}.qkv_proj", layer.quant_method.scheme, list(parts.values()))
    for name, value in fused.items():
        declared = getattr(layer, name)
        assert value.shape == declared.shape and value.dtype == declared.dtype, name
        setattr(layer, name, value.cuda())
    layer.finalize()
    x = torch.randn(7, HIDDEN, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    y = layer.forward(x.cuda()).cpu()
    ref = torch.cat([linear_reference(x.float(), p["trellis"], p["suh"], p["svh"], "mul1") for p in parts.values()], dim=1)
    assert y.shape == (7, (QO + 2 * KV) * HEAD_DIM)
    assert float((y.float() - ref).abs().max() / ref.abs().max()) < 8e-3


def test_fused_parts_with_different_bit_widths_refuse(tmp_path):
    quant = _quant(tmp_path, _parts({"q_proj": 5, "k_proj": 4, "v_proj": 5}))
    with pytest.raises(ValueError, match="mixes quantization schemes"):
        _layer(quant)


def test_wrong_multiplier_refuses(tmp_path):
    parts = _parts({"q_proj": 5, "k_proj": 5, "v_proj": 5})
    quant = _quant(tmp_path, parts)
    scheme = quant.scheme_for(f"{PREFIX}.qkv_proj")
    parts["k_proj"]["mul1"] = torch.tensor(1234, dtype=torch.int32)
    with pytest.raises(ValueError, match="multiplier"):
        quant.fuse_parts(f"{PREFIX}.qkv_proj", scheme, list(parts.values()))
