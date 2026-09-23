"""``resolve_cache_schema``: can the mirror pool's raw NVFP4 rows serve a given
cache bank layout, under whatever names that layout uses?

CPU-only, no ``MirrorExpertPool`` (which needs CUDA to pin its banks): these
exercise the pure matching/refusal logic against a minimal stand-in for the
pool's own geometry, and against the REAL kernel-method ``layout()``s
(``TritonNvfp4MoEKernel``, ``MarlinNvfp4MoEKernel``) so the "same (shape,
dtype) as the raw NVFP4 row, position for position" contract is checked
against the kernels that actually exist, not a hand-rolled shape dict.
"""
from __future__ import annotations

import types

import pytest
import torch

from freetoken.layers.quantization.moe.base import BankSpec, MoEConfig
from freetoken.layers.quantization.moe.nvfp4 import (
    MarlinNvfp4MoEKernel,
    TritonNvfp4MoEKernel,
)
from freetoken.moe.mirror_pool import nvfp4_bank_shapes, resolve_cache_schema

H, I = 64, 128


def _fake_pool(gated: bool, quant_format: str = "nvfp4"):
    shapes = nvfp4_bank_shapes(H, I, gated=gated)
    return types.SimpleNamespace(
        schema_order=tuple(shapes), shapes=shapes, quant_format=quant_format
    )


def test_identical_names_need_no_layout():
    """The common case (Nemotron-H's own loader, every GGUF family): the cache's
    bank names already are the pool's own, so the mapping is pure identity and
    ``layout`` is never even consulted."""
    pool = _fake_pool(gated=False)
    mapping = resolve_cache_schema(pool, pool.schema_order, layout=None)
    assert mapping == {name: name for name in pool.schema_order}


def test_triton_kernel_layout_resolves_by_shape_and_dtype():
    """Ornith/Qwen3.5's actual kernel: same 6 raw rows, ``_packed`` dropped from
    two of the six names. No name list, no kernel-name special case -- this is
    the real ``TritonNvfp4MoEKernel.layout()``, matched purely on geometry."""
    pool = _fake_pool(gated=True)
    cfg = MoEConfig(num_experts=4, hidden=H, intermediate=I, top_k=1)
    layout = TritonNvfp4MoEKernel().layout(cfg)
    bank_schema = tuple(layout)
    mapping = resolve_cache_schema(pool, bank_schema, layout)
    assert mapping == {
        "gate_up": "gate_up_packed",
        "gate_up_scale": "gate_up_scale",
        "gate_up_global": "gate_up_global",
        "down": "down_packed",
        "down_scale": "down_scale",
        "down_global": "down_global",
    }


def test_marlin_layout_is_refused_not_silently_mismatched():
    """Marlin pre-tiles the weights for its own GEMM and folds the global scale
    into a GPU-resident alpha -- 4 banks, not 6, and int32-tiled, not raw
    ``uint8`` rows. This must be refused by name, not partially matched."""
    pool = _fake_pool(gated=True)
    cfg = MoEConfig(num_experts=4, hidden=H, intermediate=I, top_k=1)
    layout = MarlinNvfp4MoEKernel().layout(cfg)
    bank_schema = tuple(role for role, spec in layout.items() if not spec.resident)
    with pytest.raises(ValueError, match="not a raw checkpoint-row layout"):
        resolve_cache_schema(pool, bank_schema, layout)


def test_wrong_shape_bank_is_named_in_the_refusal():
    """A same-count, same-name layout whose geometry actually differs (a bug,
    or a future kernel that is not really raw rows) must name the offending
    bank and both shapes, not just say "mismatch"."""
    pool = _fake_pool(gated=True)
    layout = {
        "gate_up": BankSpec((2 * I, H // 2), torch.uint8),
        "gate_up_scale": BankSpec((2 * I, H // 16), torch.float8_e4m3fn),
        "gate_up_global": BankSpec((2 * I,), torch.float16),
        "down": BankSpec((H, I // 2), torch.uint8),
        "down_scale": BankSpec((H, I // 16), torch.float8_e4m3fn),
        # wrong dtype for the global scale bank
        "down_global": BankSpec((H,), torch.bfloat16),
    }
    with pytest.raises(ValueError, match="down_global"):
        resolve_cache_schema(pool, tuple(layout), layout)


def test_non_nvfp4_pool_refuses_a_mismatched_cache_schema():
    """A GGUF mirror pool's own naming is unrelated to the kernel-method NVFP4
    generalization -- a mismatch there is refused as before, unconditionally."""
    pool = _fake_pool(gated=True, quant_format="gguf")
    with pytest.raises(ValueError, match="does not match cache"):
        resolve_cache_schema(pool, ("gate_up", "down"), layout=None)


def test_bank_count_mismatch_names_why():
    pool = _fake_pool(gated=True)
    with pytest.raises(ValueError, match="different bank count"):
        resolve_cache_schema(pool, ("gate_up", "down"), layout={
            "gate_up": BankSpec((2 * I, H // 2), torch.uint8),
            "down": BankSpec((H, I // 2), torch.uint8),
        })
