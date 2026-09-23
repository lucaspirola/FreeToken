"""EXL3 expert banks: the whole-model reader + ``pack`` and the RAM saver's pool rows are the
checkpoint's bytes, byte for byte (``models/exl3_banks.py``, ``layers/quantization/moe/exl3.py``).

CPU tests run over a synthetic checkpoint (``_exl3_checkpoint``); the CUDA-gated test builds a
real ``MirrorExpertPool`` over it and compares its rows with the packed banks.
"""
from __future__ import annotations

import os

import pytest
import torch
from safetensors import safe_open

import freetoken.distributed.info as di
from freetoken.layers.quantization import set_quant_config
from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.exl3 import TritonExl3MoEKernel
from freetoken.layers.quantization.scheme import exl3_scheme
from freetoken.models.exl3_banks import (
    BANK_NAMES, exl3_bank_shapes, exl3_expert_row_extents, exl3_mirror_hooks, iter_exl3_expert_pieces,
)
from freetoken.moe.offload_cache import _BANK_SCHEMAS

from tests.moe._exl3_checkpoint import install_quant, model_config, write_exl3_checkpoint

L, E, H, I, BITS = 2, 3, 256, 128, 5


@pytest.fixture(autouse=True)
def _tp1_and_quant_reset():
    try:
        di.get_tp_info()
    except RuntimeError:
        di.set_tp_info(0, 1)
    yield
    set_quant_config(None)


@pytest.fixture
def ckpt(tmp_path):
    root = str(tmp_path / "ckpt")
    tensors = write_exl3_checkpoint(root, L, E, H, I, BITS)
    install_quant(root)
    return root, tensors


def _moe_cfg():
    return MoEConfig(num_experts=E, hidden=H, intermediate=I, top_k=2, scheme=exl3_scheme(BITS, "mul1"), strategy="offload")


def _packed_banks(root):
    kernel = TritonExl3MoEKernel()
    layout = kernel.layout(_moe_cfg())
    banks = {name: torch.zeros((L, E, *spec.shape), dtype=spec.dtype) for name, spec in layout.items()}
    for layer, e0, e1, piece in iter_exl3_expert_pieces(root, model_config(L, E, H, I), batch=2):
        kernel.pack(piece, _moe_cfg(), {name: banks[name][layer, e0:e1] for name in banks})
    return banks


def _pool_row(source, flat):
    """What MirrorExpertPool._read_group writes for ``flat``, via plain preads of the same extents."""
    rows = {name: bytearray(torch.empty(tail, dtype=dt).numel() * torch.empty((), dtype=dt).element_size())
            for name, (tail, dt) in source.shapes.items()}
    for fd, pieces in source.records[flat]:
        path = os.readlink(f"/proc/self/fd/{fd}")
        plain = os.open(path, os.O_RDONLY)
        try:
            for off, length, bank, dst, broadcast in pieces:
                assert broadcast == 0
                rows[bank][dst:dst + length] = os.pread(plain, length, off)
        finally:
            os.close(plain)
    return rows


def _close(source):
    for fd in source.shard_fds.values():
        os.close(fd)


def test_layout_matches_the_bank_schema_and_the_source_shapes(ckpt):
    root, _ = ckpt
    layout = TritonExl3MoEKernel().layout(_moe_cfg())
    assert tuple(layout) == _BANK_SCHEMAS["exl3"] == BANK_NAMES
    assert all(spec.raw_row for spec in layout.values())
    shapes = exl3_bank_shapes(H, I, BITS)
    assert {n: (s.shape, s.dtype) for n, s in layout.items()} == shapes
    source = exl3_expert_row_extents(root, model_config(L, E, H, I))
    try:
        assert source.quant_format == "exl3"
        assert source.shapes == shapes  # resolve_cache_schema: identical bank names and rows
        assert len(source.records) == L * E
    finally:
        _close(source)


def test_packed_rows_are_the_checkpoint_tensors(ckpt):
    root, tensors = ckpt
    banks = _packed_banks(root)
    for layer in range(L):
        for e in range(E):
            base = f"model.language_model.layers.{layer}.mlp.experts.{e}"
            for kind in ("trellis", "suh", "svh"):
                assert torch.equal(banks[f"gate_up_{kind}"][layer, e, 0], tensors[f"{base}.gate_proj.{kind}"])
                assert torch.equal(banks[f"gate_up_{kind}"][layer, e, 1], tensors[f"{base}.up_proj.{kind}"])
                assert torch.equal(banks[f"down_{kind}"][layer, e], tensors[f"{base}.down_proj.{kind}"])


def test_pool_rows_equal_packed_rows_byte_for_byte(ckpt):
    root, _ = ckpt
    banks = _packed_banks(root)
    source = exl3_expert_row_extents(root, model_config(L, E, H, I))
    try:
        for flat in range(L * E):
            layer, e = divmod(flat, E)
            rows = _pool_row(source, flat)
            for name in BANK_NAMES:
                packed = banks[name][layer, e].contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
                assert packed == bytes(rows[name]), (flat, name)
    finally:
        _close(source)


def test_pool_extents_point_at_the_safetensors_data(ckpt):
    root, _ = ckpt
    source = exl3_expert_row_extents(root, model_config(L, E, H, I))
    try:
        for flat, groups in source.records.items():
            for fd, pieces in groups:
                assert fd in source.fd_size
                for off, length, _bank, _dst, _bc in pieces:
                    assert off + length <= source.fd_size[fd]
        assert len(source.shard_fds) == 2
    finally:
        _close(source)


def test_mirror_hooks_resolve_for_qwen35(ckpt):
    hooks = exl3_mirror_hooks(model_config(L, E, H, I))
    assert hooks is not None and hooks[1](model_config(L, E, H, I)) == exl3_bank_shapes(H, I, BITS)


def test_wrong_codebook_multiplier_refuses(tmp_path):
    root = str(tmp_path / "bad")
    write_exl3_checkpoint(root, L, E, H, I, BITS, flag=0x12345678)
    install_quant(root)
    with pytest.raises(ValueError, match="multiplier"):
        list(iter_exl3_expert_pieces(root, model_config(L, E, H, I)))
    with pytest.raises(ValueError, match="multiplier"):
        exl3_expert_row_extents(root, model_config(L, E, H, I))


def test_mixed_expert_bit_widths_refuse(tmp_path):
    root = str(tmp_path / "mixed")
    write_exl3_checkpoint(root, L, E, H, I, BITS, bits_of=lambda layer, e, proj: 4 if (layer, e, proj) == (1, 2, "down_proj") else BITS)
    install_quant(root)
    with pytest.raises(NotImplementedError, match="mix"):
        exl3_expert_row_extents(root, model_config(L, E, H, I))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mirror pool needs CUDA for host_register")
def test_mirror_pool_rows_equal_packed_rows(ckpt):
    from freetoken.moe.mirror_pool import MirrorExpertPool

    root, _ = ckpt
    banks = _packed_banks(root)
    config = model_config(L, E, H, I)
    source = exl3_expert_row_extents(root, config)
    pool = MirrorExpertPool(root, L, E, L * E, hidden_size=H, intermediate_size=I, source=source, config=config,
                            device=torch.device("cuda"), reserve_rows=0)
    try:
        pool.load_initial(set())
        for flat in range(L * E):
            layer, e = divmod(flat, E)
            row = pool.pool_row_of_id[flat]
            assert row >= 0
            for name in BANK_NAMES:
                assert torch.equal(pool.banks[name][row].cpu(), banks[name][layer, e]), (flat, name)
    finally:
        pool.close()
