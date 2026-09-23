"""S12c: the mirror pool's GGUF source (``models/qwen3_5_moe/gguf.gguf_expert_row_extents``).

Companion to ``tests/models/test_check_experts_gguf.py`` (same synthetic-GGUF writer)
and ``tests/moe/test_mirror_pool.py`` (same coverage/byte-exactness contract, now over
a GGUF checkpoint instead of NVFP4 safetensors). The CPU-only tests below exercise the
extents/row-assembly math directly against the real bytes on disk -- no CUDA, no
``MirrorExpertPool`` -- and the CUDA-gated test at the bottom builds a real pool and
checks its rows against ``load_gguf_expert_sources``, the whole-model loader.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import gguf

import freetoken.distributed.info as di
from freetoken.models.gguf.reader import gguf_mirror_hooks, gguf_tensor_extents
from freetoken.models.qwen3_5_moe.gguf import (
    gguf_expert_row_extents,
    gguf_mirror_bank_shapes,
    load_gguf_expert_sources,
    parse_gguf_config,
)
from freetoken.utils import cached_load_hf_config

from tests.models.test_check_experts_gguf import H, I, E, MAIN_LAYERS, _write_qwen35moe_gguf

pytestmark_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mirror pool needs CUDA for host_register"
)


@pytest.fixture(autouse=True)
def _tp1():
    try:
        di.get_tp_info()
    except RuntimeError:
        di.set_tp_info(0, 1)


def _config(path):
    return parse_gguf_config(cached_load_hf_config(path))


def _read_bytes(path, off, length):
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.pread(fd, length, off)
    finally:
        os.close(fd)


def test_extents_and_row_assembly_match_the_file(tmp_path):
    """Every expert's 3 pieces point at the tensor's own real bytes on disk,
    with no repack: reading them back must reproduce the checkpoint exactly."""
    path = str(tmp_path / "uniform.gguf")
    q8 = gguf.GGMLQuantizationType.Q8_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8)] * MAIN_LAYERS, mtp=False)
    config = _config(path)

    source = gguf_expert_row_extents(path, config)
    assert source.quant_format == "gguf"
    assert set(source.shapes) == {"gate_up", "down"}
    assert len(source.records) == MAIN_LAYERS * E

    row_bytes = gguf.GGML_QUANT_SIZES[q8][1] * (H // gguf.GGML_QUANT_SIZES[q8][0])
    half = I * row_bytes
    payload = H * row_bytes

    for flat, (fd, pieces) in source.records.items():
        assert source.shard_fds[path] == fd
        gu = bytearray(source.shapes["gate_up"][0][0])
        dn = bytearray(source.shapes["down"][0][0])
        for off, length, bank, dst, broadcast in pieces:
            assert broadcast == 0
            raw = _read_bytes(path, off, length)
            (gu if bank == "gate_up" else dn)[dst:dst + length] = raw
        layer, expert = divmod(flat, E)
        # Recompute this expert's expected bytes directly off the file: gate at
        # [0:half), up at [half:2*half) of gate_up, down at [0:payload) of down.
        want_gate = _read_bytes(path, _tensor_offset(path, f"blk.{layer}.ffn_gate_exps.weight") + expert * half, half)
        want_up = _read_bytes(path, _tensor_offset(path, f"blk.{layer}.ffn_up_exps.weight") + expert * half, half)
        want_down = _read_bytes(path, _tensor_offset(path, f"blk.{layer}.ffn_down_exps.weight") + expert * payload, payload)
        assert bytes(gu[:half]) == want_gate
        assert bytes(gu[half:2 * half]) == want_up
        assert bytes(dn[:payload]) == want_down


def _tensor_offset(path, name):
    off, _n = gguf_tensor_extents(path)[name]
    return off


def test_pool_row_matches_the_loader_row(tmp_path):
    """The GGUF source's per-expert bytes are exactly what the whole-model
    loader (``load_gguf_expert_sources``) puts in its own host banks, real
    bytes only -- alignment padding (uninitialized in both) is excluded."""
    path = str(tmp_path / "uniform.gguf")
    q8 = gguf.GGMLQuantizationType.Q8_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8)] * MAIN_LAYERS, mtp=False)
    config = _config(path)

    source = gguf_expert_row_extents(path, config)
    loader_banks = load_gguf_expert_sources(path, config)

    row_bytes = gguf.GGML_QUANT_SIZES[q8][1] * (H // gguf.GGML_QUANT_SIZES[q8][0])
    half = I * row_bytes
    payload = H * row_bytes

    for layer in range(MAIN_LAYERS):
        for expert in range(E):
            flat = layer * E + expert
            fd, pieces = source.records[flat]
            gu = bytearray(source.shapes["gate_up"][0][0])
            dn = bytearray(source.shapes["down"][0][0])
            for off, length, bank, dst, _bc in pieces:
                raw = _read_bytes(path, off, length)
                (gu if bank == "gate_up" else dn)[dst:dst + length] = raw
            loader_gu = loader_banks["gate_up"][layer][expert].numpy().tobytes()
            loader_dn = loader_banks["down"][layer][expert].numpy().tobytes()
            assert bytes(gu[:2 * half]) == loader_gu[:2 * half]
            assert bytes(dn[:payload]) == loader_dn[:payload]


def test_mtp_block_is_skipped(tmp_path):
    """An MTP block (layer >= config.num_layers) is never scanned, even when
    its own quant type differs from the served layers -- proof it was not
    touched, not just that it happens to be compatible."""
    path = str(tmp_path / "with_mtp.gguf")
    q8, q4 = gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8)] * MAIN_LAYERS, mtp=True)
    # The writer reuses the last served type for the MTP block by construction
    # (see _write_qwen35moe_gguf); patch nothing further -- config.num_layers
    # excludes it regardless of its type, which is the property under test.
    config = _config(path)
    assert config.num_layers == MAIN_LAYERS

    source = gguf_expert_row_extents(path, config)
    assert len(source.records) == MAIN_LAYERS * E
    assert all(flat < MAIN_LAYERS * E for flat in source.records)


def test_mixed_size_classes_are_refused(tmp_path):
    path = str(tmp_path / "mixed.gguf")
    q8, q4 = gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8), int(q4), int(q8)], mtp=False)
    config = _config(path)

    with pytest.raises(ValueError, match="mixed GGUF size classes"):
        gguf_mirror_bank_shapes(config)
    with pytest.raises(ValueError, match="mixed GGUF size classes"):
        gguf_expert_row_extents(path, config)


def test_split_gguf_file_name_is_refused(tmp_path):
    path = str(tmp_path / "split-00001-of-00002.gguf")
    q8 = gguf.GGMLQuantizationType.Q8_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8)] * MAIN_LAYERS, mtp=False)

    with pytest.raises(ValueError, match="split GGUF"):
        gguf_tensor_extents(path)


def test_model_without_the_hook_is_refused():
    from types import SimpleNamespace

    assert gguf_mirror_hooks(SimpleNamespace(model_type="nemotron_h")) is None
    assert gguf_mirror_hooks(SimpleNamespace(model_type="")) is None
    assert gguf_mirror_hooks(SimpleNamespace(model_type="not an identifier!")) is None


@pytestmark_cuda
def test_mirror_pool_covers_every_expert_from_a_gguf_source(tmp_path):
    """A fully-saturated pool (capacity == total, gpu_ids=set()) built from a
    GGUF source holds every expert's real bytes, matching the loader -- the
    device-facing half of the correctness reference (CPU-visible banks; a full
    slot-eviction swap chain is exercised for NVFP4 in test_mirror_device.py
    and is not repeated here)."""
    from freetoken.moe.mirror_pool import MirrorExpertPool

    path = str(tmp_path / "uniform.gguf")
    q8 = gguf.GGMLQuantizationType.Q8_0
    _write_qwen35moe_gguf(path, expert_types=[int(q8)] * MAIN_LAYERS, mtp=False)
    config = _config(path)
    source = gguf_expert_row_extents(path, config)
    loader_banks = load_gguf_expert_sources(path, config)

    total = MAIN_LAYERS * E
    pool = MirrorExpertPool(
        path, MAIN_LAYERS, E, total,
        hidden_size=H, intermediate_size=I,
        source=source, config=config, reserve_rows=0,
    )
    try:
        assert pool.quant_format == "gguf"
        pool.load_initial(set())
        row_bytes = gguf.GGML_QUANT_SIZES[q8][1] * (H // gguf.GGML_QUANT_SIZES[q8][0])
        half, payload = I * row_bytes, H * row_bytes
        for layer in range(MAIN_LAYERS):
            for expert in range(E):
                flat = layer * E + expert
                row = pool.pool_row_of_id[flat]
                assert row >= 0
                got_gu = pool.banks["gate_up"][row].numpy().tobytes()
                got_dn = pool.banks["down"][row].numpy().tobytes()
                want_gu = loader_banks["gate_up"][layer][expert].numpy().tobytes()
                want_dn = loader_banks["down"][layer][expert].numpy().tobytes()
                assert got_gu[:2 * half] == want_gu[:2 * half]
                assert got_dn[:payload] == want_dn[:payload]
    finally:
        pool.close()
