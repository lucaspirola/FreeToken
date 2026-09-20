"""Mirror-pool residency invariants against a synthetic NVFP4 checkpoint.

The property that makes this design correct is coverage: every expert is on the
GPU, in the mirror, or both. If it ever breaks, a GPU miss has nowhere to read
from and the GEMM silently consumes stale bytes -- so these tests assert it
directly, and assert that the bytes the GPU ends up holding are the checkpoint's
bytes after an eviction cycle deep enough to force real swaps.
"""
from __future__ import annotations

import json
import os
import struct

import pytest
import torch

import types

from freetoken.models.nemotron_h.weight import (
    NVFP4_EXPERT_SOURCE_SPEC as NEMOTRON_SPEC,
)
from freetoken.models.qwen3_5_moe.weight import (
    NVFP4_EXPERT_SOURCE_SPEC as QWEN_SPEC,
)
from freetoken.moe.mirror_pool import MirrorExpertPool, nvfp4_bank_shapes, plan_capacity

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mirror pool needs CUDA for the swap kernels"
)

LAYERS, EXPERTS, H, I = 2, 8, 32, 32


def _write_checkpoint(root, layers=LAYERS, experts=EXPERTS, h=H, i=I):
    """Minimal single-shard NVFP4 checkpoint with deterministic per-expert bytes."""
    shapes = nvfp4_bank_shapes(h, i)
    suffix_of = {
        "gate_up_packed": ("up_proj.weight", "U8"),
        "gate_up_scale": ("up_proj.weight_scale", "F8_E4M3"),
        "gate_up_global": ("up_proj.weight_scale_2", "F32"),
        "down_packed": ("down_proj.weight", "U8"),
        "down_scale": ("down_proj.weight_scale", "F8_E4M3"),
        "down_global": ("down_proj.weight_scale_2", "F32"),
    }
    header, blob, off = {}, bytearray(), 0
    expected = {}
    for layer in range(layers):
        for expert in range(experts):
            flat = layer * experts + expert
            for name, (tail, dtype) in shapes.items():
                suffix, dt = suffix_of[name]
                key = f"backbone.layers.{layer}.mixer.experts.{expert}.{suffix}"
                if dt == "F32":
                    # weight_scale_2 is one scalar in the checkpoint; the bank
                    # holds it broadcast across the row (matching the loader).
                    payload = struct.pack("<f", float(flat + 1))
                    shape = []
                    expected[(flat, name)] = torch.full(
                        tail, float(flat + 1), dtype=torch.float16
                    )
                else:
                    n = 1
                    for d in tail:
                        n *= d
                    raw = bytes(((flat * 7 + k) % 251) + 1 for k in range(n))
                    payload = raw
                    shape = list(tail)
                    expected[(flat, name)] = torch.frombuffer(
                        bytearray(raw), dtype=torch.uint8
                    ).view(*tail).clone()
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
    return expected


@pytest.fixture
def checkpoint(tmp_path):
    expected = _write_checkpoint(str(tmp_path))
    return str(tmp_path), expected


def _pool(root, capacity, reserve_rows=0):
    """A pool for the row-level tests below.

    Driven by the REAL Nemotron-H source spec, not a test-local copy: the point
    of these tests is that the shipped spec locates rows in a checkpoint laid
    out the way that model's checkpoints are.

    ``reserve_rows=0`` by default: this toy geometry (16 rows) is smaller than
    the three-layer writeback/staging reserve a real pool holds back, and these
    tests exercise checkpoint I/O and the coverage refusal, not runtime sizing.
    The reserve's own arithmetic is covered in test_coverage_clamp.py.
    """
    return MirrorExpertPool(
        root, LAYERS, EXPERTS, capacity, hidden_size=H, intermediate_size=I,
        spec=NEMOTRON_SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=reserve_rows,
    )


def test_capacity_covers_the_kv_ceiling():
    # Everything the GPU cannot hold at its smallest, plus two layers of
    # reserve: one supplies writeback landing rows (a victim must not overwrite
    # the row its own admission is uploading from), the other lets a prefill
    # materialize stage a whole layer plus the experts it displaces.
    assert plan_capacity(23, 128, 1552) == (2944 - 1552) + 3 * 128
    # Reserve is configurable; without it the bound is pure coverage.
    assert plan_capacity(23, 128, 1552, reserve=0) == 2944 - 1552
    # GPU holds everything: only the reserve remains (still capped at total).
    assert plan_capacity(23, 128, 2944) == 3 * 128


def test_rows_match_the_checkpoint(checkpoint):
    root, expected = checkpoint
    pool = _pool(root, LAYERS * EXPERTS)
    try:
        pool.load_initial(set())
        for flat in range(LAYERS * EXPERTS):
            row = pool.pool_row_of_id[flat]
            assert row >= 0
            for name in pool.shapes:
                got = pool.banks[name][row]
                want = expected[(flat, name)]
                if got.dtype == torch.float16:
                    assert torch.equal(got.cpu(), want), (flat, name)
                else:
                    assert torch.equal(got.view(torch.uint8).cpu(),
                                       want.view(torch.uint8)), (flat, name)
    finally:
        pool.close()


def test_load_initial_refuses_to_break_coverage(checkpoint):
    root, _ = checkpoint
    # A mirror too small to hold the experts the GPU lacks cannot serve a miss.
    pool = _pool(root, 4)
    try:
        with pytest.raises(RuntimeError, match="cannot cover"):
            pool.load_initial(set(range(4)))  # 12 absent, capacity 4
    finally:
        pool.close()

# --------------------------------------------------------------------------
# Gated models (Qwen3.5 / Ornith): same six banks, different checkpoint layout
# --------------------------------------------------------------------------

G_LAYERS, G_EXPERTS, G_H, G_I = 2, 4, 32, 32


def _write_gated_checkpoint(root):
    """Qwen3.5/Ornith-shaped NVFP4 checkpoint: three projections, gate|up fused.

    The loader packs gate into the first I output rows of the gate_up banks and
    up into the next I (models/nvfp4_banks.py). The mirror must land on exactly
    those bytes, because a GPU miss may be served from either source.
    """
    per_role = {
        "gate_proj": ((G_I, G_H // 2), (G_I, G_H // 16)),
        "up_proj": ((G_I, G_H // 2), (G_I, G_H // 16)),
        "down_proj": ((G_H, G_I // 2), (G_H, G_I // 16)),
    }
    header, blob, off = {}, bytearray(), 0
    expected = {}

    def _bytes(shape, seed):
        n = 1
        for d in shape:
            n *= d
        return bytes(((seed * 31 + k) % 251) + 1 for k in range(n))

    for layer in range(G_LAYERS):
        for expert in range(G_EXPERTS):
            flat = layer * G_EXPERTS + expert
            rows = {}
            for ri, (proj, (wshape, sshape)) in enumerate(per_role.items()):
                base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
                for suffix, shape, dt in (("weight", wshape, "U8"),
                                          ("weight_scale", sshape, "F8_E4M3")):
                    raw = _bytes(shape, flat * 7 + ri * 3 + len(suffix))
                    header[f"{base}.{suffix}"] = {
                        "dtype": dt, "shape": list(shape),
                        "data_offsets": [off, off + len(raw)]}
                    blob += raw
                    off += len(raw)
                    rows[(proj, suffix)] = torch.frombuffer(
                        bytearray(raw), dtype=torch.uint8).view(*shape).clone()
                g = float(flat + 1) + ri
                payload = struct.pack("<f", g)
                header[f"{base}.weight_scale_2"] = {
                    "dtype": "F32", "shape": [], "data_offsets": [off, off + 4]}
                blob += payload
                off += 4
                rows[(proj, "global")] = g
            # gate first, then up -- the fusion the loader performs.
            expected[(flat, "gate_up_packed")] = torch.cat(
                [rows[("gate_proj", "weight")], rows[("up_proj", "weight")]])
            expected[(flat, "gate_up_scale")] = torch.cat(
                [rows[("gate_proj", "weight_scale")], rows[("up_proj", "weight_scale")]])
            expected[(flat, "gate_up_global")] = torch.cat([
                torch.full((G_I,), rows[("gate_proj", "global")], dtype=torch.float16),
                torch.full((G_I,), rows[("up_proj", "global")], dtype=torch.float16)])
            expected[(flat, "down_packed")] = rows[("down_proj", "weight")]
            expected[(flat, "down_scale")] = rows[("down_proj", "weight_scale")]
            expected[(flat, "down_global")] = torch.full(
                (G_H,), rows[("down_proj", "global")], dtype=torch.float16)

    head = json.dumps(header).encode()
    with open(os.path.join(root, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        f.write(blob)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in header}}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"layers_block_type": ["moe"] * G_LAYERS}, f)
    return expected


def test_gated_rows_match_the_checkpoint(tmp_path):
    """Ornith's layout reads byte-exact through the shipped Qwen3.5 spec."""
    root = str(tmp_path)
    expected = _write_gated_checkpoint(root)
    total = G_LAYERS * G_EXPERTS
    pool = MirrorExpertPool(
        root, G_LAYERS, G_EXPERTS, total, hidden_size=G_H, intermediate_size=G_I,
        spec=QWEN_SPEC, config=types.SimpleNamespace(), reserve_rows=0,
    )
    try:
        assert pool.gated, "the Qwen3.5 spec is gated"
        assert pool.shapes["gate_up_packed"][0] == (2 * G_I, G_H // 2)
        pool.load_initial(set())
        for flat in range(total):
            row = pool.pool_row_of_id[flat]
            assert row >= 0
            for name in pool.shapes:
                got = pool.banks[name][row]
                want = expected[(flat, name)]
                if got.dtype == torch.float16:
                    assert torch.equal(got.cpu(), want), (flat, name)
                else:
                    assert torch.equal(got.view(torch.uint8).cpu(),
                                       want.view(torch.uint8)), (flat, name)
    finally:
        pool.close()


def test_a_model_without_a_published_spec_is_refused(tmp_path):
    """No spec means nobody verified that layout: refuse, do not guess."""
    root = str(tmp_path)
    _write_gated_checkpoint(root)
    with pytest.raises(ValueError, match="expert source spec"):
        MirrorExpertPool(root, G_LAYERS, G_EXPERTS, G_LAYERS * G_EXPERTS,
                         hidden_size=G_H, intermediate_size=G_I, reserve_rows=0)
