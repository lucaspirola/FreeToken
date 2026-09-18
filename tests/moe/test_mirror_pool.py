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


def _pool(root, capacity):
    return MirrorExpertPool(
        root, LAYERS, EXPERTS, capacity, hidden_size=H, intermediate_size=I
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