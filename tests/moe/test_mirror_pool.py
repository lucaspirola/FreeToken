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
    assert plan_capacity(23, 128, 1552) == (2944 - 1552) + 2 * 128
    # Reserve is configurable; without it the bound is pure coverage.
    assert plan_capacity(23, 128, 1552, reserve=0) == 2944 - 1552
    # Never more rows than the model has experts.
    assert plan_capacity(23, 128, 2944) == 128


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


def test_swap_is_a_permutation_preserving_coverage(checkpoint):
    """Host-side model of the device kernel: victims take the admitted row."""
    root, _ = checkpoint
    total = LAYERS * EXPERTS
    gpu_slots = 6
    pool = _pool(root, total - gpu_slots)
    try:
        on_gpu = list(range(gpu_slots))
        pool.load_initial(on_gpu)
        rows_before = sorted(r for r, f in enumerate(pool.id_of_pool_row) if f >= 0)
        # Force a long eviction chain: every expert not on the GPU is admitted
        # in turn, evicting the least-recently admitted one.
        for step, new_id in enumerate(range(gpu_slots, total)):
            victim = on_gpu[step % gpu_slots]
            src_row, d2h_row = pool.plan_swap(new_id, victim)
            assert src_row >= 0
            # The victim either had a duplicate (free) or takes the vacated row.
            assert d2h_row in (-1, src_row)
            on_gpu[step % gpu_slots] = new_id
            # Coverage holds after every single swap.
            for flat in range(total):
                assert pool.covers(flat, flat in on_gpu), (step, flat)
        assert sorted(
            r for r, f in enumerate(pool.id_of_pool_row) if f >= 0
        ) == rows_before, "the set of occupied rows must be invariant"

        owners = [f for f in pool.id_of_pool_row if f >= 0]
        assert len(owners) == len(set(owners)), "a row ended up owned twice"
    finally:
        pool.close()


def test_duplicates_make_evictions_free(checkpoint):
    root, _ = checkpoint
    total = LAYERS * EXPERTS
    gpu_slots = 8
    # Mandatory complement (16) + 3 duplicate rows + 2 held in reserve.
    pool = _pool(root, total - gpu_slots + 3 + 2)
    try:
        on_gpu = list(range(gpu_slots))
        pool.load_initial(on_gpu)
        seeded = pool.seed_duplicates(list(reversed(on_gpu)), reserve=2)
        assert seeded == 3, "spare rows beyond the reserve should mirror GPU residents"
        free_left = sum(1 for f in pool.id_of_pool_row if f < 0)
        assert free_left == 2, "the reserve must survive seeding"
        before = pool.free_evictions
        # Evicting a duplicated expert costs no writeback.
        duplicated = [f for f in on_gpu if pool.pool_row_of_id[f] >= 0]
        assert duplicated
        _src, d2h = pool.plan_swap(total - 1, duplicated[0])
        assert d2h == -1
        assert pool.free_evictions == before + 1
    finally:
        pool.close()


def test_missing_expert_is_a_hard_error(checkpoint):
    root, _ = checkpoint
    pool = _pool(root, LAYERS * EXPERTS)
    try:
        pool.load_initial(set())
        # Simulate the coverage invariant being violated.
        pool.pool_row_of_id[3] = -1
        with pytest.raises(RuntimeError, match="coverage violated"):
            pool.plan_swap(3, 0)
    finally:
        pool.close()
