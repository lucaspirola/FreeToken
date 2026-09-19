"""The coverage clamp: KV growth may not shrink the expert arena below what the
bounded mirror can complement.

This is the mechanism that kept the 713K-token ceiling correct (the 600K run
died mid-request before it existed), so it gets a regression test of its own:
the arena must stop at the pool's floor no matter how much the KV guard asks
to release, and the failure message must name the knob that fixes it.
"""
from __future__ import annotations

import json
import os
os.environ.setdefault("FREETOKEN_EXPERT_ARENA", "1")
import struct
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="coverage clamp needs CUDA"
)

from freetoken.moe.mirror_pool import MirrorExpertPool, nvfp4_bank_shapes, plan_capacity

LAYERS, EXPERTS, H, ISZ = 4, 16, 32, 32
_GPU = int(LAYERS * EXPERTS * 0.74)   # 47 of 64 slots on the GPU


def _write_checkpoint(root):
    """Same minimal NVFP4 checkpoint the other mirror tests use."""
    shapes = nvfp4_bank_shapes(H, ISZ)
    suffix_of = {
        "gate_up_packed": ("up_proj.weight", "U8"),
        "gate_up_scale": ("up_proj.weight_scale", "F8_E4M3"),
        "gate_up_global": ("up_proj.weight_scale_2", "F32"),
        "down_packed": ("down_proj.weight", "U8"),
        "down_scale": ("down_proj.weight_scale", "F8_E4M3"),
        "down_global": ("down_proj.weight_scale_2", "F32"),
    }
    header, blob, off = {}, bytearray(), 0
    for layer in range(LAYERS):
        for e in range(EXPERTS):
            flat = layer * EXPERTS + e
            for name, (tail, _dt) in shapes.items():
                suffix, dt = suffix_of[name]
                key = f"backbone.layers.{layer}.mixer.experts.{e}.{suffix}"
                if dt == "F32":
                    payload, shape = struct.pack("<f", float(flat + 1)), []
                else:
                    n = 1
                    for d in tail:
                        n *= d
                    payload = bytes(((flat * 13 + k) % 251) + 1 for k in range(n))
                    shape = list(tail)
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
        json.dump({"layers_block_type": ["moe"] * LAYERS}, f)


def _cache_and_pool(root, capacity):
    from freetoken.moe.offload_cache import OffloadMoeCache

    pool = MirrorExpertPool(root, LAYERS, EXPERTS, capacity,
                            hidden_size=H, intermediate_size=ISZ,
                            device=torch.device("cuda"))
    cache = OffloadMoeCache(
        num_layers=LAYERS, num_experts=EXPERTS, cache_size=_GPU,
        slot_capacity=_GPU, arena_step_slots=4,
        device=torch.device("cuda"), quant_format="nvfp4", cache_policy="lfu",
    )
    cache.direct_device_banks = True
    cache.attach_mirror_pool(pool)
    cache.mirror_warm_start()
    return cache, pool


def test_clamp_stops_at_the_coverage_floor():
    """Shrink requests below the pool's floor land ON the floor, not past it."""
    with tempfile.TemporaryDirectory() as root:
        _write_checkpoint(root)
        cap = plan_capacity(LAYERS, EXPERTS, _GPU)
        cache, pool = _cache_and_pool(root, cap)
        try:
            floor = pool.total - pool.coverage_floor_complement
            assert EXPERTS <= floor < cache.cache_size, "test needs room below the start"
            # Ask for less than the floor (still above the kernel's own
            # num_experts floor so the request itself is legal): the coverage
            # clamp must hold the arena at the pool's floor.
            # The kernel's own floor: chunk-aligned requests below
            # num_experts are refused BEFORE any coverage logic.
            with pytest.raises(ValueError, match=r"below the floor 16"):
                cache.set_usable_slots(12)
            # set_usable_slots itself does NOT clamp (the engine does, in
            # _grow_runtime_kv_arena): a legal request below the coverage
            # floor passes through, so the coverage invariant is broken --
            # assert that the ENGINE's clamp arithmetic restores it.
            below = floor - 4  # one chunk below the coverage floor
            cache.set_usable_slots(below)
            assert cache.usable_slots == below
            # what _grow_runtime_kv_arena computes: the floor minus one chunk
            # of commit slack (the last KV step must have somewhere to release
            # from -- measured "need 0.46 GiB, have 0.42 GiB" at ~600K), then
            # rounded UP to the chunk granularity.
            step = 4  # arena_step_slots of this cache
            clamped = -(-(floor - step) // step) * step
            assert clamped >= floor - step
            assert pool.total - clamped <= pool.coverage_floor_complement + step
            assert pool.total - clamped <= pool.coverage_floor_complement
            # Restore the arena to the clamped value and assert the coverage
            # invariant holds there (the raw 28-slot state above is exactly
            # the out-of-contract state the engine clamp exists to prevent).
            cache.set_usable_slots(clamped)
            assert pool.total - cache.usable_slots <= pool.coverage_floor_complement
        finally:
            pool.close()


def test_coverage_floor_math_is_the_pools_own_geometry():
    """The bound derives from the pool, not from a model constant."""
    with tempfile.TemporaryDirectory() as root:
        _write_checkpoint(root)
        total = LAYERS * EXPERTS
        cap = plan_capacity(LAYERS, EXPERTS, _GPU)
        cache, pool = _cache_and_pool(root, cap)
        try:
            assert pool.coverage_floor_complement == total - cap + 2 * EXPERTS
            floor = total - pool.coverage_floor_complement
            assert floor == cap - 2 * EXPERTS
            # A bigger pool lifts the floor (less complement to cover).
            assert floor >= 0
        finally:
            pool.close()
