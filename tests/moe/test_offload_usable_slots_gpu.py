"""GPU kernel-equivalence test for the expert-arena prerequisite.

Compares the legacy sized-cache LRU kernels (host-scalar class_begin/class_end) with
their gated (``FREETOKEN_EXPERT_ARENA=1``) pointer-based twins:

1. At usable == capacity, the two paths must produce bit-identical eviction/index
   results for the same random input sequence (no behavior change).
2. With usable < capacity, the gated path must never select (evict into, or leave
   ``id_of_slot``/``slot_for_id`` pointing at) a slot >= usable.

Requires CUDA; skipped otherwise (this suite is GPU-only work per the assignment).
"""

from __future__ import annotations

import random

import pytest
import torch

import freetoken.moe.offload_kernels as offload_kernels
from freetoken.moe.offload_cache import OffloadMoeCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_cache(num_layers, num_experts, cache_size, device) -> OffloadMoeCache:
    # cache_policy="lfu" forces ensure_experts onto the sized-cache kernel
    # (_ensure_experts_sized_gpu) regardless of GGUF size-class configuration --
    # the same dispatch condition ("cache._size_class_enabled or
    # cache.cache_policy_id == 1") the growable-KV expert-arena work targets.
    return OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=device,
        cache_policy="lfu",
    )


def _snapshot(cache: OffloadMoeCache) -> dict:
    # evict_slots/src_indices are scratch, only their [0, num_indices) prefix is
    # meaningful after a call -- comparing the whole buffer at the end would also
    # compare stale bytes left over from whichever step last wrote furthest into
    # each cache's own (independently evolving) history, which need not agree.
    # The per-step loop below compares that valid prefix; this snapshot instead
    # captures the two caches' actual persistent state.
    return {
        "slot_for_id": cache.slot_for_id.clone(),
        "id_of_slot": cache.id_of_slot.clone(),
        "usage": cache.usage.clone(),
        "step": cache.step.clone(),
        "expert_frequency": cache.expert_frequency.clone(),
        "policy_steps": cache.policy_steps.clone(),
    }


def test_gated_path_matches_legacy_path_at_usable_equals_capacity():
    device = torch.device("cuda")
    num_layers, num_experts, cache_size = 2, 6, 10
    rng = random.Random(1234)

    legacy = _make_cache(num_layers, num_experts, cache_size, device)
    gated = _make_cache(num_layers, num_experts, cache_size, device)

    steps = [
        (rng.randrange(num_layers), sorted(rng.sample(range(num_experts), k=rng.randint(1, 3))))
        for _ in range(40)
    ]

    for layer_id, experts in steps:
        legacy_ids = torch.tensor([experts], dtype=torch.int32, device=device)
        gated_ids = torch.tensor([experts], dtype=torch.int32, device=device)

        offload_kernels.FREETOKEN_EXPERT_ARENA = False
        legacy.ensure_experts(layer_id, legacy_ids)
        offload_kernels.FREETOKEN_EXPERT_ARENA = True
        try:
            gated.ensure_experts(layer_id, gated_ids)
        finally:
            offload_kernels.FREETOKEN_EXPERT_ARENA = False

        torch.cuda.synchronize()
        assert torch.equal(legacy_ids, gated_ids), (layer_id, experts)
        n = int(legacy.num_indices.item())
        assert n == int(gated.num_indices.item()), (layer_id, experts)
        assert torch.equal(legacy.evict_slots[:n], gated.evict_slots[:n]), (layer_id, experts)
        assert torch.equal(legacy.src_indices[:n], gated.src_indices[:n]), (layer_id, experts)

    legacy_snap, gated_snap = _snapshot(legacy), _snapshot(gated)
    for key in legacy_snap:
        assert torch.equal(legacy_snap[key], gated_snap[key]), key


def test_gated_path_never_selects_a_slot_at_or_beyond_usable():
    device = torch.device("cuda")
    num_layers, num_experts, cache_size, usable = 1, 4, 10, 6
    cache = _make_cache(num_layers, num_experts, cache_size, device)
    cache.usable_slots.fill_(usable)

    rng = random.Random(99)
    offload_kernels.FREETOKEN_EXPERT_ARENA = True
    try:
        for _ in range(60):
            experts = sorted(rng.sample(range(num_experts), k=rng.randint(1, num_experts)))
            ids = torch.tensor([experts], dtype=torch.int32, device=device)
            cache.ensure_experts(0, ids)
            torch.cuda.synchronize()
            # class_begin == 0 here (no size classing), so class-local == global slot id.
            n = int(cache.num_indices.item())
            if n:
                assert (cache.evict_slots[:n] < usable).all(), cache.evict_slots[:n].tolist()
            assert torch.all(ids[ids >= 0] < usable)
    finally:
        offload_kernels.FREETOKEN_EXPERT_ARENA = False

    # The region at/after usable must never have been touched.
    assert torch.all(cache.id_of_slot[usable:] == -1)
    assert torch.all(cache.usage[usable:] == 0)


def test_legacy_path_ignores_usable_slots_when_gate_is_off():
    """Sanity check that the gate genuinely controls dispatch: with the flag off,
    a shrunk usable_slots value has no effect (legacy kernel doesn't read it)."""
    device = torch.device("cuda")
    cache = _make_cache(1, 4, 10, device)
    cache.usable_slots.fill_(2)

    offload_kernels.FREETOKEN_EXPERT_ARENA = False
    ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
    cache.ensure_experts(0, ids)
    torch.cuda.synchronize()

    # All 4 experts fit in the full 10-slot cache; the (ignored) usable_slots=2
    # must not have constrained placement.
    assert torch.equal(torch.sort(ids).values, torch.tensor([[0, 1, 2, 3]], device=device))
    assert (cache.id_of_slot[:10] >= -1).all()
