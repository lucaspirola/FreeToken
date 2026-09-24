"""mirror_kernels.begin_layer: the per-layer bookkeeping in one launch equals the
two fills plus the pre-step slot-map snapshot it replaced (for the slice the
swap kernel reads)."""
from __future__ import annotations

import types

import pytest
import torch

from freetoken.moe.mirror_kernels import begin_layer


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("plan,experts", [(96, 128), (2048, 256), (7, 3)])
def test_begin_layer_matches_the_old_ops(device, plan, experts):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    layers = 5
    g = torch.Generator().manual_seed(plan + experts)
    slot_for_id = torch.randint(-1, 4000, (layers, experts), generator=g, dtype=torch.int32).to(device)
    victim = torch.randint(0, 99, (plan,), generator=g, dtype=torch.int32).to(device)
    prior = torch.randint(0, 99, (plan,), generator=g, dtype=torch.int32).to(device)
    prev0 = torch.randint(0, 99, (layers * experts,), generator=g, dtype=torch.int32).to(device)
    prev = prev0.clone()
    cache = types.SimpleNamespace(
        victim_ids=victim, prior_ids=prior, slot_for_id=slot_for_id, num_experts=experts,
        residency=types.SimpleNamespace(_mirror={"prev_slot_of_id": prev}),
    )
    for layer in (0, 3, layers - 1):
        victim.random_(0, 99)
        prior.random_(0, 99)
        begin_layer(cache, layer)
        assert torch.all(victim == -1) and torch.all(prior == -1)
        lo, hi = layer * experts, (layer + 1) * experts
        assert torch.equal(prev[lo:hi], slot_for_id[layer])
        prev0[lo:hi] = slot_for_id[layer]
        assert torch.equal(prev, prev0)   # other layers' entries untouched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_arena_ensure_kernel_does_the_same_bookkeeping(monkeypatch):
    """Under the expert arena the v2 LRU kernel does begin_layer's work itself
    (MIRROR_BOOK): victim/prior cleared past the step's misses, this layer's
    pre-step slot map snapshotted, other layers' snapshot entries untouched."""
    from freetoken.moe import offload_kernels as ok
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe.residency import MirrorResidency

    monkeypatch.setattr(ok, "FREETOKEN_EXPERT_ARENA", True)
    dev = torch.device("cuda")
    layers, experts = 3, 8
    cache = OffloadMoeCache(num_layers=layers, num_experts=experts, cache_size=12,
                            device=dev, cache_policy="lfu")
    for e in range(8):                       # layer 0 fills 8 of the 12 slots
        cache.id_of_slot[e] = e
        cache.slot_for_id[0, e] = e
    residency = MirrorResidency(pool=None, cache=cache)
    prev = torch.full((layers * experts,), 77, dtype=torch.int32, device=dev)
    residency._mirror = {
        "pool_row_of_id": torch.zeros(layers * experts, dtype=torch.int32, device=dev),
        "prev_slot_of_id": prev,
    }
    cache.residency = residency
    for layer, ids in ((1, [[2, 5]]), (0, [[1, 3]]), (2, [[0, 1, 2, 3, 4, 5]])):
        cache.victim_ids.fill_(99)
        cache.prior_ids.fill_(99)
        before = cache.slot_for_id[layer].clone()
        other = prev.clone()
        cache.ensure_experts(layer, torch.tensor(ids, dtype=torch.int32, device=dev))
        torch.cuda.synchronize()
        n = int(cache.num_indices.item())
        assert torch.all(cache.victim_ids[n:] == -1) and torch.all(cache.prior_ids[n:] == -1)
        assert not torch.any(cache.victim_ids[:n] == 99) and not torch.any(cache.prior_ids[:n] == 99)
        lo, hi = layer * experts, (layer + 1) * experts
        assert torch.equal(prev[lo:hi], before)
        other[lo:hi] = before
        assert torch.equal(prev, other)
