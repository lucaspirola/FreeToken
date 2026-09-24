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
