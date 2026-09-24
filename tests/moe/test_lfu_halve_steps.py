"""FREETOKEN_LFU_HALVE_STEPS: the LFU aging period is a switch (default 256, the historical
constant). The CPU reference and the expert-arena v2 kernel halve a layer's counts on the
same call, and at 64 they halve four times as often."""
from __future__ import annotations

import pytest
import torch

from freetoken.moe import offload_kernels as ok
from freetoken.moe.offload_cache import OffloadMoeCache


def _cpu_cache():
    cache = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=4, cache_policy="lfu",
                            device=torch.device("cpu"))
    cache._size_class_enabled = True
    cache._class_ranges = [(0, 4)]
    cache._layer_cache_class = [0, 0]
    return cache


def test_default_is_256():
    assert ok.FREETOKEN_LFU_HALVE_STEPS == 256


@pytest.mark.parametrize("period", [256, 64])
def test_cpu_reference_halves_every_period_calls(monkeypatch, period):
    monkeypatch.setattr(ok, "FREETOKEN_LFU_HALVE_STEPS", period)
    cache = _cpu_cache()
    for _ in range(period - 1):
        cache.ensure_experts(0, torch.tensor([[0]], dtype=torch.int32))
    assert int(cache.expert_frequency[0, 0]) == period - 1
    cache.ensure_experts(0, torch.tensor([[0]], dtype=torch.int32))   # call `period`: halve, +1
    assert int(cache.expert_frequency[0, 0]) == (period - 1) // 2 + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the v2 kernel needs CUDA")
@pytest.mark.parametrize("period", [256, 64])
def test_v2_kernel_matches_the_cpu_reference(monkeypatch, period):
    from freetoken.moe import offload_cache as oc

    monkeypatch.setattr(ok, "FREETOKEN_LFU_HALVE_STEPS", period)
    monkeypatch.setattr(ok, "FREETOKEN_EXPERT_ARENA", True)
    monkeypatch.setattr(oc, "FREETOKEN_EXPERT_ARENA", True)
    gpu = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=4, cache_policy="lfu",
                          device=torch.device("cuda"))
    gpu.copy_missing = lambda: None
    cpu = _cpu_cache()
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        layer = int(torch.randint(0, 2, (1,), generator=g))
        ids = torch.randperm(4, generator=g)[:2].to(torch.int32).view(1, 2)
        gpu.ensure_experts(layer, ids.cuda())
        cpu.ensure_experts(layer, ids.clone())
    assert torch.equal(gpu.expert_frequency.cpu(), cpu.expert_frequency)
