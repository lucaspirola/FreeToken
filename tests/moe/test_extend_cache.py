"""Small-M extend forwards route their MoE through the DECODE expert cache.

The prefill/extend movement path streams every expert of the layer into a
double buffer on every forward -- ``num_experts`` rows per layer per forward,
with no dependence on ``topk_ids`` -- so its cost is flat in the token count.
On Nemotron 3.5 Lightning that is 23 layers x 128 experts x ~5.35 MiB = 15.4 GiB
of PCIe traffic whether the forward carries 1 token or 8,192: free behind a full
chunk's GPU work, and the entire cost of a short extend.

These pin the gate that sends a short extend down the decode path instead, and
that the decode slot cache plus the grouped prefill GEMM computes the same thing
as the full-layer buffer does.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_LAYERS, E, CACHE_SIZE = 3, 8, 24


def _cache(**kwargs) -> OffloadMoeCache:
    defaults = dict(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=CACHE_SIZE,
        device=torch.device("cpu"),
        quant_format="nvfp4",
    )
    defaults.update(kwargs)
    return OffloadMoeCache(**defaults)


def test_threshold_is_inclusive_and_positive():
    cache = _cache(extend_cache_tokens=64)
    assert not cache.use_cached_extend(0, 0)  # no tokens: nothing to route
    assert cache.use_cached_extend(0, 1)
    assert cache.use_cached_extend(0, 64)
    assert not cache.use_cached_extend(0, 65)


def test_zero_disables():
    assert not _cache(extend_cache_tokens=0).use_cached_extend(0, 1)


def test_default_is_on():
    assert _cache().extend_cache_tokens == 64


def test_env_override():
    with patch.dict(os.environ, {"FREETOKEN_MOE_EXTEND_CACHE_TOKENS": "8"}):
        cache = _cache(extend_cache_tokens=64)
    assert cache.extend_cache_tokens == 8
    assert cache.use_cached_extend(0, 8)
    assert not cache.use_cached_extend(0, 9)
    with patch.dict(os.environ, {"FREETOKEN_MOE_EXTEND_CACHE_TOKENS": "0"}):
        assert not _cache().use_cached_extend(0, 1)


@pytest.mark.parametrize("fmt", ["nvfp4", "nvfp4_marlin", "nvfp4_b12x"])
def test_supported_formats(fmt):
    assert _cache(quant_format=fmt).use_cached_extend(0, 4)


@pytest.mark.parametrize("fmt", ["bf16", "fp8_block", "q4_0", "gguf", "mxfp4_triton"])
def test_other_formats_keep_the_full_layer_stream(fmt):
    """Only the NVFP4 GEMM entry points take arbitrary bank-row ids; the rest
    assume position == expert id and must not be pointed at the slot cache."""
    assert not _cache(quant_format=fmt).use_cached_extend(0, 4)


def test_cpu_and_hybrid_decode_targets_are_excluded():
    for target in ("cpu", "hybrid"):
        assert not _cache(decode_target=target).use_cached_extend(0, 4)


def test_cpu_layer_is_excluded():
    cache = _cache()
    cache.cpu_layer_ids = frozenset({1})
    assert cache.use_cached_extend(0, 4)
    assert not cache.use_cached_extend(1, 4)


def test_unpinned_layer_is_excluded():
    """``copy_missing`` cannot honour ensure_experts' slot remap without a device
    alias for the host bank; an unpinned layer's only copy is the whole layer."""
    cache = _cache()
    cache._unpinned_layers = frozenset({2})
    assert cache.use_cached_extend(0, 4)
    assert not cache.use_cached_extend(2, 4)


def test_engine_config_default_and_arg():
    from freetoken.engine.config import EngineConfig
    from freetoken.server.args import ServerArgs

    assert EngineConfig.moe_extend_cache_tokens == 64
    assert ServerArgs.moe_extend_cache_tokens == 64


# ---------------------------------------------------------------------------
# Numerics: the slot cache plus the grouped GEMM must equal the full-layer path.
# ---------------------------------------------------------------------------

# Ungated ReLU^2 NVFP4 geometry (Nemotron 3.5 Lightning's shape, at test size):
# models/nvfp4_banks.py allocates (E, I, H//2) / (E, I, H//16) / (E, I) when gated is
# False, and the globals are per OUTPUT ROW, not per expert -- an [E, 1] global bank
# silently reads overlapping rows and turns every comparison here into NaN == NaN.
NL, NE, HID, INT, TOPK = 2, 8, 256, 128, 2


def _rand_nvfp4_sources(seed: int = 0) -> dict[str, list[torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    total = NL * NE

    def u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def scale(*shape):
        return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

    flat = {
        "gate_up_packed": u8(total, INT, HID // 2),
        "gate_up_scale": scale(total, INT, HID // 16),
        "gate_up_global": torch.full((total, INT), 0.5, dtype=torch.float16),
        "down_packed": u8(total, HID, INT // 2),
        "down_scale": scale(total, HID, INT // 16),
        "down_global": torch.full((total, HID), 0.75, dtype=torch.float16),
    }
    return {name: list(t.pin_memory().split(NE)) for name, t in flat.items()}


def _fixture(cache_size: int = 4 * NE):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:  # OffloadMoELayer.__init__ reads the TP world
        set_tp_info(rank=0, size=1)
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=NL,
        num_experts=NE,
        cache_size=cache_size,
        device=dev,
        quant_format="nvfp4",
        prefill_overlap=True,
        cache_policy="lfu",
    )
    cache.set_bank_sources(_rand_nvfp4_sources())
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=NE,
        top_k=TOPK,
        hidden_size=HID,
        intermediate_size=INT,
        renormalize=False,
        activation="relu2",
    )
    layer.offload_cache = cache
    return cache, layer, dev


def _routing(m, dev, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, HID, device=dev, dtype=torch.bfloat16) / 4
    w = torch.rand(m, TOPK, device=dev, dtype=torch.float32)
    ids = torch.stack(
        [torch.randperm(NE, device=dev)[:TOPK] for _ in range(m)]
    ).to(torch.int32)
    return x, w, ids


@CUDA
def test_cached_extend_matches_the_full_layer_path():
    """Same routing, same weights, two movement paths -- and, deliberately, two
    kernels: the full-layer path runs the grouped prefill GEMM, the cached path the
    decode GEMV. They reduce K in a different order, so this is an agreement check
    at the tolerance ``tests/moe/test_nvfp4_backends.py`` uses for the same pair,
    not a bitwise one."""
    cache, layer, dev = _fixture()

    for m in (1, 3, 8):
        x, w, ids = _routing(m, dev, seed=m)

        cache.reset()
        cache.extend_cache_tokens = 0
        legacy = layer._prefill_routed(x, w, ids.clone())

        cache.reset()
        cache.extend_cache_tokens = 64
        cached = layer._prefill_routed(x, w, ids.clone())

        # A malformed bank makes both arms NaN and the comparison vacuous.
        assert torch.isfinite(legacy).all(), f"reference output is not finite at m={m}"
        assert torch.isfinite(cached).all(), f"cached output is not finite at m={m}"
        assert legacy.abs().max() > 0
        tol = 0.03 * float(legacy.abs().max())
        torch.testing.assert_close(cached.float(), legacy.float(), rtol=3e-2, atol=tol)


@CUDA
def test_cached_extend_fetches_only_the_routed_experts():
    """The point of the change: the rows that cross PCIe are the distinct routed
    experts, not the whole layer."""
    cache, layer, dev = _fixture()
    cache.reset()
    cache.extend_cache_tokens = 64
    x, w, _ = _routing(2, dev)
    ids = torch.tensor([[0, 1], [1, 2]], device=dev, dtype=torch.int32)
    layer._prefill_routed(x, w, ids.clone())
    torch.cuda.synchronize()
    assert int(cache.num_indices.item()) == 3  # 3 distinct experts over 4 routes
    assert int((cache.slot_for_id[0] >= 0).sum()) == 3
    assert int((cache.slot_for_id[1] >= 0).sum()) == 0  # nothing touched layer 1


@CUDA
def test_above_threshold_still_streams_the_full_layer():
    cache, layer, dev = _fixture()
    cache.reset()
    cache.extend_cache_tokens = 4
    x, w, ids = _routing(8, dev)
    layer._prefill_routed(x, w, ids.clone())
    torch.cuda.synchronize()
    # The overlap double buffer owns slots [0, 2E) and position == expert id there;
    # ensure_experts never ran, so no expert of layer 0 has a slot assigned.
    assert int((cache.slot_for_id[0] >= 0).sum()) == 0
