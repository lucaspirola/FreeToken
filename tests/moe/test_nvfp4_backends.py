"""Numerical tests for the NVFP4 MoE backends (Triton inline-dequant / Marlin / b12x).

Each verifies the full chain ``native banks -> (in-place repack) -> offload cache slot
gather -> fused forward`` against a pure-torch dequant reference, for both regimes (decode
routes slot ids into the full cache; full-layer prefill routes raw expert ids into the
materialized ``[:E]`` view or the overlap double-buffer views).

Coverage by hardware:
  - Triton (any CUDA GPU): prefill + the production fast decode GEMV, plus a fast-vs-
    baseline-kernel equality guard. This is the path used on sm_120 + CUDA 12.x.
  - Marlin (sm_80..sm_99, e.g. H100): prefill + decode + overlap.
  - b12x (sm_120 + CUDA>=13): pure-torch pack everywhere; the fused decode forward is
    gated and skipped where the kernel cannot run.
The ``--nvfp4-backend`` selection + CUDA-13 gate is checked without a GPU.
"""

from __future__ import annotations

import importlib.util
import types

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# vllm (marlin W4A16 path) is intentionally not co-installable with the core transformers
# pin; it lives in a dedicated venv. Skip rather than fail.
marlin = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="needs vllm (marlin path)",
)

L, E, S = 2, 8, 8  # layers, experts/layer, cache slots
H, I = 256, 128  # hidden, moe intermediate
TOPK = 2

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _dequant_ref(packed: torch.Tensor, scale: torch.Tensor, row_global: torch.Tensor) -> torch.Tensor:
    """[N, K//2] u8 + [N, K//16] e4m3 + [N] global -> [N, K] fp32 (low nibble first)."""
    n, k2 = packed.shape
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, 2 * k2).long()
    w = _E2M1.to(packed.device)[codes]
    s = scale.float().repeat_interleave(16, dim=1)
    return w * s * row_global.float().unsqueeze(1)


def _make_native_sources(device: torch.device, seed: int = 0) -> dict[str, list[torch.Tensor]]:
    """Random ModelOpt-style banks, CPU pinned, with one expert whose w1/w3 globals differ.

    One flat ``[L*E, ...]`` RNG draw (so seeding is unaffected) split into L
    per-layer views.
    """
    g = torch.Generator().manual_seed(seed)
    total = L * E

    def rand_u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def rand_scale(*shape):
        return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

    gate_up_global = torch.full((total, 2 * I), 1.0, dtype=torch.float16)
    gate_up_global[:, I:] = 0.5  # w3 global != w1 global: exercises the alpha fold
    down_global = torch.full((total, H), 0.75, dtype=torch.float16)
    flat = {
        "gate_up_packed": rand_u8(total, 2 * I, H // 2),
        "gate_up_scale": rand_scale(total, 2 * I, H // 16),
        "gate_up_global": gate_up_global,
        "down_packed": rand_u8(total, H, I // 2),
        "down_scale": rand_scale(total, H, I // 16),
        "down_global": down_global,
    }
    return {name: list(t.pin_memory().split(E)) for name, t in flat.items()}


def _assert_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    """bf16 grouped GEMMs round the (large) gate_up intermediates to bf16, so the
    achievable accuracy is relative to the output magnitude, not absolute."""
    tol = 0.03 * float(ref.abs().max())
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=tol)


def _swigluoai_ref(h: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """MiniMax-M3 / gpt-oss clamped swiglu over UNINTERLEAVED [gate; up] halves."""
    gate = h[:I].clamp(max=limit)
    up = h[I:].clamp(-limit, limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1.0)


def _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids, activation="silu") -> torch.Tensor:
    """Dequant + dense per-token reference for the gated MoE (silu or swigluoai)."""
    out = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    x = hidden.float()
    for t in range(hidden.size(0)):
        for j in range(topk_ids.size(1)):
            e = int(topk_ids[t, j])
            gu = _dequant_ref(
                sources["gate_up_packed"][layer_id][e].to(hidden.device),
                sources["gate_up_scale"][layer_id][e].to(hidden.device),
                sources["gate_up_global"][layer_id][e].to(hidden.device),
            )
            dn = _dequant_ref(
                sources["down_packed"][layer_id][e].to(hidden.device),
                sources["down_scale"][layer_id][e].to(hidden.device),
                sources["down_global"][layer_id][e].to(hidden.device),
            )
            h = gu @ x[t]
            if activation == "swigluoai":
                act = _swigluoai_ref(h)
            else:
                act = torch.nn.functional.silu(h[:I]) * h[I:]
            out[t] += float(topk_weights[t, j]) * (dn @ act)
    return out


def _marlin_cache(device, *, cache_size=S, prefill_overlap=False):
    from freetoken.moe.nvfp4_backends import marlin_repack_sources_inplace
    from freetoken.moe.offload_cache import OffloadMoeCache

    sources = _make_native_sources(device)
    ref_sources = {k: [t.clone() for t in v] for k, v in sources.items()}  # repack is in place
    cfg = types.SimpleNamespace(hidden_size=H, moe_intermediate_size=I)
    packed = marlin_repack_sources_inplace(sources, cfg, device, chunk=5)

    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=device,
        quant_format="nvfp4_marlin",
        prefill_overlap=prefill_overlap,
    )
    cache.set_bank_sources({name: packed[name] for name in cache.bank_schema})
    cache.set_alphas(packed["gate_up_alpha"], packed["down_alpha"])
    cache.reset()
    return cache, ref_sources


@cuda
@marlin
def test_marlin_prefill_matches_dequant_reference():
    from freetoken.moe.nvfp4_backends import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device)
    torch.manual_seed(1)
    M = 16
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    ref = _ref_moe(ref_sources, 0, hidden, topk_weights, topk_ids)

    # Synchronous full-layer prefill: slot == expert id, raw routing ids pass through.
    cache.materialize_layer(0)
    cache.copy_missing()
    g1, g2 = cache.alphas_for_layer(0)
    gu_p, gu_s, dn_p, dn_s = cache.bank_views(E)
    out = marlin_fused_experts(
        hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
        topk_weights, topk_ids, "silu", False,
    )
    _assert_close(out, ref)


@cuda
@marlin
def test_marlin_decode_matches_dequant_reference_after_prefill_stomp():
    """Decode through the slot cache, including the request-B-after-request-A pattern
    that B1 guarded against: a layer-1 full-layer prefill between two layer-0 decodes."""
    from freetoken.moe.nvfp4_backends import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device)
    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)

    def decode(layer_id, experts):
        ids = torch.tensor([experts], dtype=torch.int32, device=device)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, ids)
        cache.ensure_experts(layer_id, ids)  # rewrites ids -> slots in place
        cache.copy_missing()
        g1, g2 = cache.alphas_for_slots(layer_id)
        gu_p, gu_s, dn_p, dn_s = cache.bank_views()
        out = marlin_fused_experts(
            hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
            topk_weights, ids, "silu", False,
        )
        _assert_close(out, ref)

    decode(0, [3, 5])
    cache.materialize_layer(1)  # full-layer prefill overwrites every slot (S == E)
    cache.copy_missing()
    decode(0, [3, 5])  # must miss + reload, not serve layer-1 bytes
    decode(1, [1, 2])  # pure hits on the prefilled layer


@cuda
@marlin
def test_marlin_overlap_prefill_matches_dequant_reference():
    """prefill_overlap=True over NVFP4 banks: every layer streams through the generic
    double buffer (full-layer views, routing ids unmapped), and a decode afterwards is
    still correct -- the prefetch invalidated the bookkeeping of the stomped slots.

    The decode-after check is armed by claiming layer-0 slots *before* the prefill
    (cache_size == 2E, so every slot is buffer-backed): if the prefetch failed to
    invalidate them, the post-prefill decode would "hit" stale mappings and read other
    experts' bytes; we assert it misses both experts instead."""
    from freetoken.moe.nvfp4_backends import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device, cache_size=2 * E, prefill_overlap=True)
    torch.manual_seed(4)
    M = 16
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    warm_ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    cache.ensure_experts(0, warm_ids)
    cache.copy_missing()

    cache.begin_prefill()
    for layer_id in range(L):
        cache.prefetch_prefill_layer(layer_id)
        cache.prefetch_prefill_layer(layer_id + 1)
        gu_p, gu_s, dn_p, dn_s = cache.wait_prefill_layer(layer_id)
        g1, g2 = cache.alphas_for_layer(layer_id)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, topk_ids)
        out = marlin_fused_experts(
            hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
            topk_weights, topk_ids, "silu", False,
        )
        _assert_close(out, ref)
        cache.release_prefill_layer(layer_id)

    # Decode the pre-claimed experts after the buffers stomped the whole cache: their
    # old slot mappings must be gone (forced miss + reload), not "hit" stale entries
    # now holding other layers' prefill bytes.
    dec_hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    dec_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    ref = _ref_moe(ref_sources, 0, dec_hidden, dec_weights, ids)
    cache.ensure_experts(0, ids)
    assert int(cache.num_indices.item()) == 2, "stale slot mappings survived the prefetch"
    cache.copy_missing()
    g1, g2 = cache.alphas_for_slots(0)
    gu_p, gu_s, dn_p, dn_s = cache.bank_views()
    out = marlin_fused_experts(
        dec_hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
        dec_weights, ids, "silu", False,
    )
    _assert_close(out, ref)


@cuda
def test_triton_overlap_prefill_matches_dequant_reference():
    """The 6-bank native layout through the same generic double buffer, consumed by
    the Triton inline-dequant grouped GEMM with unmapped routing ids (n = E)."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.offload_cache import OffloadMoeCache

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=5)
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=2 * E,
        device=device,
        quant_format="nvfp4",
        prefill_overlap=True,
    )
    cache.set_bank_sources({name: sources[name] for name in cache.bank_schema})
    cache.reset()
    torch.manual_seed(6)
    M = 8
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    cache.begin_prefill()
    for layer_id in range(L):
        cache.prefetch_prefill_layer(layer_id)
        cache.prefetch_prefill_layer(layer_id + 1)
        gu_p, gu_s, gu_g, dn_p, dn_s, dn_g = cache.wait_prefill_layer(layer_id)
        ref = _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids)
        out = fused_experts_nvfp4(
            hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g,
            topk_weights, topk_ids, E, "silu", False,
        )
        _assert_close(out, ref)
        cache.release_prefill_layer(layer_id)


@cuda
def test_triton_swigluoai_matches_dequant_reference():
    """MiniMax-M3's swigluoai routed experts through the Triton prefill grouped GEMM
    and the marlin-style decode GEMV: same banks, the clamped (up+1) swiglu instead
    of silu, alpha/limit threaded through the fused entry points."""
    from freetoken.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin,
        fused_experts_nvfp4,
    )

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=11)
    torch.manual_seed(12)
    M = 8
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)
    layer_id = 0
    banks = [
        sources[name][layer_id].to(device)
        for name in (
            "gate_up_packed", "gate_up_scale", "gate_up_global",
            "down_packed", "down_scale", "down_global",
        )
    ]
    ref = _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids, activation="swigluoai")
    out = fused_experts_nvfp4(
        hidden, *banks, topk_weights, topk_ids, E, "swigluoai", False, 1.702, 7.0
    )
    _assert_close(out, ref)

    dec_hidden = hidden[:1]
    dec_ids = topk_ids[:1]
    dec_weights = topk_weights[:1]
    ref = _ref_moe(sources, layer_id, dec_hidden, dec_weights, dec_ids, activation="swigluoai")
    out = fused_experts_decode_nvfp4_marlin(
        dec_hidden, *banks, dec_weights, dec_ids, "swigluoai", False, 1.702, 7.0
    )
    _assert_close(out, ref)


def _triton_cache(device, *, cache_size=S, prefill_overlap=False):
    """Native 6-bank NVFP4 cache (no repack), consumed directly by the Triton kernels.
    The banks are not transformed, so ``sources`` doubles as the dequant reference."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    sources = _make_native_sources(device, seed=7)
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=device,
        quant_format="nvfp4",
        prefill_overlap=prefill_overlap,
    )
    cache.set_bank_sources({name: sources[name] for name in cache.bank_schema})
    cache.reset()
    return cache, sources


@cuda
def test_triton_decode_marlin_matches_dequant_reference_after_prefill_stomp():
    """The production marlin-style int32 decode GEMV through the slot cache, including the
    request-B-after-request-A pattern (a layer-1 full-layer prefill between two layer-0
    decodes) that must force a miss + reload rather than serve stale slot bytes."""
    from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

    device = torch.device("cuda")
    cache, ref_sources = _triton_cache(device)
    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)

    def decode(layer_id, experts):
        ids = torch.tensor([experts], dtype=torch.int32, device=device)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, ids)
        cache.ensure_experts(layer_id, ids)  # rewrites ids -> slots in place
        cache.copy_missing()
        gu_p, gu_s, gu_g, dn_p, dn_s, dn_g = cache.bank_views()
        out = fused_experts_decode_nvfp4_marlin(
            hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_weights, ids, "silu", False
        )
        _assert_close(out, ref)

    decode(0, [3, 5])
    cache.materialize_layer(1)  # full-layer prefill overwrites every slot (S == E)
    cache.copy_missing()
    decode(0, [3, 5])  # must miss + reload, not serve layer-1 bytes
    decode(1, [1, 2])  # pure hits on the prefilled layer


@cuda
def test_triton_decode_marlin_matches_baseline_kernel():
    """The production marlin-style decode GEMV must match the original LUT-gather decode
    within tolerance (it only reorders the dequant math: int32 wide load + deferred reduce)."""
    from freetoken.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin,
        fused_experts_decode_nvfp4_serial,
    )

    device = torch.device("cuda")
    cache, _ = _triton_cache(device)
    torch.manual_seed(11)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[1, 6]], dtype=torch.int32, device=device)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    banks = cache.bank_views()
    marlin = fused_experts_decode_nvfp4_marlin(hidden, *banks, topk_weights, ids, "silu", False)
    base = fused_experts_decode_nvfp4_serial(hidden, *banks, topk_weights, ids, "silu", False)
    torch.testing.assert_close(marlin.float(), base.float(), rtol=2e-3, atol=2e-3)


def test_nvfp4_backend_selection():
    """--nvfp4-backend selection + the flashinfer/marlin device gates -- runs without a GPU
    via the CPU branch (forced backends need a usable device, so they error loudly there)."""
    from freetoken.moe.nvfp4_backends import select_nvfp4_backend

    cpu = torch.device("cpu")
    assert select_nvfp4_backend(cpu, None, "triton") == "triton"
    assert select_nvfp4_backend(cpu, None, "auto") == "triton"  # auto on CPU
    with pytest.raises(RuntimeError):
        select_nvfp4_backend(cpu, None, "flashinfer")  # b12x needs a CUDA device
    with pytest.raises(RuntimeError):
        select_nvfp4_backend(cpu, None, "marlin")  # marlin needs a CUDA device
    with pytest.raises(ValueError):
        select_nvfp4_backend(cpu, None, "bogus")


@cuda
def test_b12x_decode_matches_dequant_reference():
    """sm_120 + CUDA>=13 only: the flashinfer b12x W4A16 fused MoE over the slot cache
    vs the dequant reference (skipped on hardware/toolkits where b12x cannot run)."""
    from freetoken.moe.nvfp4_backends import (
        _b12x_unusable_reason,
        b12x_fused_experts,
        b12x_repack_sources_inplace,
    )
    from freetoken.moe.offload_cache import OffloadMoeCache

    device = torch.device("cuda")
    reason = _b12x_unusable_reason(torch.cuda.get_device_capability(device))
    if reason is not None:
        pytest.skip(f"b12x not runnable here: {reason}")

    sources = _make_native_sources(device, seed=8)
    ref_sources = {k: [t.clone() for t in v] for k, v in sources.items()}  # repack is in place
    cfg = types.SimpleNamespace(hidden_size=H, moe_intermediate_size=I)
    packed = b12x_repack_sources_inplace(sources, cfg, device, chunk=6)

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=device, quant_format="nvfp4_b12x"
    )
    cache.set_bank_sources({name: packed[name] for name in cache.bank_schema})
    cache.set_alphas(packed["gate_up_alpha"], packed["down_alpha"])
    cache.reset()

    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    ref = _ref_moe(ref_sources, 0, hidden, topk_weights, ids)

    cache.ensure_experts(0, ids)
    cache.copy_missing()
    g1, g2 = cache.alphas_for_slots(0)
    gu_p, gu_s, dn_p, dn_s = cache.bank_views()
    out = b12x_fused_experts(
        hidden, gu_p, gu_s, g1, dn_p, dn_s, g2, topk_weights, ids, "silu", False
    )
    _assert_close(out, ref)


@cuda
def test_dummy_nvfp4_sources_match_loader_contract():
    """--use-dummy-weight banks must match the real loader's shapes/dtypes/pinning so the
    engine repack/offload path is exercised unchanged. The marlin repack + offload gather
    tail (which needs vllm) lives in test_dummy_nvfp4_sources_marlin_repack."""
    from freetoken.models.weight import dummy_nvfp4_expert_sources

    cfg = types.SimpleNamespace(
        num_layers=L, num_experts=E, hidden_size=H, moe_intermediate_size=I
    )
    sources = dummy_nvfp4_expert_sources(cfg)
    expected = {
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), torch.float8_e4m3fn),
        "gate_up_global": ((E, 2 * I), torch.float16),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), torch.float8_e4m3fn),
        "down_global": ((E, H), torch.float16),
    }
    assert sources.keys() == expected.keys()
    for name, (shape, dtype) in expected.items():
        layers = sources[name]
        assert len(layers) == L, (name, len(layers))
        for t in layers:
            assert t.shape == shape and t.dtype == dtype and t.is_pinned(), name


@cuda
@marlin
def test_dummy_nvfp4_sources_marlin_repack():
    """The --use-dummy-weight banks drop into the same marlin repack + offload path as the
    real loader's (in-place repack). The gather kernel reads the banks zero-copy from the
    GPU, which requires the allocator's memory to be device-mapped, not merely page-locked."""
    from freetoken.models.weight import dummy_nvfp4_expert_sources
    from freetoken.moe.nvfp4_backends import marlin_repack_sources_inplace
    from freetoken.moe.offload_cache import OffloadMoeCache

    cfg = types.SimpleNamespace(
        num_layers=L, num_experts=E, hidden_size=H, moe_intermediate_size=I
    )
    sources = dummy_nvfp4_expert_sources(cfg)

    device = torch.device("cuda")
    packed = marlin_repack_sources_inplace(sources, cfg, device, chunk=5)
    assert torch.isfinite(packed["gate_up_alpha"].float()).all()
    assert torch.isfinite(packed["down_alpha"].float()).all()

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=device, quant_format="nvfp4_marlin"
    )
    cache.set_bank_sources({name: packed[name] for name in cache.bank_schema})
    cache.set_alphas(packed["gate_up_alpha"], packed["down_alpha"])
    cache.reset()
    cache.materialize_layer(0)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert torch.equal(cache.bank_caches["gate_up_packed"][:E].cpu(), packed["gate_up_packed"][0])


@cuda
@pytest.mark.slow
def test_b12x_pack_is_byte_compatible_with_native_banks():
    """The b12x kernel needs sm_120, but its pack is pure torch: verify the prepared
    blocks drop into the native banks byte-for-byte (the in-place repack contract)."""
    from freetoken.moe.nvfp4_backends import b12x_repack_sources_inplace

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=3)
    cfg = types.SimpleNamespace(hidden_size=H, moe_intermediate_size=I)
    try:
        packed = b12x_repack_sources_inplace(sources, cfg, device, chunk=6)
    except Exception as exc:  # pragma: no cover - depends on flashinfer internals
        pytest.skip(f"flashinfer w4a16 prepare unavailable off-target: {exc}")
    total = L * E
    # packed banks stay per-layer lists; alphas are the one flat [L*E] exception (see
    # cache_budget.expert_bytes_per_slot).
    assert len(packed["gate_up_packed"]) == L
    assert sum(t.shape[0] for t in packed["gate_up_packed"]) == total
    assert packed["gate_up_alpha"].shape == (total,)
    assert packed["down_packed"][0].dtype == torch.int32


# ---------------------------------------------------------------------------
# Ungated ReLU^2 experts (Nemotron-3.5-Lightning): up-only [I, H] gate_up banks,
# relu(x)**2 between the two GEMMs. flashinfer's SM12x W4A16 kernel fuses that
# activation natively (SUPPORTED_MOE_ACTIVATIONS = {"silu", "relu2"}), so the b12x
# pack must skip the gated [gate,up] half-swap and pass activation= through.
# ---------------------------------------------------------------------------


def _make_ungated_sources(seed: int = 0, *, e: int = E) -> dict[str, list[torch.Tensor]]:
    """Random ungated ModelOpt NVFP4 banks: gate_up is one [I, H] up projection.

    Mirrors ``_make_native_sources`` (one flat ``[L*e, ...]`` draw split per layer,
    CPU pinned) but with the Nemotron bank shapes -- ``models/nvfp4_banks.py``
    allocates ``(E, I, H//2)`` / ``(E, I, H//16)`` / ``(E, I)`` when ``gated=False``.
    """
    g = torch.Generator().manual_seed(seed)
    total = L * e

    def rand_u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def rand_scale(*shape):
        return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

    flat = {
        "gate_up_packed": rand_u8(total, I, H // 2),
        "gate_up_scale": rand_scale(total, I, H // 16),
        # one broadcast up global per expert (the checkpoint's scalar weight_scale_2)
        "gate_up_global": torch.full((total, I), 0.5, dtype=torch.float16),
        "down_packed": rand_u8(total, H, I // 2),
        "down_scale": rand_scale(total, H, I // 16),
        "down_global": torch.full((total, H), 0.75, dtype=torch.float16),
    }
    return {name: list(t.pin_memory().split(e)) for name, t in flat.items()}


def _ref_moe_relu2(sources, layer_id, hidden, topk_weights, topk_ids) -> torch.Tensor:
    """Dequant + dense per-token reference for the ungated ReLU^2 MoE."""
    out = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    x = hidden.float()
    for t in range(hidden.size(0)):
        for j in range(topk_ids.size(1)):
            e = int(topk_ids[t, j])
            up = _dequant_ref(
                sources["gate_up_packed"][layer_id][e].to(hidden.device),
                sources["gate_up_scale"][layer_id][e].to(hidden.device),
                sources["gate_up_global"][layer_id][e].to(hidden.device),
            )
            dn = _dequant_ref(
                sources["down_packed"][layer_id][e].to(hidden.device),
                sources["down_scale"][layer_id][e].to(hidden.device),
                sources["down_global"][layer_id][e].to(hidden.device),
            )
            h = torch.relu(up @ x[t])
            out[t] += float(topk_weights[t, j]) * (dn @ (h * h))
    return out


# Nemotron-3.5-Lightning routes top-6; the fixture geometry stays small (E=8) so the
# dense dequant reference is cheap, ids just repeat experts within a token.
TOPK6 = 6
# The real Lightning moe_intermediate_size; the selection matrix needs a width above
# _b12x_min_intermediate() (1024), which the tiny fixture I == 128 is not.
LIGHTNING_I = 1856

_UNGATED_CFG = types.SimpleNamespace(
    hidden_size=H, moe_intermediate_size=I, hidden_act="relu2", expert_gated=False
)


def _b12x_ungated_banks(device, seed):
    """Pack ungated relu2 sources for b12x and return (layer-0 GPU bank views, alphas,
    untouched reference sources). The repack is in place, so the reference is cloned
    first; the banks are moved to the GPU directly (no slot cache) because both regimes
    under test read one layer's full [E] block."""
    from freetoken.moe.nvfp4_backends import b12x_repack_sources_inplace

    sources = _make_ungated_sources(seed)
    ref_sources = {k: [t.clone() for t in v] for k, v in sources.items()}
    packed = b12x_repack_sources_inplace(sources, _UNGATED_CFG, device, chunk=6)
    views = tuple(
        packed[name][0].to(device)
        for name in ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale")
    )
    return views, (packed["gate_up_alpha"][:E], packed["down_alpha"][:E]), ref_sources


def _skip_unless_b12x(device) -> None:
    from freetoken.moe.nvfp4_backends import _b12x_unusable_reason

    reason = _b12x_unusable_reason(torch.cuda.get_device_capability(device))
    if reason is not None:
        pytest.skip(f"b12x not runnable here: {reason}")


@cuda
def test_b12x_relu2_decode_matches_dequant_reference():
    """M=1 decode over ungated ReLU^2 experts: the b12x fused kernel vs the dequant
    reference. Guards the ungated pack (no [gate,up] half-swap, w13_rows == I) and the
    activation= plumbing into prepare/launch."""
    from freetoken.moe.nvfp4_backends import b12x_fused_experts

    device = torch.device("cuda")
    _skip_unless_b12x(device)
    views, (g1, g2), ref_sources = _b12x_ungated_banks(device, seed=21)

    torch.manual_seed(21)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    ids = torch.tensor([[3, 5, 0, 7, 1, 5]], dtype=torch.int32, device=device)
    weights = torch.rand(1, TOPK6, dtype=torch.float32, device=device)
    ref = _ref_moe_relu2(ref_sources, 0, hidden, weights, ids)

    gu_p, gu_s, dn_p, dn_s = views
    out = b12x_fused_experts(
        hidden, gu_p, gu_s, g1, dn_p, dn_s, g2, weights, ids, "relu2", False
    )
    _assert_close(out, ref)


@cuda
def test_b12x_relu2_prefill_matches_dequant_reference():
    """M=64 prefill over the same ungated banks (the tensor-core regime), with routing
    ids spread over every expert."""
    from freetoken.moe.nvfp4_backends import b12x_fused_experts

    device = torch.device("cuda")
    _skip_unless_b12x(device)
    views, (g1, g2), ref_sources = _b12x_ungated_banks(device, seed=22)

    torch.manual_seed(22)
    M = 64
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    ids = torch.randint(0, E, (M, TOPK6), dtype=torch.int32, device=device)
    weights = torch.rand(M, TOPK6, dtype=torch.float32, device=device)
    ref = _ref_moe_relu2(ref_sources, 0, hidden, weights, ids)

    gu_p, gu_s, dn_p, dn_s = views
    out = b12x_fused_experts(
        hidden, gu_p, gu_s, g1, dn_p, dn_s, g2, weights, ids, "relu2", False
    )
    _assert_close(out, ref)


@cuda
def test_b12x_relu2_matches_triton_backend():
    """The two GPU backends over identical ungated relu2 banks must agree: b12x's fused
    epilogue vs the Triton inline-dequant kernels' separate relu2 op (which the same
    banks feed unrepacked). Cross-checks the pack against a kernel, not just a reference."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.nvfp4_backends import b12x_fused_experts, b12x_repack_sources_inplace

    device = torch.device("cuda")
    _skip_unless_b12x(device)
    sources = _make_ungated_sources(seed=23)
    native = tuple(
        sources[name][0].to(device)
        for name in (
            "gate_up_packed", "gate_up_scale", "gate_up_global",
            "down_packed", "down_scale", "down_global",
        )
    )
    torch.manual_seed(23)
    M = 8
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    ids = torch.randint(0, E, (M, TOPK6), dtype=torch.int32, device=device)
    weights = torch.rand(M, TOPK6, dtype=torch.float32, device=device)
    triton_out = fused_experts_nvfp4(
        hidden, *native, weights, ids, E, "relu2", False
    )

    packed = b12x_repack_sources_inplace(sources, _UNGATED_CFG, device, chunk=8)
    views = tuple(
        packed[name][0].to(device)
        for name in ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale")
    )
    b12x_out = b12x_fused_experts(
        hidden, views[0], views[1], packed["gate_up_alpha"][:E],
        views[2], views[3], packed["down_alpha"][:E], weights, ids, "relu2", False,
    )
    cos = torch.nn.functional.cosine_similarity(
        b12x_out.float().flatten(), triton_out.float().flatten(), dim=0
    )
    assert float(cos) > 0.999, f"b12x vs triton cosine {float(cos)}"


def test_nvfp4_backend_selection_activation_matrix(monkeypatch):
    """The relu2 selection matrix, driven off a faked capability so it runs on any host:
    b12x on sm_120 (ungated relu2 is a b12x-fused activation), Triton elsewhere, and a
    forced marlin/relu2 rejected loudly (its epilogue is gated silu only)."""
    from freetoken.moe import nvfp4_backends as nb

    cuda_dev = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (12, 0))
    monkeypatch.setattr(nb, "_b12x_unusable_reason", lambda cc: None)
    # sm_120 + a wide enough MoE -> b12x fuses relu2 itself (Lightning's I == 1856).
    assert nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "auto", activation="relu2") == "b12x"
    assert nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "flashinfer", activation="relu2") == "b12x"
    # narrow MoE keeps the Triton M=1 GEMV, same as for silu
    assert nb.select_nvfp4_backend(cuda_dev, 512, "auto", activation="relu2") == "triton"
    # an activation neither borrowed kernel fuses still degrades to triton
    assert nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "auto", activation="swigluoai") == "triton"

    # sm_90 (marlin territory): marlin cannot do relu2, so auto must fall to triton and
    # a forced marlin must raise rather than silently mis-computing the epilogue.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    assert nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "auto", activation="relu2") == "triton"
    with pytest.raises(RuntimeError, match="relu2"):
        nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "marlin", activation="relu2")
    with pytest.raises(RuntimeError, match="swigluoai"):
        nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "flashinfer", activation="swigluoai")


def test_nvfp4_backend_selection_decode_target(monkeypatch):
    """cpu/hybrid decode reads experts out of the native ModelOpt rows, so the layout is
    pinned to triton -- auto degrades, a forced GPU backend raises."""
    from freetoken.moe import nvfp4_backends as nb

    cuda_dev = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (12, 0))
    monkeypatch.setattr(nb, "_b12x_unusable_reason", lambda cc: None)
    assert nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "auto", decode_target="gpu") == "b12x"
    for target in ("cpu", "hybrid"):
        assert nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "auto", decode_target=target) == "triton"
        assert nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "triton", decode_target=target) == "triton"
        with pytest.raises(RuntimeError, match=target):
            nb.select_nvfp4_backend(cuda_dev, LIGHTNING_I, "flashinfer", decode_target=target)


# ------------------------------------------- real Lightning layer (gated on weights)

_LIGHTNING_PATH_ENV = "FREETOKEN_NEMOTRON_LIGHTNING_PATH"


def _lightning_layer_banks(device, layer_index: int = 0, num_experts: int | None = None):
    """Load ONE real Nemotron-3.5-Lightning MoE layer's expert banks (native NVFP4).

    Reads the routed-expert tensors for that layer straight out of the checkpoint shards
    (``models/nemotron_h/weight.py``'s key regex + ``_alloc_nvfp4_host_banks(gated=False)``
    bank shapes), rather than ``load_nvfp4_expert_source_banks``, which would materialize
    all 23 MoE layers (~15.4 GiB)."""
    import glob
    import json
    import os
    from pathlib import Path

    import safetensors

    from freetoken.models.nemotron_h.config import parse_config
    from freetoken.models.nemotron_h.weight import _EXPERT_KEY_RE

    raw = os.environ.get(_LIGHTNING_PATH_ENV)
    if not raw or not Path(raw).is_dir():
        pytest.skip(f"set {_LIGHTNING_PATH_ENV} to a local Nemotron-3.5-Lightning checkpoint")
    path = Path(raw)

    from transformers import AutoConfig

    config = parse_config(AutoConfig.from_pretrained(str(path)))
    layer_id = config.moe_layer_ids[layer_index]
    e = config.num_experts if num_experts is None else num_experts
    h, i = config.expert_hidden_size, config.moe_intermediate_size

    banks = {
        "gate_up_packed": torch.empty((e, i, h // 2), dtype=torch.uint8),
        "gate_up_scale": torch.empty((e, i, h // 16), dtype=torch.float8_e4m3fn),
        "gate_up_global": torch.empty((e, i), dtype=torch.float16),
        "down_packed": torch.empty((e, h, i // 2), dtype=torch.uint8),
        "down_scale": torch.empty((e, h, i // 16), dtype=torch.float8_e4m3fn),
        "down_global": torch.empty((e, h), dtype=torch.float16),
    }
    role = {"up_proj": "gate_up", "down_proj": "down"}
    seen: set[tuple[int, str, str]] = set()
    index = path / "model.safetensors.index.json"
    if index.is_file():
        with open(index, encoding="utf-8") as f:
            files = sorted({v for v in json.load(f)["weight_map"].values()})
        shards = [str(path / f) for f in files]
    else:
        shards = sorted(glob.glob(str(path / "*.safetensors")))
    for shard in shards:
        with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                m = _EXPERT_KEY_RE.match(key)
                if m is None or int(m["layer"]) != layer_id:
                    continue
                expert = int(m["expert"])
                if expert >= e:
                    continue
                proj, kind = role[m["proj"]], m["kind"]
                t = f.get_tensor(key)
                if kind == "weight":
                    banks[f"{proj}_packed"][expert] = t
                elif kind == "weight_scale":
                    banks[f"{proj}_scale"][expert] = t
                else:  # weight_scale_2: one scalar per expert, broadcast per output row
                    banks[f"{proj}_global"][expert] = t.to(torch.float16)
                seen.add((expert, proj, kind))
    assert len(seen) == e * 2 * 3, f"missing expert tensors for layer {layer_id}: {len(seen)}"
    return config, {name: [t] for name, t in banks.items()}


@pytest.mark.needs_weights
@cuda
def test_b12x_relu2_real_lightning_layer_matches_triton():
    """One REAL Nemotron-3.5-Lightning MoE layer (H=2688, I=1856, E=128, top-6) through
    both GPU backends. Also asserts the b12x pack precondition the plan flagged as a
    checkpoint risk: the per-expert down_proj global must be row-constant."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.nvfp4_backends import b12x_fused_experts, b12x_repack_layer

    device = torch.device("cuda")
    _skip_unless_b12x(device)
    config, sources = _lightning_layer_banks(device)
    e = config.num_experts
    dn_g = sources["down_global"][0]
    assert torch.all(dn_g == dn_g[:, :1]), (
        "b12x needs a row-constant per-expert down_proj global scale; this checkpoint "
        "layer varies it per output row"
    )
    gu_g = sources["gate_up_global"][0]
    assert torch.all(gu_g == gu_g[:, :1])

    torch.manual_seed(31)
    m = 8
    top_k = config.num_experts_per_tok
    hidden = torch.randn(m, config.expert_hidden_size, dtype=torch.bfloat16, device=device) / 4
    ids = torch.randint(0, e, (m, top_k), dtype=torch.int32, device=device)
    weights = torch.rand(m, top_k, dtype=torch.float32, device=device)

    native = tuple(
        sources[name][0].to(device)
        for name in (
            "gate_up_packed", "gate_up_scale", "gate_up_global",
            "down_packed", "down_scale", "down_global",
        )
    )
    triton_out = fused_experts_nvfp4(hidden, *native, weights, ids, e, "relu2", False).float()
    del native
    torch.cuda.empty_cache()

    # in-place repack of the host banks, then one layer's blocks back onto the GPU
    post, gu_alpha, dn_alpha = b12x_repack_layer(
        {k: v[0] for k, v in sources.items()}, config, device, chunk=16
    )
    views = tuple(
        post[name].to(device)
        for name in ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale")
    )
    b12x_out = b12x_fused_experts(
        hidden, views[0], views[1], gu_alpha, views[2], views[3], dn_alpha,
        weights, ids, "relu2", False,
    ).float()
    cos = torch.nn.functional.cosine_similarity(
        b12x_out.flatten(), triton_out.flatten(), dim=0
    )
    assert float(cos) > 0.999, f"b12x vs triton cosine on a real layer: {float(cos)}"
