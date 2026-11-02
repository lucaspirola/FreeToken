"""The mirror pool's sizing estimate prices the SAME growable headroom as the ceiling
plan (commit 3b903d5's invariant, extended to the prefill transient).

``residency._mirror_final_gpu_slots`` sizes the bounded pool before the expert cache
exists, from the engine's pre-measurement ``prefill_transient_bytes``; the ceiling plan
(``GrowableKvController._plan_growable_kv``) later subtracts
``growable_headroom_bytes`` of the measured one. If the estimate subtracted less, the
pool would be sized for an arena bigger than the plan grants -- the Ornith 250K death
("coverage floor 4848, plan 4720") in a new costume. CPU-only: meta tensors, no pool.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine.cache_budget import expert_bytes_per_slot
from freetoken.engine.growable_kv import VMM_COMMIT_CUSHION_BYTES, growable_headroom_bytes
from freetoken.moe.mirror_pool import nvfp4_bank_shapes
from freetoken.moe.residency import MirrorResidency

LAYERS, EXPERTS, H, I = 2, 8, 32, 32


def _setup(monkeypatch, transient):
    import freetoken.kvcache.linear_state_pool as lsp

    total = LAYERS * EXPERTS
    per_slot = expert_bytes_per_slot({
        n: [torch.empty((EXPERTS, *tail), dtype=dt, device="meta")]
        for n, (tail, dt) in nvfp4_bank_shapes(H, I, gated=False).items()
    })
    mc = SimpleNamespace(
        expert_quant="nvfp4", hidden_size=H, expert_hidden_size=H,
        moe_intermediate_size=I, num_experts=EXPERTS, num_moe_layers=LAYERS,
        model_type="nemotron_h",
    )
    config = SimpleNamespace(model_config=mc, memory_ratio=1.0,
                             num_token_override=0, num_page_override=0, page_size=16)
    margin = max(4 * 8, -(-VMM_COMMIT_CUSHION_BYTES // per_slot))
    # Affords total - 4 slots after the bare cushion and the margin.
    budget = VMM_COMMIT_CUSHION_BYTES + (total - 4 + margin) * per_slot
    engine = SimpleNamespace(
        _pool_cls=SimpleNamespace(kv_cost=lambda config: (1, 0, 1, 0)),
        _baseline_free=budget, _weights_bytes=0,
        prefill_transient_bytes=transient,
    )
    monkeypatch.setenv("FREETOKEN_ARENA_STEP_SLOTS", "8")
    # The DMA-writeback staging ring is priced separately
    # (test_mirror_pool_golden); these identities are about the headroom term.
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_STAGE_MB", "0")
    monkeypatch.setattr(lsp, "state_pool_bytes", lambda config, num_slots=None: 0)
    return engine, config, per_slot, total


def test_estimate_prices_the_transient_exactly_like_the_ceiling_plan(monkeypatch):
    engine, config, per_slot, total = _setup(monkeypatch, transient=0)
    base = MirrorResidency._mirror_final_gpu_slots(engine, config)
    assert base == total - 4

    # A transient 3 slots above the cushion: the plan's headroom grows by 3 slots'
    # bytes, and so must the estimate's.
    engine.prefill_transient_bytes = VMM_COMMIT_CUSHION_BYTES + 3 * per_slot
    assert growable_headroom_bytes(engine.prefill_transient_bytes) - VMM_COMMIT_CUSHION_BYTES == 3 * per_slot
    assert MirrorResidency._mirror_final_gpu_slots(engine, config) == base - 3


def test_transient_below_the_cushion_changes_nothing(monkeypatch):
    engine, config, per_slot, total = _setup(monkeypatch, transient=0)
    base = MirrorResidency._mirror_final_gpu_slots(engine, config)
    engine.prefill_transient_bytes = VMM_COMMIT_CUSHION_BYTES // 2
    assert MirrorResidency._mirror_final_gpu_slots(engine, config) == base


def test_engine_without_an_estimate_prices_the_bare_cushion(monkeypatch):
    engine, config, _per_slot, total = _setup(monkeypatch, transient=0)
    del engine.prefill_transient_bytes
    assert MirrorResidency._mirror_final_gpu_slots(engine, config) == total - 4


# LFM2.5-24B-A2B EXL3 5.0 bpw: 38 MoE layers x 64 experts, ~5.63 MiB per expert slot.
LFM_LAYERS, LFM_EXPERTS, LFM_SLOT = 38, 64, 5_906_000


def _build_lfm25_auto_pool(monkeypatch, *, host_rows=-1, mem_available=64 << 30):
    """Run ``build_residency``'s auto path with the 2026-11-02 07:30 sizing (final arena
    estimate 1904 slots, so 720 auto rows) and return the capacity it built."""
    import freetoken.models.nvfp4_banks as nb
    import freetoken.moe.mirror_pool as mp
    import freetoken.moe.residency as res

    built = {}

    class _Pool:
        def __init__(self, path, layers, experts, capacity, **kwargs):
            built["capacity"] = capacity

    bank = {"rows": ((LFM_SLOT,), torch.uint8)}
    monkeypatch.setattr(mp, "MirrorExpertPool", _Pool)
    monkeypatch.setattr(mp, "nvfp4_bank_shapes", lambda *a, **k: bank)
    monkeypatch.setattr(nb, "expert_source_spec",
                        lambda mc: SimpleNamespace(gated=False, hidden_size_attr=None))
    monkeypatch.setattr(MirrorResidency, "_mirror_final_gpu_slots",
                        staticmethod(lambda engine, config: 1904))
    monkeypatch.setattr(res, "_mem_available_bytes", lambda: mem_available)
    monkeypatch.delenv("FREETOKEN_MIRROR_RESERVE_ROWS", raising=False)
    mc = SimpleNamespace(
        expert_quant="nvfp4", expert_gated=False, hidden_size=H, moe_intermediate_size=I,
        num_experts=LFM_EXPERTS, num_moe_layers=LFM_LAYERS, model_type="lfm2_moe",
    )
    config = SimpleNamespace(
        expert_residency="mirror", moe_mirror_host_rows=host_rows, moe_strategy="offload",
        moe_pageable_gpu=False, moe_cpu_layers=None, use_dummy_weight=False,
        tp_info=SimpleNamespace(size=1), nvfp4_backend="triton", model_path="/none",
        max_running_req=1,
    )
    engine = SimpleNamespace(model=SimpleNamespace(), device=torch.device("cpu"))
    res.build_residency(config, mc, engine)
    return built["capacity"]


def test_auto_pool_survives_the_0730_vram_swing(monkeypatch):
    """The refused start asked for at least 888 rows where the auto size gave 720: a
    GPU container took about 1.1 GiB between the pool sizing and the ceiling check."""
    monkeypatch.delenv("FREETOKEN_MIRROR_VRAM_MARGIN_MB", raising=False)
    capacity = _build_lfm25_auto_pool(monkeypatch)
    assert capacity >= 888
    # 1 GiB of 5.63 MiB slots, rounded up.
    assert capacity == 720 + -(-(1 << 30) // LFM_SLOT)


def test_vram_margin_is_tunable_and_explicit_rows_are_kept(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_VRAM_MARGIN_MB", "0")
    assert _build_lfm25_auto_pool(monkeypatch) == 720
    monkeypatch.delenv("FREETOKEN_MIRROR_VRAM_MARGIN_MB")
    assert _build_lfm25_auto_pool(monkeypatch, host_rows=720) == 720


def test_vram_margin_never_fails_the_start_on_host_ram(monkeypatch):
    """The margin rows stop at the host-RAM guard; the auto rows themselves still fit."""
    monkeypatch.delenv("FREETOKEN_MIRROR_VRAM_MARGIN_MB", raising=False)
    guard = 4 << 30
    room_rows = 750
    capacity = _build_lfm25_auto_pool(
        monkeypatch, mem_available=guard + room_rows * LFM_SLOT + LFM_SLOT // 2
    )
    assert capacity == room_rows
