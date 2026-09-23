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
