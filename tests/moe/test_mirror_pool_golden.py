"""S12c step 1: a golden reference for the NVFP4 mirror-pool path, computed on
the UNMODIFIED code (before ``source=`` existed), pinned as constants here.

The S12c design changes ``MirrorExpertPool.__init__`` to accept an optional
``source`` and adds a GGUF source, but promises "no line change inside
``_scan_checkpoint``, ``_row_layout``, ``_read_row``, ``nvfp4_bank_shapes``,
``plan_capacity``" -- the five functions that give the NVFP4 path its byte
layout. This test is the check on that promise: it hashes what those five
functions produce (via ``load_initial``, which calls all of them) for the
ungated (Nemotron-like) and gated (qwen-like) synthetic checkpoints the rest
of ``tests/moe/test_mirror_*`` already use, plus ``_mirror_final_gpu_slots``'s
budget arithmetic (also touched, to source shapes from either format). If any
of the hashes below ever change, either a real behavior change happened here
and the constants must move with a stated reason, or S12c regressed the NVFP4
byte-identity promise.

Both checkpoints below are built with ``capacity == total`` and
``gpu_ids=set()`` (a fully-saturated pool): every row is written by
``load_initial``, so the bank-bytes hash has no torch.empty() padding in it
that would make the hash depend on incidental garbage instead of on the code.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.mirror_pool import MirrorExpertPool, plan_capacity
from freetoken.moe.residency import MirrorResidency

from tests.moe.test_mirror_pool import (
    EXPERTS, H, I, LAYERS,
    G_EXPERTS, G_H, G_I, G_LAYERS,
    NEMOTRON_SPEC, QWEN_SPEC,
    _write_checkpoint, _write_gated_checkpoint,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mirror pool needs CUDA for host_register"
)


def _records_digest(pool: MirrorExpertPool) -> str:
    """A hash of ``_scan_checkpoint``'s own output: every row's file offsets,
    lengths, bank, destination and broadcast flag, in flat-id order. The fd
    int itself is excluded -- it is a per-run OS handle, not part of the
    checkpoint layout _scan_checkpoint computed."""
    h = hashlib.sha256()
    for flat in sorted(pool._records):
        h.update(str(flat).encode())
        # One group per shard; single-shard records hash exactly as before groups existed.
        for _fd, pieces in pool._records[flat]:
            for off, length, bank, dst, broadcast in pieces:
                h.update(f"|{off},{length},{bank},{dst},{broadcast}".encode())
    return h.hexdigest()


def _banks_digest(pool: MirrorExpertPool) -> str:
    h = hashlib.sha256()
    for name in pool.schema_order:
        bank = pool.banks[name].contiguous().view(torch.uint8).cpu().numpy()
        h.update(name.encode())
        h.update(bank.tobytes())
    return h.hexdigest()


# Golden constants. Computed on exp/reorg (pre-S12c) code: `source` did not
# exist yet, so this is exactly `MirrorExpertPool(..., spec=..., config=...)`
# followed by `load_initial(set())`, unchanged by this branch's diff (verified:
# `git diff exp/reorg -- python/freetoken/moe/mirror_pool.py` touches only
# lines inside `__init__`, and every added line there is either gated on
# `source is not None` -- never taken on this path -- or an `if source is
# None:` wrapper around the original call).
UNGATED_RECORDS_SHA256 = "1bec95ae2d6599b65d1b67dd4df678661cf6bb871d241bc8c9541c6afbbc8242"
UNGATED_BANKS_SHA256 = "69c3fc6aee1ba254826c1ce297c0d9963cc6c022252fc28cbd122c537cc3d1bb"
GATED_RECORDS_SHA256 = "1e1edd7194bc1a6dbba879b9ebb7f5125985a213194c15a4fca75473b880b673"
GATED_BANKS_SHA256 = "338044057b71b8a44b4657e3e842fcfcfdd124752e2885bd2173959a388bdec1"


def test_ungated_nvfp4_pool_is_byte_identical(tmp_path):
    root = str(tmp_path)
    _write_checkpoint(root)
    total = LAYERS * EXPERTS
    pool = MirrorExpertPool(
        root, LAYERS, EXPERTS, total, hidden_size=H, intermediate_size=I,
        spec=NEMOTRON_SPEC,
        config=SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=0,
    )
    try:
        pool.load_initial(set())
        assert _records_digest(pool) == UNGATED_RECORDS_SHA256
        assert _banks_digest(pool) == UNGATED_BANKS_SHA256
    finally:
        pool.close()


def test_gated_nvfp4_pool_is_byte_identical(tmp_path):
    root = str(tmp_path)
    _write_gated_checkpoint(root)
    total = G_LAYERS * G_EXPERTS
    pool = MirrorExpertPool(
        root, G_LAYERS, G_EXPERTS, total, hidden_size=G_H, intermediate_size=G_I,
        spec=QWEN_SPEC, config=SimpleNamespace(), reserve_rows=0,
    )
    try:
        pool.load_initial(set())
        assert _records_digest(pool) == GATED_RECORDS_SHA256
        assert _banks_digest(pool) == GATED_BANKS_SHA256
    finally:
        pool.close()


def test_plan_capacity_is_unchanged():
    # Same assertions as test_mirror_pool.py's test_capacity_covers_the_kv_ceiling,
    # pinned here too since plan_capacity is one of the five NVFP4 functions this
    # module exists to guard.
    assert plan_capacity(23, 128, 1552) == (2944 - 1552) + 3 * 128
    assert plan_capacity(23, 128, 1552, reserve=0) == 2944 - 1552
    assert plan_capacity(23, 128, 2944) == 3 * 128


def test_mirror_final_gpu_slots_is_unchanged():
    """``_mirror_final_gpu_slots`` now branches on ``mc.expert_quant`` to source
    shapes from either format (S12c); the NVFP4 branch's arithmetic must give
    the same answer as before that branch existed."""
    from freetoken.engine.growable_kv import GrowableKvController  # noqa: F401  (import guard: engine module loads)

    fixed_cache_size = 0
    mc = SimpleNamespace(
        expert_quant="nvfp4",
        hidden_size=H,
        expert_hidden_size=H,
        moe_intermediate_size=I,
        num_experts=EXPERTS,
        num_moe_layers=LAYERS,
        model_type="nemotron_h",
        linear_attention_group=lambda: None,
    )
    config = SimpleNamespace(
        model_config=mc,
        memory_ratio=1.0,
        num_token_override=0,
        num_page_override=0,
        page_size=16,
    )
    engine = SimpleNamespace(
        _pool_cls=SimpleNamespace(kv_cost=lambda config: (1, fixed_cache_size, 1, 0)),
        _baseline_free=10_000_000_000,
        _weights_bytes=0,
    )
    slots = MirrorResidency._mirror_final_gpu_slots(engine, config)
    # Generous budget (10 GB against a tiny toy geometry) -> capped at `total`
    # (min(conservative, total) in the function): the whole point of this
    # assertion is that the NVFP4 branch still reaches that same cap through
    # the same arithmetic, not the exact number itself.
    assert slots == LAYERS * EXPERTS


def test_mirror_final_gpu_slots_prices_what_the_ceiling_plan_prices(monkeypatch):
    """The estimate that sizes the pool must subtract what
    ``GrowableKV._plan_growable_kv`` subtracts -- the linear-state pool and the
    VMM commit cushion -- or the pool is sized for an arena the ceiling plan
    never grants. Ornith (0.80 GiB of GatedDeltaNet state) died at 250K on
    exactly that: floor 4848, plan 4720."""
    import freetoken.kvcache.linear_state_pool as lsp
    from freetoken.engine.cache_budget import expert_bytes_per_slot
    from freetoken.engine.growable_kv import VMM_COMMIT_CUSHION_BYTES
    from freetoken.moe.mirror_pool import nvfp4_bank_shapes

    total = LAYERS * EXPERTS
    shapes = nvfp4_bank_shapes(H, I, gated=False)
    per_slot = expert_bytes_per_slot({
        n: [torch.empty((EXPERTS, *tail), dtype=dt, device="meta")]
        for n, (tail, dt) in shapes.items()
    })
    mc = SimpleNamespace(
        expert_quant="nvfp4", hidden_size=H, expert_hidden_size=H,
        moe_intermediate_size=I, num_experts=EXPERTS, num_moe_layers=LAYERS,
        model_type="nemotron_h",
    )
    config = SimpleNamespace(model_config=mc, memory_ratio=1.0,
                             num_token_override=0, num_page_override=0, page_size=16)
    # A budget that affords total - 4 slots after the cushion and the margin,
    # so neither the num_experts floor nor the total cap is what answers
    # (with 3 slots of state it still stays above num_experts).
    margin = max(4 * 8, -(-VMM_COMMIT_CUSHION_BYTES // per_slot))
    budget = VMM_COMMIT_CUSHION_BYTES + (total - 4 + margin) * per_slot
    engine = SimpleNamespace(
        _pool_cls=SimpleNamespace(kv_cost=lambda config: (1, 0, 1, 0)),
        _baseline_free=budget, _weights_bytes=0,
    )
    monkeypatch.setenv("FREETOKEN_ARENA_STEP_SLOTS", "8")
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_STAGE_MB", "0")
    monkeypatch.setattr(lsp, "state_pool_bytes", lambda config, num_slots=None: 0)
    base = MirrorResidency._mirror_final_gpu_slots(engine, config)
    assert base == total - 4
    assert base - 3 > EXPERTS
    state_slots = 3
    monkeypatch.setattr(lsp, "state_pool_bytes",
                        lambda config, num_slots=None: state_slots * per_slot)
    assert MirrorResidency._mirror_final_gpu_slots(engine, config) == base - state_slots
    # The DMA-writeback staging ring is VRAM the live arena fill sees: it
    # must come out of the estimate slot for slot (4 rows at this budget).
    from freetoken.moe.residency import wb_stage_rows

    monkeypatch.setattr(lsp, "state_pool_bytes", lambda config, num_slots=None: 0)
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_STAGE_MB", "0.0001")
    assert wb_stage_rows(per_slot) == 4
    # Four more slots of VRAM than `base` needed: all four go to the ring
    # (unpriced, the answer would be the `total` cap instead).
    roomier = SimpleNamespace(**{**vars(engine),
                                 "_baseline_free": budget + 4 * per_slot})
    assert MirrorResidency._mirror_final_gpu_slots(roomier, config) == base
