"""Arena compaction before a shrink (moe/arena_compaction.py), CPU only.

The planner is a pure function; ``OffloadMoeCache.compact_before_shrink`` is run
on a CPU stand-in of the cache (real method, plain CPU tensors, per-row copy
path), so the map and byte bookkeeping is checked without a GPU. The CUDA arena
and mirror checks (fused copy, pool coverage, fault counters, graph replay) are
in test_arena_compaction_gpu.py.
"""

from __future__ import annotations

import random
import types

import pytest
import torch

from freetoken.moe.arena_compaction import plan_compaction
from freetoken.moe.offload_cache import OffloadMoeCache


def _apply(ids, plan):
    out = list(ids)
    for src, dst in plan.moves:
        out[dst] = ids[src]
        out[src] = -1
    return out


def test_whole_model_is_plain_lfu_compaction():
    # slots 0..3 double buffer (lo=4); survivors [4, 8), doomed [8, 12).
    ids = [0, 1, 2, 3, 10, 11, 12, -1, 20, 21, 22, -1]
    rank = [(0, 0)] * 4 + [(5, 1), (1, 1), (3, 1), (-1, -1), (9, 1), (2, 1), (0, 1), (-1, -1)]
    plan = plan_compaction(ids, rank, lambda _f: True, lo=4, n=8, current=12)
    # Hottest doomed (20, rank 9) takes the empty slot 7; 21 (rank 2) beats 11
    # (rank 1) in slot 5; 22 (rank 0) is colder than every survivor and drops.
    assert plan.moves == [(8, 7), (9, 5)]
    assert plan.displaced == [11]
    assert plan.doomed == 3 and plan.dropped_with_copy == 1 and plan.left_uncovered == 0
    assert all(dst >= 4 for _, dst in plan.moves), "the prefill double buffer is never a target"


def test_mirror_moves_gpu_only_first_and_never_displaces_one():
    ids = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]
    rank = [(0, 0)] * 4 + [(1, 1), (8, 1), (0, 1), (7, 1), (6, 1), (9, 1), (5, 1), (2, 1)]
    dup = {10, 12, 21, 23}          # experts with a pool row
    plan = plan_compaction(ids, rank, lambda f: f in dup, lo=4, n=8, current=12)
    # Targets: duplicate occupants only, coldest first -> slot 6 (12, rank 0),
    # slot 4 (10, rank 1). GPU-only 11 and 13 are never overwritten.
    # GPU-only movers first, hottest first: 20 (6), 22 (5).
    assert plan.moves == [(8, 6), (10, 4)]
    assert sorted(plan.displaced) == [10, 12]
    # 21 is a duplicate and hot (9) but no target is left: dropped for free.
    assert plan.dropped_with_copy == 2 and plan.left_uncovered == 0


def test_mirror_leaves_a_cold_gpu_only_expert_for_write_back():
    ids = [0, 1, 2, 3, 10, 11, 20, 21]
    rank = [(0, 0)] * 4 + [(9, 1), (8, 1), (1, 1), (0, 1)]
    plan = plan_compaction(ids, rank, lambda f: f in {10, 11}, lo=4, n=6, current=8)
    # Both movers are colder than both duplicates: overwriting a hot duplicate to
    # save one write-back would cost a hit, so nothing moves.
    assert plan.moves == []
    assert plan.left_uncovered == 2


@pytest.mark.parametrize("seed", range(20))
def test_plan_invariants_random(seed):
    rng = random.Random(seed)
    cap, lo = 64, 8
    n = rng.randrange(lo + 4, cap - 4)
    ids = [f if rng.random() < 0.85 else -1 for f in rng.sample(range(500), cap)]
    rank = [(rng.randrange(20), rng.randrange(1000)) if f >= 0 else (-1, -1) for f in ids]
    dup = {f for f in ids if f >= 0 and rng.random() < 0.5}
    for host_copy in (lambda _f: True, lambda f: f in dup):
        plan = plan_compaction(ids, rank, host_copy, lo=lo, n=n, current=cap)
        srcs = [s for s, _ in plan.moves]
        dsts = [d for _, d in plan.moves]
        assert len(set(srcs)) == len(srcs) and len(set(dsts)) == len(dsts)
        assert all(n <= s < cap and ids[s] >= 0 for s in srcs)
        assert all(lo <= d < n for d in dsts)
        # Only free-to-drop experts lose their GPU copy to a move.
        assert all(host_copy(f) for f in plan.displaced)
        after = _apply(ids, plan)
        kept = {f for f in after[:n] if f >= 0}
        # Nothing GPU-only disappears unaccounted: it is kept or left for write-back.
        lost = [f for f in ids if f >= 0 and not host_copy(f) and f not in kept]
        assert len(lost) == plan.left_uncovered
        assert plan.doomed == len(plan.moves) + plan.dropped_with_copy + plan.left_uncovered
    # Whole model: the survivors of [lo, n) are exactly the hottest residents of [lo, cap).
    plan = plan_compaction(ids, rank, lambda _f: True, lo=lo, n=n, current=cap)
    after = _apply(ids, plan)
    ranked = {f: rank[s] for s, f in enumerate(ids) if f >= 0}
    kept = [f for f in after[lo:n] if f >= 0]
    dropped = [f for f in (ids[s] for s in range(n, cap)) if f >= 0 and f not in kept]
    if kept and dropped:
        assert min(ranked[f] for f in kept) >= max(ranked[f] for f in dropped)


def _cpu_cache(num_layers=8, num_experts=4, cap=24, host_copy=None, prefill_overlap=True):
    """CPU stand-in carrying exactly what compact_before_shrink reads."""
    c = types.SimpleNamespace()
    c.num_layers, c.num_experts = num_layers, num_experts
    c.device = torch.device("cpu")
    c.prefill_overlap = prefill_overlap
    c._size_class_enabled = False
    c._class_arena_banks = []
    c.cache_policy_id = 1
    c.lfu_recency_tokens, c.lfu_recency_bonus = 0, 1
    c.step = torch.tensor(1000, dtype=torch.int64)
    total = num_layers * num_experts
    c.expert_frequency = torch.zeros((num_layers, num_experts), dtype=torch.int32)
    c.id_of_slot = torch.full((cap,), -1, dtype=torch.int32)
    c.slot_for_id = torch.full((num_layers, num_experts), -1, dtype=torch.int32)
    c.usage = torch.zeros((cap,), dtype=torch.int64)
    c.bank_caches = {
        "a": torch.zeros((cap, 32), dtype=torch.uint8),
        "b": torch.zeros((cap, 3, 16), dtype=torch.uint8),
    }
    c.residency = types.SimpleNamespace(host_copy_mask=lambda: host_copy)
    for name in ("compact_before_shrink", "_compaction_rank", "_copy_slot_rows"):
        setattr(c, name, types.MethodType(getattr(OffloadMoeCache, name), c))
    rng = random.Random(3)
    flats = rng.sample(range(total), cap - 3)
    for slot, flat in enumerate(flats):
        c.id_of_slot[slot] = flat
        c.slot_for_id.view(-1)[flat] = slot
        c.usage[slot] = rng.randrange(1, 1000)
        for bank in c.bank_caches.values():
            bank[slot].fill_(flat + 1)
    for flat in range(total):
        c.expert_frequency.view(-1)[flat] = rng.randrange(0, 50)
    return c


def _check_maps_and_bytes(c, n):
    ids = c.id_of_slot.tolist()
    fwd = c.slot_for_id.view(-1).tolist()
    for slot in range(n):
        flat = ids[slot]
        if flat < 0:
            continue
        assert fwd[flat] == slot
        for bank in c.bank_caches.values():
            assert torch.all(bank[slot] == flat + 1), f"slot {slot} lost expert {flat}'s bytes"
    for flat, slot in enumerate(fwd):
        if slot >= 0:
            assert ids[slot] == flat


@pytest.mark.parametrize("mirror", [False, True])
def test_compact_moves_bytes_and_maps_consistently(mirror):
    total = 32
    host_copy = [f % 2 == 0 for f in range(total)] if mirror else None
    c = _cpu_cache(host_copy=host_copy)
    before_ids = c.id_of_slot.tolist()
    before_usage = c.usage.tolist()
    n, current = 16, 24
    out = c.compact_before_shrink(n, current)
    assert out["moved"] > 0
    assert out["bytes"] == out["moved"] * (32 + 48)
    _check_maps_and_bytes(c, n)
    # Moved experts carry their usage; the vacated sources are empty. (The stand-in
    # does not run the shrink itself, so unmoved doomed experts are still there.)
    moved = 0
    for slot in range(n, current):
        new = int(c.slot_for_id.view(-1)[before_ids[slot]]) if before_ids[slot] >= 0 else -1
        if new >= 0 and new != slot:
            moved += 1
            assert new < n and int(c.usage[new]) == before_usage[slot]
            assert int(c.id_of_slot[slot]) == -1
    assert moved == out["moved"]
    # Under the mirror, a GPU-only expert that was below n stays there.
    if mirror:
        for slot in range(n):
            flat = before_ids[slot]
            if flat >= 0 and not host_copy[flat]:
                assert int(c.slot_for_id.view(-1)[flat]) == slot
    # Never into the prefill double buffer.
    for slot in range(2 * c.num_experts):
        assert int(c.id_of_slot[slot]) == before_ids[slot]


def test_compaction_off_by_env(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ARENA_COMPACTION", "0")
    c = _cpu_cache()
    before = c.id_of_slot.clone()
    assert c.compact_before_shrink(16, 24) == {}
    assert torch.equal(c.id_of_slot, before)


def test_totals_accumulate():
    c = _cpu_cache()
    c.compact_before_shrink(20, 24)
    c.compact_before_shrink(16, 20)
    assert c.compaction_totals["calls"] == 2
    assert c.compaction_totals["moved"] >= 1
