"""Pool capacity must buy something: retained duplicates, hence free evictions.

The shipped swap kernel freed the admitted expert's pool row unconditionally,
so every expert that reached the GPU immediately lost its host copy and its
next eviction paid a D2H forever. Only the duplicates seeded at warm start were
ever free to evict, and admissions consumed them one by one. Measured on
Nemotron (tasks/exclusive-expert-ram/results/sweep-memcurrent-superseded.tsv):
free evictions were 1.4% of swaps with an 1800-row pool and 16.9% with the
WHOLE 2944-row model mirrored. A knob that measures the same at 9.4 GiB and at
15.4 GiB of pinned RAM is not a knob, and that is why the first capacity sweep
came out flat.

Expert weights are read-only on the GPU, so the row an admission read from stays
a valid copy for as long as the expert is resident. Keeping it is what makes the
pool a gradient. These tests hold the two halves of that: the duplicates must
survive a long decode run, and they must still hold the right bytes.
"""
from __future__ import annotations

import tempfile
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the mirror swap kernel needs CUDA"
)


@pytest.fixture(autouse=True)
def _expert_arena():
    """See test_mirror_prefill: the gate is a module attribute, not an env var."""
    from freetoken.moe import offload_cache as oc
    from freetoken.moe import offload_kernels as ok

    prev = (oc.FREETOKEN_EXPERT_ARENA, ok.FREETOKEN_EXPERT_ARENA)
    oc.FREETOKEN_EXPERT_ARENA = True
    ok.FREETOKEN_EXPERT_ARENA = True
    try:
        yield
    finally:
        oc.FREETOKEN_EXPERT_ARENA, ok.FREETOKEN_EXPERT_ARENA = prev


from freetoken.models.nemotron_h.weight import (
    NVFP4_EXPERT_SOURCE_SPEC as NEMOTRON_SPEC,
)
from freetoken.moe.mirror_pool import MirrorExpertPool
from freetoken.moe.offload_cache import OffloadMoeCache

from tests.moe._mirror_checkpoint import write_nvfp4_checkpoint

LAYERS, EXPERTS, H, ISZ = 6, 8, 32, 32
TOTAL = LAYERS * EXPERTS      # 48 expert rows
_GPU = 20                     # decode residents; complement is 28
_STEP = 4
_RESERVE = EXPERTS            # one layer; production uses three
# The smallest pool coverage allows: complement + reserve, no room to duplicate
# anything. This is the control arm -- the capacity knob at its bottom stop.
_CAP_MIN = TOTAL - _GPU + _RESERVE          # 36 rows
_CAP_FULL = TOTAL                           # 48 rows: 12 rows spare to duplicate


def _cache(root, capacity):
    pool = MirrorExpertPool(
        root, LAYERS, EXPERTS, capacity, hidden_size=H, intermediate_size=ISZ,
        spec=NEMOTRON_SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=_RESERVE, device=torch.device("cuda"),
    )
    cache = OffloadMoeCache(
        num_layers=LAYERS, num_experts=EXPERTS, cache_size=_GPU,
        slot_capacity=_GPU, arena_step_slots=_STEP,
        device=torch.device("cuda"), quant_format="nvfp4", cache_policy="lfu",
    )
    cache.direct_device_banks = True
    cache.attach_mirror_pool(pool)
    cache.mirror_warm_start()
    return cache, pool


def _decode(cache, steps=240, seed=0):
    """A routed decode run with real eviction pressure (top-4 of 8, 6 layers)."""
    torch.manual_seed(seed)
    for step in range(steps):
        layer = step % LAYERS
        ids = torch.randperm(EXPERTS, device="cuda")[:4].to(torch.int32)
        cache.ensure_experts(layer, ids.reshape(1, 4))
        cache.copy_missing()
        cache.mirror_fault_check()
    torch.cuda.synchronize()


def _duplicates(cache):
    """Experts held BOTH on the GPU and in the pool."""
    slots = cache.slot_for_id.view(-1).cpu().tolist()
    rows = cache._mirror["pool_row_of_id"].cpu().tolist()
    return [f for f in range(TOTAL) if slots[f] >= 0 and rows[f] >= 0]


def test_a_bigger_pool_buys_free_evictions():
    """The measurement this branch exists to make: RAM must buy performance.

    A writeback is the D2H half of a swap; a free eviction is a swap that skips
    it because the victim already has a host copy. That rate IS the decode cost
    of the mirror, so it is what capacity has to move.
    """
    rates = {}
    dupes = {}
    for cap in (_CAP_MIN, _CAP_FULL):
        with tempfile.TemporaryDirectory() as root:
            write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
            cache, pool = _cache(root, cap)
            try:
                _decode(cache)
                st = cache.mirror_stats()
                assert st["swaps"] > 100, "the run must actually thrash"
                assert st["coverage_faults"] == 0
                assert st["starved_writebacks"] == 0
                rates[cap] = st["free_eviction_rate"]
                dupes[cap] = len(_duplicates(cache))
            finally:
                pool.close()

    # The bottom stop has no room to duplicate anything: its rate is whatever
    # empty slots give, and it must stay there.
    assert dupes[_CAP_MIN] == 0, (
        f"a minimum pool cannot hold duplicates, found {dupes[_CAP_MIN]}"
    )
    # 12 spare rows, and they must still be in use after 240 steps rather than
    # having decayed away -- decay is exactly what the shipped kernel did.
    assert dupes[_CAP_FULL] >= 8, (
        f"the spare rows decayed: {dupes[_CAP_FULL]} duplicates left of 12"
    )
    assert rates[_CAP_FULL] > rates[_CAP_MIN] + 0.25, (
        f"capacity bought nothing: free-eviction rate {rates[_CAP_MIN]:.3f} at "
        f"{_CAP_MIN} rows vs {rates[_CAP_FULL]:.3f} at {_CAP_FULL}"
    )


def test_a_retained_row_still_holds_its_own_expert():
    """Retention is only safe if a kept row is never handed to a writeback.

    The free stack is what feeds writeback destinations, so a retained row that
    leaked back into it would be overwritten with a DIFFERENT expert's weights
    and then served as the original. No counter would fire: the ownership maps
    would agree with themselves. Bytes are the only witness, and the checkpoint
    writes a deterministic function of the expert id precisely so they can be
    checked.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        golden = _golden(root)
        cache, pool = _cache(root, _CAP_FULL)
        try:
            _decode(cache)
            assert len(_duplicates(cache)) >= 8, "nothing was retained to check"

            rows = cache._mirror["pool_row_of_id"].cpu().tolist()
            inv = cache._mirror["id_of_pool_row"].cpu().tolist()
            slots = cache.slot_for_id.view(-1).cpu().tolist()

            for flat in range(TOTAL):
                assert slots[flat] >= 0 or rows[flat] >= 0, (
                    f"expert {flat} is neither resident nor mirrored"
                )
                if rows[flat] >= 0:
                    assert inv[rows[flat]] == flat, (
                        f"expert {flat} claims row {rows[flat]}, which the "
                        f"inverse map gives to {inv[rows[flat]]}"
                    )
                    for name, bank in pool.banks.items():
                        _assert_same(bank[rows[flat]], golden[flat][name],
                                     f"expert {flat} pool row {rows[flat]} "
                                     f"bank {name}")
        finally:
            pool.close()


def test_the_free_stack_never_starves_under_retention():
    """Retention consumes rows and nothing returns them, so it must stop.

    The floor is the pool's own reserve: above it an admission keeps its row,
    at it the row is released as before. Driving the stack to zero would drop a
    victim's only copy, which the kernel can only count -- so the count must
    stay at zero across a long run, and the stack must stay off the bottom.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cache, pool = _cache(root, _CAP_FULL)
        try:
            depths = []
            for chunk in range(12):
                _decode(cache, steps=40, seed=chunk)
                depths.append(int(cache._mirror["free_count"].item()))
            st = cache.mirror_stats()
            assert st["starved_writebacks"] == 0
            assert st["coverage_faults"] == 0
            assert st["retained_rows"] > 0, "retention never fired"
            assert min(depths) > 0, f"the free stack hit bottom: {depths}"
            # And it settles at the floor rather than draining or drifting up:
            # drifting up is the shipped behaviour (every row freed on
            # admission), which is the same as having no duplicates at all.
            assert max(depths[-4:]) <= _RESERVE + 4, (
                f"the stack drifted above the reserve: {depths}"
            )
        finally:
            pool.close()


def _golden(root):
    """Every expert row as the checkpoint has it, via a whole-model pool."""
    full = MirrorExpertPool(
        root, LAYERS, EXPERTS, TOTAL, hidden_size=H, intermediate_size=ISZ,
        spec=NEMOTRON_SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=0, device=torch.device("cuda"),
    )
    try:
        full.load_initial(set())
        return {flat: {n: full.banks[n][full.pool_row_of_id[flat]].clone()
                       for n in full.shapes}
                for flat in range(TOTAL)}
    finally:
        full.close()


def _assert_same(got, want, what):
    """Compare raw bytes: float8_e4m3 carries NaN patterns torch.equal rejects."""
    g = got.cpu().contiguous().view(torch.uint8)
    w = want.cpu().contiguous().view(torch.uint8)
    assert torch.equal(g, w), f"{what} does not hold its own weights"
