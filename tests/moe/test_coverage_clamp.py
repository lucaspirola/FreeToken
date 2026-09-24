"""The coverage clamp: KV growth may not shrink the expert arena below what the
bounded mirror can complement.

This is the mechanism that kept the 713K-token ceiling correct (the 600K run
died mid-request before it existed), so it gets a regression test of its own.

The geometry below is chosen to be NON-degenerate, which the first version of
this test was not: it used L=4, E=16, where ``plan_capacity`` clamps capacity to
the full 64 rows (a pool holding the whole model -- the one configuration that
saves no RAM and makes the bound trivial) and where ``total`` happens to equal
``2 * 2 * num_experts``. In that geometry the correct floor and an inverted one
(``capacity - 2E``) both evaluate to 32, so the test could not tell them apart
and locked in the inversion. Here ``capacity < total`` and the two formulas
disagree, which is the whole point.
"""
from __future__ import annotations

import tempfile
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="coverage clamp needs CUDA"
)


@pytest.fixture(autouse=True)
def _expert_arena():
    """Flip the gate where it is actually read: a module attribute.

    This file used to set FREETOKEN_EXPERT_ARENA in os.environ at import time,
    which only worked while it happened to be the first module importing
    ``offload_cache`` -- that module snapshots the variable into a module-level
    flag on ITS import. Run after any suite that imports it first (tests/kernels
    does), the setdefault was a no-op and every mirror test here died on
    ``attach_mirror_pool``. Patch the flags, like test_mirror_prefill.
    """
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
from freetoken.moe.mirror_pool import (
    MirrorExpertPool,
    default_reserve_rows,
    plan_capacity,
)

from tests.moe._mirror_checkpoint import write_nvfp4_checkpoint

LAYERS, EXPERTS, H, ISZ = 6, 8, 32, 32
TOTAL = LAYERS * EXPERTS          # 48 rows
_GPU = 30                         # GPU holds 30, so 18 rows must live in the pool
_STEP = 4                         # arena chunk granularity


def _pool(root, capacity):
    # The REAL Nemotron-H spec, as in test_mirror_pool: the checkpoint written
    # above is laid out the way that model's checkpoints are, and a test-local
    # copy of the spec would only prove the test agrees with itself.
    return MirrorExpertPool(root, LAYERS, EXPERTS, capacity,
                            hidden_size=H, intermediate_size=ISZ,
                            spec=NEMOTRON_SPEC,
                            config=types.SimpleNamespace(
                                moe_layer_ids=list(range(LAYERS))),
                            device=torch.device("cuda"))


def _cache_and_pool(root, capacity):
    from freetoken.moe.offload_cache import OffloadMoeCache

    pool = _pool(root, capacity)
    cache = OffloadMoeCache(
        num_layers=LAYERS, num_experts=EXPERTS, cache_size=_GPU,
        slot_capacity=_GPU, arena_step_slots=_STEP,
        device=torch.device("cuda"), quant_format="nvfp4", cache_policy="lfu",
    )
    cache.direct_device_banks = True
    cache.attach_mirror_pool(pool)
    cache.mirror_warm_start()
    return cache, pool


def test_geometry_is_not_degenerate():
    """Guard the guard: this file is worthless if the numbers collapse again."""
    cap = plan_capacity(LAYERS, EXPERTS, _GPU)
    reserve = default_reserve_rows(EXPERTS)
    assert cap < TOTAL, "capacity must be below the model size or the bound is trivial"
    correct = TOTAL - cap + reserve
    inverted = cap - 2 * EXPERTS          # the formula this test failed to catch
    assert correct != inverted, "geometry must distinguish the two formulas"


def test_floor_is_the_gpu_slot_count_coverage_needs():
    """min_gpu_slots = total - capacity + reserve, derived from the pool alone."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cap = plan_capacity(LAYERS, EXPERTS, _GPU)
        pool = _pool(root, cap)
        try:
            reserve = default_reserve_rows(EXPERTS)
            assert pool.reserve_rows == reserve
            assert pool.min_gpu_slots == TOTAL - cap + reserve
            # Auto-sizing is self-consistent: a pool planned for _GPU slots puts
            # its floor exactly at _GPU, never above it.
            assert pool.min_gpu_slots == _GPU
            # And it is NOT the inverted value that shipped.
            assert pool.min_gpu_slots != cap - 2 * EXPERTS
        finally:
            pool.close()


def test_a_bigger_pool_lowers_the_floor():
    """More host RAM must buy MORE room for KV, never less.

    The shipped arithmetic had this backwards (floor = capacity - 2E rises with
    capacity), which silently turned the RAM knob into a KV-ceiling knob.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        floors = []
        for cap in (TOTAL - 12, TOTAL - 6, TOTAL):
            pool = _pool(root, cap)
            try:
                floors.append(pool.min_gpu_slots)
            finally:
                pool.close()
        assert floors == sorted(floors, reverse=True), (
            f"floor must fall as the pool grows, got {floors}"
        )
        assert floors[0] > floors[-1], "a full-model pool must free the arena most"


def test_capacity_below_the_reserve_is_refused():
    """A pool that is all reserve covers nothing; say so at construction."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        with pytest.raises(ValueError, match=r"--moe-mirror-host-rows"):
            _pool(root, default_reserve_rows(EXPERTS))


def test_coverage_holds_at_the_floor_and_breaks_below_it():
    """The floor is tight: coverage survives exactly at it, not one chunk under."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cap = plan_capacity(LAYERS, EXPERTS, _GPU)
        cache, pool = _cache_and_pool(root, cap)
        try:
            floor = -(-pool.min_gpu_slots // _STEP) * _STEP
            coverable = pool.capacity - pool.reserve_rows
            # At the floor the complement still fits outside the reserve.
            assert pool.total - floor <= coverable
            # The kernel's own hard floor is unrelated and lower; a request
            # under it is refused before any coverage logic runs.
            with pytest.raises(ValueError, match=rf"below the floor {EXPERTS}"):
                cache.set_usable_slots(EXPERTS - _STEP)
            # set_usable_slots does not clamp -- _grow_runtime_kv_arena does,
            # by folding min_gpu_slots into its `floor`. Show what that clamp
            # is protecting: one chunk below, the complement no longer fits.
            assert pool.total - (floor - _STEP) > coverable
        finally:
            pool.close()


def test_a_lost_copy_is_a_hard_error_not_a_counter():
    """The fault counters must stop a request, not decorate /v1/stats.

    resolve_swaps cannot raise from inside a Triton kernel, so it counts. Until
    this check existed, nothing in the server ever compared those counts to
    zero: a request that lost coverage kept decoding with the wrong experts and
    the evidence sat inert in a diagnostic document.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cap = plan_capacity(LAYERS, EXPERTS, _GPU)
        cache, pool = _cache_and_pool(root, cap)
        try:
            stats = cache._mirror["stats"]
            # Clean state: the check is silent.
            stats.zero_()
            cache.mirror_fault_check()
            # An admission with no host copy (violations).
            stats[3] = 1
            with pytest.raises(RuntimeError, match=r"--moe-mirror-host-rows"):
                cache.mirror_fault_check()
            # A dropped writeback (starved) is equally fatal.
            stats.zero_()
            stats[4] = 7
            with pytest.raises(RuntimeError, match=r"7 dropped writebacks"):
                cache.mirror_fault_check()
            # The scheduler's check reads the pinned per-step snapshot.
            cache._mirror["stats_host"][4] = 7
            with pytest.raises(RuntimeError, match=r"7 dropped writebacks"):
                cache.residency.fault_check()
        finally:
            cache._mirror["stats"].zero_()
            pool.close()


def test_fault_counters_reach_the_host_without_a_sync():
    """The pinned snapshot must be wired to the device counters."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cap = plan_capacity(LAYERS, EXPERTS, _GPU)
        cache, pool = _cache_and_pool(root, cap)
        try:
            host = cache._mirror["stats_host"]
            assert host.is_pinned(), "an unpinned buffer would sync on every copy"
            assert host.shape == cache._mirror["stats"].shape
            cache._mirror["stats"][3] = 5
            # The per-step snapshot, on a side stream behind the compute stream.
            cache.residency.service_writebacks()
            torch.cuda.synchronize()
            assert host[3].item() == 5
        finally:
            cache._mirror["stats"].zero_()
            pool.close()
