"""Prefill under the bounded mirror: assembled from residency, never from disk.

The branch this file guards first ran mirrored prefill through
``materialize_layer``, which reinstalls a layer into the LRU slots and
invalidates every other resident. That emptied the mirror, so coverage had to
be rebuilt from the checkpoint at every prefill->decode transition -- 1923 +
1021 + 739 rows, 19.3 GiB, measured as 8.0 s of TTFT on an 8K prompt whose
baseline prefill is 0.34 s, and 74.7 s at 80K against the baseline's 9.8 s.

The coverage invariant already says a prefill layer never needs the disk: every
expert is a GPU resident or has a pool row, so the double buffer can be filled
from those two places. These tests hold that line -- the buffer region belongs
to prefill alone, the assembly reads no checkpoint bytes, and a full sweep
leaves coverage standing.
"""
from __future__ import annotations

import tempfile
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mirrored prefill needs CUDA"
)


@pytest.fixture(autouse=True)
def _expert_arena():
    """Turn the expert arena on for these tests, and put it back after.

    The mirror is an arena-path feature (see attach_mirror_pool). A module
    level ``os.environ.setdefault`` does not survive here: sibling modules flip
    the module attribute directly (test_expert_arena_vmm,
    test_offload_usable_slots_gpu), so whether the gate is on depends on test
    order -- and mutating the environment instead leaks the gate into tests
    that must run ungated, which is how this file first broke test_offload.
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
from freetoken.moe.mirror_pool import MirrorExpertPool, prefill_buffer_slots
from freetoken.moe.offload_cache import OffloadMoeCache

from tests.moe._mirror_checkpoint import write_nvfp4_checkpoint

LAYERS, EXPERTS, H, ISZ = 6, 8, 32, 32
TOTAL = LAYERS * EXPERTS            # 48 expert rows
_GPU = 32                           # decode residents; the buffer region is
                                     # no longer excluded from this count
_STEP = 4
# Complement (48 - 32) plus two layers of reserve. Two, not the production
# three: with no victim floor, BOTH prefill buffer halves can need a full
# writeback burst back-to-back at the start of a chunk (layer 0 into buffer
# 0, layer 1 into buffer 1), so the reserve must absorb 2*EXPERTS here, not
# the one layer a decode-only burst needs. Production's 3*num_experts covers
# this with room to spare; this toy size just needs the arithmetic to match.
_RESERVE = 2 * EXPERTS
_CAP = (TOTAL - (_GPU - prefill_buffer_slots(EXPERTS))) + _RESERVE


def _cache(root):
    pool = MirrorExpertPool(
        root, LAYERS, EXPERTS, _CAP, hidden_size=H, intermediate_size=ISZ,
        spec=NEMOTRON_SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=_RESERVE, device=torch.device("cuda"),
    )
    cache = OffloadMoeCache(
        num_layers=LAYERS, num_experts=EXPERTS, cache_size=_GPU,
        slot_capacity=_GPU, arena_step_slots=_STEP,
        device=torch.device("cuda"), quant_format="nvfp4", cache_policy="lfu",
        prefill_overlap=True,
    )
    cache.direct_device_banks = True
    cache.attach_mirror_pool(pool)
    cache.mirror_warm_start()
    return cache, pool


def _bare_cache(root):
    """Attach without warm start: every ownership/pool-row map starts at -1.

    Gives the writeback tests a known-empty slate to seat exact scenarios
    on (a retained duplicate vs. a sole GPU copy) without fighting whatever
    ``mirror_warm_start``'s even spread across layers happened to seat in
    the buffer region.
    """
    pool = MirrorExpertPool(
        root, LAYERS, EXPERTS, _CAP, hidden_size=H, intermediate_size=ISZ,
        spec=NEMOTRON_SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=_RESERVE, device=torch.device("cuda"),
    )
    cache = OffloadMoeCache(
        num_layers=LAYERS, num_experts=EXPERTS, cache_size=_GPU,
        slot_capacity=_GPU, arena_step_slots=_STEP,
        device=torch.device("cuda"), quant_format="nvfp4", cache_policy="lfu",
        prefill_overlap=True,
    )
    cache.direct_device_banks = True
    cache.attach_mirror_pool(pool)
    return cache, pool


def _assert_same(got, want, what):
    """Compare raw bytes: float8_e4m3 carries NaN patterns torch.equal rejects."""
    g = got.cpu().contiguous().view(torch.uint8)
    w = want.cpu().contiguous().view(torch.uint8)
    assert torch.equal(g, w), f"{what} does not hold its own weights"


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


def _sweep(cache, check=None):
    """One prefill sweep, in _prefill_routed's overlap choreography."""
    cache.begin_prefill()
    for layer in range(LAYERS):
        cache.prefetch_prefill_layer(layer)
        cache.prefetch_prefill_layer(layer + 1)
        views = cache.wait_prefill_layer(layer)
        if check is not None:
            torch.cuda.synchronize()
            check(layer, views)
        cache.release_prefill_layer(layer)
    torch.cuda.synchronize()


def test_warm_start_seats_the_whole_arena():
    """No slots held back for the buffer: decode gets every arena slot.

    An earlier design fenced the buffer region ([0, 2E)) out of decode with a
    victim floor, priced at 256 of 2173 arena slots on Nemotron for no
    coverage benefit once the writeback exists. Warm start now seats
    residents from slot 0, exactly like the rest of the cache.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cache, pool = _cache(root)
        try:
            base = cache._mirror_prefill_base()
            assert base == prefill_buffer_slots(EXPERTS) == 2 * EXPERTS
            # _GPU == TOTAL - complement, and warm start has nothing left to
            # hold back, so every slot -- including the buffer region -- is a
            # resident.
            assert int((cache.id_of_slot >= 0).sum()) == _GPU
            assert (cache.id_of_slot[:base] >= 0).any(), (
                "the buffer region should hold residents like any other slot"
            )
            slots = cache.slot_for_id.view(-1)
            live = slots[slots >= 0]
            assert int(live.numel()) == _GPU
        finally:
            pool.close()


def test_a_prefill_layer_is_assembled_without_reading_the_checkpoint():
    """The whole point: coverage means the disk is never needed for prefill."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        golden = _golden(root)
        cache, pool = _cache(root)
        try:
            reads: list[int] = []
            original = pool._read_row

            def counting(flat, row):
                reads.append(flat)
                return original(flat, row)

            pool._read_row = counting

            def check(layer, views):
                base_flat = layer * EXPERTS
                for name, view in zip(cache.bank_schema, views):
                    for e in range(EXPERTS):
                        # Compare BYTES, not values: these banks hold NVFP4
                        # payload reinterpreted as float8_e4m3/float16, so some
                        # rows carry NaN bit patterns and torch.equal would
                        # report every one of them as a mismatch. The pool's
                        # banks are pinned HOST rows, so the check is on CPU.
                        want = golden[base_flat + e][name]
                        got = view[e]
                        assert got.shape == want.shape, (
                            f"layer {layer} expert {e} bank {name}: buffer row "
                            f"{tuple(got.shape)} vs checkpoint {tuple(want.shape)}"
                        )
                        want_b = want.cpu().contiguous().view(torch.uint8)
                        got_b = got.cpu().contiguous().view(torch.uint8)
                        assert torch.equal(got_b, want_b), (
                            f"layer {layer} expert {e} bank {name} is not the "
                            f"checkpoint's bytes"
                        )

            _sweep(cache, check)
            assert reads == [], (
                f"mirrored prefill read {len(reads)} rows from the checkpoint; "
                f"every expert is a resident or has a pool row, so the correct "
                f"count is zero"
            )
        finally:
            pool._read_row = original
            pool.close()


def test_a_prefill_sweep_leaves_coverage_standing():
    """No boundary restore: the sweep never breaks what it would repair."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cache, pool = _cache(root)
        try:
            _sweep(cache)
            assert not cache._mirror_needs_coverage, (
                "a sweep that needs the coverage restore is a sweep that "
                "emptied the mirror -- the 19.3 GiB re-read is back"
            )
            base = cache._mirror_prefill_base()
            slots = cache.slot_for_id.view(-1).cpu().tolist()
            rows = cache._mirror["pool_row_of_id"].cpu().tolist()
            orphans = [flat for flat in range(TOTAL)
                       if slots[flat] < base and rows[flat] < 0]
            assert orphans == [], (
                f"{len(orphans)} experts are neither resident nor mirrored "
                f"after the sweep, e.g. {orphans[:8]}"
            )
            cache.mirror_fault_check()
        finally:
            pool.close()


def test_decode_may_admit_into_the_prefill_buffer_and_prefill_still_covers():
    """The exact hole an earlier design closed with a victim floor.

    That floor fenced decode out of the buffer region entirely (256 of 2173
    arena slots on Nemotron, held back for no coverage benefit once a
    writeback exists): without it, or without the writeback,
    ``_prefetch_split_mirror`` would overwrite a decode resident's only copy
    and the server died at the first real request with "expert 73 (layer 0)
    is neither a GPU resident nor in the pool". The fix keeps the floor gone
    and makes coverage survive the overwrite instead: this run drives
    admissions into the buffer region on purpose, then runs a full prefill
    sweep over the mess and checks nothing was lost.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cache, pool = _cache(root)
        try:
            base = cache._mirror_prefill_base()
            torch.manual_seed(0)
            saw_buffer_admission = False
            for step in range(60):
                layer = step % LAYERS
                ids = torch.randperm(EXPERTS, device="cuda")[:4]
                ids = ids.to(torch.int32).reshape(1, 4)
                cache.ensure_experts(layer, ids)   # rewrites ids to slot ids
                cache.copy_missing()
                torch.cuda.synchronize()
                if int(ids.min()) < base:
                    saw_buffer_admission = True
                cache.mirror_fault_check()
            assert saw_buffer_admission, (
                "the buffer region was never a candidate in 60 steps -- this "
                "run no longer exercises what it is meant to test"
            )
            # And the prefill path still works on the cache decode left behind,
            # writing back whatever decode seated in the buffer region.
            _sweep(cache)
            cache.mirror_fault_check()
            slots = cache.slot_for_id.view(-1).cpu().tolist()
            rows = cache._mirror["pool_row_of_id"].cpu().tolist()
            assert [f for f in range(TOTAL)
                    if slots[f] < base and rows[f] < 0] == []
        finally:
            pool.close()


def test_writeback_skips_retained_duplicates():
    """A buffer occupant that already has a pool row needs no D2H.

    The common case (measured: ~28% of evictions at the reserve this design
    is priced against are already free) is a retained duplicate, seeded by
    admission-time retention or by warm start's seed_duplicates. The
    writeback must recognise it and spend no PCIe traffic on it.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cache, pool = _bare_cache(root)
        try:
            E = EXPERTS
            m = cache._mirror
            slot_for_id = cache.slot_for_id.view(-1)
            flats = list(range(E))  # layer 0's experts, one per buffer-0 slot
            for slot, flat in enumerate(flats):
                cache.id_of_slot[slot] = flat
                slot_for_id[flat] = slot
                row = flat  # arbitrary distinct pool rows
                pool._read_row(flat, row)
                m["pool_row_of_id"][flat] = row
                m["id_of_pool_row"][row] = flat
            cache._mirror_publish_free_rows()
            free_before = int(m["free_count"].item())

            cache._invalidate_prefill_buffer(0)
            torch.cuda.synchronize()

            assert int(m["stats"][2].item()) == 0, "retained duplicates need no D2H"
            # Slot 6, not slot 1: a buffer invalidation is not a decode
            # admission, and counting it in slot 1 (whose denominator is
            # `swaps`) drove free_eviction_rate above 1.0.
            assert int(m["stats"][6].item()) == E, "every occupant should free-evict"
            assert int(m["stats"][1].item()) == 0, (
                "a buffer invalidation must not touch the decode free-eviction "
                "counter -- that is what made free_eviction_rate exceed 1.0"
            )
            assert (cache.id_of_slot[:E] == -1).all()
            assert int(slot_for_id[torch.tensor(flats, device="cuda")].eq(-1).all())
            assert int(m["free_count"].item()) == free_before, (
                "no free rows should be consumed for duplicates"
            )
            cache.mirror_fault_check()
        finally:
            pool.close()


def test_writeback_preserves_a_sole_copys_bytes():
    """A buffer occupant with no pool row is written back, byte-exact.

    This is the case the victim floor used to avoid altogether: a decode
    resident whose only copy sits in the slot a prefill fill is about to
    overwrite. The writeback must land it in a free pool row holding its own
    checkpoint bytes, not a neighbour's -- the same check
    test_mirror_retention.py::test_a_retained_row_still_holds_its_own_expert
    makes for a decode eviction's retained row.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        golden = _golden(root)
        cache, pool = _bare_cache(root)
        try:
            E = EXPERTS
            m = cache._mirror
            slot_for_id = cache.slot_for_id.view(-1)
            flats = list(range(E))  # layer 0's experts, one per buffer-0 slot
            for slot, flat in enumerate(flats):
                cache.id_of_slot[slot] = flat
                slot_for_id[flat] = slot
                for name, bank in cache.bank_caches.items():
                    bank[slot].copy_(golden[flat][name].to(cache.device))
                # pool_row_of_id[flat] stays -1: this is a sole GPU copy.
            cache._mirror_publish_free_rows()
            free_before = int(m["free_count"].item())
            assert free_before >= E, "the reserve must cover a whole buffer half"

            cache._invalidate_prefill_buffer(0)
            torch.cuda.synchronize()

            assert int(m["stats"][2].item()) == E, f"expected {E} writebacks"
            assert int(m["stats"][3].item()) == 0, "no coverage violations"
            assert int(m["stats"][4].item()) == 0, "no starved writebacks"
            assert (cache.id_of_slot[:E] == -1).all()
            assert int(slot_for_id[torch.tensor(flats, device="cuda")].eq(-1).all())
            assert int(m["free_count"].item()) == free_before - E

            rows = m["pool_row_of_id"].cpu().tolist()
            inv = m["id_of_pool_row"].cpu().tolist()
            for flat in flats:
                row = rows[flat]
                assert row >= 0, f"expert {flat} lost its only copy"
                assert inv[row] == flat, (
                    f"expert {flat} claims row {row}, which the inverse map "
                    f"gives to {inv[row]}"
                )
                for name, bank in pool.banks.items():
                    _assert_same(bank[row], golden[flat][name],
                                 f"expert {flat} pool row {row} bank {name}")
            cache.mirror_fault_check()
        finally:
            pool.close()


def test_admission_into_the_buffer_region_forces_retention():
    """A slot < 2E always keeps its admission's source pool row.

    MEASURED regression this guards: without this, an admission that lands
    in the buffer region at the retention floor frees its source row like
    any other slot, so the very next prefill that overwrites this half pays
    a 5.36 MiB D2H to write it back -- TTFT at 8K went from 0.68s to 1.47s,
    entirely in pass 2 (after decode has populated the buffer). Forcing
    retention for buffer-region admissions means that D2H can never happen:
    the occupant is always a duplicate, so ``_writeback_buffer_kernel``
    always takes its free-evict branch.

    Drives ``resolve_swaps`` directly (like the ``_writeback_buffer_kernel``
    tests above drive their kernel) with the free stack set exactly AT the
    retention floor -- the one condition under which a non-buffer admission
    would free its source row -- to prove the buffer-region admission
    retains anyway.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cache, pool = _bare_cache(root)
        try:
            from freetoken.moe.mirror_kernels import publish_freed_rows, resolve_swaps

            E = EXPERTS
            m = cache._mirror
            slot_for_id = cache.slot_for_id.view(-1)
            base = cache._mirror_prefill_base()
            assert base == 2 * E

            target_slot = 3  # inside buffer 0's region ([0, E))
            assert target_slot < base
            flat_new = 0     # layer 0, expert 0 -- the admission under test
            src_row = 40     # an arbitrary unowned pool row standing in for
                              # "this expert's bytes already sit in the pool"

            # The admission: slot 3 was empty (no victim), flat_new arrives
            # from the pool. ensure_experts would normally have already
            # written id_of_slot/slot_for_id and staged prior/victim before
            # resolve_swaps runs -- reproduce exactly that contract by hand.
            cache.id_of_slot[target_slot] = flat_new
            slot_for_id[flat_new] = target_slot
            cache.evict_slots[0] = target_slot
            cache.src_indices[0] = flat_new % E
            cache.num_indices[0] = 1
            cache.victim_ids[0] = -1
            cache.prior_ids[0] = -1
            m["prev_slot_of_id"][flat_new] = -1
            m["pool_row_of_id"][flat_new] = src_row
            m["id_of_pool_row"][src_row] = flat_new

            # AT the floor: free_top == retain_floor, so a non-buffer
            # admission's `free_top > retain_floor` test is false and it
            # would free src_row. Only the `slot < buffer_slots` half of the
            # OR can save it here.
            retain_floor = pool.reserve_rows
            m["free_count"][0] = retain_floor

            stats_before = m["stats"].clone()
            resolve_swaps(cache, layer_id=0)
            torch.cuda.synchronize()

            assert int(m["n_freed"].item()) == 0, (
                "the buffer-region admission staged its source row for "
                "freeing -- forced retention did not fire"
            )
            assert int(m["pool_row_of_id"][flat_new].item()) == src_row, (
                "the admission's source row was not retained"
            )
            assert int(m["id_of_pool_row"][src_row].item()) == flat_new
            stats = m["stats"]
            assert int(stats[5].item()) - int(stats_before[5].item()) == 1, (
                "retained count should have gone up by exactly this admission"
            )
            assert int(stats[0].item()) - int(stats_before[0].item()) == 1, (
                "swaps count should reflect this one admission"
            )
            publish_freed_rows(cache)  # no-op: nothing was staged
            torch.cuda.synchronize()
            assert int(m["free_count"].item()) == retain_floor, (
                "publish must not have grown the free stack"
            )

            # Now invalidate buffer 0: the occupant this admission left in
            # slot 3 must be found already-mirrored (free_evict), not D2H'd.
            d2h_before = int(m["stats"][2].item())
            free_evict_before = int(m["stats"][6].item())
            decode_free_before = int(m["stats"][1].item())
            cache._invalidate_prefill_buffer(0)
            torch.cuda.synchronize()

            assert int(m["stats"][2].item()) == d2h_before, (
                "the forced-retention occupant should cost zero D2H on eviction"
            )
            assert int(m["stats"][6].item()) == free_evict_before + 1, (
                "the forced-retention occupant should free-evict"
            )
            assert int(m["stats"][1].item()) == decode_free_before, (
                "and it must not be counted as a decode free eviction"
            )
            assert cache.id_of_slot[target_slot].item() == -1
            assert int(slot_for_id[flat_new].item()) == -1
            cache.mirror_fault_check()
        finally:
            pool.close()


def test_the_mirror_refuses_the_ungated_admission_kernel():
    """Attaching without the arena must fail loudly, not serve wrong experts.

    Only the gated ``_v2`` kernel publishes victim_ids/prior_ids. On the
    ungated one they stay -1, so ``resolve_swaps`` believes no eviction ever
    displaces anybody, skips every writeback, and decode reads experts whose
    only copy is gone -- with coverage_faults still at 0, because the kernel
    was never told there was a victim to account for. This was silent until
    the gate flipped underneath these tests and a decode run lost an expert.
    """
    from freetoken.moe import offload_cache as oc
    from freetoken.moe.offload_cache import OffloadMoeCache

    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        pool = MirrorExpertPool(
            root, LAYERS, EXPERTS, _CAP, hidden_size=H, intermediate_size=ISZ,
            spec=NEMOTRON_SPEC,
            config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
            reserve_rows=_RESERVE, device=torch.device("cuda"),
        )
        try:
            prev = oc.FREETOKEN_EXPERT_ARENA
            oc.FREETOKEN_EXPERT_ARENA = False
            try:
                cache = OffloadMoeCache(
                    num_layers=LAYERS, num_experts=EXPERTS, cache_size=_GPU,
                    slot_capacity=_GPU, arena_step_slots=_STEP,
                    device=torch.device("cuda"), quant_format="nvfp4",
                    cache_policy="lfu", prefill_overlap=True,
                )
                cache.direct_device_banks = True
                with pytest.raises(ValueError, match="FREETOKEN_EXPERT_ARENA"):
                    cache.attach_mirror_pool(pool)
            finally:
                oc.FREETOKEN_EXPERT_ARENA = prev
        finally:
            pool.close()
