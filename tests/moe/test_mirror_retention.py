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
    # Staged (DMA) writebacks reach the pool only when the host issues them;
    # the byte checks below read the pool directly.
    cache.residency.drain_writebacks()
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


# --- Lever 2: LFU victim tie-break toward candidates with a pool row -------
#
# These exercise ``_ensure_experts_sized_kernel_v2`` directly through a bare
# ``OffloadMoeCache`` (no ``MirrorExpertPool``): the kernel only ever reads
# ``cache._mirror["pool_row_of_id"]``, so a hand-built dict with just that key
# is a faithful, much smaller stand-in for engineering exact ties.

def _direct_cache(num_layers, num_experts, cache_size, device=torch.device("cuda")):
    """``cache_size`` must be >= ``num_experts`` (single-layer coverage floor);
    give it several layers so total ids (``num_layers * num_experts``) still
    exceed ``cache_size`` and evictions are possible."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    return OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=device, cache_policy="lfu",
    )


def _seat(cache, slot_of_expert: dict[int, int], usage: dict[int, int],
          frequency: dict[int, int]) -> None:
    """Directly engineer cache state: expert -> (slot, usage, frequency)."""
    for expert, slot in slot_of_expert.items():
        cache.id_of_slot[slot] = expert
        cache.slot_for_id[0, expert] = slot
        cache.usage[slot] = usage[expert]
        cache.expert_frequency[0, expert] = frequency[expert]


def _attach_fake_mirror(cache, pool_row_of_id) -> None:
    """Minimal stand-in for ``OffloadMoeCache.attach_mirror_pool``.

    ``ensure_experts`` only reads ``_mirror["pool_row_of_id"]`` (via the
    kernel) and ``_mirror["prev_slot_of_id"]`` (its own bookkeeping, unrelated
    to Lever 2) -- so this fakes just enough state to exercise the kernel
    without a real ``MirrorExpertPool``. (S2 deleted the batch-boundary
    coverage-restore flag ``ensure_experts`` used to read; it reads nothing
    else from the cache now.)
    """
    from freetoken.moe.residency import MirrorResidency

    device = cache.slot_for_id.device
    # A pool-less MirrorResidency bound to the cache: the two maps are all
    # ensure_experts (before_ensure) and the admission kernel ever read.
    residency = MirrorResidency(pool=None, cache=cache)
    residency._mirror = {
        "pool_row_of_id": torch.tensor(pool_row_of_id, dtype=torch.int32, device=device),
        "prev_slot_of_id": cache.slot_for_id.view(-1).clone(),
    }
    cache.residency = residency


def test_tiebreak_evicts_the_candidate_with_a_pool_row():
    """Equal frequency, equal usage, one candidate has a pool row: it goes.

    If the tie-break were absent (or inverted), the copy-less expert 1 would
    be an equally valid ``tl.argmin`` pick, or would be picked instead -- this
    fails either way unless the copied candidate (expert 0) is the one
    evicted.
    """
    # 2 layers x 4 experts = 8 flat ids sharing 4 slots (cache_size == num_experts
    # is the coverage floor for a single layer; the second layer is what makes
    # a miss -- and therefore an eviction -- possible at all).
    device = torch.device("cuda")
    cache = _direct_cache(num_layers=2, num_experts=4, cache_size=4, device=device)
    _seat(
        cache,
        slot_of_expert={0: 0, 1: 1, 2: 2, 3: 3},
        usage={0: 10, 1: 10, 2: 10, 3: 10},
        frequency={0: 5, 1: 5, 2: 50, 3: 50},
    )
    # flat id = layer * num_experts + expert; layer 0 experts occupy 0..3.
    _attach_fake_mirror(cache, [0, -1, -1, -1, -1, -1, -1, -1])
    # A miss on layer 1 (not resident anywhere) forces one eviction among the
    # tied layer-0 occupants.
    ids = torch.tensor([[0]], dtype=torch.int32, device=device)
    cache.ensure_experts(1, ids)
    torch.cuda.synchronize()

    n = int(cache.num_indices.item())
    assert n == 1
    evicted = int(cache.victim_ids[0].item())
    assert evicted == 0, (
        f"expected the pool-row-holding expert 0 evicted, got flat id {evicted}"
    )


def test_tiebreak_never_overrides_a_real_frequency_difference():
    """A genuinely colder copy-less expert is still evicted over a hot copy.

    Proves the pool-row bonus is one frequency bucket, not an absolute veto:
    expert 0 (no pool row, frequency 0) is far colder than expert 1 (pool row,
    frequency 100), so it must still be the one evicted despite losing the
    tie-break bucket.
    """
    device = torch.device("cuda")
    cache = _direct_cache(num_layers=2, num_experts=4, cache_size=4, device=device)
    _seat(
        cache,
        slot_of_expert={0: 0, 1: 1, 2: 2, 3: 3},
        usage={0: 10, 1: 10, 2: 10, 3: 10},
        frequency={0: 0, 1: 100, 2: 200, 3: 200},
    )
    _attach_fake_mirror(cache, [-1, 0, -1, -1, -1, -1, -1, -1])
    ids = torch.tensor([[0]], dtype=torch.int32, device=device)
    cache.ensure_experts(1, ids)
    torch.cuda.synchronize()

    evicted = int(cache.victim_ids[0].item())
    assert evicted == 0, (
        f"the +1 tie-break bucket must not beat a real 100-count gap, got flat id {evicted}"
    )


def test_no_mirror_attached_matches_pre_lever2_behaviour():
    """``HAS_MIRROR=False`` (no pool attached) must not change victim choice.

    Runs the same random step sequence on two caches -- one with no mirror
    object at all, one with a mirror attached but the tie-break knob forced
    off (``FREETOKEN_MIRROR_TIEBREAK=False``, i.e. ``HAS_MIRROR`` computed
    False despite the pool existing) -- and requires their full eviction
    trace and end state to agree byte-for-byte. If the ``if HAS_MIRROR:``
    gate leaked any effect when unset, this would drift.
    """
    import random

    from freetoken.moe import offload_kernels as ok

    device = torch.device("cuda")
    # 3 layers x 4 experts = 12 flat ids sharing 4 slots (cache_size ==
    # num_experts satisfies the single-layer coverage floor; the extra layers
    # give real churn to compare traces over).
    num_layers, num_experts, cache_size = 3, 4, 4
    rng = random.Random(4242)
    steps = [
        (rng.randrange(num_layers),
         sorted(rng.sample(range(num_experts), k=rng.randint(1, 3))))
        for _ in range(60)
    ]

    baseline = _direct_cache(num_layers, num_experts, cache_size, device)
    gated = _direct_cache(num_layers, num_experts, cache_size, device)
    _attach_fake_mirror(
        gated, [rng.choice([-1, 0]) for _ in range(num_layers * num_experts)]
    )

    prev = ok.FREETOKEN_MIRROR_TIEBREAK
    ok.FREETOKEN_MIRROR_TIEBREAK = False
    try:
        for layer_id, experts in steps:
            base_ids = torch.tensor([experts], dtype=torch.int32, device=device)
            gated_ids = torch.tensor([experts], dtype=torch.int32, device=device)
            baseline.ensure_experts(layer_id, base_ids)
            gated.ensure_experts(layer_id, gated_ids)
            torch.cuda.synchronize()
            assert torch.equal(base_ids, gated_ids), (layer_id, experts)
            n = int(baseline.num_indices.item())
            assert n == int(gated.num_indices.item())
            assert torch.equal(baseline.evict_slots[:n], gated.evict_slots[:n])
            assert torch.equal(baseline.src_indices[:n], gated.src_indices[:n])
            assert torch.equal(baseline.victim_ids[:n], gated.victim_ids[:n])
    finally:
        ok.FREETOKEN_MIRROR_TIEBREAK = prev

    for attr in ("slot_for_id", "id_of_slot", "usage", "expert_frequency", "policy_steps"):
        assert torch.equal(getattr(baseline, attr), getattr(gated, attr)), attr


def test_free_eviction_rate_rises_with_the_pool_row_tiebreak():
    """The tie-break must move the metric it exists for, on this geometry.

    Same checkpoint, same deterministic decode run (``_decode``'s internal
    ``torch.manual_seed``), same mirror capacity -- the only difference is
    whether ``FREETOKEN_MIRROR_TIEBREAK`` lets the kernel prefer a pool-row
    holder as victim. If lever 2 measured as nothing here, this must fail
    rather than be loosened.
    """
    from freetoken.moe import offload_kernels as ok

    rate = {}
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        for tiebreak in (False, True):
            cache, pool = _cache(root, _CAP_FULL)
            prev = ok.FREETOKEN_MIRROR_TIEBREAK
            ok.FREETOKEN_MIRROR_TIEBREAK = tiebreak
            try:
                _decode(cache)
                st = cache.mirror_stats()
                assert st["coverage_faults"] == 0
                assert st["starved_writebacks"] == 0
                rate[tiebreak] = st["free_eviction_rate"]
            finally:
                ok.FREETOKEN_MIRROR_TIEBREAK = prev
                pool.close()

    assert rate[True] > rate[False], (
        f"the tie-break bought nothing on this geometry: "
        f"{rate[False]:.4f} (off) -> {rate[True]:.4f} (on)"
    )


# --- Duplicate-aware eviction: the noise band (DUP_BAND) -------------------

def _evict_one(frequency, pool_row_of_id, dup_band):
    from freetoken.moe import offload_kernels as ok

    device = torch.device("cuda")
    prev = ok.FREETOKEN_MIRROR_DUP_BAND
    ok.FREETOKEN_MIRROR_DUP_BAND = dup_band
    try:
        cache = _direct_cache(num_layers=2, num_experts=4, cache_size=4, device=device)
        _seat(
            cache,
            slot_of_expert={0: 0, 1: 1, 2: 2, 3: 3},
            usage={0: 10, 1: 10, 2: 10, 3: 10},
            frequency=frequency,
        )
        _attach_fake_mirror(cache, pool_row_of_id)
        cache.ensure_experts(1, torch.tensor([[0]], dtype=torch.int32, device=device))
        torch.cuda.synchronize()
        return int(cache.victim_ids[0].item())
    finally:
        ok.FREETOKEN_MIRROR_DUP_BAND = prev


def test_band_evicts_a_duplicate_within_the_count_noise():
    """Counts 9 (no pool row) and 11 (pool row): 11 <= 9 + sqrt(9), a statistical
    tie, so the duplicate goes (a free eviction). The one-bucket tie-break
    (band off) evicts the copy-less 9 and pays a writeback."""
    freq = {0: 9, 1: 11, 2: 50, 3: 50}
    rows = [-1, 0, -1, -1, -1, -1, -1, -1]
    assert _evict_one(freq, rows, dup_band=True) == 1
    assert _evict_one(freq, rows, dup_band=False) == 0


def test_band_never_reaches_past_the_noise():
    """Counts 9 and 13: 13 > 9 + 3, a real difference, so the colder copy-less
    expert is evicted even though the warmer one has a pool row."""
    freq = {0: 9, 1: 13, 2: 50, 3: 50}
    rows = [-1, 0, -1, -1, -1, -1, -1, -1]
    assert _evict_one(freq, rows, dup_band=True) == 0


def test_free_eviction_rate_rises_with_the_band():
    """Same deterministic decode as the tie-break test: the band must buy free
    evictions over the one-bucket tie-break on this geometry."""
    from freetoken.moe import offload_kernels as ok

    rate = {}
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        for band in (False, True):
            cache, pool = _cache(root, _CAP_FULL)
            prev = ok.FREETOKEN_MIRROR_DUP_BAND
            ok.FREETOKEN_MIRROR_DUP_BAND = band
            try:
                _decode(cache)
                st = cache.mirror_stats()
                assert st["coverage_faults"] == 0
                assert st["starved_writebacks"] == 0
                rate[band] = st["free_eviction_rate"]
            finally:
                ok.FREETOKEN_MIRROR_DUP_BAND = prev
                pool.close()
    print(f"free eviction rate: tie-break {rate[False]:.4f}, band {rate[True]:.4f}")
    assert rate[True] > rate[False], rate
