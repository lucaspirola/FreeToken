"""Arena compaction on a real CUDA arena: bytes, coverage, faults, graph replay.

The shrink that funds a prefill's transient (engine/growable_kv.py) now compacts
first (``OffloadMoeCache.compact_before_shrink``). What must hold afterwards, for
both residencies:

* byte equality: every GPU resident's slot holds its own checkpoint bytes, and
  (mirror) every pool row holds its owner's bytes;
* the coverage floor: every expert is a GPU resident below ``n`` or owns a pool
  row, so the fault counters stay at 0 through more decode;
* graph replay: a decode step captured BEFORE the compaction still routes every
  expert to the slot that holds it AFTER, because the captured kernels read the
  slot maps from device memory (``_ensure_experts_sized_kernel_v2`` loads
  ``slot_for_id``) instead of baking slot numbers into the graph.

Needs a GPU with the model unloaded (CLAUDE.md: no torch tests beside a live model).
"""

from __future__ import annotations

import random
import tempfile
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _expert_arena():
    from freetoken.moe import offload_cache as oc
    from freetoken.moe import offload_kernels as ok

    prev = (oc.FREETOKEN_EXPERT_ARENA, ok.FREETOKEN_EXPERT_ARENA)
    oc.FREETOKEN_EXPERT_ARENA = True
    ok.FREETOKEN_EXPERT_ARENA = True
    try:
        yield
    finally:
        oc.FREETOKEN_EXPERT_ARENA, ok.FREETOKEN_EXPERT_ARENA = prev


LAYERS, EXPERTS, H, ISZ = 6, 8, 32, 32
TOTAL = LAYERS * EXPERTS
_CAPACITY = 40      # arena slots at the decode level
_N = 28             # after the shrink; 2E = 16 is the double buffer
_STEP = 4
_RESERVE = 2 * EXPERTS
_POOL = TOTAL - _N + _RESERVE   # exactly the floor for _N: min_gpu_slots == _N
TOPK = 4


def _same(got, want, what):
    g = got.cpu().contiguous().view(torch.uint8)
    w = want.cpu().contiguous().view(torch.uint8)
    assert torch.equal(g, w), what


def _skewed_ids(rng, layer):
    # A fixed hot set per layer plus noise, so LFU has something to rank.
    hot = [(layer * 3 + k) % EXPERTS for k in range(3)]
    picks = set()
    while len(picks) < TOPK:
        picks.add(rng.choice(hot) if rng.random() < 0.7 else rng.randrange(EXPERTS))
    return sorted(picks)


class _Graphs:
    """One captured decode step (admission + copies) per layer, static id buffers."""

    def __init__(self, cache):
        self.cache = cache
        self.bufs = [torch.zeros((1, TOPK), dtype=torch.int32, device="cuda")
                     for _ in range(LAYERS)]
        self.graphs = []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for layer in range(LAYERS):  # warm (compile) outside capture
                self.bufs[layer].copy_(torch.arange(TOPK, dtype=torch.int32).view(1, -1))
                cache.ensure_experts(layer, self.bufs[layer])
                cache.copy_missing()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for layer in range(LAYERS):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                cache.ensure_experts(layer, self.bufs[layer])
                cache.copy_missing()
            self.graphs.append(g)
        torch.cuda.synchronize()

    def step(self, layer, experts):
        """Replay; returns the slot ids the captured kernels routed ``experts`` to."""
        self.bufs[layer].copy_(torch.tensor([experts], dtype=torch.int32))
        self.graphs[layer].replay()
        torch.cuda.synchronize()
        return self.bufs[layer].view(-1).tolist()


def _swap_slots(cache, a, b):
    """Exchange two residents' slots: bytes, both maps and usage (a test fixture move,
    the same bookkeeping compaction does)."""
    ids = cache.id_of_slot
    fa, fb = int(ids[a]), int(ids[b])
    for bank in cache.bank_caches.values():
        tmp = bank[a].clone()
        bank[a].copy_(bank[b])
        bank[b].copy_(tmp)
    ids[a], ids[b] = fb, fa
    flat = cache.slot_for_id.view(-1)
    flat[fa], flat[fb] = b, a
    ua, ub = int(cache.usage[a]), int(cache.usage[b])
    cache.usage[a], cache.usage[b] = ub, ua


def _plant_hot_in_doomed(cache, k=3):
    """Put the k hottest residents of [2E, _N) into the doomed range [_N, _CAPACITY),
    swapping with the coldest doomed residents that have a host copy, so the region
    below _N gains cold, free-to-overwrite occupants. Returns the planted ids."""
    torch.cuda.synchronize()
    freq = cache.expert_frequency.view(-1).tolist()
    ids = cache.id_of_slot.tolist()
    mask = cache.residency.host_copy_mask()
    has_copy = (lambda f: True) if mask is None else (lambda f: bool(mask[f]))
    below = sorted((s for s in range(2 * EXPERTS, _N) if ids[s] >= 0),
                   key=lambda s: freq[ids[s]], reverse=True)
    doomed = sorted((s for s in range(_N, _CAPACITY) if ids[s] >= 0 and has_copy(ids[s])),
                    key=lambda s: freq[ids[s]])
    planted = []
    for hot, cold in zip(below[:k], doomed[:k]):
        if freq[ids[hot]] <= freq[ids[cold]]:
            continue
        planted.append(ids[hot])
        _swap_slots(cache, hot, cold)
    torch.cuda.synchronize()
    return planted


def _check_routed_bytes(cache, golden, layer, experts, slots, n):
    for e, slot in zip(experts, slots):
        flat = layer * EXPERTS + e
        assert 0 <= slot < n, f"expert {flat} routed to slot {slot} (usable {n})"
        assert int(cache.id_of_slot[slot]) == flat
        for name, bank in cache.bank_caches.items():
            _same(bank[slot], golden[flat][name],
                  f"replayed step routed expert {flat} to slot {slot} with wrong bytes ({name})")


def _check_residents(cache, golden, n):
    ids = cache.id_of_slot.tolist()
    fwd = cache.slot_for_id.view(-1).tolist()
    assert all(f == -1 for f in ids[n:]), "unmapped slots still claim experts"
    for slot in range(n):
        flat = ids[slot]
        if flat < 0:
            continue
        assert fwd[flat] == slot
        if slot < 2 * EXPERTS:
            continue  # the double buffer's bytes belong to the last prefill layer
        for name, bank in cache.bank_caches.items():
            _same(bank[slot], golden[flat][name], f"slot {slot} lost expert {flat} ({name})")


# ---------------------------------------------------------------- mirror


def _mirror_cache(root):
    from freetoken.models.nemotron_h.weight import NVFP4_EXPERT_SOURCE_SPEC as SPEC
    from freetoken.moe.mirror_pool import MirrorExpertPool
    from freetoken.moe.offload_cache import OffloadMoeCache

    pool = MirrorExpertPool(
        root, LAYERS, EXPERTS, _POOL, hidden_size=H, intermediate_size=ISZ, spec=SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=_RESERVE, device=torch.device("cuda"),
    )
    cache = OffloadMoeCache(
        num_layers=LAYERS, num_experts=EXPERTS, cache_size=_CAPACITY,
        slot_capacity=_CAPACITY, arena_step_slots=_STEP,
        device=torch.device("cuda"), quant_format="nvfp4", cache_policy="lfu",
        prefill_overlap=True,
    )
    cache.direct_device_banks = True
    cache.attach_mirror_pool(pool)
    cache.mirror_warm_start()
    return cache, pool


def _mirror_golden(root):
    from freetoken.models.nemotron_h.weight import NVFP4_EXPERT_SOURCE_SPEC as SPEC
    from freetoken.moe.mirror_pool import MirrorExpertPool

    full = MirrorExpertPool(
        root, LAYERS, EXPERTS, TOTAL, hidden_size=H, intermediate_size=ISZ, spec=SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=0, device=torch.device("cuda"),
    )
    try:
        full.load_initial(set())
        return {flat: {n: full.banks[n][full.pool_row_of_id[flat]].clone() for n in full.shapes}
                for flat in range(TOTAL)}
    finally:
        full.close()


@pytest.mark.parametrize("compaction", ["1", "0"])
def test_mirror_shrink_keeps_bytes_coverage_and_graphs(monkeypatch, compaction):
    from tests.moe._mirror_checkpoint import write_nvfp4_checkpoint

    monkeypatch.setenv("FREETOKEN_ARENA_COMPACTION", compaction)
    rng = random.Random(11)
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        golden = _mirror_golden(root)
        cache, pool = _mirror_cache(root)
        try:
            assert pool.min_gpu_slots <= _N, "geometry: the shrink target is above the floor"
            graphs = _Graphs(cache)
            for step in range(120):
                layer = step % LAYERS
                experts = _skewed_ids(rng, layer)
                slots = graphs.step(layer, experts)
                _check_routed_bytes(cache, golden, layer, experts, slots, _CAPACITY)
            cache.mirror_fault_check()
            planted = _plant_hot_in_doomed(cache)
            assert planted, "fixture: nothing hot to plant in the doomed range"
            cache.set_usable_slots(_N)
            torch.cuda.synchronize()
            m = cache._mirror
            fwd = cache.slot_for_id.view(-1).tolist()
            rows = m["pool_row_of_id"].tolist()
            inv = m["id_of_pool_row"].tolist()
            uncovered = [f for f in range(TOTAL) if not (0 <= fwd[f] < _N) and rows[f] < 0]
            assert uncovered == [], f"coverage lost for {uncovered}"
            for flat in range(TOTAL):
                if rows[flat] >= 0:
                    assert inv[rows[flat]] == flat
                    for name, bank in pool.banks.items():
                        _same(bank[rows[flat]], golden[flat][name],
                              f"pool row {rows[flat]} of expert {flat} ({name})")
            _check_residents(cache, golden, _N)
            if compaction == "1":
                totals = cache.compaction_totals
                assert totals["moved"] >= len(planted), totals
                assert all(0 <= fwd[f] < _N for f in planted), (
                    f"hot experts {planted} did not survive the shrink: "
                    f"{[fwd[f] for f in planted]}")
            # Decode continues on the graphs captured before the shrink.
            for step in range(120):
                layer = step % LAYERS
                experts = _skewed_ids(rng, layer)
                slots = graphs.step(layer, experts)
                _check_routed_bytes(cache, golden, layer, experts, slots, _N)
            cache.mirror_fault_check()
            stats = cache.mirror_stats()
            assert stats.get("coverage_faults", 0) == 0, stats
        finally:
            pool.close()


def test_mirror_compaction_keeps_the_hot_set(monkeypatch):
    """With compaction, the hottest doomed experts are still resident after the shrink
    and the refill writes back fewer rows than without."""
    from tests.moe._mirror_checkpoint import write_nvfp4_checkpoint

    refilled, hits_after = {}, {}
    for compaction in ("1", "0"):
        monkeypatch.setenv("FREETOKEN_ARENA_COMPACTION", compaction)
        rng = random.Random(5)
        with tempfile.TemporaryDirectory() as root:
            write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
            cache, pool = _mirror_cache(root)
            try:
                graphs = _Graphs(cache)
                for step in range(120):
                    graphs.step(step % LAYERS, _skewed_ids(rng, step % LAYERS))
                _plant_hot_in_doomed(cache)
                cache.set_usable_slots(_N)
                refilled[compaction] = getattr(cache.residency, "refill_totals", {}).get("rows", 0)
                fwd = cache.slot_for_id.view(-1).tolist()
                hot = [layer * EXPERTS + (layer * 3 + k) % EXPERTS
                       for layer in range(LAYERS) for k in range(3)]
                hits_after[compaction] = sum(1 for f in hot if 0 <= fwd[f] < _N)
            finally:
                pool.close()
    assert hits_after["1"] > hits_after["0"], hits_after
    assert refilled["1"] <= refilled["0"], refilled


# ---------------------------------------------------------------- whole model


def _whole_cache():
    from freetoken.moe.offload_cache import OffloadMoeCache

    elems = 4096
    sources = {"gate_up": [], "down": []}
    golden = {}
    for layer in range(LAYERS):
        up = torch.empty((EXPERTS, elems), dtype=torch.bfloat16, pin_memory=True)
        down = torch.empty((EXPERTS, elems // 2), dtype=torch.bfloat16, pin_memory=True)
        for e in range(EXPERTS):
            flat = layer * EXPERTS + e
            up[e].fill_(float(flat + 1))
            down[e].fill_(float(-(flat + 1)))
            golden[flat] = {"gate_up": up[e].clone(), "down": down[e].clone()}
        sources["gate_up"].append(up)
        sources["down"].append(down)
    cache = OffloadMoeCache(
        num_layers=LAYERS, num_experts=EXPERTS, cache_size=_CAPACITY,
        slot_capacity=_CAPACITY, arena_step_slots=_STEP,
        device=torch.device("cuda"), quant_format="bf16", cache_policy="lfu",
        prefill_overlap=True,
    )
    cache.direct_device_banks = True
    cache.set_bank_sources(sources)
    return cache, golden


def test_whole_model_lfu_compaction_bytes_and_graphs(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ARENA_COMPACTION", "1")
    rng = random.Random(3)
    cache, golden = _whole_cache()
    graphs = _Graphs(cache)
    for step in range(200):
        layer = step % LAYERS
        experts = _skewed_ids(rng, layer)
        slots = graphs.step(layer, experts)
        _check_routed_bytes(cache, golden, layer, experts, slots, _CAPACITY)
    planted = _plant_hot_in_doomed(cache)
    assert planted
    cache.set_usable_slots(_N)
    torch.cuda.synchronize()
    _check_residents(cache, golden, _N)
    fwd = cache.slot_for_id.view(-1).tolist()
    assert all(0 <= fwd[f] < _N for f in planted), planted
    for step in range(120):
        layer = step % LAYERS
        experts = _skewed_ids(rng, layer)
        slots = graphs.step(layer, experts)
        _check_routed_bytes(cache, golden, layer, experts, slots, _N)
    assert cache.compaction_totals["calls"] == 1
