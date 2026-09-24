"""DMA writebacks for the bounded mirror: the pool must stay byte-exact.

A victim with no duplicate used to be SM-stored straight into its pinned pool
row from inside the decode graph. SM stores into mapped host memory run at
2-14 GB/s depending on the host, against 22-29 GB/s for the copy engine, so
the swap kernel now copies the victim into a VRAM staging ring and the host
issues ring -> pool DMAs between steps (``MirrorResidency.service_writebacks``).
Only the byte path changes; every decision (which victims, which rows,
retention, the free stack) is the pre-DMA one.

What can go wrong is all timing, so these drive the three timings the server
can produce -- service after every step, service lagging several steps, never
(ring full: SM-store fallback) -- and check bytes, which are the only witness:
every admitted slot must hold its own expert the moment it is admitted
(including from a still-pending ring slot), and after a drain every owned pool
row must hold its owner's checkpoint bytes.
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
TOTAL = LAYERS * EXPERTS
_GPU = 20
_STEP = 4
_RESERVE = 2 * EXPERTS                # ring rows are capped at reserve // 3
_CAP = TOTAL                          # complement 28 + reserve 16 + 4 spare

# FREETOKEN_MIRROR_WB_STAGE_MB values: off (SM stores), a 4-row ring (the
# floor: fills within a few steps), the default (reserve // 3 = 5 rows here).
_OFF, _TINY, _DEFAULT = "0", "0.0001", ""


def _cache(root, monkeypatch, stage_mb):
    monkeypatch.setenv("FREETOKEN_MIRROR_WB_STAGE_MB", stage_mb)
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
    )
    cache.direct_device_banks = True
    cache.attach_mirror_pool(pool)
    cache.mirror_warm_start()
    return cache, pool


def _golden(root):
    full = MirrorExpertPool(
        root, LAYERS, EXPERTS, TOTAL, hidden_size=H, intermediate_size=ISZ,
        spec=NEMOTRON_SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=0, device=torch.device("cuda"),
    )
    try:
        full.load_initial(set())
        return {flat: {n: full.banks[n][full.pool_row_of_id[flat]].clone()
                       .contiguous().view(torch.uint8)
                       for n in full.shapes}
                for flat in range(TOTAL)}
    finally:
        full.close()


def _slot_ok(cache, golden, flat, slot):
    for name in cache.bank_schema:
        got = cache.bank_caches[name][slot].contiguous().view(torch.uint8).cpu()
        if not torch.equal(got, golden[flat][name]):
            return False
    return True


def _decode(cache, golden, *, steps, service_every, seed=0, check=True):
    """Routed decode with eviction pressure; every admitted slot is checked."""
    res = cache.residency
    torch.manual_seed(seed)
    bad = []
    for step in range(steps):
        layer = step % LAYERS
        perm = torch.randperm(EXPERTS, device="cuda")[:4].to(torch.int32)
        want = perm.tolist()
        ids = perm.clone().reshape(1, 4)
        cache.ensure_experts(layer, ids)          # rewrites ids to slot ids
        cache.copy_missing()
        if service_every and step % service_every == 0:
            res.service_writebacks()
        if check:
            torch.cuda.synchronize()
            for e, slot in zip(want, ids.reshape(-1).tolist()):
                if not _slot_ok(cache, golden, layer * EXPERTS + e, slot):
                    bad.append((step, layer * EXPERTS + e, slot))
        cache.mirror_fault_check()
    torch.cuda.synchronize()
    return bad


def _pool_bad(cache, pool, golden):
    """Owned pool rows whose bytes are not their owner's (after a drain)."""
    inv = cache._mirror["id_of_pool_row"].cpu().tolist()
    bad = []
    for row, flat in enumerate(inv):
        if flat < 0:
            continue
        for name in pool.shapes:
            got = pool.banks[name][row].contiguous().view(torch.uint8).cpu()
            if not torch.equal(got, golden[flat][name]):
                bad.append((row, flat, name))
                break
    return bad


def _holes(cache):
    slots = cache.slot_for_id.view(-1).cpu().tolist()
    rows = cache._mirror["pool_row_of_id"].cpu().tolist()
    return [f for f in range(TOTAL) if slots[f] < 0 and rows[f] < 0]


@pytest.mark.parametrize("stage_mb,service_every", [
    (_OFF, 1),        # the pre-DMA path through the new copy kernel
    (_DEFAULT, 1),    # steady state: DMAs land a step or two later
    (_DEFAULT, 5),    # lagging service: admissions read pending ring slots
    (_TINY, 3),       # 4-row ring: fills, SM-store fallback mixes in
    (_TINY, 0),       # never serviced: ring fills once, then pure fallback
                      # around rows whose DMA never lands before the drain
])
def test_written_back_rows_are_byte_exact(monkeypatch, stage_mb, service_every):
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        golden = _golden(root)
        cache, pool = _cache(root, monkeypatch, stage_mb)
        try:
            bad = _decode(cache, golden, steps=240, service_every=service_every)
            assert bad == [], f"admitted slots with the wrong bytes: {bad[:5]}"
            st = cache.mirror_stats()
            assert st["writebacks"] > 20, "the run must actually write back"
            assert st["coverage_faults"] == 0 and st["starved_writebacks"] == 0
            if stage_mb == _OFF:
                assert st["wb_stage_rows"] == 0 and st["staged_writebacks"] == 0
            else:
                assert st["staged_writebacks"] > 0
            if service_every != 1 and stage_mb != _OFF:
                assert st["stage_redirects"] > 0, (
                    "no admission read a pending ring slot: the redirect path "
                    "went unexercised"
                )
            if stage_mb == _TINY:
                assert st["staged_writebacks"] < st["writebacks"], (
                    "a 4-row ring never filled: the fallback went unexercised"
                )
            cache.residency.drain_writebacks()
            assert _holes(cache) == []
            assert _pool_bad(cache, pool, golden) == []
            st = cache.mirror_stats()
            assert st["dma_writebacks"] == st["staged_writebacks"]
        finally:
            pool.close()


def test_dma_writebacks_keep_every_decision(monkeypatch):
    """Same routing, SM stores vs DMA: identical residency, rows and counters.

    With the ring serviced every step it never fills, so not even the
    fallback's row choice can differ: the ownership maps and the free stack
    must match the SM-store run exactly.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        golden = _golden(root)
        runs = {}
        for stage_mb in (_OFF, _DEFAULT):
            cache, pool = _cache(root, monkeypatch, stage_mb)
            try:
                _decode(cache, golden, steps=240, service_every=1, check=False)
                cache.residency.drain_writebacks()
                m = cache._mirror
                st = cache.mirror_stats()
                runs[stage_mb] = {
                    "slot_for_id": cache.slot_for_id.view(-1).cpu().tolist(),
                    "pool_row_of_id": m["pool_row_of_id"].cpu().tolist(),
                    "free": m["free_rows"][: int(m["free_count"].item())].cpu().tolist(),
                    "counters": {k: st[k] for k in (
                        "swaps", "free_evictions", "writebacks", "retained_rows",
                        "coverage_faults", "starved_writebacks")},
                    "pool": {n: pool.banks[n].clone() for n in pool.shapes},
                }
            finally:
                pool.close()
        off, dma = runs[_OFF], runs[_DEFAULT]
        for key in ("slot_for_id", "pool_row_of_id", "free", "counters"):
            assert off[key] == dma[key], f"{key} differs between SM and DMA writebacks"
        for flat, row in enumerate(off["pool_row_of_id"]):
            if row < 0:
                continue
            for n in off["pool"]:
                a = off["pool"][n][row].contiguous().view(torch.uint8)
                b = dma["pool"][n][row].contiguous().view(torch.uint8)
                assert torch.equal(a, b), f"pool row {row} (expert {flat}) bank {n}"


@pytest.mark.parametrize("stage_mb", [_OFF, _DEFAULT])
def test_dma_writebacks_under_graph_replay(monkeypatch, stage_mb):
    """The decode step is a captured graph: ring, pointers and counts must be
    replay-stable, and the host service between replays must still land them."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        golden = _golden(root)
        cache, pool = _cache(root, monkeypatch, stage_mb)
        try:
            # One graph over every layer, like a decode step: each layer reads
            # its routing from its own persistent buffer.
            bufs = [torch.zeros((1, 4), dtype=torch.int32, device="cuda")
                    for _ in range(LAYERS)]

            def step():
                for layer, buf in enumerate(bufs):
                    cache.ensure_experts(layer, buf)
                    cache.copy_missing()

            for layer, buf in enumerate(bufs):
                buf.copy_(torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))
            step()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                step()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                step()
            torch.cuda.synchronize()

            torch.manual_seed(1)
            bad = []
            for it in range(60):
                want = []
                for buf in bufs:
                    perm = torch.randperm(EXPERTS, device="cuda")[:4].to(torch.int32)
                    want.append(perm.tolist())
                    buf.copy_(perm.reshape(1, 4))
                g.replay()
                cache.residency.service_writebacks()
                torch.cuda.synchronize()
                # 24 routed experts per step over 20 slots: a later layer may
                # evict an earlier one's admission within the same replay (the
                # real GEMM has consumed it by then). Check every admission
                # that still owns its slot after the step.
                owner = cache.slot_for_id.view(-1).cpu().tolist()
                checked = 0
                for layer, buf in enumerate(bufs):
                    for e, slot in zip(want[layer], buf.reshape(-1).tolist()):
                        flat = layer * EXPERTS + e
                        if owner[flat] != slot:
                            continue
                        checked += 1
                        if not _slot_ok(cache, golden, flat, slot):
                            bad.append((it, flat, slot))
                assert checked >= 4
                cache.mirror_fault_check()
            assert bad == [], f"admitted slots with the wrong bytes: {bad[:5]}"
            st = cache.mirror_stats()
            assert st["writebacks"] > 0
            assert (st["staged_writebacks"] > 0) == (stage_mb != _OFF)
            cache.residency.drain_writebacks()
            assert _pool_bad(cache, pool, golden) == []
        finally:
            pool.close()
