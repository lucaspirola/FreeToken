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

import os
os.environ.setdefault("FREETOKEN_EXPERT_ARENA", "1")
import tempfile
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mirrored prefill needs CUDA"
)

from freetoken.models.nemotron_h.weight import (
    NVFP4_EXPERT_SOURCE_SPEC as NEMOTRON_SPEC,
)
from freetoken.moe.mirror_pool import MirrorExpertPool, prefill_buffer_slots
from freetoken.moe.offload_cache import _PREFILL_BUFFER_USAGE, OffloadMoeCache

from tests.moe._mirror_checkpoint import write_nvfp4_checkpoint

LAYERS, EXPERTS, H, ISZ = 6, 8, 32, 32
TOTAL = LAYERS * EXPERTS            # 48 expert rows
_GPU = 32                           # 16 buffer slots + 16 decode residents
_STEP = 4
# Complement (48 - 16) plus one layer of reserve. The production reserve is
# three layers; at this toy size that would exceed the model, and what these
# tests exercise is the prefill path, not the reserve's arithmetic.
_RESERVE = EXPERTS
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


def test_the_buffer_region_belongs_to_prefill_alone():
    """Residents sit above the buffer, which carries the victim sentinel.

    This is what makes the buffer's invalidation free: nothing the next layer
    overwrites was ever an expert's only copy.
    """
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        cache, pool = _cache(root)
        try:
            base = cache._mirror_prefill_base()
            assert base == prefill_buffer_slots(EXPERTS) == 2 * EXPERTS
            assert (cache.id_of_slot[:base] == -1).all(), (
                "a decode resident in the buffer region would be dropped "
                "without a writeback the first time prefill reused the slot"
            )
            assert (cache.usage[:base] == _PREFILL_BUFFER_USAGE).all(), (
                "victim selection is an argmin over usage; without the "
                "sentinel these slots are the FIRST ones decode evicts into"
            )
            resident = cache.id_of_slot[base:]
            assert int((resident >= 0).sum()) == _GPU - base
            slots = cache.slot_for_id.view(-1)
            live = slots[slots >= 0]
            assert int(live.min()) >= base
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
