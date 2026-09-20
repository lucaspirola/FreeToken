"""The pool must pin the bytes it asked for, not the next power of two.

torch's caching host allocator rounds every allocation up to a power of two
(ATen/core/CachingHostAllocator.h: ``roundSize = PowerOf2Ceil(size)``, then
``allocate_host_memory(roundSize)``). This pool asks it for two banks of
several GiB each, so the rounding is not a rounding -- measured on Nemotron,
host RAM actually pinned against the pool asked for:

    1700 rows  pool  8.90 GiB -> 9.11 GiB   (each big bank 3.95 -> 4)
    2100 rows  pool 10.99 GiB -> 18.11 GiB  (each big bank 4.88 -> 8)
    2500 rows  pool 13.08 GiB -> 18.12 GiB  (each big bank 5.81 -> 8)
    2944 rows  pool 15.41 GiB -> 18.12 GiB  (each big bank 6.85 -> 8)

Up to 2x the pool, and a RAM knob whose cost does not move across most of its
range -- which is how the first sweeps read, and why the branch appeared to
save nothing. A direct measurement of the two allocation paths, 4.88 GiB asked
for: ``pin_memory=True`` grew resident /dev/zero by 8.07 GiB, cudaHostRegister
over an ordinary allocation grew it by 0.00 and still reported ``is_pinned()``.
"""
from __future__ import annotations

import tempfile
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinning needs CUDA"
)

from freetoken.models.nemotron_h.weight import (
    NVFP4_EXPERT_SOURCE_SPEC as NEMOTRON_SPEC,
)
from freetoken.moe.mirror_pool import MirrorExpertPool

from tests.moe._mirror_checkpoint import write_nvfp4_checkpoint

LAYERS, EXPERTS, H, ISZ = 6, 8, 32, 32
TOTAL = LAYERS * EXPERTS


def _pool(root, capacity):
    return MirrorExpertPool(
        root, LAYERS, EXPERTS, capacity, hidden_size=H, intermediate_size=ISZ,
        spec=NEMOTRON_SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=EXPERTS, device=torch.device("cuda"),
    )


def test_the_banks_are_pinned_and_exactly_the_size_asked_for():
    """Both halves matter: unpinned would make every H2D synchronous, and
    oversized is the bug above."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        pool = _pool(root, TOTAL - 8)
        try:
            total = 0
            for name, bank in pool.banks.items():
                assert bank.is_pinned(), (
                    f"bank {name} is not pinned: every admission would become "
                    f"a synchronous copy"
                )
                assert bank.is_contiguous()
                assert bank.shape[0] == pool.capacity
                total += bank.numel() * bank.element_size()
            # pool_bytes is what the server logs and what the RAM budget is
            # planned against, so it must be the truth about the allocation.
            assert total == pool.pool_bytes
        finally:
            pool.close()


def test_a_closed_pool_leaves_no_registration_behind():
    """A cudaHostRegister pin outliving its allocation is a dangling pin."""
    with tempfile.TemporaryDirectory() as root:
        write_nvfp4_checkpoint(root, LAYERS, EXPERTS, H, ISZ)
        pool = _pool(root, TOTAL - 8)
        assert pool._registered, "nothing was registered"
        pool.close()
        assert pool._registered == []
        pool.close()          # idempotent: a second close must not re-unpin
        assert pool._registered == []
