from __future__ import annotations

import gc

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_vmm_tensor_commits_suffix_without_moving_or_losing_prefix():
    from freetoken.kernel.vmm import VMMTensor

    allocation = VMMTensor(
        (4 * 1024 * 1024,), dtype=torch.uint8, device=torch.device("cuda")
    )
    granularity = allocation.granularity
    pointer = allocation.tensor.data_ptr()
    allocation.tensor[:granularity].fill_(37)
    allocation.commit_ranges([(granularity, granularity)])
    allocation.tensor[granularity : 2 * granularity].fill_(19)
    torch.cuda.synchronize()

    assert allocation.tensor.data_ptr() == pointer
    assert allocation.mapped_bytes == 2 * granularity
    assert int(allocation.tensor[:1024].sum()) == 37 * 1024
    assert int(allocation.tensor[granularity : granularity + 1024].sum()) == 19 * 1024


def test_vmm_tensor_releases_committed_physical_memory():
    from freetoken.kernel.vmm import VMMTensor

    before = torch.cuda.mem_get_info()[0]
    allocation = VMMTensor(
        (4 * 1024 * 1024,), dtype=torch.uint8, device=torch.device("cuda")
    )
    mapped = allocation.mapped_bytes
    del allocation
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.mem_get_info()[0]

    assert after >= before - mapped


def test_vmm_tensor_uncommits_suffix_and_can_recommit_it():
    from freetoken.kernel.vmm import VMMTensor

    allocation = VMMTensor(
        (8 * 1024 * 1024,), dtype=torch.uint8, device=torch.device("cuda")
    )
    granularity = allocation.granularity
    pointer = allocation.tensor.data_ptr()
    allocation.tensor[:granularity].fill_(41)
    allocation.commit_ranges([(granularity, granularity)])
    allocation.commit_ranges([(2 * granularity, granularity)])
    allocation.tensor[granularity : 3 * granularity].fill_(23)
    torch.cuda.synchronize()

    allocation.uncommit_ranges([(2 * granularity, granularity)])
    assert allocation.mapped_bytes == 2 * granularity
    assert allocation.tensor.data_ptr() == pointer
    assert int(allocation.tensor[:1024].sum()) == 41 * 1024
    assert int(allocation.tensor[granularity : granularity + 1024].sum()) == 23 * 1024

    allocation.commit_ranges([(2 * granularity, granularity)])
    allocation.tensor[2 * granularity : 2 * granularity + 1024].fill_(7)
    torch.cuda.synchronize()
    assert allocation.mapped_bytes == 3 * granularity
    assert allocation.tensor.data_ptr() == pointer
    assert int(allocation.tensor[2 * granularity : 2 * granularity + 1024].sum()) == 7 * 1024


def test_growable_mha_pool_reverses_each_growth_segment():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    step = 16_384
    pool = MHAKVCache(
        num_kv_heads=1,
        num_layers=1,
        head_dim=64,
        num_pages=3 * step + 1,
        page_size=1,
        dtype=torch.float16,
        device=torch.device("cuda"),
        grow_step_tokens=step,
    )
    pointer = pool._kv_buffer.data_ptr()
    initial_bytes = pool.mapped_bytes_for_pages(step)

    pool.commit_pages(3 * step)
    assert pool.committed_pages == 3 * step
    assert pool._kv_vmm.mapped_bytes == pool.mapped_bytes_for_pages(3 * step)
    assert pool._kv_buffer.data_ptr() == pointer

    pool.decommit_pages(step)
    assert pool.committed_pages == step
    assert pool._kv_vmm.mapped_bytes == initial_bytes
    assert pool._kv_buffer.data_ptr() == pointer

    pool.commit_pages(2 * step)
    assert pool.committed_pages == 2 * step
    assert pool._kv_buffer.data_ptr() == pointer


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_vmm_tensor_supports_the_integer_bank_dtypes(dtype):
    """``--kv-grow-step-tokens`` allocates the MoE slot cache as ``VMMTensor``s, and the
    b12x/flashinfer NVFP4 layout packs its codes into an **int32** bank -- so that pairing
    used to die at startup with ``unsupported VMM tensor dtype: torch.int32``
    (``benchmarks/results/nemotron35_lightning_5080_cache_study_2026-09-04.md``). The
    dtype table in ``kernel/vmm.py`` and ``parse_dtype`` in ``csrc/vmm_tensor.cpp`` must
    stay in step: a name missing from either side is the same startup failure."""
    from freetoken.kernel.vmm import VMMTensor

    n = 8 * 1024 * 1024 // torch.empty((), dtype=dtype).element_size()
    allocation = VMMTensor((n,), dtype=dtype, device=torch.device("cuda"))
    granularity = allocation.granularity
    elems = granularity // torch.empty((), dtype=dtype).element_size()
    assert allocation.tensor.dtype is dtype
    allocation.tensor[:elems].fill_(7)
    allocation.commit_ranges([(granularity, granularity)])
    allocation.tensor[elems : 2 * elems].fill_(-3)
    torch.cuda.synchronize()
    assert int(allocation.tensor[:16].sum()) == 7 * 16
    assert int(allocation.tensor[elems : elems + 16].sum()) == -3 * 16
    del allocation
    gc.collect()
