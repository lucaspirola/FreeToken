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
