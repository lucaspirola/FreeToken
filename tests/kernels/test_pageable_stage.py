import pytest
import torch


def test_pageable_gather_copies_selected_rows_exactly():
    if not torch.cuda.is_available():
        pytest.skip("cudaLaunchHostFunc requires CUDA")

    from freetoken.kernel import _pageable_stage

    rows = 7
    capacity = 4
    source = [
        torch.arange(rows * 129, dtype=torch.int32).view(rows, 129),
        torch.arange(rows * 67, dtype=torch.int32).view(rows, 67),
    ]
    destination = [
        torch.empty((capacity, bank.shape[1]), dtype=bank.dtype, pin_memory=True)
        for bank in source
    ]
    source_ids = torch.tensor([6, 1, 4, 0], dtype=torch.int32, pin_memory=True)
    count = torch.tensor([capacity], dtype=torch.int64, pin_memory=True)
    row_bytes = [bank[0].numel() * bank.element_size() for bank in source]
    gather = _pageable_stage.PageableGather(
        [bank.data_ptr() for bank in source],
        [bank.data_ptr() for bank in destination],
        row_bytes,
        count.data_ptr(),
        source_ids.data_ptr(),
        capacity,
        rows,
    )

    stream = torch.cuda.Stream()
    gather.launch(stream.cuda_stream)
    stream.synchronize()

    for src, dst in zip(source, destination):
        torch.testing.assert_close(dst, src[source_ids.long()], rtol=0, atol=0)
    assert gather.stats()[:2] == [1, capacity]
    assert gather.threads() >= 1


def test_pageable_gather_clamps_count_to_capacity():
    if not torch.cuda.is_available():
        pytest.skip("cudaLaunchHostFunc requires CUDA")

    from freetoken.kernel import _pageable_stage

    source = torch.arange(8 * 16, dtype=torch.uint8).view(8, 16)
    destination = torch.empty((3, 16), dtype=torch.uint8, pin_memory=True)
    source_ids = torch.tensor([7, 3, 1], dtype=torch.int32, pin_memory=True)
    count = torch.tensor([99], dtype=torch.int64, pin_memory=True)
    gather = _pageable_stage.PageableGather(
        [source.data_ptr()],
        [destination.data_ptr()],
        [16],
        count.data_ptr(),
        source_ids.data_ptr(),
        3,
        8,
    )

    stream = torch.cuda.Stream()
    gather.launch(stream.cuda_stream)
    stream.synchronize()

    torch.testing.assert_close(destination, source[source_ids.long()], rtol=0, atol=0)
    assert gather.stats()[:2] == [1, 3]


def test_pageable_gather_reuses_pool_across_small_and_large_jobs():
    if not torch.cuda.is_available():
        pytest.skip("cudaLaunchHostFunc requires CUDA")

    from freetoken.kernel import _pageable_stage

    rows = 9
    capacity = 4
    sources = [
        torch.arange(rows * width, dtype=torch.int32).view(rows, width)
        for width in (129, 67)
    ]
    destinations = [
        torch.empty((capacity, source.shape[1]), dtype=source.dtype, pin_memory=True)
        for source in sources
    ]
    source_ids = torch.tensor([8, 5, 2, 0], dtype=torch.int32, pin_memory=True)
    count = torch.tensor([1], dtype=torch.int64, pin_memory=True)
    gather = _pageable_stage.PageableGather(
        [source.data_ptr() for source in sources],
        [destination.data_ptr() for destination in destinations],
        [source[0].numel() * source.element_size() for source in sources],
        count.data_ptr(),
        source_ids.data_ptr(),
        capacity,
        rows,
    )

    stream = torch.cuda.Stream()
    for job_rows in (1, capacity, 1, 2, capacity):
        count.fill_(job_rows)
        gather.launch(stream.cuda_stream)
        stream.synchronize()
        for source, destination in zip(sources, destinations):
            torch.testing.assert_close(
                destination[:job_rows],
                source[source_ids[:job_rows].long()],
                rtol=0,
                atol=0,
            )

    assert gather.stats()[:2] == [5, 1 + capacity + 1 + 2 + capacity]
