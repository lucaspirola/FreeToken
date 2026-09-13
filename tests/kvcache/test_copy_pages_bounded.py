from unittest.mock import patch

import pytest
import torch

from freetoken.distributed.info import DistributedInfo
from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.kvcache.quant import Q6_0, Q8_0


def pool(*, asymmetric: bool):
    kwargs = {"quant_k": Q8_0, "quant_v": Q6_0} if asymmetric else {"quant": Q8_0}
    with patch("freetoken.kvcache.mha_pool.get_tp_info",
               return_value=DistributedInfo(rank=0, size=1)):
        return MHAKVCache(num_kv_heads=2, num_layers=3, head_dim=64, num_pages=17,
                          page_size=1, dtype=torch.bfloat16,
                          device=torch.device("cpu"), **kwargs)


def tensors(cache):
    values = [cache._k_buffer, cache._v_buffer] if cache._asymmetric else [cache._kv_buffer]
    if cache._scale_buffer is not None:
        values.append(cache._scale_buffer)
    return values


@pytest.mark.parametrize("asymmetric", [False, True])
def test_copy_pages_is_exact_for_chunked_quantized_storage(monkeypatch, asymmetric):
    cache = pool(asymmetric=asymmetric)
    for ordinal, value in enumerate(tensors(cache), 1):
        value.copy_(torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) + ordinal)
    src = torch.tensor([9, 10, 11, 12, 13], dtype=torch.int32)
    dst = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    expected = [value.index_select(2 if value is cache._kv_buffer or
                value is cache._scale_buffer else 1, src.long()).clone()
                for value in tensors(cache)]
    # Force at most two pages into any payload gather using the actual largest buffer geometry.
    buffers = [((value, 2) if value is cache._kv_buffer or value is cache._scale_buffer
                else (value, 1)) for value in tensors(cache)]
    per_page = max(value.numel() // value.shape[dim] * value.element_size()
                   for value, dim in buffers)
    monkeypatch.setattr("freetoken.kvcache.mha_pool.COPY_PAGE_SCRATCH_LIMIT_BYTES",
                        per_page * 2)
    assert cache._copy_page_chunk_size(buffers) == 2
    chunks = []
    original = cache._copy_page_payload
    monkeypatch.setattr(cache, "_copy_page_payload",
                        lambda payloads, source, destination: (
                            chunks.append(len(source)),
                            original(payloads, source, destination))[1])
    cache.copy_pages(src, dst)
    assert chunks == [2, 2, 1]
    for value, wanted in zip(tensors(cache), expected, strict=True):
        dim = 2 if value is cache._kv_buffer or value is cache._scale_buffer else 1
        assert torch.equal(value.index_select(dim, dst.long()), wanted)


@pytest.mark.parametrize("src,dst,error", [
    ([-1], [1], "outside"), ([17], [1], "outside"),
    ([9, 9], [1, 2], "unique"), ([9, 10], [1, 1], "unique"),
    ([9, 10], [1, 9], "overlapping"),
])
def test_copy_pages_rejects_invalid_geometry_before_writes(src, dst, error):
    cache = pool(asymmetric=False)
    cache._kv_buffer.copy_(torch.arange(cache._kv_buffer.numel(), dtype=torch.int8).reshape(
        cache._kv_buffer.shape))
    before = cache._kv_buffer.clone()
    with pytest.raises(ValueError, match=error):
        cache.copy_pages(torch.tensor(src), torch.tensor(dst))
    assert torch.equal(cache._kv_buffer, before)


@pytest.mark.parametrize("side", ["source", "destination"])
def test_copy_pages_rejects_logical_but_uncommitted_pages_before_writes(side):
    cache = pool(asymmetric=False)
    cache._committed_pages = 8
    before = cache._kv_buffer.clone()
    src, dst = torch.tensor([7]), torch.tensor([1])
    if side == "source":
        src = torch.tensor([9])
    else:
        dst = torch.tensor([9])
    with pytest.raises(ValueError, match="committed"):
        cache.copy_pages(src, dst)
    assert torch.equal(cache._kv_buffer, before)


@pytest.mark.parametrize("bad", [torch.tensor([1.5]), torch.tensor([[1]])])
def test_copy_pages_rejects_noninteger_or_nonvector_indices_before_writes(bad):
    cache = pool(asymmetric=False)
    before = cache._kv_buffer.clone()
    with pytest.raises(ValueError, match="one-dimensional integer"):
        cache.copy_pages(bad, torch.tensor([2]))
    assert torch.equal(cache._kv_buffer, before)


def test_late_payload_failure_leaves_cache_manager_metadata_unpublished(monkeypatch):
    from freetoken.scheduler.cache import CacheManager

    cache = pool(asymmetric=False)
    table = torch.zeros((1, 20), dtype=torch.int32)
    manager = CacheManager(16, 1, table, "naive", committed_pages=16,
                           page_index_offset=1)
    table[0, :2] = torch.tensor([15, 16])
    manager.free_slots = torch.tensor(list(range(1, 15)), dtype=torch.int32)
    req = type("Req", (), {"table_idx": 0, "cached_len": 2,
                            "cache_handle": object()})()
    before_table, before_free = table.clone(), manager.free_slots.clone()
    per_page = (cache._kv_buffer.numel() // cache._kv_buffer.shape[2]
                * cache._kv_buffer.element_size())
    monkeypatch.setattr("freetoken.kvcache.mha_pool.COPY_PAGE_SCRATCH_LIMIT_BYTES",
                        per_page)
    original = cache._copy_page_payload
    calls = 0

    def fail_second(payloads, source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("late copy failure")
        original(payloads, source, destination)

    monkeypatch.setattr(cache, "_copy_page_payload", fail_second)
    with pytest.raises(RuntimeError, match="late copy failure"):
        manager.compact_active_pages([req], 8, cache.copy_pages)
    assert calls == 2
    assert torch.equal(table, before_table)
    assert torch.equal(manager.free_slots, before_free)
