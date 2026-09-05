from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _reference_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_to_req: torch.Tensor,
    q_positions: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    outs = []
    group = q.shape[1] // k_cache.shape[1]
    for tok in range(q.shape[0]):
        req = int(q_to_req[tok].item())
        start = int(indptr[req].item())
        end = int(indptr[req + 1].item())
        slots = indices[start:end].to(torch.long)
        k = k_cache[slots].repeat_interleave(group, dim=1).transpose(0, 1)
        v = v_cache[slots].repeat_interleave(group, dim=1).transpose(0, 1)
        scores = torch.einsum("hd,hkd->hk", q[tok].float(), k.float()) * sm_scale
        key_pos = torch.arange(end - start, device=q.device)
        mask = key_pos <= q_positions[tok]
        if sliding_window is not None:
            mask = mask & (key_pos + sliding_window > q_positions[tok])
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        if sinks is None:
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,hkd->hd", probs, v.float())
        else:
            sink_logits = sinks.to(device=q.device, dtype=torch.float32)
            m = torch.maximum(scores.max(dim=-1).values, sink_logits)
            probs = torch.exp(scores - m[:, None])
            denom = probs.sum(dim=-1) + torch.exp(sink_logits - m)
            out = torch.einsum("hk,hkd->hd", probs, v.float()) / denom[:, None]
        outs.append(out.to(q.dtype))
    return torch.stack(outs, dim=0)


def test_triton_backend_passes_attention_sinks_to_paged_kernel(monkeypatch):
    from freetoken.attention import AttentionSpec
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata

    class FakeKVCache:
        def __init__(self):
            self.device = torch.device("cpu")
            self.k = torch.zeros(4, 1, 4)
            self.v = torch.zeros(4, 1, 4)

        def store_kv(self, k, v, out_loc, layer_id):
            _ = layer_id
            self.k[out_loc.to(torch.long)] = k.view(k.shape[0], 1, -1)
            self.v[out_loc.to(torch.long)] = v.view(v.shape[0], 1, -1)

        def k_cache(self, layer_id):
            return self.k

        def v_cache(self, layer_id):
            return self.v

    kv_cache = FakeKVCache()
    monkeypatch.setattr(
        "freetoken.attention.triton.get_global_ctx",
        lambda: SimpleNamespace(kv_cache=kv_cache),
    )

    captured = {}

    def fake_paged_attention(*args, **kwargs):
        captured["sinks"] = kwargs["sinks"]
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr("freetoken.kernel.triton.attention.paged_attention", fake_paged_attention)

    backend = TritonAttentionBackend(SimpleNamespace())
    batch = SimpleNamespace(
        attn_metadata=TritonMetadata(
            cu_seqlens_q_gpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
            indices=torch.tensor([0, 1], dtype=torch.int32),
            q_to_req=torch.tensor([0, 1], dtype=torch.int32),
            q_positions=torch.tensor([0, 0], dtype=torch.int64),
            is_decode=False,
            prefix_lens=torch.tensor([0, 0], dtype=torch.int32),
            max_q_len=1,
        ),
        out_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    q = torch.randn(2, 2, 4)
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    sinks = torch.tensor([0.25, -0.5])

    out = backend.forward(q, k, v, 0, batch, attn_spec=AttentionSpec(sinks=sinks))

    assert out.shape == q.shape
    assert captured["sinks"] is sinks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("sliding_window", [None, 3])
def test_paged_triton_attention_matches_reference(head_dim: int, sliding_window: int | None):
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_q_heads = 2
    num_kv_heads = 1
    q = torch.randn(4, num_q_heads, head_dim, device=device)
    k_cache = torch.randn(8, num_kv_heads, head_dim, device=device)
    v_cache = torch.randn(8, num_kv_heads, head_dim, device=device)
    indptr = torch.tensor([0, 5, 8], dtype=torch.int32, device=device)
    indices = torch.arange(8, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=device)
    q_positions = torch.tensor([2, 3, 4, 2], dtype=torch.int64, device=device)
    sm_scale = head_dim**-0.5

    actual = paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window=sliding_window,
        block_n=4,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_paged_triton_attention_with_sinks_matches_reference():
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(10)
    device = torch.device("cuda")
    head_dim = 256
    num_q_heads = 4
    num_kv_heads = 1
    q = torch.randn(3, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(6, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(6, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, 4, 6], dtype=torch.int32, device=device)
    indices = torch.arange(6, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    q_positions = torch.tensor([1, 3, 1], dtype=torch.int64, device=device)
    sinks = torch.tensor([1.2, -0.4, 0.7, -1.0], dtype=torch.float32, device=device)
    sm_scale = head_dim**-0.5

    actual = paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sinks=sinks,
        block_n=4,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
        sinks=sinks,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_paged_triton_attention_skips_all_masked_sliding_blocks():
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(0)
    device = torch.device("cuda")
    head_dim = 256
    q = torch.randn(1, 2, head_dim, device=device)
    k_cache = torch.randn(8, 1, head_dim, device=device)
    v_cache = torch.randn(8, 1, head_dim, device=device)
    indptr = torch.tensor([0, 8], dtype=torch.int32, device=device)
    indices = torch.arange(8, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0], dtype=torch.int32, device=device)
    q_positions = torch.tensor([7], dtype=torch.int64, device=device)
    sliding_window = 3
    sm_scale = head_dim**-0.5

    actual = paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window=sliding_window,
        block_n=4,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize(
    ("head_dim", "num_kv_heads", "sliding_window"),
    [
        (256, 8, 3),
        (512, 2, None),
    ],
)
def test_decode_triton_attention_matches_reference(
    head_dim: int,
    num_kv_heads: int,
    sliding_window: int | None,
):
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(1)
    device = torch.device("cuda")
    batch = 2
    num_q_heads = 16
    max_kv_splits = 8
    seq_lens = [5, 7]
    total_kv = sum(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=device)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    q_positions = torch.tensor([seq_lens[0] - 1, seq_lens[1] - 1], dtype=torch.int64, device=device)
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    sm_scale = head_dim**-0.5
    attn_logits = torch.empty(
        batch,
        num_q_heads,
        max_kv_splits,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)
    num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device)

    actual = decode_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        sliding_window=sliding_window,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_decode_launch_config_selects_ornith_quantized_tuning_only():
    from freetoken.kernel.triton.attention import decode_launch_config

    assert decode_launch_config(
        quant_name="int4", head_dim=256, num_q_heads=16, num_kv_heads=2,
        compute_capability=(8, 9),
    ) == (32, 32, 4)
    assert decode_launch_config(
        quant_name="int4", head_dim=256, num_q_heads=16, num_kv_heads=2,
        compute_capability=(12, 0),
    ) == (64, 64, 8)
    assert decode_launch_config(
        quant_name="q8_0", head_dim=256, num_q_heads=16, num_kv_heads=2,
        compute_capability=(8, 9),
    ) == (16, 64, 4)
    assert decode_launch_config(
        quant_name="q8_0", head_dim=256, num_q_heads=16, num_kv_heads=2,
        compute_capability=(12, 0),
    ) == (64, 64, 4)
    assert decode_launch_config(
        quant_name="q8_q6", head_dim=256, num_q_heads=16, num_kv_heads=2,
        compute_capability=(8, 9),
    ) == (32, 32, 4)
    assert decode_launch_config(
        quant_name="q8_q6", head_dim=256, num_q_heads=16, num_kv_heads=2,
        compute_capability=(12, 0),
    ) == (64, 32, 8)
    assert decode_launch_config(
        quant_name="q6_q5", head_dim=256, num_q_heads=16, num_kv_heads=2,
        compute_capability=(8, 9),
    ) == (64, 32, 8)
    assert decode_launch_config(
        quant_name="q6_q5", head_dim=256, num_q_heads=16, num_kv_heads=2,
        compute_capability=(12, 0),
    ) == (128, 32, 4)
    assert decode_launch_config(
        quant_name="quant8", head_dim=256, num_q_heads=16, num_kv_heads=2
    ) == (64, 64, 4)
    assert decode_launch_config(
        quant_name=None, head_dim=256, num_q_heads=16, num_kv_heads=2
    ) == (8, 32, 4)
    assert decode_launch_config(
        quant_name="int4", head_dim=256, num_q_heads=24, num_kv_heads=4
    ) == (8, 32, 4)


def test_decode_launch_config_keeps_tuned_branches_when_the_sm_count_is_known():
    """The measured Ornith branches win on their own geometry; the grid rule must not eat them.

    Pinned against the 2026-09-04 RTX 5080 sweep (benchmarks/bench_decode_launch.py):
    at 16Q/2KV/D256 + q8_0 the tuned (64, 64, 4) was the fastest of 32 configurations at
    both 131K and 262K, so passing ``sm_count`` may not change any tuned answer.
    """
    from freetoken.kernel.triton.attention import decode_launch_config

    ornith = {"head_dim": 256, "num_q_heads": 16, "num_kv_heads": 2}
    tuned = [
        ("int4", (8, 9), (32, 32, 4)),
        ("int4", (12, 0), (64, 64, 8)),
        ("q8_0", (8, 9), (16, 64, 4)),
        ("q8_0", (12, 0), (64, 64, 4)),
        ("q8_q6", (12, 0), (64, 32, 8)),
        ("q6_q5", (12, 0), (128, 32, 4)),
    ]
    for quant_name, capability, expected in tuned:
        kwargs = {"quant_name": quant_name, "compute_capability": capability, **ornith}
        assert decode_launch_config(**kwargs) == expected
        assert decode_launch_config(sm_count=84, **kwargs) == expected


def test_decode_launch_config_fills_the_gpu_for_untuned_head_shapes():
    """Nemotron 3.5 Lightning (32Q/2KV/D128) is the shape that had no branch at all.

    Stage 1's grid is ``batch * cdiv(num_q_heads, min(16, group)) * kv_splits``, i.e. two
    head blocks for this geometry, so the old flat 8-split fallback put 16 CTAs on 84 SMs
    and decode time grew linearly with context. 64 splits (128 CTAs) measured 8.3x/9.1x/
    9.6x/9.7x faster per layer at 131K/262K/524K/1M on the RTX 5080.
    """
    from freetoken.kernel.triton.attention import (
        _decode_head_blocks,
        _grid_filling_splits,
        decode_launch_config,
    )

    nemotron = {"quant_name": "q8_0", "head_dim": 128, "num_q_heads": 32, "num_kv_heads": 2}
    assert _decode_head_blocks(32, 2) == 2
    assert decode_launch_config(compute_capability=(12, 0), sm_count=84, **nemotron) == (64, 64, 8)
    # No SM count (CPU device, direct kernel callers): the historical conservative answer.
    assert decode_launch_config(compute_capability=(12, 0), **nemotron) == (8, 32, 4)

    # head_dim > 128 keeps the narrow tile: the wide one doubles the register footprint
    # and measured slower on the 16Q/2KV/D256 bf16 pool.
    assert decode_launch_config(
        quant_name=None, head_dim=256, num_q_heads=16, num_kv_heads=2, sm_count=84
    ) == (64, 32, 4)

    # More head blocks need fewer splits to fill the same GPU, and MHA-shaped grids
    # (one head block per query head) never fall below the historical floor.
    assert _decode_head_blocks(64, 8) == 8
    assert _grid_filling_splits(num_q_heads=64, num_kv_heads=8, sm_count=84) == 16
    assert _grid_filling_splits(num_q_heads=32, num_kv_heads=32, sm_count=84) == 8
    assert _grid_filling_splits(num_q_heads=32, num_kv_heads=2, sm_count=2048) == 128


def test_decode_launch_config_environment_override():
    from freetoken.kernel.triton import attention as attn

    attn._decode_launch_env_override.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("FREETOKEN_DECODE_KV_SPLITS", "8")
            attn._decode_launch_env_override.cache_clear()
            assert attn.decode_launch_config(
                quant_name="q8_0", head_dim=128, num_q_heads=32, num_kv_heads=2,
                compute_capability=(12, 0), sm_count=84,
            ) == (8, 64, 8)
    finally:
        attn._decode_launch_env_override.cache_clear()


def test_slot_offsets_need_int64_only_for_pools_past_the_int32_ceiling():
    """``slots * stride`` is 32-bit arithmetic in the kernels; widen only where it wraps."""
    from freetoken.kernel.triton.attention import _slot_offsets_need_int64

    # Nemotron's 2 KV heads x 128 dim: 1M slots is 268M elements, far inside int32.
    small = torch.empty((1 << 20, 2, 128), device="meta", dtype=torch.int8)
    assert not _slot_offsets_need_int64(small, small, None, None)
    # 8 KV heads x 256 dim at 1M slots is 2**31 elements exactly -- the 1M profile.
    big = torch.empty((1 << 20, 8, 256), device="meta", dtype=torch.int8)
    assert big.numel() == 2**31
    assert _slot_offsets_need_int64(small, big, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_decode_triton_attention_matches_reference_at_the_nemotron_head_shape():
    """32Q/2KV/D128 over a context long enough that the auto split count really splits."""
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(3)
    device = torch.device("cuda")
    batch, num_q_heads, num_kv_heads, head_dim = 1, 32, 2, 128
    ctx = 4096
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(ctx, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(ctx, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, ctx], dtype=torch.int32, device=device)
    indices = torch.arange(ctx, dtype=torch.int32, device=device)
    q_positions = torch.tensor([ctx - 1], dtype=torch.int64, device=device)
    q_to_req = torch.zeros(batch, dtype=torch.int32, device=device)
    sm_scale = head_dim**-0.5

    def run(splits: int):
        logits = torch.empty(
            batch, num_q_heads, splits, head_dim, dtype=torch.float32, device=device
        )
        lse = torch.empty(batch, num_q_heads, splits, dtype=torch.float32, device=device)
        counts = torch.full((batch,), splits, dtype=torch.int32, device=device)
        return decode_paged_attention(
            q, k_cache, v_cache, indptr, indices, q_positions, logits, lse, counts,
            splits, sm_scale,
        )

    expected = _reference_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_to_req, q_positions, sm_scale, None
    )
    # 64 is what the grid rule asks for on an 84-SM part; 8 is the old fallback. Split-K
    # reorders the log-sum-exp reduction, so these agree at bf16 tolerance, not bitwise.
    torch.testing.assert_close(run(64).float(), expected.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(run(64).float(), run(8).float(), atol=2e-2, rtol=2e-2)


def test_decode_runtime_splits_uses_measured_ada_batch_two_policy():
    from freetoken.kernel.triton.attention import decode_runtime_splits

    common = dict(
        preferred_splits=32,
        scratch_splits=32,
        quant_name="int4",
        head_dim=256,
        num_q_heads=16,
        num_kv_heads=2,
        compute_capability=(8, 9),
    )
    assert decode_runtime_splits(batch=1, **common) == 32
    assert decode_runtime_splits(batch=2, **common) == 16
    assert decode_runtime_splits(batch=4, **common) == 32
    assert decode_runtime_splits(batch=2, **(common | {"scratch_splits": 8})) == 8
    assert decode_runtime_splits(
        batch=2, **(common | {"compute_capability": (12, 0)})
    ) == 32


def test_decode_runtime_splits_uses_measured_ada_q8_q6_batch_four_policy():
    from freetoken.kernel.triton.attention import decode_runtime_splits

    common = dict(
        preferred_splits=32,
        scratch_splits=32,
        quant_name="q8_q6",
        head_dim=256,
        num_q_heads=16,
        num_kv_heads=2,
        compute_capability=(8, 9),
    )
    assert decode_runtime_splits(batch=1, **common) == 32
    assert decode_runtime_splits(batch=2, **common) == 32
    assert decode_runtime_splits(batch=3, **common) == 32
    assert decode_runtime_splits(batch=4, **common) == 16
    assert decode_runtime_splits(
        batch=4, **(common | {"compute_capability": (12, 0)})
    ) == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize(("num_q_heads", "num_kv_heads"), [(24, 4), (20, 4), (28, 4)])
def test_decode_triton_attention_non_pow2_group(num_q_heads: int, num_kv_heads: int):
    """GQA groups that are not a power of two (e.g. Qwen3.6-27B's 24/4 == 6). The grouped
    decode tiles the head axis to a power of two (tl.arange constraint) and masks the extra
    lanes; the result must still match the reference."""
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(3)
    device = torch.device("cuda")
    batch = 2
    head_dim = 256
    max_kv_splits = 8
    seq_lens = [5, 7]
    total_kv = sum(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=device)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    q_positions = torch.tensor([seq_lens[0] - 1, seq_lens[1] - 1], dtype=torch.int64, device=device)
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    sm_scale = head_dim**-0.5
    attn_logits = torch.empty(
        batch, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)
    num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device)

    actual = decode_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_positions,
        attn_logits, attn_lse, num_kv_splits, max_kv_splits, sm_scale,
    )
    expected = _reference_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_to_req, q_positions, sm_scale, None,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_decode_triton_attention_with_sinks_matches_reference():
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(11)
    device = torch.device("cuda")
    batch = 2
    num_q_heads = 4
    num_kv_heads = 1
    head_dim = 256
    max_kv_splits = 8
    seq_lens = [5, 7]
    total_kv = sum(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=device)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    q_positions = torch.tensor(
        [seq_lens[0] - 1, seq_lens[1] - 1],
        dtype=torch.int64,
        device=device,
    )
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    sinks = torch.tensor([1.4, -0.6, 0.25, -1.2], dtype=torch.float32, device=device)
    sm_scale = head_dim**-0.5
    attn_logits = torch.empty(
        batch,
        num_q_heads,
        max_kv_splits,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)
    num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device)

    actual = decode_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        sinks=sinks,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
        sinks=sinks,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("use_split_inputs", [False, True])
@pytest.mark.parametrize(
    ("head_dim", "num_kv_heads", "sliding_window", "cached_lens", "extend_lens"),
    [
        (256, 8, None, [0, 0], [5, 3]),
        (256, 8, 4, [3, 2], [4, 3]),
        (512, 2, None, [2, 4], [3, 2]),
    ],
)
def test_extend_triton_attention_matches_reference(
    use_split_inputs: bool,
    head_dim: int,
    num_kv_heads: int,
    sliding_window: int | None,
    cached_lens: list[int],
    extend_lens: list[int],
):
    from freetoken.kernel.triton.attention import extend_paged_attention

    torch.manual_seed(2)
    device = torch.device("cuda")
    num_q_heads = 16
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    total_q = sum(extend_lens)
    total_kv = sum(seq_lens)
    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_extend = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_extend = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor(
        [0] + extend_lens,
        dtype=torch.int32,
        device=device,
    ).cumsum_(0)
    kv_indptr = torch.tensor([0] + seq_lens, dtype=torch.int32, device=device).cumsum_(0)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)
    q_to_req = torch.empty(total_q, dtype=torch.int32, device=device)
    q_positions = torch.empty(total_q, dtype=torch.int64, device=device)
    offset = 0
    kv_offset = 0
    for req_idx, (cached_len, extend_len) in enumerate(zip(cached_lens, extend_lens)):
        q_to_req[offset : offset + extend_len].fill_(req_idx)
        q_positions[offset : offset + extend_len] = torch.arange(
            cached_len,
            cached_len + extend_len,
            dtype=torch.int64,
            device=device,
        )
        k_cache[kv_offset + cached_len : kv_offset + cached_len + extend_len] = k_extend[
            offset : offset + extend_len
        ]
        v_cache[kv_offset + cached_len : kv_offset + cached_len + extend_len] = v_extend[
            offset : offset + extend_len
        ]
        offset += extend_len
        kv_offset += cached_len + extend_len
    sm_scale = head_dim**-0.5

    actual = extend_paged_attention(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        indices,
        prefix_lens,
        max(extend_lens),
        sm_scale,
        sliding_window=sliding_window,
        k_extend=k_extend if use_split_inputs else None,
        v_extend=v_extend if use_split_inputs else None,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("use_split_inputs", [False, True])
def test_extend_triton_attention_with_sinks_matches_reference(use_split_inputs: bool):
    from freetoken.kernel.triton.attention import extend_paged_attention

    torch.manual_seed(12)
    device = torch.device("cuda")
    head_dim = 256
    num_q_heads = 4
    num_kv_heads = 1
    cached_lens = [1, 2]
    extend_lens = [2, 1]
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    total_q = sum(extend_lens)
    total_kv = sum(seq_lens)
    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_extend = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_extend = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor([0] + extend_lens, dtype=torch.int32, device=device).cumsum_(0)
    kv_indptr = torch.tensor([0] + seq_lens, dtype=torch.int32, device=device).cumsum_(0)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)
    q_to_req = torch.empty(total_q, dtype=torch.int32, device=device)
    q_positions = torch.empty(total_q, dtype=torch.int64, device=device)
    offset = 0
    kv_offset = 0
    for req_idx, (cached_len, extend_len) in enumerate(zip(cached_lens, extend_lens)):
        q_to_req[offset : offset + extend_len].fill_(req_idx)
        q_positions[offset : offset + extend_len] = torch.arange(
            cached_len,
            cached_len + extend_len,
            dtype=torch.int64,
            device=device,
        )
        k_cache[kv_offset + cached_len : kv_offset + cached_len + extend_len] = k_extend[
            offset : offset + extend_len
        ]
        v_cache[kv_offset + cached_len : kv_offset + cached_len + extend_len] = v_extend[
            offset : offset + extend_len
        ]
        offset += extend_len
        kv_offset += cached_len + extend_len
    sinks = torch.tensor([1.0, -0.25, 0.5, -1.1], dtype=torch.float32, device=device)
    sm_scale = head_dim**-0.5

    actual = extend_paged_attention(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        indices,
        prefix_lens,
        max(extend_lens),
        sm_scale,
        sinks=sinks,
        k_extend=k_extend if use_split_inputs else None,
        v_extend=v_extend if use_split_inputs else None,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
        sinks=sinks,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("head_dim", "smem_optin", "expected"),
    [
        # datacenter opt-in smem (A100 ~164KB / H100 ~227KB): keep the fast tiles where they fit
        (128, 232448, (128, 64)),
        (256, 232448, (128, 64)),
        (512, 232448, (32, 64)),
        (256, 167936, (128, 64)),  # A100: hd256 fast tile fits
        (512, 167936, (16, 16)),  # A100: hd512 fast tile does not fit -> shrink (no smem overflow)
        # consumer opt-in smem (sm_89 ~99KB): shrink once head_dim >= 256
        (256, 101376, (64, 32)),
        (512, 101376, (16, 16)),
        # unknown budget -> conservative small tiles (prior consumer-safe behavior)
        (256, 0, (64, 32)),
        (512, 0, (16, 16)),
    ],
)
def test_select_extend_tile_is_shared_memory_aware(head_dim, smem_optin, expected):
    import triton

    from freetoken.kernel.triton.attention import _select_extend_tile

    block_d = triton.next_power_of_2(head_dim)
    assert _select_extend_tile(head_dim, block_d, smem_optin) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_triton_backend_stores_kv_and_matches_reference(monkeypatch):
    from freetoken.attention import AttentionSpec
    from freetoken.attention.triton import TritonAttentionBackend

    class FakeKVCache:
        def __init__(self, device: torch.device, head_dim: int):
            self.device = device
            self.dtype = torch.float32
            self.k = torch.zeros(4, 1, head_dim, device=device)
            self.v = torch.zeros(4, 1, head_dim, device=device)

        def store_kv(self, k, v, out_loc, layer_id):
            self.k[out_loc.to(torch.long)] = k.view(k.shape[0], 1, -1)
            self.v[out_loc.to(torch.long)] = v.view(v.shape[0], 1, -1)

        def k_cache(self, layer_id):
            return self.k

        def v_cache(self, layer_id):
            return self.v

    device = torch.device("cuda")
    head_dim = 256
    page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    kv_cache = FakeKVCache(device, head_dim)
    ctx = SimpleNamespace(kv_cache=kv_cache, page_table=page_table)
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)

    backend = TritonAttentionBackend(SimpleNamespace())
    batch = SimpleNamespace(
        padded_reqs=[
            SimpleNamespace(extend_len=1, device_len=2, cached_len=1, table_idx=0),
            SimpleNamespace(extend_len=1, device_len=2, cached_len=1, table_idx=1),
        ],
        positions=torch.tensor([1, 1], dtype=torch.int64, device=device),
        out_loc=torch.tensor([1, 3], dtype=torch.int32, device=device),
    )
    kv_cache.k[0] = torch.randn(1, head_dim, device=device)
    kv_cache.v[0] = torch.randn(1, head_dim, device=device)
    kv_cache.k[2] = torch.randn(1, head_dim, device=device)
    kv_cache.v[2] = torch.randn(1, head_dim, device=device)
    q = torch.randn(2, 2, head_dim, device=device)
    k = torch.randn(2, head_dim, device=device)
    v = torch.randn(2, head_dim, device=device)

    backend.prepare_metadata(batch)
    actual = backend.forward(
        q,
        k,
        v,
        layer_id=0,
        batch=batch,
        attn_spec=AttentionSpec(sliding_window=None, sm_scale=head_dim**-0.5),
    )
    expected = _reference_paged_attention(
        q,
        kv_cache.k,
        kv_cache.v,
        batch.attn_metadata.indptr,
        batch.attn_metadata.indices,
        batch.attn_metadata.q_to_req,
        batch.attn_metadata.q_positions,
        head_dim**-0.5,
        sliding_window=None,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_triton_backend_replay_metadata_uses_capture_buffers(monkeypatch):
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata

    class FakeKVCache:
        def __init__(self, device: torch.device):
            self.device = device

    device = torch.device("cuda")
    page_table = torch.arange(16, dtype=torch.int32, device=device).view(2, 8)
    ctx = SimpleNamespace(kv_cache=FakeKVCache(device), page_table=page_table)
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)

    backend = TritonAttentionBackend(SimpleNamespace())
    backend.init_capture_graph(max_seq_len=8, bs_list=[2])
    assert backend.capture is not None

    capture_batch = SimpleNamespace(size=2)
    backend.prepare_for_capture(capture_batch)
    capture_metadata = capture_batch.attn_metadata
    assert isinstance(capture_metadata, TritonMetadata)
    assert capture_metadata.cu_seqlens_q_gpu.data_ptr() == backend.capture.cu_seqlens_q.data_ptr()
    assert capture_metadata.indptr.data_ptr() == backend.capture.cu_seqlens_k.data_ptr()
    assert capture_metadata.indices.data_ptr() == backend.capture.page_table.view(-1).data_ptr()
    assert capture_metadata.q_positions.data_ptr() == backend.capture.positions.data_ptr()

    runtime_batch = SimpleNamespace(
        padded_size=2,
        padded_reqs=[
            SimpleNamespace(extend_len=1, device_len=3, cached_len=2, table_idx=0),
            SimpleNamespace(extend_len=1, device_len=5, cached_len=4, table_idx=1),
        ],
        positions=torch.tensor([2, 4], dtype=torch.int64, device=device),
    )
    backend.prepare_metadata(runtime_batch)
    runtime_metadata = runtime_batch.attn_metadata
    assert isinstance(runtime_metadata, TritonMetadata)
    expected_indptr = runtime_metadata.indptr.clone()
    expected_indices = runtime_metadata.indices.clone()
    expected_positions = runtime_metadata.q_positions.clone()

    backend.prepare_for_replay(runtime_batch)
    replay_metadata = runtime_batch.attn_metadata
    assert isinstance(replay_metadata, TritonMetadata)
    assert replay_metadata.cu_seqlens_q_gpu.data_ptr() == backend.capture.cu_seqlens_q.data_ptr()
    assert replay_metadata.indptr.data_ptr() == backend.capture.cu_seqlens_k.data_ptr()
    assert replay_metadata.indices.data_ptr() == backend.capture.page_table.view(-1).data_ptr()
    assert replay_metadata.q_positions.data_ptr() == backend.capture.positions.data_ptr()
    assert replay_metadata.attn_logits is not None
    assert replay_metadata.attn_lse is not None
    assert replay_metadata.num_kv_splits is not None
    assert replay_metadata.attn_logits.data_ptr() == backend.capture.attn_logits.data_ptr()
    assert replay_metadata.attn_lse.data_ptr() == backend.capture.attn_lse.data_ptr()
    assert replay_metadata.num_kv_splits.data_ptr() == backend.capture.num_kv_splits.data_ptr()

    torch.testing.assert_close(backend.capture.cu_seqlens_k[:3], expected_indptr)
    torch.testing.assert_close(backend.capture.page_table.view(-1)[:8], expected_indices)
    torch.testing.assert_close(backend.capture.positions[:2].to(torch.int64), expected_positions)


def test_triton_metadata_keeps_full_indices_and_optional_swa_indices(monkeypatch):
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata

    page_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )
    # Global-paged SWA: swa_indices = translate(full page-table indices) via the full->swa map.
    ctx = SimpleNamespace(
        kv_cache=SimpleNamespace(
            device=torch.device("cpu"),
            swa_paged=True,
            translate_loc_from_full_to_swa=lambda idx: idx + 100,
        ),
        page_table=page_table,
    )
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)

    backend = TritonAttentionBackend(SimpleNamespace())
    batch = SimpleNamespace(
        padded_reqs=[
            SimpleNamespace(extend_len=1, device_len=2, cached_len=1, table_idx=0),
            SimpleNamespace(extend_len=2, device_len=3, cached_len=1, table_idx=1),
        ],
        positions=torch.tensor([1, 1, 2], dtype=torch.int64),
    )

    backend.prepare_metadata(batch)
    metadata = batch.attn_metadata

    assert isinstance(metadata, TritonMetadata)
    assert metadata.indices.tolist() == [10, 11, 20, 21, 22]
    assert metadata.swa_indices is not None
    assert metadata.swa_indices.tolist() == [110, 111, 120, 121, 122]


@pytest.mark.parametrize(
    ("block_dv", "num_warps", "expected"),
    [
        (128, 4, 64),  # sm_120 Nemotron 32Q/2KV/D128 -- the 2026-09-05 prefill fix
        (128, 8, 128),  # 8-warp devices still admit the historical 128-row tile
        (256, 4, 32),
        (256, 8, 64),
        (512, 4, 16),
        (1024, 4, 16),  # never below the floor
    ],
)
def test_extend_block_m_cap_follows_the_accumulator(block_dv, num_warps, expected):
    """BLOCK_M is capped by fp32 accumulator registers per thread, not by a constant."""
    from freetoken.kernel.triton.attention import _EXTEND_ACC_REGS, _extend_block_m_cap

    assert _extend_block_m_cap(block_dv, num_warps) == expected
    if expected > 16:
        assert expected * block_dv / (32 * num_warps) <= _EXTEND_ACC_REGS


@pytest.mark.parametrize(
    ("head_dim", "smem_optin", "capability", "formats", "expected"),
    [
        # THE FIX: sm_120 takes 4 warps, so the hard-coded 128-row tile spilled
        # (396 spill slots vs 14) and ran at 29.3 TFLOP/s instead of 70.4.
        (128, 101376, (12, 0), (0, 0), (64, 64, 4, 1)),
        # 8-warp devices are untouched: same tile, same warps, same stages as before.
        (128, 101376, (8, 9), (0, 0), (128, 64, 8, 1)),
        (128, 232448, (9, 0), (0, 0), (128, 64, 8, 1)),
        # Measured head_dim >= 256 branches are overrides and must not move.
        (256, 101376, (12, 0), (0, 0), (64, 32, 4, 2)),
        (256, 101376, (12, 0), (3, 4), (64, 32, 8, 2)),  # q6/q5 unpacking lanes
        (256, 232448, (9, 0), (0, 0), (128, 64, 8, 1)),
        (512, 101376, (12, 0), (0, 0), (16, 16, 4, 1)),  # gemma4 full attention
    ],
)
def test_extend_launch_config(head_dim, smem_optin, capability, formats, expected):
    import triton

    from freetoken.kernel.triton.attention import extend_launch_config

    got = extend_launch_config(
        head_dim=head_dim,
        block_d=triton.next_power_of_2(head_dim),
        smem_optin=smem_optin,
        capability=capability,
        k_format=formats[0],
        v_format=formats[1],
    )
    assert got == expected


def test_extend_launch_env_override(monkeypatch):
    """The prefill twin of FREETOKEN_DECODE_*, so a launch change is A/B-able."""
    import triton

    from freetoken.kernel.triton import attention as attn

    def cfg():
        attn._extend_launch_env_override.cache_clear()
        return attn.extend_launch_config(
            head_dim=128,
            block_d=triton.next_power_of_2(128),
            smem_optin=101376,
            capability=(12, 0),
        )

    try:
        assert cfg() == (64, 64, 4, 1)
        monkeypatch.setenv("FREETOKEN_EXTEND_BLOCK_M", "128")
        monkeypatch.setenv("FREETOKEN_EXTEND_NUM_WARPS", "8")
        assert cfg() == (128, 64, 8, 1)
        monkeypatch.setenv("FREETOKEN_EXTEND_BLOCK_N", "32")
        monkeypatch.setenv("FREETOKEN_EXTEND_NUM_STAGES", "2")
        assert cfg() == (128, 32, 8, 2)
    finally:
        attn._extend_launch_env_override.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_extend_launch_change_agrees_with_the_previous_tile(monkeypatch):
    """The 64-row tile must agree with the 128-row one it replaces.

    Changing BLOCK_M changes the order of the flash accumulation, so the two are not
    bit-identical and a bitwise gate would reject a correct 2.46x. Gate on agreement
    against both the old launch and the dense reference instead.
    """
    from freetoken.kernel.triton import attention as attn

    torch.manual_seed(7)
    device = torch.device("cuda")
    # Nemotron 3.5 Lightning attention geometry.
    num_q_heads, num_kv_heads, head_dim = 32, 2, 128
    cached_len, extend_len = 384, 192
    total_kv = cached_len + extend_len
    q = torch.randn(extend_len, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_extend = k_cache[cached_len:].clone()
    v_extend = v_cache[cached_len:].clone()
    qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, total_kv], dtype=torch.int32, device=device)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([cached_len], dtype=torch.int32, device=device)
    q_to_req = torch.zeros(extend_len, dtype=torch.int32, device=device)
    q_positions = torch.arange(
        cached_len, total_kv, dtype=torch.int64, device=device
    )
    sm_scale = head_dim**-0.5

    def run():
        attn._extend_launch_env_override.cache_clear()
        return extend_paged_attention_call()

    def extend_paged_attention_call():
        return attn.extend_paged_attention(
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            indices,
            prefix_lens,
            extend_len,
            sm_scale,
            k_extend=k_extend,
            v_extend=v_extend,
        )

    try:
        new = run()
        monkeypatch.setenv("FREETOKEN_EXTEND_BLOCK_M", "128")
        monkeypatch.setenv("FREETOKEN_EXTEND_BLOCK_N", "64")
        monkeypatch.setenv("FREETOKEN_EXTEND_NUM_WARPS", "4")
        monkeypatch.setenv("FREETOKEN_EXTEND_NUM_STAGES", "1")
        old = run()
    finally:
        attn._extend_launch_env_override.cache_clear()

    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
    )
    torch.testing.assert_close(new.float(), old.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(new.float(), expected.float(), atol=2e-2, rtol=2e-2)
