"""flashinfer extend path (freetoken/kernel/extend_flashinfer.py) against the triton kernel."""
from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.quant import BLOCK, FP8_E4M3, NONE, Q8_0

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
flashinfer = pytest.importorskip("flashinfer")


def _rand(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.bfloat16)


def _pool(spec, k, v):
    if not spec.enabled:
        return k, None, v, None
    from freetoken.kernel.triton.kv_quant import store_kv_quant

    slots, heads, dim = k.shape
    kq = torch.zeros(slots, heads, dim, device="cuda", dtype=spec.storage_dtype)
    vq = torch.zeros_like(kq)
    ks = torch.zeros(slots, heads, dim // BLOCK, device="cuda", dtype=torch.float16)
    vs = torch.zeros_like(ks)
    store_kv_quant(kq, ks, vq, vs, torch.arange(slots, device="cuda", dtype=torch.int32), k, v, spec)
    return kq, ks, vq, vs


@cuda_only
@pytest.mark.parametrize("spec", [Q8_0, FP8_E4M3, NONE], ids=["q8_0", "fp8", "bf16"])
@pytest.mark.parametrize("geom", [(16, 2, 256), (32, 2, 128), (8, 8, 64)], ids=["ornith", "nemotron", "mha64"])
def test_flashinfer_extend_matches_triton(monkeypatch, spec, geom):
    """Two sequences in one batch, scattered slots, one prefix spanning several
    dequant blocks (block 256) and one empty: flashinfer and triton agree to
    bf16 kernel noise."""
    from freetoken.kernel.triton.attention import extend_paged_attention

    monkeypatch.setenv("FREETOKEN_EXTEND_FI_BLOCK", "256")
    hq, hkv, d = geom
    qo_lens, prefix_lens = [300, 130], [0, 1000]
    kv_lens = [q + p for q, p in zip(qo_lens, prefix_lens)]
    slots = sum(kv_lens) + 64
    perm = torch.randperm(slots, generator=torch.Generator().manual_seed(3))[: sum(kv_lens)]
    kv_indices = perm.to(device="cuda", dtype=torch.int32)
    kc, ks, vc, vs = _pool(spec, _rand((slots, hkv, d), 1), _rand((slots, hkv, d), 2))
    t = sum(qo_lens)
    q = _rand((t, hq, d), 4)
    ke, ve = _rand((t, hkv, d), 5), _rand((t, hkv, d), 6)
    i32 = dict(device="cuda", dtype=torch.int32)
    kw = dict(
        q=q, k_cache=kc, v_cache=vc, k_scale=ks, v_scale=vs, k_extend=ke, v_extend=ve,
        qo_indptr=torch.tensor([0, qo_lens[0], t], **i32),
        kv_indptr=torch.tensor([0, kv_lens[0], sum(kv_lens)], **i32),
        kv_indices=kv_indices, prefix_lens=torch.tensor(prefix_lens, **i32),
        max_q_len=max(qo_lens), sm_scale=d ** -0.5,
    )
    want = extend_paged_attention(**kw)  # no host_lens: triton
    got = extend_paged_attention(**kw, host_lens=(qo_lens, prefix_lens, kv_lens))
    monkeypatch.setenv("FREETOKEN_EXTEND_BACKEND", "triton")
    forced = extend_paged_attention(**kw, host_lens=(qo_lens, prefix_lens, kv_lens))
    assert torch.equal(forced, want)
    err = (got.float() - want.float()).abs().max().item()
    assert err < 2e-2, err
    assert not torch.equal(got, want)  # it really ran the other kernel


@cuda_only
def test_flashinfer_extend_skips_sinks_and_windows():
    from freetoken.kernel import extend_flashinfer as fx

    q = torch.empty(4, 8, 128, device="cuda", dtype=torch.bfloat16)
    lens = ([4], [0], [4])
    assert fx.eligible(q, 1, 1, None, None, None, q, lens)
    assert not fx.eligible(q, 1, 1, 128, None, None, q, lens)
    assert not fx.eligible(q, 1, 1, None, q, None, q, lens)
    assert not fx.eligible(q, 2, 1, None, None, None, q, lens)   # Q4_0 pool
    assert not fx.eligible(q, 1, 1, None, None, None, q, None)   # no host lengths
