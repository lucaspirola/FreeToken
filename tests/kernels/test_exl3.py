"""EXL3 kernels (``kernel/triton/exl3.py``, ``moe/fused_exl3.py``) against two oracles.

* exllamav3 itself, through ``tests/fixtures/exl3/exllamav3_ref.npz`` (written by
  ``tests/fixtures/exl3/make_exllamav3_ref.py`` in an exllamav3 venv): its ``reconstruct``
  output and its ``LinearEXL3`` forward on seeded random tensors.
* the torch reference decoder (``reconstruct_reference`` / ``linear_reference``), itself
  pinned to exllamav3 by the first test.

Error metric: ``max|y - ref| / max|ref|`` over the whole output ("rel"). The kernels round the
rotated input to fp16 (as exllamav3 does), so a few 1e-3 is expected; decode (codebook values)
must be bit-exact.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from freetoken.kernel.triton.exl3 import CODEBOOKS, linear_reference, reconstruct_reference

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "exl3", "exllamav3_ref.npz")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton EXL3 kernels need CUDA")
ALL_BITS = tuple(range(2, 9))
REL_TOL = 4e-3


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12))


@pytest.fixture(scope="module")
def fx():
    if not os.path.isfile(FIXTURE):
        pytest.skip(f"{FIXTURE} missing (run make_exllamav3_ref.py in an exllamav3 venv)")
    return dict(np.load(FIXTURE))


def _rand_trellis(k, n, bits, g, device="cpu"):
    return torch.randint(-(1 << 15), 1 << 15, (k // 16, n // 16, 16 * bits), generator=g, dtype=torch.int32).to(torch.int16).to(device)


# ---------------------------------------------------------------------------
# vs exllamav3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
def test_reference_decoder_is_exllamav3_reconstruct(fx, bits, codebook):
    tr = torch.from_numpy(fx[f"rec_trellis_{bits}_{codebook}"])
    want = torch.from_numpy(fx[f"rec_w_{bits}_{codebook}"])
    assert torch.equal(reconstruct_reference(tr, codebook).view(torch.int16), want.view(torch.int16))


@cuda
@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
def test_triton_reconstruct_is_exllamav3_reconstruct(fx, bits, codebook):
    from freetoken.kernel.triton.exl3 import reconstruct

    tr = torch.from_numpy(fx[f"rec_trellis_{bits}_{codebook}"]).cuda()
    want = torch.from_numpy(fx[f"rec_w_{bits}_{codebook}"])
    assert torch.equal(reconstruct(tr, codebook).cpu().view(torch.int16), want.view(torch.int16))


def _fixture_linears(fx):
    return sorted({k.split("lin_trellis_", 1)[1] for k in fx if k.startswith("lin_trellis_")})


def _linear_case(fx, tag):
    return {name: torch.from_numpy(fx[f"lin_{name}_{tag}"]) for name in ("trellis", "suh", "svh", "x", "y_gemv", "y_recon")}


def test_linear_reference_matches_exllamav3_forward(fx):
    tags = _fixture_linears(fx)
    assert len(tags) == 21
    for tag in tags:
        c = _linear_case(fx, tag)
        y = linear_reference(c["x"], c["trellis"], c["suh"], c["svh"], tag.split("_")[1])
        assert rel_err(y, c["y_recon"]) < REL_TOL, tag
        assert rel_err(y, c["y_gemv"]) < REL_TOL, tag


@cuda
def test_triton_linear_matches_exllamav3_forward(fx):
    """exl3_forward vs LinearEXL3.forward (its bsz GEMV kernel, and its reconstruct + hgemm path)."""
    from freetoken.kernel.triton.exl3 import Exl3Parts
    from freetoken.layers.quantization.linear.exl3 import exl3_forward

    worst = {}
    for tag in _fixture_linears(fx):
        c = {k: v.cuda() for k, v in _linear_case(fx, tag).items()}
        bits, codebook = int(tag.split("_")[0]), tag.split("_")[1]
        k, n = c["suh"].shape[0], c["svh"].shape[0]
        parts = Exl3Parts.build(k, (n,), bits, codebook, "cuda")
        y = exl3_forward(c["x"], c["trellis"].reshape(-1), c["suh"].view(1, -1), c["svh"], parts, torch.float16)
        for ref in ("y_gemv", "y_recon"):
            err = rel_err(y, c[ref])
            worst[(tag, ref)] = err
            assert err < REL_TOL, (tag, ref, err)
    print("\nexl3_forward vs exllamav3 rel err:", {f"{t}/{r}": f"{e:.2e}" for (t, r), e in worst.items()})


# ---------------------------------------------------------------------------
# Triton vs the torch reference (all bits, all codebooks, fused parts, every row regime)
# ---------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
def test_triton_reconstruct_matches_reference(bits, codebook):
    from freetoken.kernel.triton.exl3 import reconstruct

    g = torch.Generator().manual_seed(bits * 7 + CODEBOOKS[codebook])
    tr = _rand_trellis(256, 384, bits, g)
    assert torch.equal(reconstruct(tr.cuda(), codebook).cpu().view(torch.int16), reconstruct_reference(tr, codebook).view(torch.int16))


def _dense_case(k, sizes, bits, codebook, seed):
    g = torch.Generator().manual_seed(seed)
    trs = [_rand_trellis(k, s, bits, g) for s in sizes]
    suh = torch.stack([(torch.randn(k, generator=g) * 0.5).half() for _ in sizes])
    svh = torch.cat([(torch.randn(s, generator=g) * 0.02).half() for s in sizes])
    return trs, suh, svh


@cuda
@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
@pytest.mark.parametrize("rows", (1, 3, 8, 9, 40, 300))
def test_triton_linear_matches_reference(bits, codebook, rows):
    from freetoken.kernel.triton.exl3 import Exl3Parts
    from freetoken.layers.quantization.linear.exl3 import exl3_forward

    k, sizes = 512, (256, 128, 384)  # a fused q|k|v-like projection, part boundaries off the 256 grid
    trs, suh, svh = _dense_case(k, sizes, bits, codebook, seed=rows * 100 + bits)
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16)
    ref, col = [], 0
    for j, (tr, s) in enumerate(zip(trs, sizes)):
        ref.append(linear_reference(x.float(), tr, suh[j], svh[col:col + s], codebook))
        col += s
    ref = torch.cat(ref, dim=1)
    parts = Exl3Parts.build(k, sizes, bits, codebook, "cuda")
    flat = torch.cat([t.reshape(-1) for t in trs]).cuda()
    y32 = exl3_forward(x.cuda(), flat, suh.cuda(), svh.cuda(), parts, torch.float32)
    assert rel_err(y32.cpu(), ref) < REL_TOL
    # the model's bf16 output adds its own rounding (up to 2^-9 of each element)
    y = exl3_forward(x.cuda(), flat, suh.cuda(), svh.cuda(), parts, torch.bfloat16)
    assert y.shape == (rows, sum(sizes)) and y.dtype == torch.bfloat16
    assert rel_err(y.cpu(), ref) < 2 * REL_TOL


# ---------------------------------------------------------------------------
# MoE forward
# ---------------------------------------------------------------------------

E, H, I, TOPK = 8, 256, 128, 3


def _moe_banks(bits, codebook, n, seed):
    from freetoken.models.exl3_banks import exl3_bank_shapes

    g = torch.Generator().manual_seed(seed)
    banks = []
    for name, (shape, dtype) in exl3_bank_shapes(H, I, bits).items():
        if dtype == torch.int16:
            t = torch.randint(-(1 << 15), 1 << 15, (n, *shape), generator=g, dtype=torch.int32).to(torch.int16)
        elif name.endswith("suh"):
            t = (torch.randn((n, *shape), generator=g) * 0.5).half()
        else:
            t = (torch.randn((n, *shape), generator=g) * 0.05).half()
        banks.append(t)
    return tuple(banks)


def _routing(tokens, n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(n, generator=g)[:TOPK] for _ in range(tokens)]).to(torch.int32)
    w = torch.softmax(torch.randn(tokens, TOPK, generator=g), dim=-1)
    x = torch.randn(tokens, H, generator=g).to(torch.bfloat16)
    return x, w, ids


@cuda
@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
@pytest.mark.parametrize("mode,tokens", (("decode", 1), ("decode", 4), ("prefill", 5), ("prefill", 300)))
def test_moe_matches_reference(bits, codebook, mode, tokens):
    from freetoken.moe.fused_exl3 import fused_experts_exl3, fused_experts_exl3_reference

    n = 12 if mode == "decode" else E  # decode ids are cache slots: a bank of S != E rows
    banks = _moe_banks(bits, codebook, n, seed=bits)
    x, w, ids = _routing(tokens, n, seed=tokens)
    ref = fused_experts_exl3_reference(x, banks, w, ids, codebook=codebook)
    y = fused_experts_exl3(
        x.cuda(), tuple(b.cuda() for b in banks), w.cuda(), ids.cuda(), bits=bits, codebook=codebook,
        is_prefill=mode == "prefill", num_experts=n if mode == "prefill" else None,
    )
    assert y.shape == x.shape and y.dtype == x.dtype
    assert rel_err(y.cpu(), ref) < 8e-3, rel_err(y.cpu(), ref)


@cuda
def test_moe_prefill_chunks_agree_with_one_chunk(monkeypatch):
    import freetoken.moe.fused_exl3 as fe

    banks = tuple(b.cuda() for b in _moe_banks(5, "mul1", E, seed=1))
    x, w, ids = (t.cuda() for t in _routing(700, E, seed=3))
    kw = dict(bits=5, codebook="mul1", is_prefill=True, num_experts=E)
    whole = fe.fused_experts_exl3(x, banks, w, ids, **kw)
    monkeypatch.setattr(fe, "PREFILL_CHUNK_TOKENS", 128)
    chunked = fe.fused_experts_exl3(x, banks, w, ids, **kw)
    assert rel_err(chunked, whole) < 1e-2


@cuda
def test_moe_decode_replays_under_a_cuda_graph():
    from freetoken.moe.fused_exl3 import fused_experts_exl3

    S, tokens = 12, 2
    banks = tuple(b.cuda() for b in _moe_banks(5, "mul1", S, seed=5))
    x, w, ids = (t.cuda() for t in _routing(tokens, S, seed=7))
    kw = dict(bits=5, codebook="mul1", is_prefill=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # warm up: Triton compiles, lru caches fill
            fused_experts_exl3(x, banks, w, ids, **kw)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_experts_exl3(x, banks, w, ids, **kw)
    for seed in (11, 12, 13):
        x2, w2, ids2 = (t.cuda() for t in _routing(tokens, S, seed=seed))
        x.copy_(x2), w.copy_(w2), ids.copy_(ids2)
        graph.replay()
        eager = fused_experts_exl3(x2, banks, w2, ids2, **kw)
        torch.cuda.synchronize()
        assert rel_err(out, eager) < 1e-3
