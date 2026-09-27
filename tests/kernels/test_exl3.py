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
@pytest.mark.parametrize("bits", (2, 5, 8))
def test_moe_prefill_folded_input_matches_rotated_path(monkeypatch, bits):
    """The decoded prefill with diag(suh) H folded into W_hat (raw activations into both GEMMs)
    equals the had_rows + GEMM path up to fp16 rounding of the folded weight."""
    import freetoken.moe.fused_exl3 as fe

    banks = tuple(b.cuda() for b in _moe_banks(bits, "mul1", E, seed=bits))
    x, w, ids = (t.cuda() for t in _routing(300, E, seed=5))
    kw = dict(bits=bits, codebook="mul1", is_prefill=True, num_experts=E)
    monkeypatch.setattr(fe, "PREFILL_FOLD_INPUT", True)
    folded = fe.fused_experts_exl3(x, banks, w, ids, **kw)
    monkeypatch.setattr(fe, "PREFILL_FOLD_INPUT", False)
    rotated = fe.fused_experts_exl3(x, banks, w, ids, **kw)
    assert rel_err(folded, rotated) < 3e-3, rel_err(folded, rotated)


@cuda
@pytest.mark.parametrize("m,k,n", ((300, 512, 768), (1024, 2048, 384), (77, 4096, 256)))
def test_gemm_f16acc_matches_fp32_accumulation(m, k, n):
    from freetoken.kernel.triton.exl3 import gemm_f16acc

    g = torch.Generator().manual_seed(m)
    a = (torch.randn(m, k, generator=g) * 0.5).half().cuda()
    b = (torch.randn(k, n, generator=g) * 0.05).half().cuda()
    ref = a.float() @ b.float()
    y = gemm_f16acc(a, b, out_dtype=torch.float32)
    assert rel_err(y, ref) < 5e-3, rel_err(y, ref)
    wide = torch.zeros(m, n + 128, dtype=torch.bfloat16, device="cuda")
    gemm_f16acc(a, b, out=wide[:, 64:64 + n])  # a column slice of a wider bf16 output
    assert torch.equal(wide[:, 64:64 + n], y.to(torch.bfloat16))


@cuda
def test_f16acc_prefill_paths_match_reference(monkeypatch):
    """FREETOKEN_EXL3_F16ACC=1: the MoE decoded-slab GEMM and the dense folded GEMM accumulate in
    fp16; both stay close to the fp32 references (the logits gate decides the default)."""
    import freetoken.moe.fused_exl3 as fe
    from freetoken.kernel.triton.exl3 import Exl3Parts
    from freetoken.layers.quantization.linear import exl3 as lin

    monkeypatch.setenv("FREETOKEN_EXL3_F16ACC", "1")
    banks = _moe_banks(5, "mul1", E, seed=21)
    x, w, ids = _routing(300, E, seed=22)
    ref = fe.fused_experts_exl3_reference(x, banks, w, ids, codebook="mul1")
    y = fe.fused_experts_exl3(x.cuda(), tuple(b.cuda() for b in banks), w.cuda(), ids.cuda(), bits=5,
                              codebook="mul1", is_prefill=True, num_experts=E)
    assert rel_err(y.cpu(), ref) < 1.5e-2, rel_err(y.cpu(), ref)
    k, sizes = 512, (256, 128, 384)
    trs, suh, svh = _dense_case(k, sizes, 5, "mul1", seed=23)
    xd = torch.randn(1100, k, generator=torch.Generator().manual_seed(24)).to(torch.bfloat16)
    dref, col = [], 0
    for j, (tr, s) in enumerate(zip(trs, sizes)):
        dref.append(linear_reference(xd.float(), tr, suh[j], svh[col:col + s], "mul1"))
        col += s
    dref = torch.cat(dref, dim=1)
    parts = Exl3Parts.build(k, sizes, 5, "mul1", "cuda")
    flat = torch.cat([t.reshape(-1) for t in trs]).cuda()
    monkeypatch.setattr(lin, "FOLD_MIN_ROWS", 16)
    monkeypatch.setattr(lin, "RECONSTRUCT_SLAB", 256)
    yd = lin.exl3_forward(xd.cuda(), flat, suh.cuda(), svh.cuda(), parts, torch.bfloat16)
    assert rel_err(yd.cpu(), dref) < 1.5e-2, rel_err(yd.cpu(), dref)


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
        assert torch.equal(out, eager)  # deterministic split-K: replay == eager bit for bit


@cuda
def test_split_k_decode_is_deterministic():
    """Split-K partials are summed in a fixed order, so decode repeats bit for bit (with
    atomics the real model's saver and whole-model runs drifted apart after ~15 tokens)."""
    from freetoken.kernel.triton.exl3 import Exl3Parts
    from freetoken.layers.quantization.linear.exl3 import exl3_forward, pick_split_k
    from freetoken.moe.fused_exl3 import fused_experts_exl3

    k, n = 2048, 1024
    assert pick_split_k(1, n // 128, k, torch.device("cuda")) > 1
    trs, suh, svh = _dense_case(k, (n,), 5, "mul1", seed=3)
    parts = Exl3Parts.build(k, (n,), 5, "mul1", "cuda")
    x = torch.randn(1, k, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16).cuda()
    args = (trs[0].reshape(-1).cuda(), suh.cuda(), svh.cuda(), parts, torch.bfloat16)
    first = exl3_forward(x, *args)
    assert all(torch.equal(first, exl3_forward(x, *args)) for _ in range(20))

    banks = tuple(b.cuda() for b in _moe_banks(5, "mul1", 12, seed=5))
    xm, w, ids = (t.cuda() for t in _routing(1, 12, seed=6))
    first = fused_experts_exl3(xm, banks, w, ids, bits=5, codebook="mul1", is_prefill=False)
    assert all(torch.equal(first, fused_experts_exl3(xm, banks, w, ids, bits=5, codebook="mul1", is_prefill=False)) for _ in range(20))


@cuda
def test_concurrent_split_k_gemvs_on_their_own_counter_lanes_match_serial():
    """The EXL3 decode shared expert runs on a side stream beside the routed GEMVs
    (models/qwen3_5_moe/moe.py). Split-K GEMVs running concurrently must count arrivals in
    separate counter lanes; on one lane they corrupt each other's reductions."""
    from freetoken.kernel.triton.exl3 import Exl3Parts, split_counter_lane
    from freetoken.layers.quantization.linear.exl3 import exl3_forward, pick_split_k
    from freetoken.moe.fused_exl3 import fused_experts_exl3

    k, n = 2048, 1024
    assert pick_split_k(1, n // 128, k, torch.device("cuda")) > 1
    trs, suh, svh = _dense_case(k, (n,), 5, "mul1", seed=7)
    parts = Exl3Parts.build(k, (n,), 5, "mul1", "cuda")
    x = torch.randn(1, k, generator=torch.Generator().manual_seed(8)).to(torch.bfloat16).cuda()
    dense = (trs[0].reshape(-1).cuda(), suh.cuda(), svh.cuda(), parts, torch.bfloat16)
    banks = tuple(b.cuda() for b in _moe_banks(5, "mul1", 12, seed=9))
    xm, w, ids = (t.cuda() for t in _routing(1, 12, seed=10))
    kw = dict(bits=5, codebook="mul1", is_prefill=False)
    ref_dense = exl3_forward(x, *dense)
    ref_moe = fused_experts_exl3(xm, banks, w, ids, **kw)
    with split_counter_lane(1):
        exl3_forward(x, *dense)  # allocates lane 1
    main, side = torch.cuda.current_stream(), torch.cuda.Stream()
    for _ in range(50):
        side.wait_stream(main)
        with torch.cuda.stream(side), split_counter_lane(1):
            d = [exl3_forward(x, *dense) for _ in range(4)]
        m = [fused_experts_exl3(xm, banks, w, ids, **kw) for _ in range(4)]
        main.wait_stream(side)
        torch.cuda.synchronize()
        assert all(torch.equal(t, ref_dense) for t in d)
        assert all(torch.equal(t, ref_moe) for t in m)


@cuda
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
def test_reconstruct_path_matches_gemm_path(monkeypatch, codebook):
    """Long prefills decode W_hat once per part and use cuBLAS; slabs split a part's columns. The
    result must agree with the tl.dot GEMM (same fp16 operands, fp32 accumulation, same epilogue)."""
    from freetoken.kernel.triton.exl3 import Exl3Parts
    from freetoken.layers.quantization.linear import exl3 as lin

    k, sizes, bits, rows = 512, (256, 128, 384), 4, 333
    trs, suh, svh = _dense_case(k, sizes, bits, codebook, seed=7)
    parts = Exl3Parts.build(k, sizes, bits, codebook, "cuda")
    flat = torch.cat([t.reshape(-1) for t in trs]).cuda()
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(8)).to(torch.bfloat16).cuda()
    args = (x, flat, suh.cuda(), svh.cuda(), parts, torch.float32)
    monkeypatch.setattr(lin, "RECONSTRUCT_MIN_ROWS", 1 << 30)
    gemm = lin.exl3_forward(*args)
    monkeypatch.setattr(lin, "RECONSTRUCT_MIN_ROWS", 16)
    monkeypatch.setattr(lin, "RECONSTRUCT_SLAB", 128)  # every part split into 128-column slabs
    recon = lin.exl3_forward(*args)
    assert rel_err(recon, gemm) < 1e-5, rel_err(recon, gemm)


@cuda
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
def test_moe_prefill_decoded_scratch_matches_in_kernel_decode(monkeypatch, codebook):
    """Long prefills decode W_hat into an fp16 scratch a group of experts at a time and the GEMM
    reads it; the product must be bitwise the in-kernel trellis decode's (same W_hat values, same
    16-wide mma k-steps), which lets PREFILL_DECODED_MIN_ROUTES_PER_EXPERT be a pure speed knob."""
    import freetoken.moe.fused_exl3 as fe

    banks = tuple(b.cuda() for b in _moe_banks(4, codebook, E, seed=11))
    x, w, ids = (t.cuda() for t in _routing(300, E, seed=12))
    kw = dict(bits=4, codebook=codebook, is_prefill=True, num_experts=E)
    monkeypatch.setattr(fe, "PREFILL_FOLD_INPUT", False)  # the folded weight is a different rounding
    monkeypatch.setattr(fe, "PREFILL_DECODED_MIN_ROUTES_PER_EXPERT", 1 << 30)
    in_kernel = fe.fused_experts_exl3(x, banks, w, ids, **kw)
    monkeypatch.setattr(fe, "PREFILL_DECODED_MIN_ROUTES_PER_EXPERT", 0)
    monkeypatch.setattr(fe, "PREFILL_DECODE_GROUP", 5)  # groups that do not divide the expert count
    scratch = fe.fused_experts_exl3(x, banks, w, ids, **kw)
    assert torch.equal(scratch.view(torch.int16), in_kernel.view(torch.int16))


@cuda
@pytest.mark.parametrize("tokens", (1, 4))
def test_fused_decode_epilogues_match_the_unfused_path(monkeypatch, tokens):
    """Split-K reduction folded into silu*up + down rotation and into the top-k combine:
    the same arithmetic up to summation order and the exp of silu, so within fp16 noise."""
    from freetoken.moe.fused_exl3 import fused_experts_exl3

    banks = tuple(b.cuda() for b in _moe_banks(5, "mul1", 12, seed=9))
    x, w, ids = (t.cuda() for t in _routing(tokens, 12, seed=10))
    kw = dict(bits=5, codebook="mul1", is_prefill=False)
    fused = fused_experts_exl3(x, banks, w, ids, **kw)
    monkeypatch.setenv("FREETOKEN_EXL3_FUSED_EPILOGUE", "0")
    unfused = fused_experts_exl3(x, banks, w, ids, **kw)
    assert rel_err(fused, unfused) < 2e-3, rel_err(fused, unfused)


@cuda
@pytest.mark.parametrize("out_dtype", (torch.float32, torch.bfloat16))
def test_inkernel_split_k_reduction_matches_torch_sum(monkeypatch, out_dtype):
    """The last split of each (row, 128 columns) sums the planes in split order inside the
    GEMV; the separate torch.sum pass it replaces agrees to fp32 summation-order noise, and
    the counters are left armed for the next launch (repeat == first, bit for bit)."""
    from freetoken.kernel.triton.exl3 import Exl3Parts
    from freetoken.layers.quantization.linear.exl3 import exl3_forward, pick_split_k

    k, n, rows = 2048, 1024, 3
    assert pick_split_k(rows, n // 128, k, torch.device("cuda")) > 1
    trs, suh, svh = _dense_case(k, (n,), 5, "mul1", seed=21)
    parts = Exl3Parts.build(k, (n,), 5, "mul1", "cuda")
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(22)).to(torch.bfloat16).cuda()
    args = (trs[0].reshape(-1).cuda(), suh.cuda(), svh.cuda(), parts, out_dtype)
    fused = exl3_forward(x, *args)
    assert all(torch.equal(fused, exl3_forward(x, *args)) for _ in range(5))
    monkeypatch.setenv("FREETOKEN_EXL3_SPLITK_INKERNEL", "0")
    summed = exl3_forward(x, *args)
    assert rel_err(fused.float(), summed.float()) < (1e-6 if out_dtype == torch.float32 else 4e-3)


@cuda
@pytest.mark.parametrize("rows", (1, 3, 8))
@pytest.mark.parametrize("k,sizes", ((2048, (1024,)), (4096, (2048,)), (2048, (512, 256, 256)), (384, (256,))),
                         ids=("2048x1024", "4096x2048", "3parts", "k384"))
def test_gemv_prologue_rotation_matches_had_rows(monkeypatch, rows, k, sizes):
    """The input rotation computed inside the GEMV prologue (H8 x H16 factorisation, per
    program over the 128-blocks of its K range) against the separate had_rows launch: the
    rotated inputs differ at most by fp16 rounding of a reordered fp32 sum, and the prologue
    path is deterministic (repeat == first, bit for bit)."""
    from freetoken.kernel.triton.exl3 import Exl3Parts, had_rows
    from freetoken.layers.quantization.linear.exl3 import exl3_forward

    trs, suh, svh = _dense_case(k, sizes, 5, "mul1", seed=31)
    parts = Exl3Parts.build(k, sizes, 5, "mul1", "cuda")
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(32)).to(torch.bfloat16).cuda()
    tr = torch.cat([t.reshape(-1) for t in trs]).cuda()
    args = (tr, suh.cuda(), svh.cuda(), parts, torch.float32)
    monkeypatch.setenv("FREETOKEN_EXL3_GEMV_PREROT", "1")
    pre = exl3_forward(x, *args)
    assert all(torch.equal(pre, exl3_forward(x, *args)) for _ in range(3))
    monkeypatch.setenv("FREETOKEN_EXL3_GEMV_PREROT", "0")
    sep = exl3_forward(x, *args)
    assert rel_err(pre, sep) < 2e-3, rel_err(pre, sep)
    # the rotation itself: FWHT-by-Kronecker in fp32 vs had_rows' split-fp16 tensor-core dot
    xh = had_rows(x.contiguous(), suh.cuda(), parts).float()
    from freetoken.kernel.triton.exl3 import hadamard_reference
    h = hadamard_reference().double().cuda() / 128 ** 0.5
    want = torch.stack([((x.double() * suh.cuda()[j].double()).view(rows, -1, 128) @ h).view(rows, k)
                        for j in range(len(sizes))])
    assert (xh.double() - want).abs().max() <= 2e-3 * want.abs().max()


@cuda
@pytest.mark.parametrize("tokens", (1, 4))
def test_moe_decode_prologue_rotation_matches_had_rows(monkeypatch, tokens):
    """Routed gate|up with the rotation in the GEMV prologue (x[p // top_k] * suh[ids[p]])
    against had_rows + GEMV, at 1 and 4 tokens (routes share the source row)."""
    from freetoken.moe.fused_exl3 import fused_experts_exl3

    banks = tuple(b.cuda() for b in _moe_banks(5, "mul1", 12, seed=41))
    x, w, ids = (t.cuda() for t in _routing(tokens, 12, seed=42))
    kw = dict(bits=5, codebook="mul1", is_prefill=False)
    monkeypatch.setenv("FREETOKEN_EXL3_GEMV_PREROT", "1")
    pre = fused_experts_exl3(x, banks, w, ids, **kw)
    assert torch.equal(pre, fused_experts_exl3(x, banks, w, ids, **kw))
    monkeypatch.setenv("FREETOKEN_EXL3_GEMV_PREROT", "0")
    sep = fused_experts_exl3(x, banks, w, ids, **kw)
    assert rel_err(pre, sep) < 2e-3, rel_err(pre, sep)


@cuda
@pytest.mark.parametrize("bits", (2, 5, 8))
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
def test_folded_reconstruct_path_matches_reference(monkeypatch, bits, codebook):
    """Prefill with both rotations folded into W_full (reconstruct_folded + one fp16 GEMM on the raw
    activations) against the fp32 reference and against the rotate-activations reconstruct path:
    the only new rounding is W_full's (and the fp16 GEMM output), same order as the xh rounding."""
    from freetoken.kernel.triton.exl3 import Exl3Parts, reconstruct_folded, reconstruct_reference, hadamard_reference
    from freetoken.layers.quantization.linear import exl3 as lin

    k, sizes, rows = 512, (256, 128, 384), 300
    trs, suh, svh = _dense_case(k, sizes, bits, codebook, seed=11 + bits)
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(12)).to(torch.bfloat16)
    ref, col = [], 0
    for j, (tr, s) in enumerate(zip(trs, sizes)):
        ref.append(linear_reference(x.float(), tr, suh[j], svh[col:col + s], codebook))
        col += s
    ref = torch.cat(ref, dim=1)
    parts = Exl3Parts.build(k, sizes, bits, codebook, "cuda")
    flat = torch.cat([t.reshape(-1) for t in trs]).cuda()
    args = (x.cuda(), flat, suh.cuda(), svh.cuda(), parts, torch.float32)
    monkeypatch.setattr(lin, "FOLD_MIN_ROWS", 16)
    monkeypatch.setenv("FREETOKEN_EXL3_FOLD", "1")
    folded = lin.exl3_forward(*args)
    assert rel_err(folded.cpu(), ref) < REL_TOL, rel_err(folded.cpu(), ref)
    monkeypatch.setenv("FREETOKEN_EXL3_FOLD", "0")
    rot = lin.exl3_forward(*args)
    assert rel_err(folded, rot) < 2e-3, rel_err(folded, rot)
    # W_full itself against diag(suh) H W_hat H diag(svh) in float64
    w = reconstruct_folded(flat, suh.cuda(), svh.cuda(), parts)[0].double().cpu()
    h = hadamard_reference().double() / 128 ** 0.5
    col = 0
    for j, (tr, s) in enumerate(zip(trs, sizes)):
        wh = reconstruct_reference(tr, codebook).double()
        hk = torch.block_diag(*[h] * (k // 128))
        hn = torch.block_diag(*[h] * (s // 128))
        want = suh[j].double()[:, None] * (hk @ wh @ hn) * svh[col:col + s].double()[None, :]
        got = w[:, col:col + s]
        assert (got - want).abs().max() <= 2e-3 * want.abs().max(), (got - want).abs().max()
        col += s
    # column slabs (crossing part boundaries) build the same W_full columns bit for bit
    full = reconstruct_folded(flat, suh.cuda(), svh.cuda(), parts)[0]
    for c0, c1 in ((0, 128), (128, 512), (384, 768)):
        assert torch.equal(reconstruct_folded(flat, suh.cuda(), svh.cuda(), parts, cols=(c0, c1))[0], full[:, c0:c1])
    monkeypatch.setenv("FREETOKEN_EXL3_FOLD", "1")
    monkeypatch.setattr(lin, "RECONSTRUCT_SLAB", 256)
    slabbed = lin.exl3_forward(*args)
    assert rel_err(slabbed, folded) < 1e-3, rel_err(slabbed, folded)
