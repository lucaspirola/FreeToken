"""Bit-exact rewrites of EXL3 kernels, checked for identical bits (not a tolerance):

* the 32-bit codebook decode (``_decode_words``, which ``_exl3_decode`` now calls) against the former
  uint64 formula, exhaustively: every 16-bit window at every shift 0..31, all three codebooks, with
  random bits around the window; both also against the torch reference ``_decode_codes``;
* ``had_rows`` with H in column slices / 64-row tiles (``HAD_ROWS_BIG``) against the held-H tiling;
* ``reconstruct_experts`` decoding in bitstream order (``_decode_tile_rows``) against the
  (row, column)-indexed ``_exl3_decode``, every bit width 1..8 and codebook, several part layouts;
* the fused MoE prefill with group-ranged decoded GEMM launches and L2-sized decode groups against
  the plain full-grid launches with 32-expert groups (uniform and skewed routing, idle experts)."""
from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.exl3 import (
    CODEBOOKS, Exl3Parts, _decode_codes, _decode_words, _exl3_decode, _words, had_rows, reconstruct_experts,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton EXL3 kernels need CUDA")


@triton.jit
def _decode_u64(a, b, shift, CB: tl.constexpr):
    """``_exl3_decode``'s arithmetic before the 32-bit rewrite (b967140), verbatim."""
    a = a.to(tl.uint64)
    b = b.to(tl.uint64)
    shift = shift.to(tl.uint64)
    w = (((a << 32) | b) >> shift) & 0xFFFF
    if CB == 2:
        x = (w * 0x83DCD12D) & 0xFFFFFFFF
        s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        v = (s + 1024).to(tl.float32) * 0.00676727294921875 + (-10.3828125)
        return v.to(tl.float16)
    else:
        if CB == 1:
            x = (w * 0xCBAC1FED) & 0xFFFFFFFF
        else:
            x = (w * 89226354 + 64248484) & 0xFFFFFFFF
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = (x & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        hi = (x >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
        return lo + hi


@triton.jit
def _both_kernel(a_ptr, b_ptr, s_ptr, new_ptr, old_ptr, n, CB: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    a = tl.load(a_ptr + o, mask=m).to(tl.uint32, bitcast=True)
    b = tl.load(b_ptr + o, mask=m).to(tl.uint32, bitcast=True)
    s = tl.load(s_ptr + o, mask=m).to(tl.uint32)
    tl.store(new_ptr + o, _decode_words(a, b, s, CB), mask=m)
    tl.store(old_ptr + o, _decode_u64(a, b, s, CB), mask=m)


@cuda
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
def test_decode32_equals_u64_every_window_every_shift(codebook):
    g = torch.Generator().manual_seed(1)
    w = torch.arange(1 << 16, dtype=torch.int64).repeat(32)                       # every state ...
    shift = torch.arange(32, dtype=torch.int64).repeat_interleave(1 << 16)        # ... at every shift
    noise = torch.randint(0, 1 << 62, w.shape, generator=g, dtype=torch.int64) << 2 | torch.randint(0, 4, w.shape, generator=g)
    word64 = (noise & ~(0xFFFF << shift)) | (w << shift)                           # window planted at shift
    a = (word64 >> 32).to(torch.int32)                                             # int32 wraps: the uint32 bits
    b = word64.to(torch.int32)
    assert torch.equal(((a.to(torch.int64) & 0xFFFFFFFF) << 32 | (b.to(torch.int64) & 0xFFFFFFFF)), word64)
    assert torch.equal((word64 >> shift) & 0xFFFF, w)
    n = w.numel()
    new = torch.empty(n, dtype=torch.float16, device="cuda")
    old = torch.empty_like(new)
    _both_kernel[(triton.cdiv(n, 1024),)](a.cuda(), b.cuda(), shift.to(torch.int32).cuda(), new, old, n,
                                          CB=CODEBOOKS[codebook], BLOCK=1024)
    new_bits = new.cpu().view(torch.int16)
    assert torch.equal(new_bits, old.cpu().view(torch.int16))
    assert torch.equal(new_bits, _decode_codes(w, codebook).view(torch.int16))


@cuda
@pytest.mark.parametrize("k,sizes,src_div,rows", [(2048, (1024,), 8, 8 * 1000 + 8), (512, (2048,), 1, 4133), (256, (128, 256), 1, 4096)])
def test_had_rows_slices_equal_held_h(k, sizes, src_div, rows):
    g = torch.Generator(device="cpu").manual_seed(rows)
    parts = Exl3Parts.build(k, sizes, 5, "mul1", "cuda")
    e = 16
    x = (torch.randn(rows // src_div, k, generator=g) * 3).half().cuda()
    suh = torch.randn(e, len(sizes) * k, generator=g).half().cuda()
    ids = torch.randint(0, e, (rows,), generator=g, dtype=torch.int32).cuda()
    kw = dict(src_div=src_div, experts=ids, suh_expert_stride=suh.stride(0))
    held = had_rows(x, suh, parts, block_m=16, num_warps=4, k_blocks=1, n_slices=1, **kw)
    for bm, ns, kb in ((64, 2, 2), (64, 2, 4), (32, 4, 1), (16, 2, 16)):
        sliced = had_rows(x, suh, parts, block_m=bm, num_warps=4, k_blocks=kb, n_slices=ns, **kw)
        assert torch.equal(sliced.view(torch.int16), held.view(torch.int16)), (bm, ns, kb)
    assert torch.equal(had_rows(x, suh, parts, **kw).view(torch.int16), held.view(torch.int16))  # shipped launcher


@triton.jit
def _indexed_reconstruct_kernel(tr_ptr, tr_expert_stride, out_ptr, out_expert_stride,
                                part_of_nb_ptr, word_off_ptr, ntiles_ptr, nstart_ptr,
                                BITS: tl.constexpr, CB: tl.constexpr):
    """``_reconstruct_experts_kernel`` before the bitstream-order rewrite (757caee), verbatim."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2).to(tl.int64)
    part = tl.load(part_of_nb_ptr + pid_n)
    n_tiles = tl.load(ntiles_ptr + part)
    n_local = pid_n * 128 - tl.load(nstart_ptr + part) + tl.arange(0, 128)
    tr = tr_ptr + g * tr_expert_stride + tl.load(word_off_ptr + part)
    kk = pid_k * 16 + tl.arange(0, 16)
    w = _exl3_decode(tr, kk[:, None], n_local[None, :], n_tiles, BITS, CB)
    n = tl.num_programs(1) * 128
    tl.store(out_ptr + g * out_expert_stride + kk[:, None].to(tl.int64) * n + (pid_n * 128 + tl.arange(0, 128))[None, :], w)


@cuda
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("codebook", tuple(CODEBOOKS))
def test_bitstream_reconstruct_equals_indexed_decode(bits, codebook, monkeypatch):
    import freetoken.kernel.triton.exl3 as k3

    g = torch.Generator().manual_seed(100 * bits + CODEBOOKS[codebook])
    for k, sizes in ((2048, (512, 512)), (512, (2048,)), (256, (128, 384))):
        parts = Exl3Parts.build(k, sizes, bits, codebook, "cuda")
        e = 3
        bank = torch.randint(-(1 << 15), 1 << 15, (e, parts.words * 2), generator=g, dtype=torch.int32).to(torch.int16).cuda()
        ref = torch.empty((e, k, parts.n), dtype=torch.float16, device="cuda")
        _indexed_reconstruct_kernel[(k // 16, parts.n // 128, e)](
            _words(bank), bank.stride(0) // 2, ref, ref.stride(0),
            parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart, BITS=bits, CB=CODEBOOKS[codebook])
        for kt, warps in ((1, 4), (2, 4), (4, 8), (3, 2)):  # KT 3 falls back to 1 where it does not divide
            monkeypatch.setattr(k3, "RECON_KT", kt)
            monkeypatch.setattr(k3, "RECON_WARPS", warps)
            new = reconstruct_experts(bank, parts, 0, e)
            assert torch.equal(new.view(torch.int16), ref.view(torch.int16)), (k, sizes, kt, warps)
            part = reconstruct_experts(bank, parts, 1, 3)  # an expert range not starting at 0
            assert torch.equal(part.view(torch.int16), ref[1:3].view(torch.int16)), (k, sizes, kt, warps)


@cuda
@pytest.mark.parametrize("skew", (False, True))
def test_ranged_l2_groups_equal_plain_launches(monkeypatch, skew):
    """Group-ranged decoded GEMM launches (``mb_range``) and per-projection decode groups change
    which program computes a tile, never its arithmetic: bitwise the plain 32-expert-group path."""
    import freetoken.moe.fused_exl3 as fe
    from freetoken.models.exl3_banks import exl3_bank_shapes

    H, I, E, TOPK, BITS, M = 512, 256, 40, 4, 4, 700
    g = torch.Generator().manual_seed(7 + skew)
    B = []
    for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
        if dtype == torch.int16:
            B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), generator=g, dtype=torch.int32).to(torch.int16).cuda())
        else:
            B.append((torch.randn((E, *shape), generator=g) * (0.5 if name.endswith("suh") else 0.05)).half().cuda())
    x = torch.randn(M, H, generator=g).to(torch.bfloat16).cuda()
    w = torch.softmax(torch.randn(M, TOPK, generator=g), -1).cuda()
    if skew:  # a few hot experts, several idle ones (no rows: empty group runs)
        p = torch.zeros(E)
        p[:6] = torch.tensor([40.0, 20, 10, 5, 3, 2])
        p[20:23] = 1.0
        ids = torch.multinomial(p.expand(M, E), TOPK, generator=g)
    else:
        ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)])
    ids = ids.to(torch.int32).cuda()
    run = lambda: fe.fused_experts_exl3(x, tuple(B), w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)
    monkeypatch.setattr(fe, "PREFILL_FOLD_INPUT", False)
    monkeypatch.setattr(fe, "PREFILL_DECODED_MIN_TOKENS", 1)
    monkeypatch.setattr(fe, "PREFILL_RANGED", False)
    monkeypatch.setattr(fe, "PREFILL_DECODE_GROUP_OVERRIDE", 32)
    ref = run()
    monkeypatch.setattr(fe, "PREFILL_RANGED", True)
    for group in (32, 7, 1, 64):
        monkeypatch.setattr(fe, "PREFILL_DECODE_GROUP_OVERRIDE", group)
        assert torch.equal(run().view(torch.int16), ref.view(torch.int16)), group
    monkeypatch.setattr(fe, "PREFILL_DECODE_GROUP_OVERRIDE", None)
    for frac in (0.5, 1e-9, 0.001):  # L2-sized groups; a tiny budget falls back to PREFILL_DECODE_GROUP
        monkeypatch.setattr(fe, "PREFILL_DECODE_L2_FRACTION", frac)
        assert torch.equal(run().view(torch.int16), ref.view(torch.int16)), frac


@cuda
@pytest.mark.parametrize("m,k,n", [(1152, 2048, 2048), (1808, 2048, 1024), (2048, 512, 2048), (3001, 4096, 2048), (8192, 2048, 2048)])
def test_gemm_cast_equals_cast_cublas_cast(m, k, n):
    """gemm_cast: fused bf16->fp16 input cast, fp32 accumulation, fp16 rounding, bf16 output cast --
    bitwise the host cast + torch.mm + copy-cast it replaces (these shapes keep cuBLAS in one k pass)."""
    from freetoken.kernel.triton.exl3 import gemm_cast

    g = torch.Generator().manual_seed(m + k + n)
    x = torch.randn(m, k, generator=g).to(torch.bfloat16).cuda()
    w = (torch.randn(k, n + 256, generator=g) * 0.03).half().cuda()[:, 128 : 128 + n]  # strided slab
    w = w.contiguous()
    ref = torch.mm(x.to(torch.float16), w).to(torch.bfloat16)
    wide = torch.zeros(m, n + 384, dtype=torch.bfloat16, device="cuda")
    out = gemm_cast(x, w, wide[:, 256 : 256 + n])  # a column slice, as the folded path writes
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    assert not wide[:, :256].any() and not wide[:, 256 + n :].any()
    for cfg in (dict(BM=128, BN=128, BK=32), dict(BM=64, BN=256, BK=64, num_warps=8)):
        assert torch.equal(gemm_cast(x, w, torch.empty_like(ref), **cfg).view(torch.int16), ref.view(torch.int16)), cfg


@cuda
@pytest.mark.parametrize("rows", (1152, 2000))
def test_folded_fused_cast_path_equals_cublas_path(monkeypatch, rows):
    import freetoken.layers.quantization.linear.exl3 as lin

    g = torch.Generator().manual_seed(rows)
    parts = Exl3Parts.build(2048, (2048, 1024, 512), 5, "mul1", "cuda")
    tr = torch.randint(-(1 << 15), 1 << 15, (parts.words * 2,), generator=g, dtype=torch.int32).to(torch.int16).cuda()
    suh = (torch.randn(3 * 2048, generator=g) * 0.5).half().cuda()
    svh = (torch.randn(parts.n, generator=g) * 0.05).half().cuda()
    x = torch.randn(rows, 2048, generator=g).to(torch.bfloat16).cuda()
    fused = lin.exl3_forward(x, tr, suh, svh, parts, torch.bfloat16)
    monkeypatch.setattr(lin, "FUSED_CAST_MIN_ROWS", 1 << 30)
    cublas = lin.exl3_forward(x, tr, suh, svh, parts, torch.bfloat16)
    assert torch.equal(fused.view(torch.int16), cublas.view(torch.int16))
