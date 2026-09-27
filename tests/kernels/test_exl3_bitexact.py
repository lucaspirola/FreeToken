"""Bit-exact rewrites of EXL3 kernels, checked for identical bits (not a tolerance):

* the 32-bit codebook decode (``_decode_words``, which ``_exl3_decode`` now calls) against the former
  uint64 formula, exhaustively: every 16-bit window at every shift 0..31, all three codebooks, with
  random bits around the window; both also against the torch reference ``_decode_codes``;
* ``had_rows`` with H in column slices / 64-row tiles (``HAD_ROWS_BIG``) against the held-H tiling."""
from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.exl3 import CODEBOOKS, Exl3Parts, _decode_codes, _decode_words, had_rows

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
