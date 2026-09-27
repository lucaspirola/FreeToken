"""EXL3 (exllamav3 trellis) weight kernels.

An EXL3 linear ``X`` stores ``trellis`` int16 ``[K/16, N/16, 16*bits]`` (one 16x16 tile of
``bits``-bit trellis codes per ``[k_tile, n_tile]``), ``suh`` fp16 ``[K]`` and ``svh`` fp16
``[N]``. With ``H`` the 128x128 Sylvester Hadamard matrix scaled by ``1/sqrt(128)`` and applied
blockwise over each 128 channels, the layer computes::

    y = H_blocks( H_blocks(x * suh) @ W_hat ) * svh

where ``W_hat`` is the decoded trellis. The decode rules (bit order, codebooks, the tile's
tensor-core permutation) are in ``tasks/ornith-exl3/design.md`` and were checked bit-exact
against exllamav3's own ``reconstruct`` for bits 2..8 and all three codebooks.

Three kernels, all accumulating in fp32:

* ``had_rows``: rotated inputs ``xh[part, p] = H_blocks(x[src(p)] * suh[expert(p), part])`` in fp16.
* ``exl3_gemm``: grouped ``tl.dot`` GEMM (rows sorted per expert, or one expert for a dense
  layer) with the trellis decoded in registers and the output Hadamard + ``svh`` in the epilogue.
* ``exl3_gemv``: one program per (row, 128 output columns[, K split]) for decode, where every
  row may belong to a different expert slot; ``tl.dot`` would waste 15/16 of each MMA there. It
  walks each tile in bitstream order (see ``_exl3_gemv_kernel``) so all decode indexing is static.

A weight is addressed as *parts*: a fused projection (q|k|v, gate|up) keeps each part's tiles
contiguous ("part-major"), with its own suh. Every part boundary is a multiple of 128, so each
128-wide output block belongs to one part. ``Exl3Parts`` holds the device tables.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

HAD = 128
HAD_SCALE = 1.0 / math.sqrt(HAD)
CODEBOOKS = {"3inst": 0, "mcg": 1, "mul1": 2}
# the multipliers the procedural codebooks hardcode; a checkpoint's mul1 / mcg tensor must hold these
MUL1_MULT = 0x83DCD12D
MCG_MULT = 0xCBAC1FED
# fp16 bit patterns 0x1eee and 0xc931 (exllamav3 codebook.cuh), as exact floats
_MUL1_INV = 0.00676727294921875
_MUL1_BIAS = -10.3828125


# ---------------------------------------------------------------------------
# Reference decoder (torch, any device): the oracle the Triton kernels are tested against
# ---------------------------------------------------------------------------


def tile_stream_index() -> torch.Tensor:
    """``t[r, c]``: the trellis stream position of tile element (row r, column c)."""
    r = torch.arange(16).view(16, 1)
    c = torch.arange(16).view(1, 16)
    return 32 * (c & 7) + 16 * ((r >> 2) & 1) + 8 * ((r >> 1) & 1) + 4 * (c >> 3) + 2 * (r >> 3) + (r & 1)


def _decode_codes(w: torch.Tensor, codebook: str) -> torch.Tensor:
    """16-bit trellis states (int64) -> fp16 codebook values."""
    if codebook == "mul1":
        x = (w * MUL1_MULT) & 0xFFFFFFFF
        s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        return ((s + 1024).to(torch.float64) * _MUL1_INV + _MUL1_BIAS).to(torch.float16)
    if codebook == "mcg":
        x = (w * MCG_MULT) & 0xFFFFFFFF
    elif codebook == "3inst":
        x = (w * 89226354 + 64248484) & 0xFFFFFFFF
    else:
        raise ValueError(f"unknown EXL3 codebook {codebook!r}")
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60

    def as_half(v: torch.Tensor) -> torch.Tensor:
        return v.to(torch.int32).to(torch.int16).view(torch.float16)

    return as_half(x & 0xFFFF) + as_half(x >> 16)


def reconstruct_reference(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    """Decode a ``[K/16, N/16, 16*bits]`` trellis to the rotated-basis ``W_hat`` fp16 ``[K, N]``."""
    kt, nt, width = trellis.shape
    if width % 16:
        raise ValueError(f"trellis tile width {width} is not 16 * bits (half-integer bitrates are not supported)")
    bits = width // 16
    words = trellis.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF  # [kt, nt, 8*bits]
    t = tile_stream_index().to(trellis.device)
    nwords = 8 * bits
    b0 = t * bits + bits - 16 + 256 * bits
    b1 = b0 + 16
    i1_raw = (b1 - 1) // 32
    shift = (i1_raw + 1) * 32 - b1
    a = words[..., (b0 // 32) % nwords]  # [kt, nt, 16, 16]
    b = words[..., i1_raw % nwords]
    w = (((a << 32) | b) >> shift) & 0xFFFF
    vals = _decode_codes(w, codebook)
    return vals.permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def hadamard_reference(n: int = HAD) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def linear_reference(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: str) -> torch.Tensor:
    """fp32 ``y = H(H(x * suh) @ W_hat) * svh`` for one part."""
    w = reconstruct_reference(trellis, codebook).float()
    h = hadamard_reference().to(x.device) * HAD_SCALE
    k, n = w.shape
    xs = (x.float() * suh.float()).view(-1, k // HAD, HAD) @ h
    y = xs.view(-1, k) @ w
    return ((y.view(-1, n // HAD, HAD) @ h).view(-1, n)) * svh.float()


@functools.lru_cache(maxsize=None)
def hadamard_pm1(device: torch.device) -> torch.Tensor:
    """The +-1 Sylvester matrix in fp16 (exact); kernels scale by ``HAD_SCALE`` in fp32."""
    return hadamard_reference().to(torch.float16).to(device).contiguous()


# ---------------------------------------------------------------------------
# Part tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exl3Parts:
    """How one expert's (or one dense layer's) weight splits into parts along N.

    ``word_off`` / ``ntiles`` / ``nstart`` are per part: the part's first tile (int32 words from
    the expert base), its N/16 tile count and its first global output column. ``part_of_nb`` maps
    each 128-wide output block to its part. ``suh_part_stride`` is the suh distance between parts.
    """

    k: int
    n: int
    bits: int
    codebook: str
    part_of_nb: torch.Tensor
    word_off: torch.Tensor
    ntiles: torch.Tensor
    nstart: torch.Tensor
    num_parts: int
    suh_part_stride: int
    sizes: tuple[int, ...] = ()  # host copy of each part's N (the reconstruct path slices by it)

    @staticmethod
    def build(k: int, sizes: tuple[int, ...], bits: int, codebook: str, device, *, suh_part_stride: int | None = None) -> "Exl3Parts":
        if k % HAD:
            raise ValueError(f"EXL3 needs in_features divisible by {HAD}, got {k}")
        if any(s % HAD for s in sizes):
            raise ValueError(f"EXL3 fused parts must be multiples of {HAD} wide, got {sizes}")
        if codebook not in CODEBOOKS:
            raise ValueError(f"unknown EXL3 codebook {codebook!r}")
        words_per_tile = 8 * bits
        word_off, ntiles, nstart, part_of_nb = [], [], [], []
        off = col = 0
        for j, s in enumerate(sizes):
            word_off.append(off)
            ntiles.append(s // 16)
            nstart.append(col)
            part_of_nb += [j] * (s // HAD)
            off += (k // 16) * (s // 16) * words_per_tile
            col += s

        def t(v):
            return torch.tensor(v, dtype=torch.int32, device=device)

        return Exl3Parts(
            k, col, bits, codebook, t(part_of_nb), t(word_off), t(ntiles), t(nstart), len(sizes),
            k if suh_part_stride is None else suh_part_stride, tuple(sizes),
        )

    @property
    def words(self) -> int:
        """int32 words of trellis for all parts of one expert."""
        return (self.k // 16) * (self.n // 16) * 8 * self.bits


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _exl3_decode(tr_ptr, kk, nn, n_tiles, BITS: tl.constexpr, CB: tl.constexpr):
    """Decode W_hat[kk, nn] (kk a column vector of rows, nn a row vector of part-local columns) to fp16."""
    kt = kk // 16
    r = kk % 16
    nt = nn // 16
    c = nn % 16
    t = 32 * (c & 7) + 16 * ((r >> 2) & 1) + 8 * ((r >> 1) & 1) + 4 * (c >> 3) + 2 * (r >> 3) + (r & 1)
    WORDS: tl.constexpr = 8 * BITS
    base = (kt * n_tiles + nt) * WORDS
    b0 = t * BITS + (BITS - 16 + 256 * BITS)
    b1 = b0 + 16
    i1 = (b1 - 1) // 32
    # 32-bit words through _decode_words: the uint64 window (a << 32 | b) >> shift made ptxas emit
    # I2F.U64 (FP64 pipe on GeForce) for the byte-sum conversion. Bit-exact: shift is in [0, 31],
    # so the funnel shift is the same window; the multiplies only ever kept the low 32 bits; and
    # s + 1024 <= 2044 converts exactly from either width (tests/kernels/test_exl3_bitexact.py).
    shift = ((i1 + 1) * 32 - b1).to(tl.uint32)
    a = tl.load(tr_ptr + base + (b0 // 32) % WORDS).to(tl.uint32, bitcast=True)
    b = tl.load(tr_ptr + base + i1 % WORDS).to(tl.uint32, bitcast=True)
    return _decode_words(a, b, shift, CB)


@triton.jit
def _had_rows_kernel(
    x_ptr, stride_xm,
    out_ptr, stride_opart, stride_om,
    suh_ptr, suh_expert_stride, suh_part_stride,
    expert_ptr, had_ptr, P,
    SRC_DIV: tl.constexpr, HAS_EXPERT: tl.constexpr, BM: tl.constexpr, KB: tl.constexpr, NS: tl.constexpr = 1,
):
    """``NS`` > 1 multiplies by H in NS column slices of 128 / NS (each slice loaded per block, not
    held): every output element keeps the same K=128 hi and lo dots, so the result is bitwise the
    same for any BM / NS / KB / warps; the smaller H operand frees the registers for BM 64."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    part = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < P
    src = rows // SRC_DIV
    if HAS_EXPERT:
        e = tl.load(expert_ptr + rows, mask=rmask, other=0).to(tl.int64)
    else:
        e = tl.zeros([BM], dtype=tl.int64)
    idx = tl.arange(0, 128)
    SW: tl.constexpr = 128 // NS
    jn = tl.arange(0, SW)
    if NS == 1:
        h = tl.load(had_ptr + idx[:, None] * 128 + idx[None, :])  # once per program, KB column blocks
    for i in tl.static_range(KB):
        cols = (pid_k * KB + i) * 128 + idx
        x = tl.load(x_ptr + src[:, None].to(tl.int64) * stride_xm + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        s = tl.load(suh_ptr + e[:, None] * suh_expert_stride + part * suh_part_stride + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        xs = x * s
        hi = xs.to(tl.float16)
        lo = (xs - hi.to(tl.float32)).to(tl.float16)
        if NS == 1:
            y = (tl.dot(hi, h) + tl.dot(lo, h)) * 0.08838834764831843
            tl.store(out_ptr + part * stride_opart + rows[:, None].to(tl.int64) * stride_om + cols[None, :], y.to(tl.float16), mask=rmask[:, None])
        else:
            for n in tl.static_range(NS):
                hs = tl.load(had_ptr + idx[:, None] * 128 + n * SW + jn[None, :])
                y = (tl.dot(hi, hs) + tl.dot(lo, hs)) * 0.08838834764831843
                oc = (pid_k * KB + i) * 128 + n * SW + jn
                tl.store(out_ptr + part * stride_opart + rows[:, None].to(tl.int64) * stride_om + oc[None, :], y.to(tl.float16), mask=rmask[:, None])


@triton.jit
def _exl3_gemm_kernel(
    xh_ptr, stride_xpart, stride_xm,
    out_ptr, stride_om,
    tr_ptr, tr_expert_stride,
    svh_ptr, svh_expert_stride,
    had_ptr,
    part_of_nb_ptr, word_off_ptr, ntiles_ptr, nstart_ptr,
    sorted_ptr, expert_ids_ptr, npad_ptr,
    P, K,
    w_ptr, w_expert_stride, e_lo, e_hi,
    BM: tl.constexpr, BK: tl.constexpr, BITS: tl.constexpr, CB: tl.constexpr, SORTED: tl.constexpr,
    DECODED: tl.constexpr, SRC_DIV: tl.constexpr, F16ACC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if SORTED:
        if pid_m * BM >= tl.load(npad_ptr):
            return
        expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if expert < 0:
            return
        if DECODED:  # W_hat of experts [e_lo, e_hi) only: other blocks belong to another group's launch
            if expert < e_lo or expert >= e_hi:
                return
        rows = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    else:
        rows = pid_m * BM + tl.arange(0, BM)
        expert = tl.zeros([], dtype=tl.int64)
    rmask = rows < P
    rows64 = rows.to(tl.int64)
    part = tl.load(part_of_nb_ptr + pid_n)
    n_tiles = tl.load(ntiles_ptr + part)
    n_local0 = pid_n * 128 - tl.load(nstart_ptr + part)
    tr = tr_ptr + expert * tr_expert_stride + tl.load(word_off_ptr + part)
    a_ptr = xh_ptr + part * stride_xpart + (rows64 // SRC_DIV)[:, None] * stride_xm
    if F16ACC:  # mma.sync with fp16 accumulators: 2x the fp32-accumulate rate on GeForce parts
        acc = tl.zeros([BM, 128], dtype=tl.float16)
    else:
        acc = tl.zeros([BM, 128], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + kk[None, :], mask=rmask[:, None], other=0.0)
        if DECODED:
            w = tl.load(w_ptr + (expert - e_lo) * w_expert_stride + kk[:, None].to(tl.int64) * (tl.num_programs(1) * 128)
                        + (pid_n * 128 + tl.arange(0, 128))[None, :])
        else:
            # decoded tile-major, [BK/16, 8 tiles, 16, 16], then permuted to [BK, 128]: the same
            # values, but a warp's gathers stay inside one tile's words (few L1 wavefronts)
            k4 = k0 + tl.arange(0, BK // 16)[:, None, None, None] * 16 + tl.arange(0, 16)[None, None, :, None]
            n4 = n_local0 + tl.arange(0, 8)[None, :, None, None] * 16 + tl.arange(0, 16)[None, None, None, :]
            w4 = _exl3_decode(tr, k4, n4, n_tiles, BITS, CB)
            w = tl.reshape(tl.permute(w4, (0, 2, 1, 3)), (BK, 128))
        if F16ACC:
            acc = tl.dot(a, w, acc, out_dtype=tl.float16)
        else:
            acc = tl.dot(a, w, acc)
    idx = tl.arange(0, 128)
    h = tl.load(had_ptr + idx[:, None] * 128 + idx[None, :])
    if F16ACC:
        y = tl.dot(acc, h)
    else:
        hi = acc.to(tl.float16)
        lo = (acc - hi.to(tl.float32)).to(tl.float16)
        y = tl.dot(hi, h) + tl.dot(lo, h)
    cols = pid_n * 128 + idx
    sv = tl.load(svh_ptr + expert * svh_expert_stride + cols).to(tl.float32)
    y = y * (sv * 0.08838834764831843)[None, :]
    tl.store(out_ptr + rows64[:, None] * stride_om + cols[None, :], y.to(out_ptr.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _decode_words(a, b, shift, CB: tl.constexpr):
    """W_hat values from the two words around each 16-bit window (32-bit ops only: funnel shift,
    wrapping multiply, dp4a byte sum). Same values as ``_exl3_decode``."""
    w = tl.inline_asm_elementwise(
        "shf.r.wrap.b32 $0, $1, $2, $3;", "=r,r,r,r", [b, a, shift], dtype=tl.uint32, is_pure=True, pack=1,
    ) & 0xFFFF
    if CB == 2:
        x = w * tl.full([], 0x83DCD12D, tl.uint32)
        s = tl.inline_asm_elementwise(
            "dp4a.u32.u32 $0, $1, 16843009, 0;", "=r,r", [x], dtype=tl.uint32, is_pure=True, pack=1,
        )
        return ((s + 1024).to(tl.float32) * 0.00676727294921875 + (-10.3828125)).to(tl.float16)
    else:
        if CB == 1:
            x = w * tl.full([], 0xCBAC1FED, tl.uint32)
        else:
            x = w * tl.full([], 89226354, tl.uint32) + tl.full([], 64248484, tl.uint32)
        x = (x & tl.full([], 0x8FFF8FFF, tl.uint32)) ^ tl.full([], 0x3B603B60, tl.uint32)
        lo = (x & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        hi = (x >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
        return lo + hi


@triton.jit
def _had128_8x16(v):
    """128-point Sylvester Hadamard (natural order, unnormalised) of ``v`` viewed ``[8, 16]``
    (element ``16 a + b``): H128 = H8 (x) H16, so ``Y = H8 @ V @ H16`` -- 8 x 8 x 16 + 8 x 16 x 16
    fp32 products instead of a 128 x 128 matrix that a single-warp program cannot hold in registers.
    Entries are (-1)^popcount(i & j), generated, not loaded."""
    i8 = tl.arange(0, 8)
    i16 = tl.arange(0, 16)
    a8 = i8[:, None] & i8[None, :]
    a8 = a8 ^ (a8 >> 2)
    a8 = a8 ^ (a8 >> 1)
    h8 = 1.0 - 2.0 * (a8 & 1).to(tl.float32)
    a16 = i16[:, None] & i16[None, :]
    a16 = a16 ^ (a16 >> 2)
    a16 = a16 ^ (a16 >> 1)
    h16 = 1.0 - 2.0 * (a16 & 1).to(tl.float32)
    t = tl.sum(h8[:, :, None] * v[None, :, :], axis=1)
    return tl.sum(t[:, None, :] * h16[None, :, :], axis=2)


@triton.jit
def _exl3_gemv_kernel(
    xh_ptr, stride_xpart, stride_xm,
    out_ptr, stride_om, stride_osplit,
    tr_ptr, tr_expert_stride,
    svh_ptr, svh_expert_stride,
    had_ptr,
    part_of_nb_ptr, word_off_ptr, ntiles_ptr, nstart_ptr,
    expert_ptr, K_SPLIT,
    final_ptr, stride_fm, count_ptr,
    x_ptr, stride_x, suh_ptr, suh_expert_stride, suh_part_stride, rot_ptr,
    BITS: tl.constexpr, CB: tl.constexpr, HAS_EXPERT: tl.constexpr, REDUCE: tl.constexpr,
    PRE_ROT: tl.constexpr = False, SRC_DIV: tl.constexpr = 1, KB: tl.constexpr = 1,
    K_BLOCKS: tl.constexpr = 1, BANDS: tl.constexpr = 1,
):
    """GEMV in bitstream order. Code ``t`` of a 16x16 tile sits at bit ``t * BITS`` of the tile's
    stream and is W_hat[r, c] with ``t = 32 (c & 7) + 16 r2 + 8 r1 + 4 (c >> 3) + 2 r3 + r0``, so the
    32 codes of one ``c & 7`` value ("group") are one contiguous bit run. Each lane owns a group of a
    16-row band: every word, shift and row of its 32 codes is a compile-time constant, the few words
    are loaded once (identical addresses CSE), and it accumulates columns ``c`` and ``c + 8``."""
    p = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    if HAS_EXPERT:
        expert = tl.load(expert_ptr + p).to(tl.int64)
    else:
        expert = tl.zeros([], dtype=tl.int64)
    part = tl.load(part_of_nb_ptr + pid_n)
    n_tiles = tl.load(ntiles_ptr + part)
    WORDS: tl.constexpr = 8 * BITS
    OFF0: tl.constexpr = 257 * BITS - 16  # bit of code 0's window: (BITS - 16 + 256 * BITS)
    g = tl.arange(0, 64)
    nt = (pid_n * 128 - tl.load(nstart_ptr + part)) // 16 + g // 8
    cl = g % 8
    start = (32 * BITS * cl + OFF0) // 32  # first word of the group's run (before the wrap)
    tile = tr_ptr + expert * tr_expert_stride + tl.load(word_off_ptr + part) + nt.to(tl.int64) * WORDS
    row_words = n_tiles.to(tl.int64) * WORDS
    acc_lo = tl.zeros([64], dtype=tl.float32)
    acc_hi = tl.zeros([64], dtype=tl.float32)
    k_begin = pid_k * K_SPLIT
    if PRE_ROT:
        # Input rotation in the prologue (replaces the separate had_rows launch): this
        # program rotates only the KB 128-blocks its K range touches, from the raw row
        # x[p // SRC_DIV], into a private fp16 scratch that the loop below reads like xh.
        # H128 = H8 (x) H16 (Sylvester, natural order), so Y = H8 @ X @ H16 on X = x*suh
        # viewed [8, 16]; entries are (-1)^popcount(a & b), generated, not loaded.
        i8 = tl.arange(0, 8)
        i16 = tl.arange(0, 16)
        a8 = i8[:, None] & i8[None, :]
        a8 = a8 ^ (a8 >> 2)
        a8 = a8 ^ (a8 >> 1)
        h8 = 1.0 - 2.0 * (a8 & 1).to(tl.float32)
        a16 = i16[:, None] & i16[None, :]
        a16 = a16 ^ (a16 >> 2)
        a16 = a16 ^ (a16 >> 1)
        h16 = 1.0 - 2.0 * (a16 & 1).to(tl.float32)
        blk0 = k_begin // 128
        rot = rot_ptr + ((p.to(tl.int64) * tl.num_programs(1) + pid_n) * tl.num_programs(2) + pid_k) * (KB * 128)
        src = (p // SRC_DIV).to(tl.int64)
        within = i8[:, None] * 16 + i16[None, :]
        for b in tl.static_range(KB):
            blk = blk0 + b
            bmask = (within >= 0) & (blk < K_BLOCKS)
            c2 = blk * 128 + within
            xv = tl.load(x_ptr + src * stride_x + c2, mask=bmask, other=0.0).to(tl.float32)
            sv_in = tl.load(suh_ptr + expert * suh_expert_stride + part * suh_part_stride + c2,
                            mask=bmask, other=0.0).to(tl.float32)
            xs = xv * sv_in
            t = tl.sum(h8[:, :, None] * xs[None, :, :], axis=1)
            y = tl.sum(t[:, None, :] * h16[None, :, :], axis=2) * 0.08838834764831843
            tl.store(rot + b * 128 + within, y.to(tl.float16))
        tl.debug_barrier()
        a_ptr = rot - blk0 * 128
    else:
        a_ptr = xh_ptr + part * stride_xpart + p.to(tl.int64) * stride_xm
    # BANDS 16-row bands per iteration: their loads are all independent, so a single-warp program
    # keeps BANDS x more bytes in flight (the GEMV is latency-bound at 8 warps per SM)
    for k00 in range(k_begin, k_begin + K_SPLIT, 16 * BANDS):
        for bd in tl.static_range(BANDS):
            k0 = k00 + 16 * bd
            band = tile + (k0 // 16) * row_words
            for i in tl.static_range(32):
                b0 = (OFF0 % 32) + i * BITS  # relative to word `start`
                j0 = b0 // 32
                j1 = (b0 + 15) // 32
                sh = (j1 + 1) * 32 - (b0 + 16)
                r = (i & 1) + 8 * ((i >> 1) & 1) + 2 * ((i >> 3) & 1) + 4 * ((i >> 4) & 1)
                hi = tl.load(band + (start + j0) % WORDS).to(tl.uint32, bitcast=True)
                lo = tl.load(band + (start + j1) % WORDS).to(tl.uint32, bitcast=True)
                w = _decode_words(hi, lo, tl.full([64], sh, tl.uint32), CB).to(tl.float32)
                a = tl.load(a_ptr + k0 + r).to(tl.float32)
                if (i >> 2) & 1:
                    acc_hi += w * a
                else:
                    acc_lo += w * a
    if REDUCE:
        # Split-K: each split stores its UNROTATED partial (columns col, col + 8 of its groups);
        # the last split to finish sums the planes in split order -- the same fixed order whichever
        # program is last, so the result is deterministic -- and applies the 128-point output
        # rotation + svh ONCE, instead of every split rotating its own partial. Re-arms the counter
        # for the next launch (CUDA-graph replays included).
        col_lo = (g // 8) * 16 + cl
        plane = out_ptr + pid_k * stride_osplit + p.to(tl.int64) * stride_om + pid_n * 128
        tl.store(plane + col_lo, acc_lo)
        tl.store(plane + col_lo + 8, acc_hi)
        tl.debug_barrier()
        cnt = count_ptr + p * tl.num_programs(1) + pid_n
        done = tl.atomic_add(cnt, 1, sem="acq_rel")
        if done == tl.num_programs(2) - 1:
            i8 = tl.arange(0, 8)
            i16 = tl.arange(0, 16)
            o816 = i8[:, None] * 16 + i16[None, :]
            vsum = tl.zeros([8, 16], dtype=tl.float32)
            for sp in range(0, tl.num_programs(2)):
                vsum += tl.load(out_ptr + sp * stride_osplit + p.to(tl.int64) * stride_om + pid_n * 128 + o816,
                                cache_modifier=".cg")
            ocols = pid_n * 128 + o816
            svo = tl.load(svh_ptr + expert * svh_expert_stride + ocols).to(tl.float32)
            yo = _had128_8x16(vsum) * svo * 0.08838834764831843
            tl.store(final_ptr + p.to(tl.int64) * stride_fm + ocols, yo.to(final_ptr.dtype.element_ty))
            tl.atomic_xchg(cnt, 0)
        return
    # acc_lo[g] / acc_hi[g] hold column 16 (g // 8) + (g % 8) / + 8: as [8 tiles, 16 columns]
    v = tl.reshape(tl.permute(tl.join(tl.reshape(acc_lo, (8, 8)), tl.reshape(acc_hi, (8, 8))), (0, 2, 1)), (8, 16))
    o816 = tl.arange(0, 8)[:, None] * 16 + tl.arange(0, 16)[None, :]
    cols = pid_n * 128 + o816
    sv = tl.load(svh_ptr + expert * svh_expert_stride + cols).to(tl.float32)
    y = _had128_8x16(v) * sv * 0.08838834764831843
    tl.store(out_ptr + pid_k * stride_osplit + p.to(tl.int64) * stride_om + cols, y.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


# had_rows tile: rows per program, warps, 128-column blocks per program (each program loads the
# 128x128 H once and reuses it across its blocks). The kernel is mma-bound, not DRAM-bound: the
# exact hi/lo split runs two 128-wide dots per element, 137 GFLOP for the MoE gate_up input at
# 8192 tokens x top-8 (1.13 ms at 121.5 TF/s). Box sweep (perf/bench_had_rows.py), gate_up input:
# bm16/4 warps/kb1 2.01-2.08 ms -> kb16 1.46 ms; larger bm or 8 warps spill and lose.
HAD_ROWS_BM = 16
HAD_ROWS_WARPS = 4
HAD_ROWS_KB = 16  # clamped to the divisors of K / 128 (down input: 4)
# ... only from this many rows on: a decode step (a few rows) needs one program per column block
# to fill the SMs
HAD_ROWS_KB_MIN_ROWS = 4096
# From HAD_ROWS_KB_MIN_ROWS rows on: H in 2 column slices loaded per block (bit-exact, see the kernel)
# frees the registers of the held 128x128 H (255 -> fewer per thread), so 64-row tiles fit. RTX 5080,
# 8192 tokens x top-8, interleaved medians (tasks/ornith-exl3/kfix/results/rerun2-had-rows.txt):
# gate_up input 0.744 -> 0.673 ms, down input 0.244 -> 0.201 ms. Holding both slices spills (x2.5).
HAD_ROWS_BIG = dict(BM=64, KB=2, NS=2, num_warps=4)


def had_rows(
    x: torch.Tensor,
    suh: torch.Tensor,
    parts: Exl3Parts,
    *,
    rows: int | None = None,
    src_div: int = 1,
    experts: torch.Tensor | None = None,
    suh_expert_stride: int = 0,
    out: torch.Tensor | None = None,
    block_m: int | None = None,
    num_warps: int | None = None,
    k_blocks: int | None = None,
    n_slices: int | None = None,
) -> torch.Tensor:
    """``out[part, p] = H_blocks(x[p // src_div] * suh[experts[p], part])`` as fp16 ``[parts, rows, K]``.

    ``suh`` is addressed as ``suh.data_ptr() + expert * suh_expert_stride + part * parts.suh_part_stride``
    (elements); a dense layer passes ``experts=None``."""
    assert x.stride(-1) == 1 and suh.stride(-1) == 1
    k = parts.k
    rows = x.shape[0] * src_div if rows is None else rows
    if out is None:
        out = torch.empty((parts.num_parts, rows, k), dtype=torch.float16, device=x.device)
    if rows == 0:
        return out
    big = HAD_ROWS_BIG if rows >= HAD_ROWS_KB_MIN_ROWS else {}
    bm = block_m or big.get("BM", HAD_ROWS_BM)
    kb = k_blocks or big.get("KB", HAD_ROWS_KB if rows >= HAD_ROWS_KB_MIN_ROWS else 1)
    ns = n_slices or big.get("NS", 1)
    while (k // HAD) % kb:
        kb //= 2
    grid = (triton.cdiv(rows, bm), k // HAD // kb, parts.num_parts)
    _had_rows_kernel[grid](
        x, x.stride(0),
        out, out.stride(0), out.stride(1),
        suh, suh_expert_stride, parts.suh_part_stride,
        experts if experts is not None else x, hadamard_pm1(x.device), rows,
        SRC_DIV=src_div, HAS_EXPERT=experts is not None, BM=bm, KB=kb, NS=ns,
        num_warps=num_warps or big.get("num_warps", HAD_ROWS_WARPS),
    )
    return out


def _words(trellis: torch.Tensor) -> torch.Tensor:
    return trellis.view(torch.int32) if trellis.dtype != torch.int32 else trellis


def exl3_gemm(
    xh: torch.Tensor,
    trellis: torch.Tensor,
    svh: torch.Tensor,
    parts: Exl3Parts,
    *,
    out: torch.Tensor,
    tr_expert_stride: int = 0,
    svh_expert_stride: int = 0,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_post_pad: torch.Tensor | None = None,
    block_m: int = 64,
    decoded: torch.Tensor | None = None,
    expert_range: tuple[int, int] = (0, 0),
    block_k: int = 16,
    num_stages: int = 1,
    num_warps: int = 4,
    src_div: int = 0,
    f16acc: bool = False,
) -> torch.Tensor:
    """Grouped GEMM: ``out[p] = svh * H(xh[part(n), p] @ W_hat)``; rows sorted per expert when
    ``sorted_ids`` is given (``tr_expert_stride`` in int32 words), else every row uses expert 0.

    ``decoded`` (sorted mode): W_hat already decoded by ``reconstruct_experts`` for the experts in
    ``expert_range`` = [lo, hi), fp16 ``[hi - lo, K, N]``; only those experts' row blocks run.

    ``src_div > 0``: ``xh`` is the RAW activation ``[T, K]`` fp16 shared by all parts (route ``p``
    reads row ``p // src_div``) and ``decoded`` must carry the input rotation
    (``reconstruct_folded(..., fold_out=False)``); rows = ``T * src_div``.

    The defaults are the in-kernel-decode tile (the decoded path passes its own): one stage, since
    software-pipelining the decode gathers routes them through shared memory and leaves the kernel
    MIO-bound (RTX 5080, fused MoE prefill at 64 tokens 12.5 -> 2.4 ms per layer, bk 32 / 2 stages ->
    bk 16 / 1 stage, tasks/ornith-exl3/kfix/results/rerun-inline-gemm.txt). Tile shape and stages
    keep each output's 16-wide mma k-step sequence, so the result is bitwise the same."""
    raw = src_div > 0
    rows = xh.shape[0] * src_div if raw else xh.shape[1]
    if rows == 0:
        return out
    words = _words(trellis)
    sorted_mode = sorted_ids is not None
    # sorted mode: blocks at or past num_post_pad return at once, so the grid may overshoot
    n_mblocks = triton.cdiv(sorted_ids.numel(), block_m) if sorted_mode else triton.cdiv(rows, block_m)
    grid = (n_mblocks, parts.n // HAD)
    dummy = parts.part_of_nb
    _exl3_gemm_kernel[grid](
        xh, 0 if raw else xh.stride(0), xh.stride(0) if raw else xh.stride(1),
        out, out.stride(0),
        words, tr_expert_stride,
        svh, svh_expert_stride,
        hadamard_pm1(xh.device),
        parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart,
        sorted_ids if sorted_mode else dummy, expert_ids if sorted_mode else dummy,
        num_post_pad if sorted_mode else dummy,
        rows, parts.k,
        decoded if decoded is not None else xh, decoded.stride(0) if decoded is not None else 0,
        expert_range[0], expert_range[1],
        BM=block_m, BK=block_k, BITS=parts.bits, CB=CODEBOOKS[parts.codebook], SORTED=sorted_mode,
        DECODED=decoded is not None, SRC_DIV=src_div if raw else 1, F16ACC=f16acc,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# 16-row bands per GEMV loop iteration (clamped to divide the split's K range)
GEMV_BANDS = 1
# Register cap of the single-warp GEMV program. Uncapped, ptxas gives the loop 80 registers (25
# warps per SM); 64 is spill-free (ptxas -v, sm_120) and fills the 32 resident blocks an SM allows.
# The old epilogue (a 128 x 128 H product per program) needed 168 registers plus 1.4 KB of spills,
# capping the GEMV at 12 warps per SM.
GEMV_MAXNREG = 64


def exl3_gemv(
    xh: torch.Tensor | None,
    trellis: torch.Tensor,
    svh: torch.Tensor,
    parts: Exl3Parts,
    *,
    out: torch.Tensor,
    experts: torch.Tensor | None = None,
    tr_expert_stride: int = 0,
    svh_expert_stride: int = 0,
    split_k: int = 1,
    x: torch.Tensor | None = None,
    suh: torch.Tensor | None = None,
    suh_expert_stride: int = 0,
    src_div: int = 1,
    bands: int | None = None,
) -> torch.Tensor:
    """Per-row GEMV (decode) into ``out`` ``[rows, N]`` (any float dtype). With ``split_k > 1``
    each K split writes a rotated fp32 partial to its own plane of a scratch buffer and the
    planes are summed in order (the output rotation is linear, so rotating partials is exact):
    deterministic, and fixed-shape for CUDA graphs.

    ``xh=None`` with ``x``/``suh`` (had_rows' arguments) rotates the input inside the GEMV
    prologue instead of reading a had_rows result: rows = ``x.shape[0] * src_div``, row ``p``
    reads ``x[p // src_div]`` scaled by ``suh[experts[p]]`` (dense: ``experts=None``)."""
    pre_rot = xh is None
    if pre_rot:
        assert x is not None and suh is not None and x.stride(-1) == 1 and suh.stride(-1) == 1
        rows = x.shape[0] * src_div
        device = x.device
    else:
        rows = xh.shape[1]
        device = xh.device
    if rows == 0:
        return out
    k = parts.k
    if k % (split_k * 16):
        raise ValueError(f"split_k {split_k} does not divide K {k} into 16-row bands")
    dst = out if split_k == 1 else torch.empty((split_k, rows, parts.n), dtype=torch.float32, device=device)
    words = _words(trellis)
    grid = (rows, parts.n // HAD, split_k)
    reduce = split_k > 1 and _inkernel_reduce()
    counts = _split_counters(rows * (parts.n // HAD), device) if reduce else parts.part_of_nb
    nb = bands or GEMV_BANDS
    while (k // split_k) % (16 * nb):
        nb //= 2
    kb = 1
    if pre_rot:
        k_split = k // split_k
        kb = max((((i + 1) * k_split - 1) // HAD) - (i * k_split) // HAD + 1 for i in range(split_k))
        rot = torch.empty((rows * (parts.n // HAD) * split_k, kb * HAD), dtype=torch.float16, device=device)
        xh = rot  # placeholder for the unused xh pointer
    _exl3_gemv_kernel[grid](
        xh, xh.stride(0) if not pre_rot else 0, xh.stride(1) if not pre_rot else 0,
        dst, dst.stride(-2), dst.stride(0) if split_k > 1 else 0,
        words, tr_expert_stride,
        svh, svh_expert_stride,
        hadamard_pm1(device),
        parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart,
        experts if experts is not None else parts.part_of_nb, k // split_k,
        out, out.stride(-2), counts,
        x if pre_rot else xh, x.stride(0) if pre_rot else 0,
        suh if pre_rot else xh, suh_expert_stride, parts.suh_part_stride,
        rot if pre_rot else xh,
        BITS=parts.bits, CB=CODEBOOKS[parts.codebook],
        HAS_EXPERT=experts is not None, REDUCE=reduce,
        PRE_ROT=pre_rot, SRC_DIV=src_div, KB=kb, K_BLOCKS=k // HAD, BANDS=nb,
        num_warps=1, maxnreg=GEMV_MAXNREG,
    )
    if split_k > 1 and not reduce:
        torch.sum(dst, dim=0, out=out) if out.dtype == torch.float32 else out.copy_(dst.sum(dim=0))
    return out


def gemv_pre_rot() -> bool:
    """``FREETOKEN_EXL3_GEMV_PREROT=0`` restores the separate ``had_rows`` launch before a decode GEMV."""
    return os.getenv("FREETOKEN_EXL3_GEMV_PREROT", "1").strip() != "0"


def f16acc_enabled() -> bool:
    """``FREETOKEN_EXL3_F16ACC=1``: prefill GEMMs (MoE decoded slab, dense folded W_full) accumulate
    in fp16 (A/B, off by default: default-on only after the logits gate)."""
    return os.getenv("FREETOKEN_EXL3_F16ACC", "0").strip() == "1"


def _inkernel_reduce() -> bool:
    """``FREETOKEN_EXL3_SPLITK_INKERNEL=0`` restores the separate ``torch.sum`` reduction."""
    return os.getenv("FREETOKEN_EXL3_SPLITK_INKERNEL", "1").strip() != "0"


_COUNTERS: dict = {}
# Which counter buffer the next GEMVs use. GEMVs issued in stream order may share one buffer
# (each launch leaves it zeroed), but GEMVs that can run CONCURRENTLY on another stream must not:
# the shared expert on its side stream (models/qwen3_5_moe/moe.py) takes lane 1 while the routed
# GEMVs use lane 0, else both launches count arrivals in the same slots and a split's partial
# is summed by the wrong program (seen as non-identical greedy output, 2026-09-24).
_COUNTER_LANE = 0


class split_counter_lane:
    """``with split_counter_lane(1): ...`` -- GEMVs launched inside use counter lane 1."""

    def __init__(self, lane: int):
        self.lane = lane

    def __enter__(self):
        global _COUNTER_LANE
        self.prev, _COUNTER_LANE = _COUNTER_LANE, self.lane
        return self

    def __exit__(self, *exc):
        global _COUNTER_LANE
        _COUNTER_LANE = self.prev
        return False


def _split_counters(n: int, device: torch.device) -> torch.Tensor:
    """Zeroed int32 arrival counters for the in-kernel split-K reduction, one per (row, 128
    output columns). Every launch leaves them zero again, so one buffer per device and counter
    lane serves all GEMVs issued in stream order on that lane; it is allocated (or grown) outside
    graph capture, on the warm-up pass that precedes it."""
    key = (device, _COUNTER_LANE)
    buf = _COUNTERS.get(key)
    if buf is None or buf.numel() < n:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 split-K counters must be allocated before CUDA-graph capture")
        buf = torch.zeros(max(n, 1 << 14), dtype=torch.int32, device=device)
        _COUNTERS[key] = buf
    return buf


@triton.jit
def _splitk_silu_had_kernel(
    pl_ptr, stride_ps, stride_pm,
    out_ptr, stride_om,
    suh_ptr, suh_expert_stride, expert_ptr, had_ptr, P, INTER,
    S: tl.constexpr, BM: tl.constexpr,
):
    """Decode epilogue of gate|up and prologue of down in one pass: sum the split-K planes of
    ``g = [gate | up]`` (fixed order), round to fp16 as the unfused path does, ``a = silu(gate) *
    up`` in fp32 rounded to fp16, then ``ah = H_blocks(a * suh[expert])`` exactly as had_rows."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < P
    cols = pid_k * 128 + tl.arange(0, 128)
    base = pl_ptr + rows[:, None].to(tl.int64) * stride_pm + cols[None, :]
    g = tl.zeros([BM, 128], dtype=tl.float32)
    u = tl.zeros([BM, 128], dtype=tl.float32)
    for sp in tl.static_range(S):
        g += tl.load(base + sp * stride_ps, mask=rmask[:, None], other=0.0)
        u += tl.load(base + sp * stride_ps + INTER, mask=rmask[:, None], other=0.0)
    g = g.to(tl.float16).to(tl.float32)
    u = u.to(tl.float16).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g)) * u).to(tl.float16)
    e = tl.load(expert_ptr + rows, mask=rmask, other=0).to(tl.int64)
    sc = tl.load(suh_ptr + e[:, None] * suh_expert_stride + cols[None, :], mask=rmask[:, None], other=0.0)
    xs = a.to(tl.float32) * sc.to(tl.float32)
    hi = xs.to(tl.float16)
    lo = (xs - hi.to(tl.float32)).to(tl.float16)
    idx = tl.arange(0, 128)
    h = tl.load(had_ptr + idx[:, None] * 128 + idx[None, :])
    y = (tl.dot(hi, h) + tl.dot(lo, h)) * 0.08838834764831843
    tl.store(out_ptr + rows[:, None].to(tl.int64) * stride_om + cols[None, :], y.to(tl.float16), mask=rmask[:, None])


def splitk_silu_had(planes: torch.Tensor, suh: torch.Tensor, experts: torch.Tensor, suh_expert_stride: int) -> torch.Tensor:
    """``planes`` ``[S, P, 2I]`` fp32 gate|up partials -> ``[1, P, I]`` fp16 rotated down input
    (the ``xh`` layout ``exl3_gemv`` reads)."""
    splits, rows, n2 = planes.shape
    inter = n2 // 2
    assert inter % HAD == 0 and planes.stride(-1) == 1 and suh.stride(-1) == 1
    out = torch.empty((1, rows, inter), dtype=torch.float16, device=planes.device)
    if rows:
        bm = 16
        _splitk_silu_had_kernel[(triton.cdiv(rows, bm), inter // HAD)](
            planes, planes.stride(0), planes.stride(1),
            out, out.stride(1),
            suh, suh_expert_stride, experts, hadamard_pm1(planes.device), rows, inter,
            S=splits, BM=bm, num_warps=4,
        )
    return out


@triton.jit
def _splitk_combine_kernel(
    pl_ptr, stride_ps, stride_pm, w_ptr, out_ptr, stride_ot,
    S: tl.constexpr, TOP_K: tl.constexpr, BN: tl.constexpr,
):
    """``out[t] = sum_k w[t, k] * sum_s planes[s, t * TOP_K + k]`` in fp32, fixed order."""
    t = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = t * TOP_K + k
        o = tl.zeros([BN], dtype=tl.float32)
        for sp in tl.static_range(S):
            o += tl.load(pl_ptr + sp * stride_ps + p.to(tl.int64) * stride_pm + cols)
        acc += o * tl.load(w_ptr + p).to(tl.float32)
    tl.store(out_ptr + t.to(tl.int64) * stride_ot + cols, acc.to(out_ptr.dtype.element_ty))


def splitk_combine(planes: torch.Tensor, topk_weights: torch.Tensor, tokens: int, top_k: int, dtype: torch.dtype) -> torch.Tensor:
    """Down-projection split-K partials ``[S, tokens * top_k, H]`` -> routed output ``[tokens, H]``."""
    splits, rows, n = planes.shape
    assert rows == tokens * top_k and n % HAD == 0 and planes.stride(-1) == 1
    w = topk_weights.reshape(tokens * top_k)
    if w.stride(0) != 1:
        w = w.contiguous()
    out = torch.empty((tokens, n), dtype=dtype, device=planes.device)
    if tokens:
        _splitk_combine_kernel[(tokens, n // HAD)](
            planes, planes.stride(0), planes.stride(1), w, out, out.stride(0),
            S=splits, TOP_K=top_k, BN=HAD, num_warps=4,
        )
    return out


@triton.jit
def _had_cols_kernel(acc_ptr, stride_am, out_ptr, stride_om, svh_ptr, had_ptr, rows, BM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    r = pid_m * BM + tl.arange(0, BM)
    rmask = r < rows
    r64 = r.to(tl.int64)
    idx = tl.arange(0, 128)
    cols = pid_n * 128 + idx
    acc = tl.load(acc_ptr + r64[:, None] * stride_am + cols[None, :], mask=rmask[:, None], other=0.0)
    h = tl.load(had_ptr + idx[:, None] * 128 + idx[None, :])
    hi = acc.to(tl.float16)
    lo = (acc - hi.to(tl.float32)).to(tl.float16)
    y = tl.dot(hi, h) + tl.dot(lo, h)
    sv = tl.load(svh_ptr + cols).to(tl.float32)
    y = y * (sv * 0.08838834764831843)[None, :]
    tl.store(out_ptr + r64[:, None] * stride_om + cols[None, :], y.to(out_ptr.dtype.element_ty), mask=rmask[:, None])


def had_cols(acc: torch.Tensor, svh: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out = svh * H_128blocks(acc)`` (the GEMM epilogue on its own): ``acc`` fp32 ``[rows, N]``,
    ``out`` ``[rows, N]`` of any float dtype, both with unit column stride; ``svh`` ``[N]``."""
    rows, n = acc.shape
    assert n % HAD == 0 and acc.stride(1) == 1 and out.stride(1) == 1 and out.shape == acc.shape
    if rows:
        _had_cols_kernel[(triton.cdiv(rows, 64), n // HAD)](
            acc, acc.stride(0), out, out.stride(0), svh, hadamard_pm1(acc.device), rows, BM=64, num_warps=4,
        )
    return out


@triton.jit
def _reconstruct_kernel(tr_ptr, out_ptr, n_tiles, N, BITS: tl.constexpr, CB: tl.constexpr):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    kk = pid_k * 16 + tl.arange(0, 16)
    nn = pid_n * 128 + tl.arange(0, 128)
    w = _exl3_decode(tr_ptr, kk[:, None], nn[None, :], n_tiles, BITS, CB)
    tl.store(out_ptr + kk[:, None].to(tl.int64) * N + nn[None, :], w)


@triton.jit
def _reconstruct_experts_kernel(
    tr_ptr, tr_expert_stride, out_ptr, out_expert_stride, e_lo,
    part_of_nb_ptr, word_off_ptr, ntiles_ptr, nstart_ptr,
    BITS: tl.constexpr, CB: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2).to(tl.int64)
    part = tl.load(part_of_nb_ptr + pid_n)
    n_tiles = tl.load(ntiles_ptr + part)
    n_local = pid_n * 128 - tl.load(nstart_ptr + part) + tl.arange(0, 128)
    tr = tr_ptr + (e_lo + g) * tr_expert_stride + tl.load(word_off_ptr + part)
    kk = pid_k * 16 + tl.arange(0, 16)
    w = _exl3_decode(tr, kk[:, None], n_local[None, :], n_tiles, BITS, CB)
    n = tl.num_programs(1) * 128
    tl.store(out_ptr + g * out_expert_stride + kk[:, None].to(tl.int64) * n + (pid_n * 128 + tl.arange(0, 128))[None, :], w)


def reconstruct_experts(bank: torch.Tensor, parts: Exl3Parts, lo: int, hi: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """Decode experts ``[lo, hi)`` of a ``[E, words]`` trellis bank (all parts, in the GEMM's column
    order) to W_hat fp16 ``[hi - lo, K, N]`` -- the ``decoded`` operand of ``exl3_gemm``."""
    g = hi - lo
    if out is None:
        out = torch.empty((g, parts.k, parts.n), dtype=torch.float16, device=bank.device)
    assert out.shape[1:] == (parts.k, parts.n) and out.shape[0] >= g and out.is_contiguous()
    if g > 0:
        _reconstruct_experts_kernel[(parts.k // 16, parts.n // HAD, g)](
            _words(bank), bank.stride(0) // (2 if bank.dtype == torch.int16 else 1), out, out.stride(0), lo,
            parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart,
            BITS=parts.bits, CB=CODEBOOKS[parts.codebook],
        )
    return out


@triton.jit
def _reconstruct_folded_kernel(
    tr_ptr, tr_expert_stride, out_ptr, out_expert_stride, out_row_stride, e_lo, nb0,
    suh_ptr, suh_expert_stride, suh_part_stride, svh_ptr, svh_expert_stride, had_ptr,
    part_of_nb_ptr, word_off_ptr, ntiles_ptr, nstart_ptr,
    BITS: tl.constexpr, CB: tl.constexpr, FOLD_OUT: tl.constexpr,
):
    """One 128x128 tile of ``W_full = diag(suh) H W_hat H diag(svh) / 128`` (both rotations and
    both sign vectors folded into the weight, exllamav3's reconstruct_had): ``x @ W_full`` equals
    ``svh * H((H(x * suh)) @ W_hat)`` without rotating any activation."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1) + nb0  # global 128-column block; the output holds blocks [nb0, ...)
    g = tl.program_id(2).to(tl.int64)
    e = e_lo + g
    part = tl.load(part_of_nb_ptr + pid_n)
    n_tiles = tl.load(ntiles_ptr + part)
    idx = tl.arange(0, 128)
    n_local = pid_n * 128 - tl.load(nstart_ptr + part) + idx
    tr = tr_ptr + e * tr_expert_stride + tl.load(word_off_ptr + part)
    kk = pid_k * 128 + idx
    w = _exl3_decode(tr, kk[:, None], n_local[None, :], n_tiles, BITS, CB)
    h = tl.load(had_ptr + idx[:, None] * 128 + idx[None, :])
    a = tl.dot(h, w)  # H W: 128 exact +-w terms per entry, fp32
    s = tl.load(suh_ptr + e * suh_expert_stride + part * suh_part_stride + kk).to(tl.float32)
    cols = pid_n * 128 + idx
    if FOLD_OUT:
        hi = a.to(tl.float16)
        lo = (a - hi.to(tl.float32)).to(tl.float16)
        b = tl.dot(hi, h) + tl.dot(lo, h)
        v = tl.load(svh_ptr + e * svh_expert_stride + cols).to(tl.float32)
        y = b * (s * 0.0078125)[:, None] * v[None, :]
    else:  # input side only: diag(suh) H W_hat / sqrt(128); the GEMM epilogue keeps H + svh
        y = a * (s * 0.08838834764831843)[:, None]
    lcols = (pid_n - nb0) * 128 + idx
    tl.store(out_ptr + g * out_expert_stride + kk[:, None].to(tl.int64) * out_row_stride + lcols[None, :],
             y.to(out_ptr.dtype.element_ty))


def reconstruct_folded(
    trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, parts: Exl3Parts, *,
    lo: int = 0, hi: int = 1, suh_expert_stride: int = 0, svh_expert_stride: int = 0,
    out: torch.Tensor | None = None, fold_out: bool = True, cols: tuple[int, int] | None = None,
) -> torch.Tensor:
    """``W_full`` fp16 ``[hi - lo, K, N]`` for experts ``[lo, hi)`` of a ``[E, words]`` bank (a dense
    layer: a flat trellis, ``lo=0, hi=1``): ``x @ W_full[e] = svh * H(H(x * suh) @ W_hat)`` per part,
    so a plain GEMM on the raw activations replaces had_rows + GEMM + had_cols.

    ``fold_out=False`` folds the input side only (``diag(suh) H W_hat``): the GEMM then reads raw
    activations and keeps its Hadamard + svh epilogue (the MoE prefill's decoded operand).

    ``cols=(c0, c1)`` (multiples of 128) builds only those output columns, ``[hi - lo, K, c1 - c0]``."""
    g = hi - lo
    c0, c1 = cols if cols is not None else (0, parts.n)
    assert c0 % HAD == 0 and c1 % HAD == 0 and 0 <= c0 < c1 <= parts.n
    if out is None:
        out = torch.empty((g, parts.k, c1 - c0), dtype=torch.float16, device=trellis.device)
    assert out.shape[1:] == (parts.k, c1 - c0) and out.shape[0] >= g and out.is_contiguous()
    words = _words(trellis)
    tr_stride = words.stride(0) if words.dim() > 1 else 0
    if g > 0:
        _reconstruct_folded_kernel[(parts.k // HAD, (c1 - c0) // HAD, g)](
            words, tr_stride, out, out.stride(0), out.stride(1), lo, c0 // HAD,
            suh, suh_expert_stride, parts.suh_part_stride, svh, svh_expert_stride, hadamard_pm1(trellis.device),
            parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart,
            BITS=parts.bits, CB=CODEBOOKS[parts.codebook], FOLD_OUT=fold_out, num_warps=8,
        )
    return out


@triton.jit
def _gemm_f16acc_kernel(
    a_ptr, stride_am, b_ptr, stride_bk, c_ptr, stride_cm, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    per_group = GROUP * n_n
    first_m = (pid // per_group) * GROUP
    group_m = tl.minimum(n_m - first_m, GROUP)
    pid_m = first_m + (pid % per_group) % group_m
    pid_n = (pid % per_group) // group_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None].to(tl.int64) * stride_am + rk[None, :]
    b = b_ptr + rk[:, None].to(tl.int64) * stride_bk + rn[None, :]
    acc = tl.zeros([BM, BN], dtype=tl.float16)
    for k0 in range(0, K, BK):
        x = tl.load(a, mask=rm[:, None] < M, other=0.0).to(tl.float16)  # bf16 activations cast here
        w = tl.load(b, mask=rn[None, :] < N, other=0.0)
        acc = tl.dot(x, w, acc, out_dtype=tl.float16)
        a += BK
        b += BK * stride_bk
    tl.store(c_ptr + rm[:, None].to(tl.int64) * stride_cm + rn[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


# tiles of gemm_f16acc: box sweep at M=8192 (perf/bench_f16acc.py) put BM128/BK32/4 warps/3 stages
# first on every Ornith shape, BN256 on the wide ones (in_proj 217, qkv 209 TF/s) and BN128 on the
# narrow ones (o_proj 196, shared 169-172 TF/s); cuBLAS fp32-accumulate reaches 104-120 TF/s
GEMM_F16ACC_CFG = dict(BM=128, BN=128, BK=32, GROUP=8, num_warps=4, num_stages=3)
GEMM_F16ACC_WIDE_N = 4096  # BN=256 from this output width on


def gemm_f16acc(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None, *, out_dtype=torch.float16, **cfg) -> torch.Tensor:
    """``a @ b`` for fp16 ``b [K, N]`` and fp16 or bf16 ``a [M, K]`` (rounded to fp16 on load, as the
    host cast would) with FP16 accumulation (exllamav3's hgemm numerics); K a multiple of the K
    tile. ``out`` may be a column slice of a wider tensor."""
    m, k = a.shape
    n = b.shape[1]
    assert a.dtype in (torch.float16, torch.bfloat16) and b.dtype == torch.float16
    assert a.stride(1) == 1 and b.stride(1) == 1
    c = {**GEMM_F16ACC_CFG, **({"BN": 256} if n >= GEMM_F16ACC_WIDE_N else {}), **cfg}
    assert k % c["BK"] == 0
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    assert out.stride(1) == 1
    grid = (triton.cdiv(m, c["BM"]) * triton.cdiv(n, c["BN"]),)
    _gemm_f16acc_kernel[grid](a, a.stride(0), b, b.stride(0), out, out.stride(0), m, n, k,
                              BM=c["BM"], BN=c["BN"], BK=c["BK"], GROUP=c["GROUP"],
                              num_warps=c["num_warps"], num_stages=c["num_stages"])
    return out


def reconstruct(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    """Triton decode of a ``[K/16, N/16, 16*bits]`` trellis to ``W_hat`` fp16 ``[K, N]`` (tests, tools)."""
    kt, nt, width = trellis.shape
    bits = width // 16
    k, n = kt * 16, nt * 16
    if n % HAD:
        raise ValueError(f"N {n} is not a multiple of {HAD}")
    out = torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    _reconstruct_kernel[(kt, n // HAD)](_words(trellis.contiguous()), out, nt, n, BITS=bits, CB=CODEBOOKS[codebook])
    return out


__all__ = [
    "CODEBOOKS", "Exl3Parts", "HAD", "HAD_SCALE", "MCG_MULT", "MUL1_MULT",
    "exl3_gemm", "exl3_gemv", "f16acc_enabled", "gemm_f16acc", "had_cols", "had_rows", "reconstruct_experts", "reconstruct_folded", "hadamard_pm1", "hadamard_reference",
    "linear_reference", "reconstruct", "reconstruct_reference", "tile_stream_index",
]
