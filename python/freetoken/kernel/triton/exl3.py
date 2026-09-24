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
    shift = ((i1 + 1) * 32 - b1).to(tl.uint64)
    a = tl.load(tr_ptr + base + (b0 // 32) % WORDS).to(tl.uint32, bitcast=True).to(tl.uint64)
    b = tl.load(tr_ptr + base + i1 % WORDS).to(tl.uint32, bitcast=True).to(tl.uint64)
    w = (((a << 32) | b) >> shift) & 0xFFFF
    if CB == 2:
        x = (w * 0x83DCD12D) & 0xFFFFFFFF
        s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        # (1024 + s) * inv + bias is exact in fp32; one rounding to fp16 matches exllamav3's hfma
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
def _had_rows_kernel(
    x_ptr, stride_xm,
    out_ptr, stride_opart, stride_om,
    suh_ptr, suh_expert_stride, suh_part_stride,
    expert_ptr, had_ptr, P,
    SRC_DIV: tl.constexpr, HAS_EXPERT: tl.constexpr, BM: tl.constexpr,
):
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
    cols = pid_k * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + src[:, None].to(tl.int64) * stride_xm + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    s = tl.load(suh_ptr + e[:, None] * suh_expert_stride + part * suh_part_stride + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    xs = x * s
    hi = xs.to(tl.float16)
    lo = (xs - hi.to(tl.float32)).to(tl.float16)
    idx = tl.arange(0, 128)
    h = tl.load(had_ptr + idx[:, None] * 128 + idx[None, :])
    y = (tl.dot(hi, h) + tl.dot(lo, h)) * 0.08838834764831843
    tl.store(out_ptr + part * stride_opart + rows[:, None].to(tl.int64) * stride_om + cols[None, :], y.to(tl.float16), mask=rmask[:, None])


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
    DECODED: tl.constexpr,
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
    n_local = pid_n * 128 - tl.load(nstart_ptr + part) + tl.arange(0, 128)
    tr = tr_ptr + expert * tr_expert_stride + tl.load(word_off_ptr + part)
    a_ptr = xh_ptr + part * stride_xpart + rows64[:, None] * stride_xm
    acc = tl.zeros([BM, 128], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + kk[None, :], mask=rmask[:, None], other=0.0)
        if DECODED:
            w = tl.load(w_ptr + (expert - e_lo) * w_expert_stride + kk[:, None].to(tl.int64) * (tl.num_programs(1) * 128)
                        + (pid_n * 128 + tl.arange(0, 128))[None, :])
        else:
            w = _exl3_decode(tr, kk[:, None], n_local[None, :], n_tiles, BITS, CB)
        acc = tl.dot(a, w, acc)
    idx = tl.arange(0, 128)
    h = tl.load(had_ptr + idx[:, None] * 128 + idx[None, :])
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
def _exl3_gemv_kernel(
    xh_ptr, stride_xpart, stride_xm,
    out_ptr, stride_om, stride_osplit,
    tr_ptr, tr_expert_stride,
    svh_ptr, svh_expert_stride,
    had_ptr,
    part_of_nb_ptr, word_off_ptr, ntiles_ptr, nstart_ptr,
    expert_ptr, K_SPLIT,
    final_ptr, stride_fm, count_ptr,
    BITS: tl.constexpr, CB: tl.constexpr, HAS_EXPERT: tl.constexpr, REDUCE: tl.constexpr,
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
    a_ptr = xh_ptr + part * stride_xpart + p.to(tl.int64) * stride_xm
    row_words = n_tiles.to(tl.int64) * WORDS
    acc_lo = tl.zeros([64], dtype=tl.float32)
    acc_hi = tl.zeros([64], dtype=tl.float32)
    k_begin = pid_k * K_SPLIT
    for k0 in range(k_begin, k_begin + K_SPLIT, 16):
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
    idx = tl.arange(0, 128)
    col_lo = (g // 8) * 16 + cl
    h_lo = tl.load(had_ptr + col_lo[:, None] * 128 + idx[None, :]).to(tl.float32)
    h_hi = tl.load(had_ptr + (col_lo + 8)[:, None] * 128 + idx[None, :]).to(tl.float32)
    y = tl.sum(acc_lo[:, None] * h_lo, axis=0) + tl.sum(acc_hi[:, None] * h_hi, axis=0)
    cols = pid_n * 128 + idx
    sv = tl.load(svh_ptr + expert * svh_expert_stride + cols).to(tl.float32)
    y = y * sv * 0.08838834764831843
    tl.store(out_ptr + pid_k * stride_osplit + p.to(tl.int64) * stride_om + cols, y.to(out_ptr.dtype.element_ty))
    if REDUCE:
        # Split-K reduced in place: the last split of this (row, 128 columns) to finish sums
        # every split's plane in split order -- the same fixed order whichever program is
        # last, so the result is deterministic -- writes the output, and re-arms the counter
        # for the next launch (CUDA-graph replays included).
        tl.debug_barrier()
        cnt = count_ptr + p * tl.num_programs(1) + pid_n
        done = tl.atomic_add(cnt, 1, sem="acq_rel")
        if done == tl.num_programs(2) - 1:
            acc = tl.zeros([128], dtype=tl.float32)
            for sp in range(0, tl.num_programs(2)):
                acc += tl.load(out_ptr + sp * stride_osplit + p.to(tl.int64) * stride_om + cols,
                               cache_modifier=".cg")
            tl.store(final_ptr + p.to(tl.int64) * stride_fm + cols, acc.to(final_ptr.dtype.element_ty))
            tl.atomic_xchg(cnt, 0)


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


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
    bm = 16
    grid = (triton.cdiv(rows, bm), k // HAD, parts.num_parts)
    _had_rows_kernel[grid](
        x, x.stride(0),
        out, out.stride(0), out.stride(1),
        suh, suh_expert_stride, parts.suh_part_stride,
        experts if experts is not None else x, hadamard_pm1(x.device), rows,
        SRC_DIV=src_div, HAS_EXPERT=experts is not None, BM=bm,
        num_warps=4,
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
) -> torch.Tensor:
    """Grouped GEMM: ``out[p] = svh * H(xh[part(n), p] @ W_hat)``; rows sorted per expert when
    ``sorted_ids`` is given (``tr_expert_stride`` in int32 words), else every row uses expert 0.

    ``decoded`` (sorted mode): W_hat already decoded by ``reconstruct_experts`` for the experts in
    ``expert_range`` = [lo, hi), fp16 ``[hi - lo, K, N]``; only those experts' row blocks run."""
    rows = xh.shape[1]
    if rows == 0:
        return out
    words = _words(trellis)
    sorted_mode = sorted_ids is not None
    # sorted mode: blocks at or past num_post_pad return at once, so the grid may overshoot
    n_mblocks = triton.cdiv(sorted_ids.numel(), block_m) if sorted_mode else triton.cdiv(rows, block_m)
    grid = (n_mblocks, parts.n // HAD)
    dummy = parts.part_of_nb
    _exl3_gemm_kernel[grid](
        xh, xh.stride(0), xh.stride(1),
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
        BM=block_m, BK=32, BITS=parts.bits, CB=CODEBOOKS[parts.codebook], SORTED=sorted_mode,
        DECODED=decoded is not None,
        num_warps=4, num_stages=2,
    )
    return out


def exl3_gemv(
    xh: torch.Tensor,
    trellis: torch.Tensor,
    svh: torch.Tensor,
    parts: Exl3Parts,
    *,
    out: torch.Tensor,
    experts: torch.Tensor | None = None,
    tr_expert_stride: int = 0,
    svh_expert_stride: int = 0,
    split_k: int = 1,
) -> torch.Tensor:
    """Per-row GEMV (decode) into ``out`` ``[rows, N]`` (any float dtype). With ``split_k > 1``
    each K split writes a rotated fp32 partial to its own plane of a scratch buffer and the
    planes are summed in order (the output rotation is linear, so rotating partials is exact):
    deterministic, and fixed-shape for CUDA graphs."""
    rows = xh.shape[1]
    if rows == 0:
        return out
    k = parts.k
    if k % (split_k * 16):
        raise ValueError(f"split_k {split_k} does not divide K {k} into 16-row bands")
    dst = out if split_k == 1 else torch.empty((split_k, rows, parts.n), dtype=torch.float32, device=xh.device)
    words = _words(trellis)
    grid = (rows, parts.n // HAD, split_k)
    reduce = split_k > 1 and _inkernel_reduce()
    counts = _split_counters(rows * (parts.n // HAD), xh.device) if reduce else parts.part_of_nb
    _exl3_gemv_kernel[grid](
        xh, xh.stride(0), xh.stride(1),
        dst, dst.stride(-2), dst.stride(0) if split_k > 1 else 0,
        words, tr_expert_stride,
        svh, svh_expert_stride,
        hadamard_pm1(xh.device),
        parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart,
        experts if experts is not None else parts.part_of_nb, k // split_k,
        out, out.stride(-2), counts,
        BITS=parts.bits, CB=CODEBOOKS[parts.codebook],
        HAS_EXPERT=experts is not None, REDUCE=reduce,
        num_warps=1,
    )
    if split_k > 1 and not reduce:
        torch.sum(dst, dim=0, out=out) if out.dtype == torch.float32 else out.copy_(dst.sum(dim=0))
    return out


def _inkernel_reduce() -> bool:
    """``FREETOKEN_EXL3_SPLITK_INKERNEL=0`` restores the separate ``torch.sum`` reduction."""
    return os.getenv("FREETOKEN_EXL3_SPLITK_INKERNEL", "1").strip() != "0"


_COUNTERS: dict = {}


def _split_counters(n: int, device: torch.device) -> torch.Tensor:
    """Zeroed int32 arrival counters for the in-kernel split-K reduction, one per (row, 128
    output columns). Every launch leaves them zero again, so one buffer per device serves all
    GEMVs issued in stream order; it is allocated (or grown) outside graph capture, on the
    warm-up pass that precedes it."""
    buf = _COUNTERS.get(device)
    if buf is None or buf.numel() < n:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 split-K counters must be allocated before CUDA-graph capture")
        buf = torch.zeros(max(n, 1 << 14), dtype=torch.int32, device=device)
        _COUNTERS[device] = buf
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
    "exl3_gemm", "exl3_gemv", "had_cols", "had_rows", "reconstruct_experts", "hadamard_pm1", "hadamard_reference",
    "linear_reference", "reconstruct", "reconstruct_reference", "tile_stream_index",
]
