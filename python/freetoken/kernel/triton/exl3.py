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
  row may belong to a different expert slot; ``tl.dot`` would waste 15/16 of each MMA there.

A weight is addressed as *parts*: a fused projection (q|k|v, gate|up) keeps each part's tiles
contiguous ("part-major"), with its own suh. Every part boundary is a multiple of 128, so each
128-wide output block belongs to one part. ``Exl3Parts`` holds the device tables.
"""

from __future__ import annotations

import functools
import math
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
def _exl3_gemv_kernel(
    xh_ptr, stride_xpart, stride_xm,
    out_ptr, stride_om, stride_osplit,
    tr_ptr, tr_expert_stride,
    svh_ptr, svh_expert_stride,
    had_ptr,
    part_of_nb_ptr, word_off_ptr, ntiles_ptr, nstart_ptr,
    expert_ptr, K_SPLIT,
    BK: tl.constexpr, BITS: tl.constexpr, CB: tl.constexpr,
    HAS_EXPERT: tl.constexpr,
):
    p = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    if HAS_EXPERT:
        expert = tl.load(expert_ptr + p).to(tl.int64)
    else:
        expert = tl.zeros([], dtype=tl.int64)
    part = tl.load(part_of_nb_ptr + pid_n)
    n_tiles = tl.load(ntiles_ptr + part)
    n_local = pid_n * 128 - tl.load(nstart_ptr + part) + tl.arange(0, 128)
    tr = tr_ptr + expert * tr_expert_stride + tl.load(word_off_ptr + part)
    a_ptr = xh_ptr + part * stride_xpart + p.to(tl.int64) * stride_xm
    acc = tl.zeros([128], dtype=tl.float32)
    k_begin = pid_k * K_SPLIT
    for k0 in range(k_begin, k_begin + K_SPLIT, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + kk).to(tl.float32)
        w = _exl3_decode(tr, kk[:, None], n_local[None, :], n_tiles, BITS, CB).to(tl.float32)
        acc += tl.sum(a[:, None] * w, axis=0)
    idx = tl.arange(0, 128)
    h = tl.load(had_ptr + idx[:, None] * 128 + idx[None, :]).to(tl.float32)
    y = tl.sum(acc[:, None] * h, axis=0)
    cols = pid_n * 128 + idx
    sv = tl.load(svh_ptr + expert * svh_expert_stride + cols).to(tl.float32)
    y = y * sv * 0.08838834764831843
    # split K: each split writes its own plane; the launcher sums the planes in a fixed order
    # (atomics would make decode run-to-run nondeterministic)
    tl.store(out_ptr + pid_k * stride_osplit + p.to(tl.int64) * stride_om + cols, y.to(out_ptr.dtype.element_ty))


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
    bk = 32
    k = parts.k
    if k % (split_k * bk):
        raise ValueError(f"split_k {split_k} does not divide K {k} into {bk}-row steps")
    dst = out if split_k == 1 else torch.empty((split_k, rows, parts.n), dtype=torch.float32, device=xh.device)
    words = _words(trellis)
    grid = (rows, parts.n // HAD, split_k)
    _exl3_gemv_kernel[grid](
        xh, xh.stride(0), xh.stride(1),
        dst, dst.stride(-2), dst.stride(0) if split_k > 1 else 0,
        words, tr_expert_stride,
        svh, svh_expert_stride,
        hadamard_pm1(xh.device),
        parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart,
        experts if experts is not None else parts.part_of_nb, k // split_k,
        BK=bk, BITS=parts.bits, CB=CODEBOOKS[parts.codebook],
        HAS_EXPERT=experts is not None,
        num_warps=4,
    )
    if split_k > 1:
        torch.sum(dst, dim=0, out=out) if out.dtype == torch.float32 else out.copy_(dst.sum(dim=0))
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
