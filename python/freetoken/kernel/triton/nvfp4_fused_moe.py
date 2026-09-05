"""Inline-dequant NVFP4 fused-MoE Triton kernels.

These kernels read the NVFP4 expert cache (packed FP4 codes + fp8 block scales +
fp16 per-row global scale) *directly* and dequantize inside the GEMM K-loop, so the
grouped MoE GEMM never materializes a BF16 copy of the experts (no separate dequant
pass, no HBM round-trip of the 4x larger BF16 weights).

Layout, for an expert weight ``W[N, K]``:
  - ``packed[slot, n, k//2]`` uint8: two FP4 codes per byte, low nibble = even k.
  - ``scale[slot, n, k//16]`` fp8-e4m3: per 16-wide block scale.
  - ``global[slot, n]`` fp16: per-output-row scale (``weight_scale_2``).
  - ``W[n, k] = E2M1[code] * scale[n, k//16] * global[n]``.

The global scale is constant along K, so it is applied once after the K-loop. The
``E2M1[code]`` step is arithmetic (:func:`_e2m1_decode`), not a LUT gather -- see there.

Decode (M=1) is HBM-bandwidth bound. :func:`_decode_nvfp4_marlin_kernel` is the production
decode GEMV (int32 wide loads + deferred K reduction); :func:`_decode_nvfp4_moe_kernel` is
the original LUT-gather variant kept for A/B comparison. The marlin-style wide load lifts
MiniMax-M2 decode toward the RTX 5090 read-bandwidth ceiling (~87%); the residual gap is FP4
dequant ALU, which a swizzled-layout tensor-core path (marlin / flashinfer b12x) closes but
those need sm_80-99 / CUDA>=13 respectively.

All three kernels carry an ``ACT`` constexpr epilogue (0 = none, 1 = ReLU^2) so the
ungated ReLU^2 geometries (Nemotron 3.5 Lightning) skip the separate activation pass
and its bf16 store/load round trip of the whole ``[M*top_k, I]`` intermediate. ReLU^2
is applied *last*, after the global scale and the routed weight, which is exactly
where ``_run_act`` sat, so the fused form is the (slightly more accurate) same math.
Gated activations (silu/gelu/swigluoai) mix the two halves of ``gate_up`` and cannot
be fused into a per-tile epilogue; they keep ``ACT=0`` plus the separate op.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl
from triton.language import target_info
from triton.runtime.jit import constexpr_function

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32

_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


@functools.lru_cache(maxsize=None)
def _e2m1_lut(device_index: int) -> torch.Tensor:
    """The 16 E2M1 values as a device tensor, for the decode GEMVs (see
    :func:`_e2m1_decode` for why prefill does not use it)."""
    return torch.tensor(
        _E2M1_VALUES, dtype=torch.float32, device=torch.device("cuda", device_index)
    )


@triton.jit
def _e2m1_decode(code):
    """FP4 E2M1 code (0..15) -> its fp32 value, by bit construction.

    Used by the **prefill** GEMM in place of the 16-entry LUT gather
    (``tl.load(lut_ptr + code)``). However small and L1-resident the table is, that is one
    *indexed* load per element of the dequantized tile: at the prefill GEMM's
    [BLOCK_KB, BLOCK_N] operand it issues more LSU traffic than the ``tl.dot`` it feeds,
    and the grouped GEMM was bound by it -- 8192x6 routes went 48.4 -> 38.8 ms and M=256
    3.92 -> 2.10 ms just from this swap, at identical tiles.

    The **decode** GEMVs deliberately keep the gather. They are HBM-bandwidth bound with
    idle LSU capacity, so the L1 hits hide under the weight stream while these integer ops
    sit on the dependency chain instead: measured, the arithmetic form costs decode 8-10%
    at M=8/16 and nothing at M=1. Same values either way, so the choice is pure scheduling.

    The code is ``s|ee|m``, magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6. For magnitude >= 2 the
    value is ``2**(mag // 2 - 1) * (1 + .5 * (mag & 1))``, whose fp32 bits are exactly
    ``(126 + (mag >> 1)) << 23 | (mag & 1) << 22``. Magnitudes 0 and 1 (0.0 and 0.5) are
    the two exceptions and fold in with one select. Everything stays in the integer
    domain so the sign is an OR of the fp32 sign bit rather than a negation -- that is
    what keeps code 8 at -0.0 (as the table had it) instead of +0.0, and makes the result
    bit-identical to the LUT for all 16 codes.
    """
    mag = code & 7
    normal = ((126 + (mag >> 1)) << 23) | ((mag & 1) << 22)
    bits = tl.where(mag > 1, normal, tl.where(mag == 1, 0x3F000000, 0))  # 0x3F000000 == 0.5
    bits = tl.where(code > 7, bits | -2147483648, bits)  # -2147483648 == the sign bit
    return bits.to(tl.float32, bitcast=True)


def _force_no_native_e2m1() -> bool:
    return os.environ.get("FREETOKEN_NVFP4_NO_NATIVE_CVT", "").lower() in ("1", "true", "yes", "on")


_NO_NATIVE_E2M1 = _force_no_native_e2m1()


@constexpr_function
def e2m1_native_cvt_cx():
    """Compile-time: does the target have the hardware FP4 decode?

    ``cvt.rn.f16x2.e2m1x2`` (PTX ISA 8.6) exists from Blackwell on -- sm_100 datacenter
    and sm_120 consumer alike. It turns :func:`_e2m1_decode`'s ~14-op bit construction
    into ONE instruction per *pair* of codes, and it is bit-identical to it for all 256
    byte values (``tests/moe/test_nvfp4_backends.py``), so the branch is a pure
    scheduling choice. ``FREETOKEN_NVFP4_NO_NATIVE_CVT=1`` forces the arithmetic form
    (read once at import, for A/B)."""
    return not _NO_NATIVE_E2M1 and target_info.cuda_capability_geq(10, 0)


@triton.jit
def _e2m1_decode_pair_native(byte_):
    """One packed NVFP4 byte -> (low nibble value, high nibble value) as fp16.

    The low nibble -- the even-k code -- lands in the low half of the ``f16x2`` pair,
    which is the same order :func:`_e2m1_decode` produces from ``code & 0xF`` and
    ``(code >> 4) & 0xF``."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8  b0;
            .reg .b32 h2;
            cvt.u8.u32 b0, $2;
            cvt.rn.f16x2.e2m1x2 h2, b0;
            mov.b32 {$0, $1}, h2;
        }
        """,
        "=h,=h,r",
        [byte_],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _apply_act(acc, ACT: tl.constexpr):
    """gemm1 epilogue activation. ``ACT`` 0 = identity, 1 = ReLU^2 (``relu(x)**2``).

    Runs after the per-row global scale and the routed weight, i.e. on exactly the value
    :func:`freetoken.moe.fused_nvfp4._run_act` used to read back from HBM."""
    if ACT == 1:
        acc = tl.where(acc > 0, acc * acc, 0.0)
    return acc


@triton.jit
def _decode_nvfp4_moe_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    lut_ptr,           # [16] fp32
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,  # bytes processed per K-iter (covers 2*KB k-values)
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ACT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Original LUT-gather serial-K decode GEMV. Kept as the A/B baseline; the production
    decode path is :func:`_decode_nvfp4_marlin_kernel`."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    K_BYTES = K // 2
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kb_start in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
        byte_idx = kb_start * BLOCK_SIZE_KB + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_slot + offs_n[None, :] * stride_pn + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(
            p_ptrs, mask=byte_mask[:, None] & n_mask[None, :], other=0
        ).to(tl.int32)
        lo = bytes_ & 0xF
        hi = (bytes_ >> 4) & 0xF
        b_lo = tl.load(lut_ptr + lo)
        b_hi = tl.load(lut_ptr + hi)

        sblk = byte_idx // 8
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + sblk[:, None] * stride_sblk
        s_mask = byte_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))
        b_lo = b_lo * scale
        b_hi = b_hi * scale

        a_lo = tl.load(a_base + (2 * byte_idx) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        a_hi = tl.load(a_base + (2 * byte_idx + 1) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        accumulator += tl.sum(a_lo[:, None] * b_lo, axis=0)
        accumulator += tl.sum(a_hi[:, None] * b_hi, axis=0)

    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    accumulator = _apply_act(accumulator, ACT)

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _decode_nvfp4_marlin_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 8] int32 (8 fp4 codes per word, nibble j -> k=8*w+j)
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    lut_ptr,           # [16] fp32
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,  # int32 words per K-iter (covers 8*KW k-values)
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ACT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Marlin-style NVFP4 decode GEMV: wide int32 weight loads + deferred reduction.

    Versus :func:`_decode_nvfp4_moe_kernel` (which loads the packed codes one byte at a
    time and runs a cross-lane reduction *every* K-iter) this:

      * Loads the FP4 codes as **int32 words** (8 codes / 4 bytes per element) so the HBM
        read issues wide coalesced transactions -- the single biggest lever for the
        gate/up GEMM, whose mem-only ceiling jumps from ~65% to ~85% of peak read BW.
      * **Defers** the K reduction: the ``[BLOCK_KW, BLOCK_N]`` partial accumulates across
        *all* K-iters and is reduced once at the end, removing the per-iter ``tl.sum``
        barrier so the weight loads of successive iters pipeline.

    The e2m1 codes share one fp8 block scale per 16 k-values (== 2 words), applied per
    word. Mirrors the fast kernel's route/tile mapping and epilogue so it is a drop-in
    decode GEMV (CUDA-graph safe: fixed shapes, no host sync)."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    K_WORDS = K // 8
    partial = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW)):
        widx = kw_start * BLOCK_SIZE_KW + offs_kw
        w_mask = widx < K_WORDS

        word = tl.load(
            packed_slot + offs_n[None, :] * stride_pn + widx[:, None] * stride_pkw,
            mask=w_mask[:, None] & n_mask[None, :], other=0,
        )
        # 8 codes/word fall in the same or adjacent 16-wide block -> one scale per word.
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + (widx[:, None] // 2) * stride_sblk
        s_mask = w_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))

        kbase = 8 * widx
        acc_w = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        for j in tl.static_range(8):
            code = (word >> (4 * j)) & 0xF
            b = tl.load(lut_ptr + code)
            a_j = tl.load(a_base + (kbase + j) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
            acc_w += a_j[:, None] * b
        partial += acc_w * scale

    accumulator = tl.sum(partial, axis=0)
    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    accumulator = _apply_act(accumulator, ACT)

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _prefill_nvfp4_moe_kernel(
    a_ptr,             # [M, K] activations
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [num_valid_tokens, N] output (flat over M*top_k)
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,    # cache slot per M-block
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    ACT: tl.constexpr,
    compute_type: tl.constexpr,
    DEINTERLEAVED_A: tl.constexpr = False,
):
    """Grouped GEMM over the packed NVFP4 experts, two ``tl.dot``s per packed byte.

    ``DEINTERLEAVED_A`` selects A's k-layout (a compile-time branch; the False path is
    the historical one, byte-for-byte):

      * False -- A is the plain ``[M, K]`` activation, so the even-k and odd-k halves of
        a byte-block are two stride-2 gathers on the contiguous axis (neither vectorizes).
      * True  -- A is pre-deinterleaved on the host into two contiguous k-planes,
        logically ``[M, 2, K // 2]`` flattened to ``[M, K]``: even k at column ``kk`` and
        odd k at column ``K // 2 + kk``.  Both loads are then unit-stride.

    Only the *addressing* of A changes; masks, the ``// top_k`` row indexing, the
    per-``tl.dot`` reduction order, the accumulator and the epilogue are identical, so
    the two paths are bit-identical up to the loads themselves.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    a_row = offs_token[:, None] // top_k * stride_am
    if DEINTERLEAVED_A:
        # Even-k plane then odd-k plane: both gathers are unit-stride in k.
        a_ptrs_lo = a_ptr + (a_row + offs_kb[None, :] * stride_ak)
        a_ptrs_hi = a_ptrs_lo + (K // 2) * stride_ak
    else:
        a_ptrs_lo = a_ptr + (a_row + (2 * offs_kb)[None, :] * stride_ak)
        a_ptrs_hi = a_ptr + (a_row + (2 * offs_kb + 1)[None, :] * stride_ak)

    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    packed_base = packed_ptr + slot * stride_pe + offs_bn[None, :] * stride_pn
    scale_base = scale_ptr + slot * stride_se + offs_bn[None, :] * stride_sn

    # One e4m3 scale covers 16 k-values == 8 packed bytes, so a [BLOCK_KB, BLOCK_N]
    # scale tile holds each value 8 times. Load the [NGRP, BLOCK_SIZE_N] distinct rows and
    # broadcast: that is the single largest term in this kernel (measured 1.73x at the
    # Nemotron geometry -- and it also takes the tile OUT of shared memory, 28 KB -> 12 KB).
    tl.static_assert(BLOCK_SIZE_KB % 8 == 0, "BLOCK_SIZE_KB must be a multiple of 8 bytes")
    NGRP: tl.constexpr = BLOCK_SIZE_KB // 8
    offs_grp = tl.arange(0, NGRP)
    n_grp_total = K // 16

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    K_BYTES = K // 2
    for kb in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
        byte_idx = kb * BLOCK_SIZE_KB + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_base + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(p_ptrs, mask=byte_mask[:, None], other=0).to(tl.int32)

        grp_idx = kb * NGRP + offs_grp
        grp_mask = grp_idx[:, None] < n_grp_total
        s_ptrs = scale_base + grp_idx[:, None] * stride_sblk
        if e4m3_native_cx():
            s_grp = tl.load(s_ptrs, mask=grp_mask, other=0.0).to(tl.float32)
        else:
            s_grp = e4m3_u8_to_f32(tl.load(s_ptrs, mask=grp_mask, other=0))
        scale = tl.reshape(
            tl.broadcast_to(s_grp[:, None, :], (NGRP, 8, BLOCK_SIZE_N)),
            (BLOCK_SIZE_KB, BLOCK_SIZE_N),
        )

        if e2m1_native_cvt_cx():
            d_lo, d_hi = _e2m1_decode_pair_native(bytes_)
            b_lo = d_lo.to(tl.float32) * scale  # [BLOCK_KB, BLOCK_N]
            b_hi = d_hi.to(tl.float32) * scale
        else:
            b_lo = _e2m1_decode(bytes_ & 0xF) * scale
            b_hi = _e2m1_decode((bytes_ >> 4) & 0xF) * scale

        a_lo = tl.load(a_ptrs_lo, mask=token_mask[:, None] & byte_mask[None, :], other=0.0)
        a_hi = tl.load(a_ptrs_hi, mask=token_mask[:, None] & byte_mask[None, :], other=0.0)
        accumulator += tl.dot(a_lo, b_lo.to(a_lo.dtype))
        accumulator += tl.dot(a_hi, b_hi.to(a_hi.dtype))

        if DEINTERLEAVED_A:
            a_ptrs_lo += BLOCK_SIZE_KB * stride_ak
            a_ptrs_hi += BLOCK_SIZE_KB * stride_ak
        else:
            a_ptrs_lo += BLOCK_SIZE_KB * 2 * stride_ak
            a_ptrs_hi += BLOCK_SIZE_KB * 2 * stride_ak

    g = tl.load(global_ptr + slot * stride_ge + offs_bn * stride_gn).to(tl.float32)
    accumulator = accumulator * g[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = _apply_act(accumulator, ACT)
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


__all__ = [
    "_apply_act",
    "_e2m1_decode",
    "_e2m1_decode_pair_native",
    "_e2m1_lut",
    "e2m1_native_cvt_cx",
    "_decode_nvfp4_moe_kernel",
    "_decode_nvfp4_marlin_kernel",
    "_prefill_nvfp4_moe_kernel",
]
