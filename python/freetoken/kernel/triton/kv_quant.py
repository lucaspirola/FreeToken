"""Quantizing store into a compact KV pool.

The unquantized path stores K/V with ``kernel/store.py``'s CUDA kernel, which is a
pure byte copy parameterized by element size. Quantized storage has to compute a scale
per block of :data:`~freetoken.kvcache.quant.BLOCK` elements along ``head_dim`` on the
way in, so it gets its own kernel here. The int4 path packs two signed values per byte.

One program handles one ``(token, kv_head)`` pair: it loads that head's ``head_dim``
values as a ``[head_dim // BLOCK, BLOCK]`` tile, reduces max-abs along the block, and
writes the quantized values plus one scale per block. K and V are done in the same
program -- they share the token's slot index and the tile geometry, so doing both
halves the launch count and the index math.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import round_e4m3


@triton.jit
def _store_kv_quant_kernel(
    k_ptr,  # [tokens, heads, D] source, compute dtype
    v_ptr,
    kc_ptr,  # [slots, heads, D // EPB] destination, storage dtype
    vc_ptr,
    ks_ptr,  # [slots, heads, D // BLOCK] scales, fp16
    vs_ptr,
    indices_ptr,  # [tokens] destination slot per token
    stride_kt,
    stride_kh,
    stride_ct,
    stride_ch,
    stride_st,
    stride_sh,
    MAX_MAG: tl.constexpr,
    IS_INT: tl.constexpr,
    EPB: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(indices_ptr + tok).to(tl.int64)

    if EPB == 2:
        # Nibble-packed int4: tile the source as [NBLOCK, BLOCK // 2, 2], where the last
        # axis is a (even, odd) element pair that becomes one byte. The scale is still
        # per BLOCK (per row), computed over all BLOCK elements of the row.
        pair_offs = (
            tl.arange(0, NBLOCK)[:, None, None] * BLOCK
            + tl.arange(0, BLOCK // 2)[None, :, None] * 2
            + tl.arange(0, 2)[None, None, :]
        )
        byte_offs = tl.arange(0, NBLOCK)[:, None] * (BLOCK // 2) + tl.arange(0, BLOCK // 2)[None, :]
        scale_offs = tl.arange(0, NBLOCK)

        for is_v in tl.static_range(2):
            src_ptr = v_ptr if is_v else k_ptr
            dst_ptr = vc_ptr if is_v else kc_ptr
            sc_ptr = vs_ptr if is_v else ks_ptr

            x = tl.load(src_ptr + tok * stride_kt + head * stride_kh + pair_offs).to(tl.float32)
            flat_x = x.reshape(NBLOCK, BLOCK)
            abs_x = tl.abs(flat_x)
            amax = tl.max(abs_x, axis=1)  # [NBLOCK]
            # Match GGML Q4_0 exactly, including its first-element tie break for
            # equally large positive and negative extrema.
            extreme_idx = tl.argmax(abs_x, axis=1, tie_break_left=True)
            block_idx = tl.arange(0, BLOCK)[None, :]
            extreme = tl.sum(
                tl.where(block_idx == extreme_idx[:, None], flat_x, 0.0), axis=1
            )
            scale = tl.where(amax > 0, extreme / -8.0, 1.0)
            scale = scale.to(sc_ptr.dtype.element_ty).to(tl.float32)
            # GGML reference truncates x / d + 8.5 into the unsigned nibble range.
            # The value is nonnegative here, so floor is equivalent to truncation.
            q = tl.math.div_rn(x, scale[:, None, None])
            q = tl.minimum(tl.maximum(tl.floor(q + 8.5), 0.0), 15.0).to(tl.int8)
            # Low nibble = even (index 0), high nibble = odd (index 1). Triton has no
            # integer indexing, so split the last axis and pack arithmetically.
            even, odd = tl.split(q)
            packed = even + odd * 16
            tl.store(
                dst_ptr + slot * stride_ct + head * stride_ch + byte_offs,
                packed.to(dst_ptr.dtype.element_ty),
            )
            tl.store(
                sc_ptr + slot * stride_st + head * stride_sh + scale_offs,
                scale.to(sc_ptr.dtype.element_ty),
            )
        return

    # EPB == 1: 8-bit element-shaped storage, one byte per element.
    # [NBLOCK, BLOCK] tile over head_dim: rows are quant blocks, columns the elements
    # sharing one scale.
    offs = tl.arange(0, NBLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    scale_offs = tl.arange(0, NBLOCK)

    for is_v in tl.static_range(2):
        src_ptr = v_ptr if is_v else k_ptr
        dst_ptr = vc_ptr if is_v else kc_ptr
        sc_ptr = vs_ptr if is_v else ks_ptr

        x = tl.load(src_ptr + tok * stride_kt + head * stride_kh + offs).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        # An all-zero block quantizes to zeros under any positive scale; 1.0 keeps the
        # division finite.
        scale = tl.where(amax > 0, amax / MAX_MAG, 1.0)
        # Round to the stored precision before dividing, so the value written here and
        # the value the attention kernels read back are scaled by the identical number.
        scale = scale.to(sc_ptr.dtype.element_ty).to(tl.float32)
        # div_rn, not `/`: the plain operator is free to lower to a reciprocal multiply,
        # which disagrees with the torch reference on values sitting exactly between two
        # quantization steps. IEEE round-to-nearest divide makes the two bit-identical.
        q = tl.math.div_rn(x, scale[:, None])
        if IS_INT:
            # Round half away from zero (what GGUF's Q8_0 does), then clamp -- the
            # float->int cast truncates.
            q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
            q = tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG)
        else:
            # The native fp32 -> float8e4nv downcast does not round to nearest on
            # every arch (it lowers as a truncating fp32 -> fp16 -> e4m3 double-round
            # on sm_89), so values just above a grid midpoint collapse downward and
            # disagree with the RNE torch reference. Round explicitly first.
            q = round_e4m3(tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG))

        tl.store(
            dst_ptr + slot * stride_ct + head * stride_ch + offs,
            q.to(dst_ptr.dtype.element_ty),
        )
        tl.store(
            sc_ptr + slot * stride_st + head * stride_sh + scale_offs,
            scale.to(sc_ptr.dtype.element_ty),
        )


@triton.jit
def _store_one_quant_kernel(
    src_ptr,  # [tokens, heads, D]
    dst_ptr,  # [slots, heads, storage_D]
    scale_ptr,  # [slots, heads, D // BLOCK]
    indices_ptr,
    stride_xt,
    stride_xh,
    stride_ct,
    stride_ch,
    stride_st,
    stride_sh,
    BITS: tl.constexpr,
    MAX_MAG: tl.constexpr,
    IS_INT: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
):
    """Quantize one side of an asymmetric K/V pair.

    The existing fused kernel remains the fast path whenever K and V share a format.
    Different formats necessarily have different pointer dtypes and physical strides,
    so compiling one small launch per side keeps Triton's type system honest.
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(indices_ptr + tok).to(tl.int64)

    offs = tl.arange(0, NBLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    scale_offs = tl.arange(0, NBLOCK)
    x = tl.load(src_ptr + tok * stride_xt + head * stride_xh + offs).to(tl.float32)

    abs_x = tl.abs(x)
    amax = tl.max(abs_x, axis=1)
    extreme_idx = tl.argmax(abs_x, axis=1, tie_break_left=True)
    block_idx = tl.arange(0, BLOCK)[None, :]
    extreme = tl.sum(tl.where(block_idx == extreme_idx[:, None], x, 0.0), axis=1)

    if BITS == 6:
        initial = tl.where(amax > 0, extreme / -32.0, 1.0)
        q = tl.minimum(
            tl.maximum(tl.floor(tl.math.div_rn(x, initial[:, None]) + 32.5), 0.0),
            63.0,
        )
        signed = q - 32.0
        weights = x * x
        sumqx = tl.sum(weights * signed * x, axis=1)
        sumq2 = tl.sum(weights * signed * signed, axis=1)
        scale = tl.where(sumq2 > 0, sumqx / sumq2, initial)
        scale = scale.to(scale_ptr.dtype.element_ty).to(tl.float32)
        # Cache-native contiguous planes: D/2 adjacent low-nibble pairs followed by
        # D/4 adjacent upper-two-bit quads. This avoids divides in attention loads.
        q = q.to(tl.uint8).reshape(NBLOCK * BLOCK)
        even, odd = tl.split(q.reshape(NBLOCK * BLOCK // 2, 2))
        low = (even & 15) | ((odd & 15) << 4)
        left, right = tl.split(q.reshape(NBLOCK * BLOCK // 4, 2, 2))
        q0, q2 = tl.split(left)
        q1, q3 = tl.split(right)
        high = (
            ((q0 >> 4) & 3)
            | (((q1 >> 4) & 3) << 2)
            | (((q2 >> 4) & 3) << 4)
            | (((q3 >> 4) & 3) << 6)
        )
        low_offs = tl.arange(0, NBLOCK * BLOCK // 2)
        high_offs = tl.arange(0, NBLOCK * BLOCK // 4)
        tl.store(dst_ptr + slot * stride_ct + head * stride_ch + low_offs, low)
        tl.store(
            dst_ptr + slot * stride_ct + head * stride_ch
            + NBLOCK * BLOCK // 2 + high_offs,
            high,
        )
        tl.store(scale_ptr + slot * stride_st + head * stride_sh + scale_offs, scale)
        return

    if BITS == 4:
        scale = tl.where(amax > 0, extreme / -8.0, 1.0)
        scale = scale.to(scale_ptr.dtype.element_ty).to(tl.float32)
        q = tl.minimum(
            tl.maximum(tl.floor(tl.math.div_rn(x, scale[:, None]) + 8.5), 0.0), 15.0
        ).to(tl.uint8)
        even, odd = tl.split(q.reshape(NBLOCK, BLOCK // 2, 2))
        packed = even | (odd << 4)
        byte_offs = scale_offs[:, None] * (BLOCK // 2) + tl.arange(0, BLOCK // 2)[None, :]
        tl.store(dst_ptr + slot * stride_ct + head * stride_ch + byte_offs, packed)
    else:
        scale = tl.where(amax > 0, amax / MAX_MAG, 1.0)
        scale = scale.to(scale_ptr.dtype.element_ty).to(tl.float32)
        q = tl.math.div_rn(x, scale[:, None])
        if IS_INT:
            q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
            q = tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG)
        else:
            q = round_e4m3(tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG))
        tl.store(dst_ptr + slot * stride_ct + head * stride_ch + offs, q.to(dst_ptr.dtype.element_ty))
    tl.store(scale_ptr + slot * stride_st + head * stride_sh + scale_offs, scale)


def _store_one_quant(cache, scale, indices, source, spec) -> None:
    from freetoken.kvcache.quant import BLOCK

    tokens, heads, head_dim = source.shape
    assert cache.shape[1:] == (heads, spec.storage_dim(head_dim))
    _store_one_quant_kernel[(tokens, heads)](
        source,
        cache,
        scale,
        indices,
        source.stride(0),
        source.stride(1),
        cache.stride(0),
        cache.stride(1),
        scale.stride(0),
        scale.stride(1),
        BITS=spec.bits,
        MAX_MAG=spec.max_magnitude,
        IS_INT=spec.is_integer,
        BLOCK=BLOCK,
        NBLOCK=head_dim // BLOCK,
        num_warps=4,
    )


def store_kv_quant(
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_cache: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec,
    v_spec=None,
) -> None:
    """Quantize ``k``/``v`` ``[tokens, heads, D]`` into the pool slots ``indices``.

    ``k_cache``/``v_cache`` are ``[slots, heads, D // EPB]`` in the spec's storage dtype
    (``D // EPB == D`` for 8-bit, ``D // 2`` for packed int4) and ``k_scale``/``v_scale``
    ``[slots, heads, D // BLOCK]`` in fp16.
    """
    from freetoken.kvcache.quant import BLOCK

    v_spec = spec if v_spec is None else v_spec
    num_tokens, num_heads, head_dim = k.shape
    if num_tokens == 0:
        return
    assert head_dim % BLOCK == 0, f"head_dim {head_dim} not a multiple of {BLOCK}"
    if v_spec != spec or spec.bits == 6:
        _store_one_quant(k_cache, k_scale, indices, k, spec)
        _store_one_quant(v_cache, v_scale, indices, v, v_spec)
        return

    assert spec.bits in (4, 8), "symmetric q6 uses the asymmetric-capable store path"
    assert k_cache.shape[1:] == (num_heads, spec.storage_dim(head_dim)), (
        f"packed cache geometry {tuple(k_cache.shape[1:])} != "
        f"{(num_heads, spec.storage_dim(head_dim))}"
    )
    _store_kv_quant_kernel[(num_tokens, num_heads)](
        k,
        v,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        indices,
        k.stride(0),
        k.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        MAX_MAG=spec.max_magnitude,
        IS_INT=spec.is_integer,
        EPB=spec.elements_per_byte,
        BLOCK=BLOCK,
        NBLOCK=head_dim // BLOCK,
        num_warps=4,
    )


__all__ = ["store_kv_quant"]
