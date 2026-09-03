# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Vendored from vLLM `vllm/model_executor/layers/mamba/ops/ssd_bmm.py`
# (sequence-aligned varlen chunk variant), which itself adapts
# https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_bmm.py
#
# FreeToken changes: `vllm.triton_utils` -> plain triton; autotune configs pruned
# to 8 that fit ~100 KB smem on sm_120 (GB203) and match the chunk 128 x chunk 128
# x dstate 128 shape; `autotune_cache_kwargs`.

# ruff: noqa: E501

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.autotune_cache import autotune_cache_kwargs

# `tl.dot` defaults to TF32 (10 mantissa bits) for fp32 inputs, which leaves the
# fp32 path at ~4e-4 RMS-relative against an fp64 gold -- fp16 grade, not fp32.
# Triton ignores `input_precision` for fp16/bf16 operands, so asking for "ieee"
# costs the bf16 hot path nothing and makes the fp32 path fp32-exact (~1e-6).
DOT_PRECISION = tl.constexpr("ieee")

# smem ~= (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * 2 B * num_stages. The output tile
# is (chunk_size x chunk_size) = 128x128, so BLOCK_N 256 buys nothing and the
# vLLM (128,256,64,s3) config would need 144 KB. Retained set, all <= 64 KB:
#   (128,128,32,s4)=64K  (128,64,32,s4)=48K  (64,128,32,s4)=48K  (64,64,64,s3)=48K
#   (64,64,32,s4)=32K    (128,32,32,s4)=40K  (64,32,32,s5)=30K   (32,64,32,s5)=30K
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["chunk_size", "K", "IS_CAUSAL"],
    **autotune_cache_kwargs,
)
@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    out_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    chunk_size: tl.constexpr,
    K: tl.constexpr,
    ngroups: tl.constexpr,
    stride_a_seqlen: tl.int64,
    stride_a_head: tl.int64,
    stride_ak: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_bk: tl.constexpr,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_outm: tl.int64,
    stride_outn: tl.constexpr,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_ch = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    a_ptr += chunk_seqlen_start * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += chunk_seqlen_start * stride_b_seqlen + pid_h * stride_b_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # compute a * b.T
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ).to(dot_dtype)
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
            & (offs_n[None, :] < chunk_size_limit),
            other=0.0,
        ).to(dot_dtype)
        acc += tl.dot(a, b, input_precision=DOT_PRECISION)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out = acc.to(out_ptr.dtype.element_ty)
    out_ptr += pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size),
    )


def _bmm_chunk_fwd(a, b, chunk_size, cu_chunk_seqlens, causal=False, output_dtype=None):
    """
    Argument:
        a: (seqlen, ngroups, k)
        b: (seqlen, ngroups, k)
        chunk_size: int
        cu_chunk_seqlens: (nchunks+1,)
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (nchunks, ngroups, chunk_size, chunk_size)
    """
    seqlen, ngroups, k = a.shape
    assert b.shape == a.shape
    if a.stride(-1) != 1 and a.stride(0) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(0) != 1:
        b = b.contiguous()

    nchunks = len(cu_chunk_seqlens) - 1
    # Allocates output.
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.empty(
        (nchunks, ngroups, chunk_size, chunk_size), device=a.device, dtype=out_dtype
    )
    dot_dtype = (
        tl.bfloat16
        if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16
        else (
            tl.float16
            if a.dtype == torch.float16 or b.dtype == torch.float16
            else tl.float32
        )
    )

    def grid(META):
        return (
            triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
            * triton.cdiv(chunk_size, META["BLOCK_SIZE_N"]),
            nchunks * ngroups,
        )

    with torch.accelerator.device_index(a.device.index):
        _bmm_chunk_fwd_kernel[grid](
            a_ptr=a,
            b_ptr=b,
            out_ptr=out,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            chunk_size=chunk_size,
            K=k,
            ngroups=ngroups,
            stride_a_seqlen=a.stride(0),
            stride_a_head=a.stride(1),
            stride_ak=a.stride(2),
            stride_b_seqlen=b.stride(0),
            stride_b_head=b.stride(1),
            stride_bk=b.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_outm=out.stride(-2),
            stride_outn=out.stride(-1),
            IS_CAUSAL=causal,
            dot_dtype=dot_dtype,
        )
    return out
