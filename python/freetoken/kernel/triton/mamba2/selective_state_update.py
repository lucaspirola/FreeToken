# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2024, Tri Dao, Albert Gu.
# The Triton kernel below is vendored from vLLM
# `vllm/model_executor/layers/mamba/ops/mamba_ssm.py::_selective_scan_update_kernel`,
# which itself adapts
# https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py
#
# FreeToken changes:
#   * no `vllm.*` imports; `fast_exp` / `softplus` inlined (vLLM keeps them in
#     `ops/triton_helpers.py` and `ops/mamba_ssm.py`).
#   * only the paths this engine needs: `state_batch_indices` with a pad id,
#     `dt_softplus`, `HAS_DT_BIAS`, `HAS_D`; dropped `z` (the gate is applied by
#     the gated RMSNorm, see `gated_norm.py`), speculative decoding
#     (`num_accepted_tokens` / `dst_state_batch_indices` / `cu_seqlens`) and
#     stochastic fp16 state rounding.
#   * the tied-head-dim path only: `dt`/`A`/`dt_bias`/`D` are one value per head
#     (`[H]`, `[B, H]`), which is the Mamba-2 (as opposed to Mamba-1) layout, so
#     `dA` is a scalar per program instead of a `[BLOCK_M, DSTATE]` tile.
#   * one fixed launch config (`BLOCK_SIZE_M = 64`, `num_warps = 8`) instead of
#     vLLM's JSON config table: the kernel updates the recurrent pool in place,
#     so it must never be autotuned (an autotuner replays a kernel many times).
#   * `do_not_specialize` on every runtime integer so a different batch size or
#     group count never triggers a recompile -- `warm_mamba2_decode` must be
#     able to build the module before CUDA-graph capture.

"""Mamba-2 decode: one-token selective state update (+ flashinfer front end).

Task 2A3 of `tasks/nemotron35-plan.md`. `mamba2_decode` is the decode-time twin
of `mamba2_prefill`: one token per sequence, the recurrent pool read and written
in place at `indices`, no host sync and no shape-dependent control flow, so a
CUDA graph can capture it.

Two backends, selected by `FREETOKEN_MAMBA2_DECODE`:

``flashinfer`` (default when importable)
    `flashinfer.mamba.selective_state_update`, a hand-written CUDA kernel.
    It wants `A`/`D`/`dt_bias`/`dt` expanded over the head dim; those views are
    stride-0 and cached per parameter tensor, so a steady-state call allocates
    nothing but `out` (and nothing at all when `out` is given).
``triton``
    The kernel in this file. Same numerics to fp32 roundoff, no JIT-compiled
    C++ dependency.

``auto`` (the default) picks flashinfer when it imports *and* its JIT module
builds; the first build failure demotes the process to Triton for good.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

__all__ = [
    "mamba2_decode",
    "resolve_decode_backend",
    "warm_mamba2_decode",
]

PAD_SLOT_ID = -1
_BLOCK_SIZE_M = 64
_NUM_WARPS = 8
_ENV = "FREETOKEN_MAMBA2_DECODE"


# --------------------------------------------------------------------------
# Triton kernel
# --------------------------------------------------------------------------
@triton.jit
def fast_exp(x):
    """exp(x) via the hardware ex2.approx.f32 instruction."""
    return tl.math.exp2(1.4426950408889634 * x)


@triton.jit
def softplus(dt):
    return tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.jit(
    do_not_specialize=[
        "dim",
        "dstate",
        "nheads_ngroups_ratio",
        "pad_slot_id",
        "stride_state_slot",
        "stride_state_head",
        "stride_x_batch",
        "stride_x_head",
        "stride_dt_batch",
        "stride_dt_head",
        "stride_B_batch",
        "stride_B_group",
        "stride_C_batch",
        "stride_C_group",
        "stride_out_batch",
        "stride_out_head",
        "stride_idx_batch",
    ]
)
def _selective_state_update_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    # dimensions
    dim,
    dstate,
    nheads_ngroups_ratio,
    # strides
    stride_state_slot,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_idx_batch,
    # meta
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    slot = tl.load(state_batch_indices_ptr + pid_b * stride_idx_batch).to(tl.int64)
    state_ptr += slot * stride_state_slot + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    # A padded row (`slot == pad_slot_id`) masks every state load and store, so
    # the slot it names is never read and never written. The pointer arithmetic
    # above still runs; Triton never dereferences a fully masked pointer.
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate) & (slot != pad_slot_id)
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    x = tl.load(x_ptr + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0).to(
        tl.float32
    )
    dt = tl.load(dt_ptr).to(tl.float32)
    if HAS_DT_BIAS:
        dt += tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
    if DT_SOFTPLUS:
        dt = softplus(dt)
    a = tl.load(A_ptr).to(tl.float32)
    dA = fast_exp(a * dt)

    b = tl.load(B_ptr + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0).to(
        tl.float32
    )
    c = tl.load(C_ptr + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(
        tl.float32
    )

    state = state * dA + (b * dt)[None, :] * x[:, None]
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)

    out = tl.sum(state * c[None, :], axis=1)
    if HAS_D:
        out += x * tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
    tl.store(
        out_ptr + offs_m * stride_out_dim,
        out.to(out_ptr.dtype.element_ty),
        mask=offs_m < dim,
    )


def _decode_triton(
    x, dt, B, C, *, A, D, dt_bias, state_source, indices, out, dt_softplus
):
    bs, nheads, dim = x.shape
    dstate = B.shape[-1]
    ngroups = B.shape[1]
    grid = (triton.cdiv(dim, _BLOCK_SIZE_M), bs, nheads)
    _selective_state_update_kernel[grid](
        state_source,
        x,
        dt,
        dt_bias,
        A,
        B,
        C,
        D,
        out,
        indices,
        PAD_SLOT_ID,
        dim,
        dstate,
        nheads // ngroups,
        state_source.stride(0),
        state_source.stride(1),
        state_source.stride(2),
        state_source.stride(3),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dt.stride(0),
        dt.stride(1),
        dt_bias.stride(0) if dt_bias is not None else 0,
        A.stride(0),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        D.stride(0) if D is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        DT_SOFTPLUS=dt_softplus,
        BLOCK_SIZE_M=_BLOCK_SIZE_M,
        BLOCK_SIZE_DSTATE=triton.next_power_of_2(dstate),
        num_warps=_NUM_WARPS,
    )
    return out


# --------------------------------------------------------------------------
# flashinfer front end
# --------------------------------------------------------------------------
@functools.cache
def _flashinfer_ssu():
    """The flashinfer entry point, or None when the package is not importable."""
    try:
        from flashinfer.mamba import selective_state_update
    except Exception:  # pragma: no cover - depends on the install
        return None
    return selective_state_update


# (id(A), id(D), id(dt_bias), dt dtype) -> (A_exp, D_exp, dtb_exp, keepalive)
#
# The expansions are stride-0 views (and, for D / dt_bias, one dtype cast that
# flashinfer requires to match `dt`), so caching them is what keeps a steady
# state decode call allocation-free. The source tensors are kept alive in the
# value: without that, a freed parameter could have its `id()` recycled by a
# different tensor and silently hit this cache.
_PARAM_CACHE: dict[tuple, tuple] = {}


def _expand_params(A, D, dt_bias, nheads, headdim, dstate, dt_dtype):
    key = (
        id(A),
        id(D),
        id(dt_bias),
        dt_dtype,
        nheads,
        headdim,
        dstate,
    )
    hit = _PARAM_CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1], hit[2]
    # A caller that rebuilds `A` every step (e.g. `-torch.exp(A_log)` inline)
    # would both allocate per call and grow this dict without bound. Hold the
    # parameters on the module instead. The cap is a leak guard, not a policy:
    # 4 tensors per layer x 23 Mamba layers is 92 live entries.
    if len(_PARAM_CACHE) >= 512:
        _PARAM_CACHE.clear()
    a_exp = A[:, None, None].expand(nheads, headdim, dstate)
    d_exp = None if D is None else D.to(dt_dtype)[:, None].expand(nheads, headdim)
    b_exp = (
        None
        if dt_bias is None
        else dt_bias.to(dt_dtype)[:, None].expand(nheads, headdim)
    )
    _PARAM_CACHE[key] = (a_exp, d_exp, b_exp, (A, D, dt_bias))
    return a_exp, d_exp, b_exp


def _decode_flashinfer(
    x, dt, B, C, *, A, D, dt_bias, state_source, indices, out, dt_softplus
):
    ssu = _flashinfer_ssu()
    assert ssu is not None
    _, nheads, headdim = x.shape
    dstate = B.shape[-1]
    a_exp, d_exp, b_exp = _expand_params(
        A, D, dt_bias, nheads, headdim, dstate, dt.dtype
    )
    ssu(
        state_source,
        x,
        dt.unsqueeze(-1).expand(-1, -1, headdim),
        a_exp,
        B,
        C,
        d_exp,
        z=None,
        dt_bias=b_exp,
        dt_softplus=dt_softplus,
        state_batch_indices=indices,
        pad_slot_id=PAD_SLOT_ID,
        out=out,
    )
    return out


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
_flashinfer_broken: str | None = None


def _env_mode() -> str:
    mode = (os.environ.get(_ENV) or "auto").strip().lower()
    if mode not in ("auto", "flashinfer", "triton"):
        raise ValueError(
            f"{_ENV}={mode!r}; expected one of 'auto', 'flashinfer', 'triton'"
        )
    return mode


def resolve_decode_backend() -> str:
    """The backend the next :func:`mamba2_decode` call will use ('flashinfer' or
    'triton'). In ``auto`` mode a flashinfer JIT build failure has to be observed
    once (by a real call, e.g. :func:`warm_mamba2_decode`) before this reports
    ``triton``."""
    mode = _env_mode()
    if mode == "triton":
        return "triton"
    if mode == "flashinfer":
        if _flashinfer_ssu() is None:
            raise RuntimeError(f"{_ENV}=flashinfer but flashinfer is not importable")
        return "flashinfer"
    if _flashinfer_broken is not None or _flashinfer_ssu() is None:
        return "triton"
    return "flashinfer"


def _demote_flashinfer(exc: BaseException) -> None:
    global _flashinfer_broken
    if _flashinfer_broken is None:
        from freetoken.utils import init_logger

        _flashinfer_broken = f"{type(exc).__name__}: {exc}"
        init_logger(__name__).warning(
            "flashinfer selective_state_update unusable (%s); "
            "falling back to the Triton Mamba-2 decode kernel",
            _flashinfer_broken,
        )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def mamba2_decode(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    A: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    state_source: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor | None = None,
    dt_softplus: bool = True,
) -> torch.Tensor:
    """One decode step of the Mamba-2 recurrence, in place on the pool.

    ``h <- exp(dt * A) * h + dt * x (x) B`` then ``y <- C . h + D * x``, with
    ``dt = softplus(dt_raw + dt_bias)`` and head ``h`` reading group
    ``h // (H // G)``.

    Args:
        x: ``[B, H, P]`` post-conv SSM input (bf16).
        dt: ``[B, H]`` raw timestep, pre-bias and pre-softplus (bf16).
        B: ``[B, G, N]``.
        C: ``[B, G, N]``.
        A: ``[H]`` fp32, already ``-exp(A_log)``. Hold this (and ``D`` /
            ``dt_bias``) on the module: the flashinfer backend caches their
            expanded views keyed on ``id()``, so rebuilding ``A`` every step
            allocates on every call and defeats the cache.
        D: ``[H]`` skip connection, or None.
        dt_bias: ``[H]``, or None.
        state_source: ``[slots, H, P, N]`` fp32 contiguous recurrent pool, read
            and written in place.
        indices: ``[B]`` int32 pool slot per row. ``-1`` marks a padded row: its
            state is neither read nor written, and its ``out`` row is
            unspecified (both backends leave it holding ``D * x``).
        out: optional preallocated ``[B, H, P]`` output in ``x``'s dtype.
        dt_softplus: apply ``softplus`` to ``dt + dt_bias``.

    Returns:
        ``out`` ``[B, H, P]``.

    No host sync, no data-dependent shapes: CUDA-graph capturable once
    :func:`warm_mamba2_decode` has built the kernel.
    """
    bs, nheads, headdim = x.shape
    assert dt.shape == (bs, nheads), f"dt is {tuple(dt.shape)}, expected {(bs, nheads)}"
    assert B.dim() == 3 and C.shape == B.shape, "B and C must both be [B, G, N]"
    ngroups, dstate = B.shape[1], B.shape[2]
    assert B.shape[0] == bs
    assert nheads % ngroups == 0, (
        f"{nheads} heads is not a multiple of {ngroups} groups"
    )
    assert state_source.dim() == 4 and state_source.shape[1:] == (
        nheads,
        headdim,
        dstate,
    ), (
        f"state pool is {tuple(state_source.shape)}, expected "
        f"[slots, {nheads}, {headdim}, {dstate}]"
    )
    assert state_source.dtype == torch.float32, "the recurrent pool must be fp32"
    assert state_source.is_contiguous(), "the recurrent pool must be contiguous"
    assert indices.shape == (bs,) and indices.dtype == torch.int32, (
        f"indices must be int32 [{bs}], got {indices.dtype} {tuple(indices.shape)}"
    )
    assert A.shape == (nheads,) and A.dtype == torch.float32
    assert D is None or D.shape == (nheads,)
    assert dt_bias is None or dt_bias.shape == (nheads,)

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape and out.dtype == x.dtype

    kwargs = dict(
        A=A,
        D=D,
        dt_bias=dt_bias,
        state_source=state_source,
        indices=indices,
        out=out,
        dt_softplus=dt_softplus,
    )
    mode = _env_mode()
    if resolve_decode_backend() == "flashinfer":
        try:
            return _decode_flashinfer(x, dt, B, C, **kwargs)
        except Exception as exc:
            if mode == "flashinfer":
                raise
            # `auto`: a JIT build / launch failure demotes the process. The
            # build happens before any launch, so the pool is still untouched.
            _demote_flashinfer(exc)
    return _decode_triton(x, dt, B, C, **kwargs)


def warm_mamba2_decode(
    state_source: torch.Tensor,
    bs: int,
    *,
    ngroups: int = 1,
    dt_softplus: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Build the decode kernel (flashinfer JIT or Triton) before graph capture.

    Runs one throw-away step on a scratch pool with the same per-slot geometry
    and strides as ``state_source``, so nothing in the real pool is touched.
    ``ngroups`` only has to match reality for flashinfer's module cache; the
    Triton kernel takes the group ratio as a non-specialised runtime argument.

    Returns the backend name that :func:`mamba2_decode` resolved to.
    """
    assert state_source.dim() == 4 and state_source.dtype == torch.float32
    _, nheads, headdim, dstate = state_source.shape
    assert nheads % ngroups == 0
    dev = state_source.device
    # flashinfer requires `state.size(0) >= x.size(0)`, so the scratch pool has
    # one slot per row; the per-slot strides (all that a compiled module keys
    # on) are identical to `state_source`'s.
    scratch = torch.zeros(bs, nheads, headdim, dstate, dtype=torch.float32, device=dev)
    x = torch.zeros(bs, nheads, headdim, dtype=dtype, device=dev)
    dt = torch.zeros(bs, nheads, dtype=dtype, device=dev)
    b = torch.zeros(bs, ngroups, dstate, dtype=dtype, device=dev)
    a = torch.full((nheads,), -1.0, dtype=torch.float32, device=dev)
    d = torch.zeros(nheads, dtype=torch.float32, device=dev)
    idx = torch.zeros(bs, dtype=torch.int32, device=dev)
    mamba2_decode(
        x,
        dt,
        b,
        b.clone(),
        A=a,
        D=d,
        dt_bias=d,
        state_source=scratch,
        indices=idx,
        out=torch.zeros_like(x),
        dt_softplus=dt_softplus,
    )
    torch.cuda.synchronize(dev)
    return resolve_decode_backend()
