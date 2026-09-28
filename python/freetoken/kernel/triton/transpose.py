"""Tiled 2-D transpose (a pure copy): ``out[c, r] = x[r, c]`` for any row stride of ``x`` and ``out``.

``x.t().contiguous()`` goes through TensorIterator's generic strided copy, which reads or writes one
side a column at a time: on the GDN prefill ([8192, 8192] bf16 per layer and chunk) it ran at ~270
GB/s. Tiles loaded along one side's rows and stored along the other's (Triton transposes the tile
through shared memory) keep both sides coalesced."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(x_ptr, stride_xr, out_ptr, stride_or, R, C, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = (r[:, None] < R) & (c[None, :] < C)
    v = tl.load(x_ptr + r[:, None].to(tl.int64) * stride_xr + c[None, :], mask=m)
    tl.store(out_ptr + c[None, :].to(tl.int64) * stride_or + r[:, None], v, mask=m)


TRANSPOSE_TILE = dict(BR=64, BC=64, num_warps=4)


def transpose_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out[:] = x.t()`` for 2-D ``x [R, C]`` and ``out [C, R]``, both unit-stride along their last dim."""
    R, C = x.shape
    assert out.shape == (C, R) and x.stride(1) == 1 and out.stride(1) == 1 and x.dtype == out.dtype
    if R and C:
        t = TRANSPOSE_TILE
        _transpose_kernel[(triton.cdiv(R, t["BR"]), triton.cdiv(C, t["BC"]))](
            x, x.stride(0), out, out.stride(0), R, C, BR=t["BR"], BC=t["BC"], num_warps=t["num_warps"])
    return out


def transpose(x: torch.Tensor) -> torch.Tensor:
    """``x.t().contiguous()`` for 2-D ``x`` with a unit last-dim stride."""
    return transpose_into(x, torch.empty((x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device))


__all__ = ["transpose", "transpose_into"]
