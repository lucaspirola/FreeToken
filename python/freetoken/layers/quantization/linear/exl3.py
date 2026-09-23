"""EXL3 (exllamav3 trellis) linears: decoded in the kernel, never materialized.

The layer holds ``trellis`` int16 flat and part-major (a fused projection's parts back to back,
each ``[K/16, N_j/16, 16*bits]``), ``suh`` fp16 ``[parts, K]`` and ``svh`` fp16 ``[N]``; see
``freetoken.kernel.triton.exl3`` for the math and the dialect's ``fuse_parts`` for the packing.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod

# At most this many rows go through the per-row GEMV; more use the tl.dot GEMM.
GEMV_MAX_ROWS = 8


def exl3_facts(scheme) -> tuple[int, str]:
    """``(bits, codebook)`` from an EXL3 scheme's weight element tag ``exl3_<codebook>_<bits>``."""
    _, codebook, bits = scheme.weight.elem.split("_")
    return int(bits), codebook


@functools.lru_cache(maxsize=None)
def _num_sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def pick_split_k(rows: int, n_blocks: int, k: int, device: torch.device) -> int:
    """Split K until the decode grid covers the GPU about twice (each split must be >= 256 rows of K)."""
    target = 2 * _num_sms(device.index or 0)
    split = 1
    while rows * n_blocks * split < target and k % (split * 2 * 32) == 0 and k // (split * 2) >= 256:
        split *= 2
    return split


def exl3_forward(x2: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, parts, out_dtype) -> torch.Tensor:
    from freetoken.kernel.triton.exl3 import exl3_gemm, exl3_gemv, had_rows

    rows = x2.shape[0]
    xh = had_rows(x2, suh, parts)
    if rows <= GEMV_MAX_ROWS:
        split = pick_split_k(rows, parts.n // 128, parts.k, x2.device)
        if split > 1:
            acc = torch.zeros((rows, parts.n), dtype=torch.float32, device=x2.device)
            exl3_gemv(xh, trellis, svh, parts, out=acc, split_k=split)
            return acc.to(out_dtype)
        out = torch.empty((rows, parts.n), dtype=out_dtype, device=x2.device)
        return exl3_gemv(xh, trellis, svh, parts, out=out)
    out = torch.empty((rows, parts.n), dtype=out_dtype, device=x2.device)
    block_m = 16 if rows <= 32 else (32 if rows <= 128 else 64)
    return exl3_gemm(xh, trellis, svh, parts, out=out, block_m=block_m)


class TritonExl3LinearKernel(LinearKernel):
    name = "triton"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if cfg.in_features % 128 or any(s % 128 for s in cfg.output_sizes):
            return f"EXL3 needs 128-aligned dims, got K={cfg.in_features} parts {cfg.output_sizes}"
        return None

    def finalize(self, layer: Any) -> None:
        from freetoken.kernel.triton.exl3 import Exl3Parts

        bits, codebook = exl3_facts(layer.quant_method.scheme)
        layer._exl3_parts = Exl3Parts.build(layer.in_features, tuple(layer.output_sizes), bits, codebook, layer.trellis.device)

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        *lead, k = x.shape
        x2 = x.reshape(-1, k)
        if x2.stride(-1) != 1 or x2.stride(0) != k:
            x2 = x2.contiguous()
        out = exl3_forward(x2, layer.trellis, layer.suh, layer.svh, layer._exl3_parts, x.dtype)
        if layer.bias is not None:
            out = out + layer.bias.to(out.dtype)
        return out.reshape(*lead, out.shape[-1])


class EmulationExl3LinearKernel(LinearKernel):
    """Decode to fp32 with the torch reference decoder, then a plain matmul (any device; tests)."""

    name = "emulation"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.exl3 import linear_reference

        bits, codebook = exl3_facts(layer.quant_method.scheme)
        *lead, k = x.shape
        x2 = x.reshape(-1, k)
        outs, off = [], 0
        for j, n in enumerate(layer.output_sizes):
            count = (k // 16) * (n // 16) * 16 * bits
            tr = layer.trellis[off : off + count].view(k // 16, n // 16, 16 * bits)
            col = sum(layer.output_sizes[:j])
            outs.append(linear_reference(x2, tr, layer.suh[j], layer.svh[col : col + n], codebook))
            off += count
        out = torch.cat(outs, dim=-1).to(x.dtype)
        if layer.bias is not None:
            out = out + layer.bias.to(out.dtype)
        return out.reshape(*lead, out.shape[-1])


@register_method(QuantKind.EXL3, LayerKind.LINEAR)
class Exl3LinearMethod(LinearMethod):
    candidates = (TritonExl3LinearKernel, EmulationExl3LinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        bits, _ = exl3_facts(self.scheme)
        layer.trellis = torch.empty((g.in_features // 16) * (g.out_features // 16) * 16 * bits, dtype=torch.int16)
        layer.suh = torch.empty(len(g.output_sizes), g.in_features, dtype=torch.float16)
        layer.svh = torch.empty(g.out_features, dtype=torch.float16)
