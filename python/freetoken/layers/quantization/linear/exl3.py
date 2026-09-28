"""EXL3 (exllamav3 trellis) linears: decoded in the kernel; long prefills decode one part at a time.

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
# From this many rows on, decode W_hat once per part to fp16 and let cuBLAS (fp32 accumulation) do
# the product, then the Hadamard/svh epilogue: the tl.dot GEMM re-decodes each weight column once
# per row block and runs at ~6.5 TF/s on an RTX 5080, 13-18x slower than reconstruct + cuBLAS at
# M >= 1024; reconstruct is ahead or level from M=16 on every Ornith shape (bench_dense_crossover-box-2026-09-24.txt).
RECONSTRUCT_MIN_ROWS = 16
# Decode GEMVs with at most this many 128-column blocks rotate their input in the GEMV
# prologue instead of a separate had_rows launch. Every program rotates the 128-blocks of
# its own K range, so a K block is rotated once per column block: a small projection
# saves the launch (o_proj 4096->2048, 16 blocks: 22.7 -> 21.1 us per CUDA-graph replay),
# a wide one pays more rotation than the launch cost (GDN in_proj 2048->12288, 96
# blocks: 50.4 -> 51.6 us). Box bench, tasks/ornith-exl3/fuse.
PREROT_MAX_NBLOCKS = 32
# Output columns per cuBLAS call: bounds the fp32 accumulator at rows x this.
RECONSTRUCT_SLAB = 2048
# From this many rows on, reconstruct W_full = diag(suh) H W_hat H diag(svh) (both rotations folded
# into the weight, as exllamav3's reconstruct_had) and run one plain GEMM on the raw activations:
# the per-token had_rows + had_cols passes cost 36-60% of every dense prefill projection at 8K rows
# (tasks/ornith-exl3/perf/RESEARCH.md). FREETOKEN_EXL3_FOLD=0 keeps the rotate-activations path.
FOLD_MIN_ROWS = 1024
# From this many rows on, the folded product runs in gemm_cast (Triton, fp32 accumulation) with the
# bf16 -> fp16 input cast and the fp16 -> bf16 output cast fused, instead of a host cast, cuBLAS and a
# copy-cast per slab. Bitwise the cuBLAS path wherever cuBLAS keeps one k pass: on the RTX 5080 cuBLAS
# splits K only for o_proj/out_proj (K 4096) at 1024 and 1059-1088 rows among the Ornith shapes at
# >= 1024 rows (tasks/ornith-exl3/popt-exl3/results/split-map2.log), hence 1152.
FUSED_CAST_MIN_ROWS = 1152


def fold_enabled() -> bool:
    import os

    return os.getenv("FREETOKEN_EXL3_FOLD", "1").strip() != "0"


def exl3_facts(scheme) -> tuple[int, str]:
    """``(bits, codebook)`` from an EXL3 scheme's weight element tag ``exl3_<codebook>_<bits>``."""
    _, codebook, bits = scheme.weight.elem.split("_")
    return int(bits), codebook


@functools.lru_cache(maxsize=None)
def _num_sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


GEMV_PROGRAMS_PER_SM = 20
# the last split sums every split's partial plane serially: past 32 splits that costs more than the
# extra programs gain (box: o_proj s32 13.3 us, s64 14.4 us)
GEMV_MAX_SPLIT = 32


def pick_split_k(rows: int, n_blocks: int, k: int, device: torch.device) -> int:
    """Split K while the decode grid stays within GEMV_PROGRAMS_PER_SM single-warp programs per SM
    (each split keeps >= 32 rows of K, a whole number of 16-row bands). Box sweep
    (tasks/ornith-exl3/perf/d2-gemv): with the 64-register GEMV (32 resident warps per SM) the big
    Ornith shapes are fastest at 1024-1536 programs (GDN in_proj s16, attn qkv s16, MoE gate|up s16,
    MoE down s8); at 168 registers (12 warps per SM) it was 256-768."""
    target = GEMV_PROGRAMS_PER_SM * _num_sms(device.index or 0)
    split = 1
    while (rows * n_blocks * split * 2 <= target and split * 2 <= GEMV_MAX_SPLIT
           and k % (split * 2 * 16) == 0 and k // (split * 2) >= 32):
        split *= 2
    return split


def exl3_forward(x2: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, parts, out_dtype) -> torch.Tensor:
    from freetoken.kernel.triton.exl3 import exl3_gemm, exl3_gemv, gemv_pre_rot, had_rows

    rows = x2.shape[0]
    if rows <= GEMV_MAX_ROWS:
        split = pick_split_k(rows, parts.n // 128, parts.k, x2.device)
        out = torch.empty((rows, parts.n), dtype=out_dtype, device=x2.device)
        if gemv_pre_rot() and parts.n // 128 <= PREROT_MAX_NBLOCKS:
            # the input rotation runs in the GEMV prologue: no had_rows launch
            return exl3_gemv(None, trellis, svh, parts, out=out, split_k=split, x=x2, suh=suh)
        return exl3_gemv(had_rows(x2, suh, parts), trellis, svh, parts, out=out, split_k=split)
    if rows >= FOLD_MIN_ROWS and parts.sizes and fold_enabled():
        return _forward_folded(x2, trellis, suh, svh, parts, out_dtype)
    xh = had_rows(x2, suh, parts)
    out = torch.empty((rows, parts.n), dtype=out_dtype, device=x2.device)
    if rows >= RECONSTRUCT_MIN_ROWS and parts.sizes:
        return _forward_reconstruct(xh, trellis, svh, parts, out)
    block_m = 16 if rows <= 32 else (32 if rows <= 128 else 64)
    return exl3_gemm(xh, trellis, svh, parts, out=out, block_m=block_m)


def _forward_reconstruct(xh: torch.Tensor, trellis: torch.Tensor, svh: torch.Tensor, parts, out: torch.Tensor) -> torch.Tensor:
    """``out = svh * H(xh[part] @ W_hat[part])`` part by part: W_hat decoded once (fp16, exact),
    the product in cuBLAS with an fp32 result, the 128-block Hadamard + svh in ``had_cols``."""
    from freetoken.kernel.triton.exl3 import had_cols, reconstruct

    k, bits = parts.k, parts.bits
    off = col = 0
    for j, n in enumerate(parts.sizes):
        count = (k // 16) * (n // 16) * 16 * bits
        w = reconstruct(trellis[off : off + count].view(k // 16, n // 16, 16 * bits), parts.codebook)
        for c0 in range(0, n, RECONSTRUCT_SLAB):
            c1 = min(c0 + RECONSTRUCT_SLAB, n)
            acc = torch.mm(xh[j], w[:, c0:c1], out_dtype=torch.float32)
            had_cols(acc, svh[col + c0 : col + c1], out[:, col + c0 : col + c1])
        off += count
        col += n
    return out


def _forward_folded(x2: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, parts, out_dtype) -> torch.Tensor:
    """``out = x2 @ W_full``: W_full decoded once with both rotations and sign vectors folded in
    (fp16, one rounding), the product in cuBLAS (fp16 in, fp32 accumulation, fp16 out)."""
    from freetoken.kernel.triton.exl3 import f16acc_enabled, gemm_f16acc, reconstruct_folded

    n = parts.n
    if f16acc_enabled() and x2.stride(1) == 1:
        # one Triton GEMM per slab: casts x on load and writes out_dtype, no host-side glue
        out = torch.empty((x2.shape[0], n), dtype=out_dtype, device=x2.device)
        for c0 in range(0, n, RECONSTRUCT_SLAB):
            c1 = min(c0 + RECONSTRUCT_SLAB, n)
            gemm_f16acc(x2, reconstruct_folded(trellis, suh, svh, parts, cols=(c0, c1))[0], out=out[:, c0:c1])
        return out
    if (x2.shape[0] >= FUSED_CAST_MIN_ROWS and x2.dtype in (torch.float16, torch.bfloat16) and x2.stride(1) == 1
            and parts.k % 64 == 0):
        from freetoken.kernel.triton.exl3 import gemm_cast

        out = torch.empty((x2.shape[0], n), dtype=out_dtype, device=x2.device)
        for c0 in range(0, n, RECONSTRUCT_SLAB):
            c1 = min(c0 + RECONSTRUCT_SLAB, n)
            gemm_cast(x2, reconstruct_folded(trellis, suh, svh, parts, cols=(c0, c1))[0], out[:, c0:c1])
        return out
    xf = x2 if x2.dtype == torch.float16 else x2.to(torch.float16)
    if out_dtype == torch.float16 and n <= RECONSTRUCT_SLAB:
        return torch.mm(xf, reconstruct_folded(trellis, suh, svh, parts)[0])
    # column slabs: W_full and the fp16 product exist one slab at a time (the prefill transient
    # the server reserves), the result lands in out_dtype directly
    out = torch.empty((x2.shape[0], n), dtype=out_dtype, device=x2.device)
    for c0 in range(0, n, RECONSTRUCT_SLAB):
        c1 = min(c0 + RECONSTRUCT_SLAB, n)
        w = reconstruct_folded(trellis, suh, svh, parts, cols=(c0, c1))[0]
        if out_dtype == torch.float16:
            torch.mm(xf, w, out=out[:, c0:c1])
        else:
            out[:, c0:c1] = torch.mm(xf, w)
    return out


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
        # an EXL3 layer has no dense weight; drop the placeholder a base class may have
        # declared (ParallelLMHead inherits VocabParallelEmbedding's [V, H] ``weight``)
        if isinstance(getattr(layer, "weight", None), torch.Tensor):
            layer.weight = None
