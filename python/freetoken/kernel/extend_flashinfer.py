"""Prefill (extend) attention through flashinfer's dense prefill kernel.

The shared triton extend kernel reaches 49-76 TFLOP/s on an RTX 5080 at 8K-120K
prefixes, flashinfer's FA2 prefill 110-113 (tasks/attn-prefill, box numbers):
at a 120K prefix, 351 -> 152 ms per layer for 16q/2kv D256 and 239 -> 151 ms
for 32q/2kv D128. flashinfer cannot read the paged q8_0 pool, so the prefix is
gathered and dequantized block by block into a bounded bf16 buffer, attended
non-causally with its log-sum-exp returned, and merged with the causal
attention over the chunk's own keys. The block bound keeps the transient
independent of the context length (the prefill headroom is measured once, at
startup, on a short prefix).

Covers what the production models need: no sliding window, no sinks, no
multimodal blocks, unquantized or byte-per-value (q8_0 / fp8) pools. Anything
else stays on the triton kernel (``eligible``).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Prefix tokens dequantized per flashinfer call. 16K tokens is 32 MiB of bf16
# K+V for 2 KV heads at D256; each block costs one extra launch and one merge.
_BLOCK_ENV = "FREETOKEN_EXTEND_FI_BLOCK"
_DEFAULT_BLOCK = 16384


def prefix_block() -> int:
    raw = os.getenv(_BLOCK_ENV, "").strip()
    return max(int(raw), 256) if raw else _DEFAULT_BLOCK


def enabled() -> bool:
    return os.getenv("FREETOKEN_EXTEND_BACKEND", "flashinfer").strip().lower() == "flashinfer"


_FI = None
_LOGGED = False


def _flashinfer():
    global _FI
    if _FI is None:
        try:
            import flashinfer  # noqa: PLC0415

            _FI = flashinfer
        except Exception:  # not installed / no build for this GPU
            _FI = False
    return _FI or None


def eligible(q, k_format, v_format, sliding_window, sinks, block_ends, k_extend, host_lens) -> bool:
    return (
        host_lens is not None
        and k_extend is not None
        and sliding_window is None
        and sinks is None
        and block_ends is None
        and k_format in (0, 1)
        and v_format in (0, 1)
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.shape[-1] in (64, 128, 256)
        and enabled()
        and _flashinfer() is not None
    )


@triton.jit
def _gather_dequant_kernel(
    src, scale, idx, dst,
    s_src0, s_src1, s_sc0, s_sc1,
    H: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr, QBLOCK: tl.constexpr,
    FORMAT: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(idx + t).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(src + slot * s_src0 + h * s_src1 + offs, mask=mask).to(tl.float32)
    if FORMAT == 1:
        # Same arithmetic as the triton kernels' _load_kv: value * scale in fp32,
        # rounded once to the compute dtype.
        sc = tl.load(scale + slot * s_sc0 + h * s_sc1 + offs // QBLOCK, mask=mask, other=0.0)
        x = x * sc.to(tl.float32)
    tl.store(dst + (t * H + h) * D + offs, x.to(dst.dtype.element_ty), mask=mask)


def _gather(cache, scale, fmt, idx, dst):
    n = idx.numel()
    heads, d = dst.shape[1], dst.shape[2]
    qblock = d // scale.shape[-1] if scale is not None else 1
    sc = scale if scale is not None else cache
    _gather_dequant_kernel[(n, heads)](
        cache, sc, idx, dst,
        cache.stride(0), cache.stride(1),
        sc.stride(0) if scale is not None else 0, sc.stride(1) if scale is not None else 0,
        H=heads, D=d, BLOCK_D=triton.next_power_of_2(d), QBLOCK=qblock, FORMAT=fmt,
        num_warps=2 if d <= 128 else 4,
    )


def extend_attention(
    *, q, k_cache, v_cache, k_scale, v_scale, k_format, v_format, kv_indices,
    k_extend, v_extend, sm_scale, out, host_lens,
):
    """``host_lens`` = (qo_lens, prefix_lens, kv_lens) per sequence, host ints;
    kv_indices lists each sequence's kv_lens slots, prefix first. ``out`` may be
    None: a single sequence then returns flashinfer's own output tensor, which
    saves a q-sized buffer and a copy (the single-lane prefill case)."""
    global _LOGGED
    fi = _flashinfer()
    if not _LOGGED:
        _LOGGED = True
        logger.info_rank0(
            "extend attention: flashinfer %s prefill, prefix dequantized in %d-token blocks "
            "(FREETOKEN_EXTEND_BACKEND=triton for the triton kernel)", fi.__version__, prefix_block())
    qo_lens, prefix_lens, kv_lens = host_lens
    heads_kv, d = k_extend.shape[1], k_extend.shape[2]
    block = prefix_block()
    max_prefix = max(prefix_lens, default=0)
    kbuf = vbuf = None
    if max_prefix:
        # Full block size whatever the prefix: the transient must not depend on
        # the context length, or the startup headroom measurement undercounts it.
        kbuf = torch.empty((block, heads_kv, d), dtype=q.dtype, device=q.device)
        vbuf = torch.empty_like(kbuf)
    qo = kv = 0
    for ql, pl, kl in zip(qo_lens, prefix_lens, kv_lens):
        if ql:
            qb = q[qo:qo + ql]
            o, lse = fi.single_prefill_with_kv_cache(
                qb, k_extend[qo:qo + ql], v_extend[qo:qo + ql],
                causal=True, sm_scale=sm_scale, return_lse=True,
            )
            for s in range(0, pl, block):
                e = min(pl, s + block)
                idx = kv_indices[kv + s:kv + e]
                kb, vb = kbuf[:e - s], vbuf[:e - s]
                _gather(k_cache, k_scale, k_format, idx, kb)
                _gather(v_cache, v_scale, v_format, idx, vb)
                op, lp = fi.single_prefill_with_kv_cache(
                    qb, kb, vb, causal=False, sm_scale=sm_scale, return_lse=True,
                )
                fi.merge_state_in_place(o, lse, op, lp)
                del op, lp
            if out is None and ql == q.shape[0]:
                return o
            if out is None:
                out = torch.empty_like(q)
            out[qo:qo + ql].copy_(o)
            del o, lse
        qo += ql
        kv += kl
    return out if out is not None else torch.empty_like(q)
