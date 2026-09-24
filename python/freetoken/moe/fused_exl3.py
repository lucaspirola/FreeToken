"""EXL3 routed experts: the forward over the six raw-row banks (``models/exl3_banks.py``).

Per route ``p`` (token ``p // top_k``, expert ``ids[p]``)::

    xh   = H(x * gate_up_suh[e])                  both parts, [2, P, H] fp16  (had_rows)
    g    = [gate | up] = gemm/gemv(xh, gate_up)   [P, 2I]
    a    = act(gate) * up                         [P, I]
    ah   = H(a * down_suh[e])                     [1, P, I]
    o    = gemm/gemv(ah, down)                    [P, H]
    out  = sum_k w[t, k] * o[t * top_k + k]

Decode (``is_prefill=False``): ``ids`` are slot ids into the GPU cache's ``[S, ...]`` views and
every step is a fixed-shape launch, so it records in a CUDA graph. Prefill: ``ids`` are expert
ids into ``[E, ...]`` views; routes are sorted per expert (``moe_align_block_size``) and run
through the tl.dot GEMM, a chunk of tokens at a time to bound the ``[2, P, H]`` rotation buffer.
A long prefill chunk touches every expert, so W_hat is decoded once per chunk into an fp16 scratch,
``PREFILL_DECODE_GROUP`` experts at a time, and the GEMM reads it instead of re-decoding the trellis
in every row block (tasks/ornith-exl3/perf: the in-kernel decode ran the MoE at 3-6 TF/s).
"""

from __future__ import annotations

import functools
import os

import torch

from freetoken.kernel.triton.exl3 import (
    Exl3Parts,
    exl3_gemm,
    exl3_gemv,
    gemv_pre_rot,
    had_rows,
    reconstruct_experts,
    splitk_combine,
    splitk_silu_had,
)

# tokens per MoE prefill sub-chunk. Each sub-chunk decodes every expert's W_hat once and runs every
# expert's rows through one grouped GEMM, so the sub-chunk sets how many rows a decoded weight tile
# serves: at 2048 tokens (~64 rows per expert of 256) the GEMM was bandwidth-starved (25 TF/s) and the
# reconstruct rewrote all experts 4x per 8K chunk. One sub-chunk per 8K prefill chunk: MoE layer
# 31.2 -> 13.6 ms at 8192 tokens (bench_moe_prefill.py, box). The rotated input is
# 2 * tokens * top_k * H fp16 (512 MiB at 8192 x 8 x 2048) and the down output the same in fp32.
PREFILL_CHUNK_TOKENS = 8192
# a chunk of at least this many tokens decodes W_hat into the scratch; shorter ones (extends of a
# few hundred tokens touch only part of the experts) keep the in-kernel decode
PREFILL_DECODED_MIN_TOKENS = 256
# experts per decoded scratch group: gate_up + down W_hat are 3 * H * I * 2 bytes per expert
# (6 MiB for Ornith), so 32 experts hold 192 MiB
PREFILL_DECODE_GROUP = 32
# the gate/up output and the activation between the two GEMMs: fp16, as in exllamav3 (whose
# fp16 forward the checkpoint was quantized and calibrated against); bf16 here cost 8x the
# rounding error for nothing, since the down projection rounds its rotated input to fp16 anyway
ACT_DTYPE = torch.float16


@functools.lru_cache(maxsize=16)
def expert_parts(hidden: int, inter: int, bits: int, codebook: str, device: torch.device) -> tuple[Exl3Parts, Exl3Parts]:
    """Part tables of one expert: gate_up (two parts ``[H -> I]``, suh parts H apart) and down (``[I -> H]``)."""
    gate_up = Exl3Parts.build(hidden, (inter, inter), bits, codebook, device, suh_part_stride=hidden)
    down = Exl3Parts.build(inter, (hidden,), bits, codebook, device)
    return gate_up, down


def _combine(o: torch.Tensor, topk_weights: torch.Tensor, tokens: int, top_k: int, dtype: torch.dtype) -> torch.Tensor:
    w = topk_weights.reshape(tokens, top_k, 1).to(torch.float32)
    return (o.view(tokens, top_k, -1).to(torch.float32) * w).sum(dim=1).to(dtype)


def _act(g: torch.Tensor, activation: str, alpha: float, limit: float) -> torch.Tensor:
    from freetoken.layers.activation import gated_act_and_mul

    a = torch.empty((g.shape[0], g.shape[1] // 2), dtype=g.dtype, device=g.device)
    gated_act_and_mul(activation, g, a, alpha=alpha, limit=limit)
    return a


def _fused_epilogues() -> bool:
    return os.getenv("FREETOKEN_EXL3_FUSED_EPILOGUE", "1").strip() != "0"


def _decode(x, banks, topk_weights, ids, top_k, parts, activation, alpha, limit):
    from freetoken.layers.quantization.linear.exl3 import pick_split_k

    gu_tr, gu_suh, gu_svh, dn_tr, dn_suh, dn_svh = banks
    gu, dn = parts
    tokens = x.shape[0]
    routes = tokens * top_k
    dev = x.device
    if activation == "silu" and _fused_epilogues():
        # The gate|up output's fp16 rounding, silu*up and the down input rotation are one
        # launch, and the top-k combine another (was: cast, act, had_rows, and four torch
        # ops for the combine). The split-K sums happen inside exl3_gemv, and with
        # FREETOKEN_EXL3_GEMV_PREROT (default) so does the gate|up input rotation.
        g = torch.empty((routes, gu.n), dtype=torch.float32, device=dev)
        gu_split = pick_split_k(routes, gu.n // 128, gu.k, dev)
        if gemv_pre_rot():
            exl3_gemv(None, gu_tr, gu_svh, gu, out=g, experts=ids, tr_expert_stride=gu_tr.stride(0) // 2,
                      svh_expert_stride=gu_svh.stride(0), split_k=gu_split,
                      x=x, suh=gu_suh, suh_expert_stride=gu_suh.stride(0), src_div=top_k)
        else:
            xh = had_rows(x, gu_suh, gu, src_div=top_k, experts=ids, suh_expert_stride=gu_suh.stride(0))
            exl3_gemv(xh, gu_tr, gu_svh, gu, out=g, experts=ids, tr_expert_stride=gu_tr.stride(0) // 2,
                      svh_expert_stride=gu_svh.stride(0), split_k=gu_split)
        ah = splitk_silu_had(g.unsqueeze(0), dn_suh, ids, dn_suh.stride(0))
        o = torch.empty((routes, dn.n), dtype=torch.float32, device=dev)
        exl3_gemv(ah, dn_tr, dn_svh, dn, out=o, experts=ids, tr_expert_stride=dn_tr.stride(0) // 2,
                  svh_expert_stride=dn_svh.stride(0), split_k=pick_split_k(routes, dn.n // 128, dn.k, dev))
        return splitk_combine(o.unsqueeze(0), topk_weights, tokens, top_k, x.dtype)
    xh = had_rows(x, gu_suh, gu, src_div=top_k, experts=ids, suh_expert_stride=gu_suh.stride(0))
    split = pick_split_k(routes, gu.n // 128, gu.k, dev)
    g = torch.empty((routes, gu.n), dtype=torch.float32, device=dev)
    exl3_gemv(xh, gu_tr, gu_svh, gu, out=g, experts=ids, tr_expert_stride=gu_tr.stride(0) // 2,
              svh_expert_stride=gu_svh.stride(0), split_k=split)
    a = _act(g.to(ACT_DTYPE), activation, alpha, limit)
    ah = had_rows(a, dn_suh, dn, experts=ids, suh_expert_stride=dn_suh.stride(0))
    split = pick_split_k(routes, dn.n // 128, dn.k, dev)
    o = torch.empty((routes, dn.n), dtype=torch.float32, device=dev)
    exl3_gemv(ah, dn_tr, dn_svh, dn, out=o, experts=ids, tr_expert_stride=dn_tr.stride(0) // 2,
              svh_expert_stride=dn_svh.stride(0), split_k=split)
    return _combine(o, topk_weights, tokens, top_k, x.dtype)


def _prefill_block_m(routes: int, num_experts: int) -> int:
    per_expert = routes / max(num_experts, 1)
    return 16 if per_expert < 16 else (32 if per_expert < 64 else 64)


# tile config of the decoded-slab grouped GEMM (exl3_gemm with ``decoded``). Box sweep at 8192
# tokens (bench_moe_prefill.py): BK 64, 3 stages, 8 warps, BM 32 is the best of 72 configs (GEMM
# 16.3 -> 6.5 ms per layer with the larger sub-chunk); BM 64/128 lose 5-10%. Overridable for sweeps.
PREFILL_GEMM: dict | None = None


def _prefill_gemm(routes: int, num_experts: int) -> dict:
    per_expert = routes / max(num_experts, 1)
    cfg = dict(block_m=16 if per_expert < 16 else 32, block_k=64, num_stages=3, num_warps=8)
    if PREFILL_GEMM:
        cfg.update(PREFILL_GEMM)
    return cfg


def _prefill(x, banks, topk_weights, topk_ids, top_k, parts, activation, alpha, limit, num_experts):
    from freetoken.moe.fused import moe_align_block_size

    gu_tr, gu_suh, gu_svh, dn_tr, dn_suh, dn_svh = banks
    gu, dn = parts
    out = torch.empty_like(x)
    decoded = x.shape[0] >= PREFILL_DECODED_MIN_TOKENS
    if decoded:
        group = min(PREFILL_DECODE_GROUP, num_experts)
        w_gu = torch.empty((group, gu.k, gu.n), dtype=torch.float16, device=x.device)
        w_dn = torch.empty((group, dn.k, dn.n), dtype=torch.float16, device=x.device)
    for t0 in range(0, x.shape[0], PREFILL_CHUNK_TOKENS):
        t1 = min(t0 + PREFILL_CHUNK_TOKENS, x.shape[0])
        xc = x[t0:t1]
        ids2 = topk_ids[t0:t1]
        ids = ids2.reshape(-1).contiguous()
        routes = ids.numel()
        cfg = _prefill_gemm(routes, num_experts) if decoded else dict(block_m=_prefill_block_m(routes, num_experts))
        sorted_ids, expert_ids, npad = moe_align_block_size(ids2.contiguous(), cfg["block_m"], num_experts)
        sort = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_post_pad=npad, **cfg)
        xh = had_rows(xc, gu_suh, gu, src_div=top_k, experts=ids, suh_expert_stride=gu_suh.stride(0))
        g = torch.empty((routes, gu.n), dtype=ACT_DTYPE, device=x.device)
        gu_args = dict(tr_expert_stride=gu_tr.stride(0) // 2, svh_expert_stride=gu_svh.stride(0), **sort)
        if decoded:
            for lo in range(0, num_experts, group):
                hi = min(lo + group, num_experts)
                reconstruct_experts(gu_tr, gu, lo, hi, out=w_gu)
                exl3_gemm(xh, gu_tr, gu_svh, gu, out=g, decoded=w_gu, expert_range=(lo, hi), **gu_args)
        else:
            exl3_gemm(xh, gu_tr, gu_svh, gu, out=g, **gu_args)
        del xh
        a = _act(g, activation, alpha, limit)
        del g
        ah = had_rows(a, dn_suh, dn, experts=ids, suh_expert_stride=dn_suh.stride(0))
        o = torch.empty((routes, dn.n), dtype=torch.float32, device=x.device)
        dn_args = dict(tr_expert_stride=dn_tr.stride(0) // 2, svh_expert_stride=dn_svh.stride(0), **sort)
        if decoded:
            for lo in range(0, num_experts, group):
                hi = min(lo + group, num_experts)
                reconstruct_experts(dn_tr, dn, lo, hi, out=w_dn)
                exl3_gemm(ah, dn_tr, dn_svh, dn, out=o, decoded=w_dn, expert_range=(lo, hi), **dn_args)
        else:
            exl3_gemm(ah, dn_tr, dn_svh, dn, out=o, **dn_args)
        # one kernel, fixed k order, no [routes, H] fp32 temporaries (was: cast, mul, sum)
        out[t0:t1] = splitk_combine(o.unsqueeze(0), topk_weights[t0:t1], t1 - t0, top_k, x.dtype)
    return out


def fused_experts_exl3(
    x: torch.Tensor,
    banks: tuple[torch.Tensor, ...],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    bits: int,
    codebook: str,
    activation: str = "silu",
    alpha: float = 1.0,
    limit: float = float("inf"),
    is_prefill: bool,
    num_experts: int | None = None,
) -> torch.Tensor:
    """``banks`` = (gate_up_trellis, gate_up_suh, gate_up_svh, down_trellis, down_suh, down_svh),
    each ``[S or E, ...]`` with the expert dim outermost; see the module docstring."""
    if x.stride(-1) != 1 or x.stride(0) != x.shape[1]:
        x = x.contiguous()
    top_k = topk_ids.shape[1]
    hidden = x.shape[1]
    inter = banks[4].shape[-1]
    parts = expert_parts(hidden, inter, bits, codebook, x.device)
    if x.shape[0] == 0:
        return torch.empty_like(x)
    if is_prefill:
        if num_experts is None:
            raise ValueError("EXL3 prefill needs the view's expert count")
        return _prefill(x, banks, topk_weights, topk_ids, top_k, parts, activation, alpha, limit, num_experts)
    ids = topk_ids.reshape(-1).contiguous()
    return _decode(x, banks, topk_weights, ids, top_k, parts, activation, alpha, limit)


def fused_experts_exl3_reference(x, banks, topk_weights, topk_ids, *, codebook: str, activation: str = "silu") -> torch.Tensor:
    """Torch reference (decode by ``linear_reference``), fp32 throughout and returned as fp32: for the tests only."""
    import torch.nn.functional as F

    from freetoken.kernel.triton.exl3 import linear_reference

    gu_tr, gu_suh, gu_svh, dn_tr, dn_suh, dn_svh = banks
    assert activation == "silu"
    tokens, top_k = topk_ids.shape
    inter = dn_suh.shape[-1]
    out = torch.zeros((tokens, x.shape[1]), dtype=torch.float32, device=x.device)
    for t in range(tokens):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            xi = x[t : t + 1].float()
            gate = linear_reference(xi, gu_tr[e, 0], gu_suh[e, 0], gu_svh[e, 0], codebook).float()
            up = linear_reference(xi, gu_tr[e, 1], gu_suh[e, 1], gu_svh[e, 1], codebook).float()
            a = F.silu(gate) * up
            o = linear_reference(a, dn_tr[e], dn_suh[e], dn_svh[e], codebook).float()
            out[t] += float(topk_weights[t, k]) * o[0]
    assert inter == gu_svh.shape[-1]
    return out


__all__ = ["PREFILL_CHUNK_TOKENS", "expert_parts", "fused_experts_exl3", "fused_experts_exl3_reference"]
