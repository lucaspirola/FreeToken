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
    group_blocks,
    had_rows,
    reconstruct_experts,
    f16acc_enabled,
    reconstruct_folded,
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
# few hundred tokens touch only part of the experts) keep the in-kernel decode. The two paths are
# bitwise equal (same W_hat values, same 16-wide mma k-steps). Offline the in-kernel GEMM is faster up
# to ~1100 tokens (Ornith), but moving the switch there did not shorten the server's 300 / 1000-token
# TTFT, where the prefill layer stream dominates (tasks/ornith-exl3/kfix/README.md), so it stays.
PREFILL_DECODED_MIN_TOKENS = 256
# experts per decoded scratch group, per projection: as many as fit in PREFILL_DECODE_L2_FRACTION of
# the GPU's L2, so the GEMM reads W_hat back from L2 instead of DRAM (RTX 5080, 64 MB L2, Ornith:
# gate_up 4 MiB per expert -> 8, down 2 MiB -> 16; the fused MoE at 8192 tokens 11.39 -> ~10 ms,
# tasks/ornith-exl3/popt-exl3/README.md). A GPU whose L2 holds fewer than PREFILL_DECODE_GROUP_MIN
# experts gains nothing from residency and keeps PREFILL_DECODE_GROUP (fewer launches).
# PREFILL_DECODE_GROUP_OVERRIDE (int) forces one size for both (sweeps).
PREFILL_DECODE_GROUP = 32
PREFILL_DECODE_L2_FRACTION = 0.5
PREFILL_DECODE_GROUP_MIN = 4
PREFILL_DECODE_GROUP_OVERRIDE: int | None = None
# decoded GEMM launches cover only their group's row blocks (``exl3_gemm(mb_range=...)``) instead of
# every expert's blocks with all but the group's returning at once (~60 us per launch at 8192
# tokens). A/B switch.
PREFILL_RANGED = True


@functools.lru_cache(maxsize=None)
def _l2_bytes(index: int) -> int:
    return int(getattr(torch.cuda.get_device_properties(index), "L2_cache_size", 0) or 0)


def _decode_group(parts: Exl3Parts, num_experts: int, device: torch.device) -> int:
    if PREFILL_DECODE_GROUP_OVERRIDE:
        return min(PREFILL_DECODE_GROUP_OVERRIDE, num_experts)
    per_expert = parts.k * parts.n * 2
    fit = int(_l2_bytes(device.index or 0) * PREFILL_DECODE_L2_FRACTION) // per_expert
    return min(fit if fit >= PREFILL_DECODE_GROUP_MIN else PREFILL_DECODE_GROUP, PREFILL_DECODE_GROUP, num_experts)


# the gate/up output and the activation between the two GEMMs: fp16, as in exllamav3 (whose
# fp16 forward the checkpoint was quantized and calibrated against); bf16 here cost 8x the
# rounding error for nothing, since the down projection rounds its rotated input to fp16 anyway
ACT_DTYPE = torch.float16
# decoded prefill: fold the input rotation ``diag(suh) H`` into the decoded W_hat
# (reconstruct_folded(fold_out=False)), so both GEMMs read the raw activations (route p reads
# token p // top_k) and the two had_rows launches plus the [2, P, H] rotated buffer disappear.
# OFF by default: unlike a dense layer, the fold pays the 128x128 Hadamard on every expert's weights
# (256 x 3 x H x I elements, ~206 GFLOP per 8K chunk) instead of on the routed activations (~43
# GFLOP), so reconstruct_folded costs 6.18 ms against reconstruct 2.69 + had_rows 2.30 ms and the
# MoE layer goes 12.18 -> 13.24 ms at 8192 tokens (bench_moe_prefill.py --quick, box).
# FREETOKEN_EXL3_MOE_FOLD=1 turns it on (A/B only).
PREFILL_FOLD_INPUT = os.environ.get("FREETOKEN_EXL3_MOE_FOLD", "0") == "1"


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
    if f16acc_enabled():
        # fp16 accumulators halve the register tile, so BM 64 / 2 stages wins: GEMM 6.37 -> 4.77 ms
        # per layer at 8192 tokens (the same 72-config sweep, FREETOKEN_EXL3_F16ACC=1)
        bm = 16 if per_expert < 16 else (32 if per_expert < 64 else 64)
        cfg = dict(block_m=bm, block_k=64, num_stages=2, num_warps=8, f16acc=True)
    else:
        cfg = dict(block_m=16 if per_expert < 16 else 32, block_k=64, num_stages=3, num_warps=8)
    if PREFILL_GEMM:
        cfg.update(PREFILL_GEMM)
    return cfg


def _decoded_gemms(xin, tr, suh, svh, parts, out, w, group, sort, gemm_args, num_experts, fold, src_div):
    """Every expert group: decode its W_hat into ``w`` (``fold``: with the input rotation folded in),
    then the grouped GEMM over the group's row blocks."""
    gb = group_blocks(sort["expert_ids"], sort["num_post_pad"], sort["block_m"], num_experts, group) if PREFILL_RANGED else None
    for lo in range(0, num_experts, group):
        hi = min(lo + group, num_experts)
        if fold:
            reconstruct_folded(tr, suh, svh, parts, lo=lo, hi=hi, suh_expert_stride=suh.stride(0),
                               svh_expert_stride=svh.stride(0), out=w, fold_out=False)
        else:
            reconstruct_experts(tr, parts, lo, hi, out=w)
        rng = dict(mb_range=gb[lo // group : lo // group + 2], num_experts=num_experts) if gb is not None else {}
        exl3_gemm(xin, tr, svh, parts, out=out, decoded=w, expert_range=(lo, hi), src_div=src_div, **gemm_args, **sort, **rng)


def _prefill(x, banks, topk_weights, topk_ids, top_k, parts, activation, alpha, limit, num_experts):
    from freetoken.moe.fused import moe_align_block_size

    gu_tr, gu_suh, gu_svh, dn_tr, dn_suh, dn_svh = banks
    gu, dn = parts
    out = torch.empty_like(x)
    decoded = x.shape[0] >= PREFILL_DECODED_MIN_TOKENS
    fold = decoded and PREFILL_FOLD_INPUT
    if decoded:
        g_gu = _decode_group(gu, num_experts, x.device)
        g_dn = _decode_group(dn, num_experts, x.device)
        w_gu = torch.empty((g_gu, gu.k, gu.n), dtype=torch.float16, device=x.device)
        w_dn = torch.empty((g_dn, dn.k, dn.n), dtype=torch.float16, device=x.device)
    gu_args = dict(tr_expert_stride=gu_tr.stride(0) // 2, svh_expert_stride=gu_svh.stride(0))
    dn_args = dict(tr_expert_stride=dn_tr.stride(0) // 2, svh_expert_stride=dn_svh.stride(0))
    for t0 in range(0, x.shape[0], PREFILL_CHUNK_TOKENS):
        t1 = min(t0 + PREFILL_CHUNK_TOKENS, x.shape[0])
        xc = x[t0:t1]
        ids2 = topk_ids[t0:t1]
        ids = ids2.reshape(-1).contiguous()
        routes = ids.numel()
        cfg = _prefill_gemm(routes, num_experts) if decoded else dict(block_m=_prefill_block_m(routes, num_experts))
        sorted_ids, expert_ids, npad = moe_align_block_size(ids2.contiguous(), cfg["block_m"], num_experts)
        sort = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_post_pad=npad, **cfg)
        g = torch.empty((routes, gu.n), dtype=ACT_DTYPE, device=x.device)
        if fold:
            xs = xc.to(torch.float16)
            _decoded_gemms(xs, gu_tr, gu_suh, gu_svh, gu, g, w_gu, g_gu, sort, gu_args, num_experts, True, top_k)
            del xs
        else:
            xh = had_rows(xc, gu_suh, gu, src_div=top_k, experts=ids, suh_expert_stride=gu_suh.stride(0))
            if decoded:
                _decoded_gemms(xh, gu_tr, gu_suh, gu_svh, gu, g, w_gu, g_gu, sort, gu_args, num_experts, False, 0)
            else:
                exl3_gemm(xh, gu_tr, gu_svh, gu, out=g, **gu_args, **sort)
            del xh
        a = _act(g, activation, alpha, limit)
        del g
        # FREETOKEN_EXL3_F16ACC also keeps the per-route down output in fp16 (exllamav3 does): the
        # combine then reads half the bytes (it sums in fp32 either way)
        o_dtype = torch.float16 if f16acc_enabled() else torch.float32
        o = torch.empty((routes, dn.n), dtype=o_dtype, device=x.device)
        if fold:
            _decoded_gemms(a, dn_tr, dn_suh, dn_svh, dn, o, w_dn, g_dn, sort, dn_args, num_experts, True, 1)
        else:
            ah = had_rows(a, dn_suh, dn, experts=ids, suh_expert_stride=dn_suh.stride(0))
            if decoded:
                _decoded_gemms(ah, dn_tr, dn_suh, dn_svh, dn, o, w_dn, g_dn, sort, dn_args, num_experts, False, 0)
            else:
                exl3_gemm(ah, dn_tr, dn_svh, dn, out=o, **dn_args, **sort)
            del ah
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
