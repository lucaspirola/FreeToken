"""Small fused epilogues for gated shared-expert MoE blocks."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_expert_add_kernel(
    routed_ptr,
    shared_ptr,
    gate_ptr,
    hidden: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    token = offsets // hidden
    routed = tl.load(routed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + token, mask=mask, other=0.0).to(tl.float32)
    out = routed + shared * tl.sigmoid(gate)
    tl.store(routed_ptr + offsets, out, mask=mask)


def fused_shared_expert_add_(
    routed: torch.Tensor, shared: torch.Tensor, gate_logits: torch.Tensor
) -> torch.Tensor:
    """In-place ``routed += sigmoid(gate_logits) * shared`` in one launch.

    Qwen3.5 previously materialized sigmoid, multiply, and add as separate
    pointwise launches for every MoE layer. The routed tensor is dead after the
    merge, so using it as the output also avoids a fourth full-hidden allocation.
    """
    assert routed.is_cuda and shared.is_cuda and gate_logits.is_cuda
    assert routed.shape == shared.shape and routed.dim() == 2
    assert gate_logits.numel() == routed.shape[0]
    assert routed.is_contiguous() and shared.is_contiguous() and gate_logits.is_contiguous()
    n_elements = routed.numel()
    _shared_expert_add_kernel[(triton.cdiv(n_elements, 256),)](
        routed,
        shared,
        gate_logits,
        hidden=routed.shape[1],
        n_elements=n_elements,
        BLOCK=256,
        num_warps=4,
    )
    return routed


@triton.jit
def _shared_route_reduce_kernel(
    routes_ptr,
    routed_weights_ptr,
    gate_ptr,
    out_ptr,
    hidden: tl.constexpr,
    routed_top_k: tl.constexpr,
    total_top_k: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    token = offsets // hidden
    feature = offsets - token * hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for route in tl.static_range(routed_top_k):
        value = tl.load(
            routes_ptr + (token * total_top_k + route) * hidden + feature,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            routed_weights_ptr + token * routed_top_k + route,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += value * weight
    shared = tl.load(
        routes_ptr + (token * total_top_k + routed_top_k) * hidden + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(gate_ptr + token, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, acc + shared * tl.sigmoid(gate), mask=mask)


def fused_shared_route_reduce(
    routes: torch.Tensor, routed_weights: torch.Tensor, gate_logits: torch.Tensor
) -> torch.Tensor:
    """Reduce top-k routed rows and the final gated shared-expert row."""
    tokens, routed_top_k = routed_weights.shape
    total_top_k = routed_top_k + 1
    assert routes.dim() == 2 and routes.shape[0] == tokens * total_top_k
    assert gate_logits.numel() == tokens
    hidden = routes.shape[1]
    out = torch.empty((tokens, hidden), dtype=routes.dtype, device=routes.device)
    n_elements = out.numel()
    _shared_route_reduce_kernel[(triton.cdiv(n_elements, 256),)](
        routes,
        routed_weights,
        gate_logits,
        out,
        hidden=hidden,
        routed_top_k=routed_top_k,
        total_top_k=total_top_k,
        n_elements=n_elements,
        BLOCK=256,
        num_warps=4,
    )
    return out


__all__ = ["fused_shared_expert_add_", "fused_shared_route_reduce"]
