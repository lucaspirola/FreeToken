"""Shared-expert epilogues compile one kernel for every prompt length and stay byte-identical to
the per-length build (a constexpr token count compiled a new kernel -- 180-280 ms of stalled
prefill -- for each new length, i.e. on every file-read extend)."""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@triton.jit
def _per_length_add(routed_ptr, shared_ptr, gate_ptr, hidden: tl.constexpr, n_elements: tl.constexpr,
                    BLOCK: tl.constexpr):
    # the previous kernel, verbatim but for the name: n_elements specialised per length
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    token = offsets // hidden
    routed = tl.load(routed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + token, mask=mask, other=0.0).to(tl.float32)
    out = routed + shared * tl.sigmoid(gate)
    tl.store(routed_ptr + offsets, out, mask=mask)


def _variants(kernel) -> int:
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        return sum(len(v[0]) for v in caches.values())
    return sum(len(v) for v in kernel.cache.values())


def test_shared_expert_add_one_build_for_every_length_and_bit_exact():
    from freetoken.kernel.triton.shared_expert import _shared_expert_add_kernel, fused_shared_expert_add_

    hidden = 2048
    counts = []
    for tokens in (1, 7, 300, 1000, 4097, 12345):
        g = torch.Generator(device="cuda").manual_seed(tokens)
        routed = torch.randn(tokens, hidden, device="cuda", generator=g).to(torch.bfloat16)
        shared = torch.randn(tokens, hidden, device="cuda", generator=g).to(torch.bfloat16)
        gate = torch.randn(tokens, 1, device="cuda", generator=g).to(torch.bfloat16)
        ref = routed.clone()
        n = ref.numel()
        _per_length_add[(triton.cdiv(n, 256),)](ref, shared, gate, hidden=hidden, n_elements=n, BLOCK=256,
                                               num_warps=4)
        out = fused_shared_expert_add_(routed.clone(), shared, gate)
        assert torch.equal(out, ref), tokens
        counts.append(_variants(_shared_expert_add_kernel))
    # tokens * hidden is a multiple of 16 for every length: one build serves them all
    assert counts[-1] == counts[0] == 1, counts


def test_shared_route_reduce_one_build_for_every_length():
    from freetoken.kernel.triton.shared_expert import _shared_route_reduce_kernel, fused_shared_route_reduce

    hidden, top_k = 1024, 8
    counts = []
    for tokens in (1, 300, 1000, 2049):
        g = torch.Generator(device="cuda").manual_seed(tokens)
        routes = torch.randn(tokens * (top_k + 1), hidden, device="cuda", generator=g).to(torch.bfloat16)
        weights = torch.rand(tokens, top_k, device="cuda", generator=g)
        gate = torch.randn(tokens, device="cuda", generator=g).to(torch.bfloat16)
        r = routes.view(tokens, top_k + 1, hidden).float()
        ref = (r[:, :top_k] * weights[:, :, None]).sum(1) + r[:, top_k] * torch.sigmoid(gate.float())[:, None]
        out = fused_shared_route_reduce(routes, weights, gate)
        torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=1e-2)
        counts.append(_variants(_shared_route_reduce_kernel))
    assert counts[-1] == counts[0] == 1, counts
