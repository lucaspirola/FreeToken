"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    return sum(t[0][0].numel() * t[0].element_size() for t in sources.values())


def expert_slot_signatures(
    sources: dict[str, "list[torch.Tensor]"],
) -> tuple[tuple[int, ...], ...]:
    """Per-layer expert row bytes, retaining bank boundaries for size classes."""
    if not sources:
        return ()
    banks = tuple(sources.values())
    num_layers = len(banks[0])
    assert all(len(bank) == num_layers for bank in banks)
    return tuple(
        tuple(bank[layer][0].numel() * bank[layer].element_size() for bank in banks)
        for layer in range(num_layers)
    )


def expert_cache_bytes(
    cache_size: int,
    *,
    slot_signatures: tuple[tuple[int, ...], ...] | None,
    num_experts: int,
    prefill_overlap: bool,
    fallback_per_expert_bytes: int,
) -> int:
    """Exact GPU bytes for uniform or mixed-size expert slot caches.

    Mixed GGUF reserves a legacy-width two-layer prefill buffer, then distributes
    decode capacity across compact signature classes in proportion to layer count.
    Keep this arithmetic identical to ``OffloadMoeCache._set_gguf_size_class_sources``
    so auto-sizing can reinvest its savings instead of stranding them as free VRAM.
    """
    signatures = tuple(slot_signatures or ())
    unique = tuple(dict.fromkeys(signatures))
    if len(unique) <= 1:
        return cache_size * fallback_per_expert_bytes

    reserve = 2 * num_experts if prefill_overlap else 0
    usable = cache_size - reserve
    if usable < len(unique) * num_experts:
        raise ValueError("mixed-size cache is below its per-class decode floor")
    counts = [signatures.count(signature) for signature in unique]
    remaining = usable - len(unique) * num_experts
    capacities = [
        num_experts + remaining * count // len(signatures) for count in counts
    ]
    for class_id in range(usable - sum(capacities)):
        capacities[class_id % len(capacities)] += 1
    max_slot_bytes = sum(max(signature[i] for signature in unique) for i in range(len(unique[0])))
    return reserve * max_slot_bytes + sum(
        capacity * sum(unique[class_id])
        for class_id, capacity in enumerate(capacities)
    )


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
    expert_slot_signatures: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    unique_signatures = tuple(dict.fromkeys(expert_slot_signatures or ()))
    if len(unique_signatures) > 1:
        reserve = 2 * num_experts if overlap else 0
        lo = reserve + len(unique_signatures) * num_experts
    else:
        lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    cache_bytes = lambda size: expert_cache_bytes(
        size,
        slot_signatures=expert_slot_signatures,
        num_experts=num_experts,
        prefill_overlap=overlap,
        fallback_per_expert_bytes=per_expert_bytes,
    )
    # MoE-priority: reserve KV first, then experts greedily take the remaining
    # budget. Mixed-size GGUF is piecewise-linear, so solve it exactly by count.
    available_for_experts = budget_bytes - kv_reserve_bytes
    if expert_slot_signatures and len(unique_signatures) > 1:
        low, high = lo, hi
        while low < high:
            mid = (low + high + 1) // 2
            if cache_bytes(mid) <= available_for_experts:
                low = mid
            else:
                high = mid - 1
        moe_cache_size = low
    else:
        raw = available_for_experts // per_expert_bytes
        moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    moe_bytes = cache_bytes(moe_cache_size)
    remaining = budget_bytes - moe_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    quant_format: str,
    expert_slot_signatures: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = 992 if quant_format == "nvfp4_marlin" else total_experts
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
        expert_slot_signatures=expert_slot_signatures,
    )
