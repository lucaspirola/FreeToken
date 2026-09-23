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


def gguf_class_capacities(
    usable: int, layer_counts: "list[int]", num_layers: int, num_experts: int
) -> "list[int]":
    """Per-class decode-slot capacities for ``usable`` slots total, distributed in
    proportion to each class's layer count (``layer_counts``, same order as the
    classes), remainder handed out round-robin, each floored at ``num_experts``.

    This is the exact arithmetic ``OffloadMoeCache._set_gguf_size_class_sources``
    uses to split ``cache_size`` (today) across signature classes; it is factored
    out here so the same split can be applied to other joint totals -- notably the
    expert arena's ``slot_capacity`` ceiling (S12b) -- without the two call sites
    drifting apart. ``sum(result) == usable`` always (assuming
    ``usable >= len(layer_counts) * num_experts``, checked by callers).
    """
    remaining = usable - len(layer_counts) * num_experts
    capacities = [
        num_experts + remaining * count // num_layers for count in layer_counts
    ]
    for i in range(usable - sum(capacities)):
        capacities[i % len(capacities)] += 1
    return capacities


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
    capacities = gguf_class_capacities(usable, counts, len(signatures), num_experts)
    max_slot_bytes = sum(max(signature[i] for signature in unique) for i in range(len(unique[0])))
    return reserve * max_slot_bytes + sum(
        capacity * sum(unique[class_id])
        for class_id, capacity in enumerate(capacities)
    )


def _arena_chunk_boundaries(capacity: int, step_slots: int) -> tuple[int, ...]:
    """Usable-slot values landing exactly on a commit-chunk boundary, ascending:
    ``[0, step_slots, 2*step_slots, ..., capacity]`` (final chunk partial if ``capacity``
    is not a multiple of ``step_slots``). These are the only ``usable_slots`` values a
    real commit/uncommit ladder can sit at (``uncommit_ranges`` must exactly match a
    prior ``commit_ranges`` mapping -- see ``kernel/csrc/vmm_tensor.cpp``)."""
    assert step_slots > 0
    boundaries = [0]
    while boundaries[-1] < capacity:
        boundaries.append(min(boundaries[-1] + step_slots, capacity))
    return tuple(boundaries)


def arena_bytes_for_usable(
    usable_slots: int,
    capacity: int,
    step_slots: int,
    bank_row_bytes: "list[int] | tuple[int, ...]",
    granule: int = 2 * 1024 * 1024,
) -> int:
    """Bytes actually mapped across all banks when ``usable_slots`` of ``capacity`` are
    usable, given a ``step_slots``-slot commit ladder (last chunk possibly partial).

    The ladder only commits whole chunks, so ``usable_slots`` rounds up to the next
    chunk boundary (capped at ``capacity``); each bank's mapped bytes for that boundary
    is ``round_up(boundary * row_bytes, granule)`` -- a cumulative granule-aligned
    partition of the linear ``slot * row_bytes`` layout, so there is never any padding
    between chunks and plain slot indexing stays exact. Bytes are a step function of
    ``usable_slots``, not linear, since the rounding remainder differs per bank.

    ``bank_row_bytes``: one entry per independent commit-tracked allocation (per
    ``(bank, layer)`` pair, as allocated by ``OffloadMoeCache._alloc_device_bank_cache``),
    each entry being that allocation's per-slot row byte count. Flat across banks and
    layers, in no particular grouping.
    """
    assert 0 <= usable_slots <= capacity
    assert step_slots > 0
    if usable_slots == 0:
        return 0
    boundary = min(div_ceil(usable_slots, step_slots) * step_slots, capacity)
    return sum(div_ceil(boundary * row_bytes, granule) * granule for row_bytes in bank_row_bytes)


def usable_for_target_free_bytes(
    target_free_bytes: int,
    capacity: int,
    step_slots: int,
    bank_row_bytes: "list[int] | tuple[int, ...]",
    granule: int = 2 * 1024 * 1024,
) -> int:
    """Inverse of :func:`arena_bytes_for_usable` for shrinking: the largest chunk-boundary
    ``usable`` (i.e. giving up the FEWEST slots) such that freeing everything above it
    (going from ``capacity`` usable down to ``usable``) releases at least
    ``target_free_bytes``.

    Only chunk-boundary values are considered because a real shrink can only uncommit
    whole chunks (``uncommit_ranges`` must exactly match a prior ``commit_ranges`` mapping;
    see ``kernel/csrc/vmm_tensor.cpp``). The planner must never promise more free slots
    than it can actually hand back, so this always returns a boundary that frees AT LEAST
    the target, never less -- when no boundary can free that much (the target exceeds
    everything the arena could ever release), it returns ``0`` (the minimum, i.e. shrink
    all the way down), which is the closest the arena can get.
    """
    total_bytes = arena_bytes_for_usable(capacity, capacity, step_slots, bank_row_bytes, granule)
    boundaries = _arena_chunk_boundaries(capacity, step_slots)
    best = 0
    for usable in boundaries:
        freed = total_bytes - arena_bytes_for_usable(usable, capacity, step_slots, bank_row_bytes, granule)
        if freed >= target_free_bytes:
            best = max(best, usable)
    return best


def joint_arena_boundaries(
    class_capacities: "list[int]", class_step_slots: "list[int]"
) -> tuple[int, ...]:
    """Valid joint ``usable`` values for S12b's one-arena-per-size-class layout,
    sorted ascending.

    Each class ``c`` owns a fixed, disjoint sub-range of the joint slot-id space:
    ``[begin_c, begin_c + class_capacities[c])`` with ``begin_c`` the cumulative sum
    of the earlier classes' capacities (ceilings). A single joint ``usable`` cutoff
    sweeps across that concatenated space top-to-bottom (mirroring the existing
    single-class ``usable_slots`` scalar the gated kernels already read via
    ``tl.load`` -- see ``offload_kernels._ensure_experts_sized_kernel_v2``'s
    ``off_c < usable`` mask): class ``c``'s own live usable count is
    ``clamp(usable - begin_c, 0, class_capacities[c])``. A commit/uncommit ladder
    can only land on one of class ``c``'s OWN chunk boundaries
    (``_arena_chunk_boundaries(class_capacities[c], class_step_slots[c])``), so the
    only joint values a real transition can target are ``begin_c + local_boundary``
    for some class and one of its local boundaries -- the union computed here.
    """
    assert len(class_capacities) == len(class_step_slots)
    begin = 0
    values = {0}
    for capacity, step in zip(class_capacities, class_step_slots):
        for local in _arena_chunk_boundaries(capacity, step):
            values.add(begin + local)
        begin += capacity
    values.add(begin)
    return tuple(sorted(values))


def joint_arena_bytes_for_usable(
    usable: int,
    class_capacities: "list[int]",
    class_step_slots: "list[int]",
    class_bank_row_bytes: "list[list[int]]",
    granule: int = 2 * 1024 * 1024,
) -> int:
    """Bytes mapped across EVERY size class's own arena at joint cutoff ``usable``
    (see :func:`joint_arena_boundaries`): the sum, over classes in order, of
    :func:`arena_bytes_for_usable` applied to that class's own
    ``clamp(usable - begin_c, 0, capacity_c)``.

    With exactly one class this reduces to ``arena_bytes_for_usable(usable,
    class_capacities[0], class_step_slots[0], class_bank_row_bytes[0])`` term for
    term (``begin_0 == 0``), i.e. today's single-class arena byte model,
    unchanged -- the N=1 regression this function must satisfy.
    """
    assert len(class_capacities) == len(class_step_slots) == len(class_bank_row_bytes)
    begin = 0
    total = 0
    for capacity, step, row_bytes in zip(
        class_capacities, class_step_slots, class_bank_row_bytes
    ):
        local = min(max(usable - begin, 0), capacity)
        total += arena_bytes_for_usable(local, capacity, step, row_bytes, granule)
        begin += capacity
    return total


def joint_arena_floor(
    class_capacities: "list[int]", class_floors: "list[int]"
) -> int:
    """Smallest joint ``usable`` at which EVERY class simultaneously meets its own
    floor (``>= num_experts``, or ``>= 2*num_experts`` with prefill overlap).

    The joint cutoff fills classes in order (class 0 first): class ``c`` sees any
    slots at all only once ``usable > begin_c``, and by the time class ``c+1``
    needs any coverage, ``usable >= begin_{c+1} = begin_c + capacity_c``, i.e.
    class ``c`` is already at its own ceiling (which is always ``>= its own
    floor`` by construction) -- so every class before the last is automatically
    covered once the last class is. The binding constraint is therefore the last
    class's own floor, reached at ``sum(class_capacities[:-1]) + class_floors[-1]``.
    """
    assert len(class_capacities) == len(class_floors)
    if not class_capacities:
        return 0
    return sum(class_capacities[:-1]) + class_floors[-1]


def plan_joint_arena_usable(
    budget_bytes: int,
    kv_bytes: int,
    class_capacities: "list[int]",
    class_step_slots: "list[int]",
    class_bank_row_bytes: "list[list[int]]",
    class_floors: "list[int]",
) -> int:
    """Largest joint ``usable`` (see :func:`joint_arena_boundaries`) whose arena
    bytes plus ``kv_bytes`` fit ``budget_bytes``, among values that meet every
    class's floor (:func:`joint_arena_floor`). MoE-priority, same shape as
    ``plan_cache_budget``'s single-class binary search: KV takes whatever the
    largest affordable expert footprint leaves.

    Raises ``ValueError`` if even the floor does not fit.
    """
    boundaries = joint_arena_boundaries(class_capacities, class_step_slots)
    floor = joint_arena_floor(class_capacities, class_floors)
    candidates = [b for b in boundaries if b >= floor]
    if not candidates:
        raise ValueError("no joint arena boundary meets every class floor")

    def bytes_at(n: int) -> int:
        return joint_arena_bytes_for_usable(
            n, class_capacities, class_step_slots, class_bank_row_bytes
        )

    if bytes_at(candidates[0]) + kv_bytes > budget_bytes:
        raise ValueError(
            f"joint arena floor {candidates[0]} needs "
            f"{bytes_at(candidates[0]) + kv_bytes} B > budget {budget_bytes} B"
        )
    lo, hi = 0, len(candidates) - 1
    best = candidates[0]
    while lo <= hi:
        mid = (lo + hi) // 2
        n = candidates[mid]
        if bytes_at(n) + kv_bytes <= budget_bytes:
            best = n
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def joint_usable_for_target_free_bytes(
    target_free_bytes: int,
    class_capacities: "list[int]",
    class_step_slots: "list[int]",
    class_bank_row_bytes: "list[list[int]]",
    granule: int = 2 * 1024 * 1024,
) -> int:
    """Multi-class counterpart of :func:`usable_for_target_free_bytes`: the
    largest joint boundary (:func:`joint_arena_boundaries`) such that freeing
    everything above it (shrinking from the full joint total down to it)
    releases at least ``target_free_bytes``. Same contract, same "closest the
    arena can get" fallback (0) when the target is unreachable -- see that
    function's docstring; this is exactly its shape run over
    :func:`joint_arena_bytes_for_usable` instead of :func:`arena_bytes_for_usable`.
    """
    boundaries = joint_arena_boundaries(class_capacities, class_step_slots)
    total = boundaries[-1]
    total_bytes = joint_arena_bytes_for_usable(
        total, class_capacities, class_step_slots, class_bank_row_bytes, granule
    )
    best = 0
    for usable in boundaries:
        freed = total_bytes - joint_arena_bytes_for_usable(
            usable, class_capacities, class_step_slots, class_bank_row_bytes, granule
        )
        if freed >= target_free_bytes:
            best = max(best, usable)
    return best


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
    quant_format: str = "",
    expert_slot_signatures: tuple[tuple[int, ...], ...] | None = None,
    max_slots: int | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    if max_slots is None and quant_format == "nvfp4_marlin":
        # marlin's fused MoE entry addresses at most 992 expert slots; upstream states
        # the same limit through ``method.slot_limit()``, which the fork's
        # moe/nvfp4_backends.py path does not bind (method is None there).
        max_slots = 992
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
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
