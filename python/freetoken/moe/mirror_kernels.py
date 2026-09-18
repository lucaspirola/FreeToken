"""Device-side swap resolution for the host expert mirror.

``ensure_experts`` leaves the step's misses in ``src_indices`` (layer-local
expert ids) and their target GPU slots in ``evict_slots``. With the whole model
pinned in RAM the H2D source row *is* the expert id, so the baseline feeds those
tensors straight to ``fast_index_copy_multi_jit``.

A bounded mirror breaks that identity: expert ``e`` lives at pool row
``pool_row_of_id[e]``, and the expert being displaced must be written back to
host RAM unless a duplicate already covers it. Both decisions are pure functions
of state the GPU already holds, so they are made here, on the device, in one
launch -- never round-tripping to Python. That is what keeps decode
CUDA-graph capturable, which is where the baseline's 176 tok/s comes from.

Three hazards, all found by measurement
---------------------------------------
1. **The victim cannot reuse the row its admission is vacating.** Both copies
   are issued back-to-back on one stream::

       D2H: gpu[slot] -> pool[r]      (writes r)
       H2D: pool[r]   -> gpu[slot]    (reads r)

   a read-after-write on ``r``: the upload would carry the victim's bytes.
   Reversing the order just moves the hazard onto the GPU slot. Writebacks
   therefore land on rows from a **free stack** of unowned rows.

2. **Rows freed this step cannot be recycled this step.** A later miss would pop
   the row an earlier admission is still uploading from. Observed with four
   misses: ``h2d_src=[0,1,2,3]`` against ``d2h_dst=[19,0,1,2]``, three experts
   landed holding a neighbour's weights. Freed rows go to ``freed_rows`` and are
   folded in by ``publish_freed_rows`` once every copy is issued.

3. **An expert already in its target slot needs no copy at all.** The prefill
   materialize schedules every expert of the layer, including GPU-resident ones
   that ``_mirror_stage_layer`` relocated into place. Those have no mirror row
   by design, so copying them would be both a spurious coverage fault and a
   wasted transfer. The kernel now emits a descriptor only for slots whose
   current owner differs from the expert being installed, which is why
   ``h2d_src``/``h2d_dst`` are compacted with their own count.
"""
from __future__ import annotations

import triton
import triton.language as tl


def resolve_swaps(cache, layer_id: int) -> None:
    """Translate this step's misses into mirror copy descriptors (device-side)."""
    m = cache._mirror
    plan = m["h2d_src"].numel()
    _resolve_swaps_kernel[(1,)](
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        cache.victim_ids,
        cache.prior_ids,
        m["pool_row_of_id"],
        m["id_of_pool_row"],
        m["free_rows"],
        m["free_count"],
        m["freed_rows"],
        m["n_freed"],
        m["h2d_src"],
        m["h2d_dst"],
        m["n_h2d"],
        m["d2h_src"],
        m["d2h_dst"],
        m["n_d2h"],
        m["stats"],
        layer_id,
        cache.num_experts,
        BLOCK=triton.next_power_of_2(max(plan, 1)),
    )


def publish_freed_rows(cache) -> None:
    """Fold rows freed this step into the free stack (after the copies)."""
    m = cache._mirror
    _publish_freed_kernel[(1,)](
        m["freed_rows"],
        m["n_freed"],
        m["free_rows"],
        m["free_count"],
        BLOCK=triton.next_power_of_2(max(m["freed_rows"].numel(), 1)),
    )


@triton.jit(do_not_specialize=["layer_id", "num_experts"])
def _resolve_swaps_kernel(
    src_indices_ptr,      # int32 [plan]  layer-local expert id of each miss
    evict_slots_ptr,      # int32 [plan]  GPU slot each miss lands in
    num_indices_ptr,      # int64 [1]     miss count
    victim_ids_ptr,       # int32 [plan]  expert losing its GPU copy, -1 if none
    prior_ids_ptr,        # int32 [plan]  slot's owner before this step, -1 if empty
    pool_row_of_id_ptr,   # int32 [L*E]   pool row holding each expert, or -1
    id_of_pool_row_ptr,   # int32 [cap]   inverse of the above
    free_rows_ptr,        # int32 [cap]   stack of unowned pool rows
    free_count_ptr,       # int32 [1]     stack depth
    freed_rows_ptr,       # int32 [plan]  rows freed this step (published later)
    n_freed_ptr,          # int32 [1]
    h2d_src_ptr,          # int32 [plan]  pool row feeding each admission
    h2d_dst_ptr,          # int32 [plan]  GPU slot receiving it
    n_h2d_ptr,            # int64 [1]     admission count (<= miss count)
    d2h_src_ptr,          # int32 [plan]  GPU slot of each victim needing writeback
    d2h_dst_ptr,          # int32 [plan]  pool row receiving it
    n_d2h_ptr,            # int64 [1]     writeback count
    stats_ptr,            # int64 [5]     swaps, free_evict, d2h, violations, starved
    layer_id,
    num_experts,
    BLOCK: tl.constexpr,
):
    """One program: the miss count is <= top_k * batch (decode) or num_experts
    (prefill materialize), small either way. Serial within the program so the
    free stack and the ownership updates need no atomics; correctness depends
    on that single-program shape.
    """
    n = tl.load(num_indices_ptr)
    n_h2d = 0
    n_d2h = 0
    n_freed = 0
    swaps = 0
    free_evict = 0
    violations = 0
    starved = 0
    free_top = tl.load(free_count_ptr)

    for i in range(0, n):
        expert = tl.load(src_indices_ptr + i)
        slot = tl.load(evict_slots_ptr + i)
        flat_new = layer_id * num_experts + expert
        # Hazard 3: the slot already holds this expert, so there is nothing to
        # copy. id_of_slot carries the NEW owner by the time this kernel runs,
        # so the pre-step owner comes from prior_ids, published by the LRU and
        # materialize kernels before they clobber it.
        prior = tl.load(prior_ids_ptr + i)
        victim = tl.load(victim_ids_ptr + i)
        if prior != flat_new:
            src_row = tl.load(pool_row_of_id_ptr + flat_new)
            if src_row < 0:
                # Coverage invariant broken: not on the GPU, not mirrored.
                violations += 1
            else:
                swaps += 1
                tl.store(h2d_src_ptr + n_h2d, src_row)
                tl.store(h2d_dst_ptr + n_h2d, slot)
                n_h2d += 1
                writeback = False
                if victim >= 0:
                    if tl.load(pool_row_of_id_ptr + victim) < 0:
                        writeback = True
                if writeback:
                    if free_top > 0:
                        free_top -= 1
                        dst_row = tl.load(free_rows_ptr + free_top)
                        tl.store(d2h_src_ptr + n_d2h, slot)
                        tl.store(d2h_dst_ptr + n_d2h, dst_row)
                        n_d2h += 1
                        tl.store(id_of_pool_row_ptr + dst_row, victim)
                        tl.store(pool_row_of_id_ptr + victim, dst_row)
                    else:
                        # Reserve exhausted: the victim's only copy would be
                        # lost. plan_capacity sizes against this; the host
                        # treats a nonzero count as a hard error.
                        starved += 1
                else:
                    # Victim mirrored already (duplicate) or slot never used.
                    free_evict += 1
                # The admission's old row falls free -- staged, not pushed
                # (hazard 2).
                tl.store(pool_row_of_id_ptr + flat_new, -1)
                tl.store(id_of_pool_row_ptr + src_row, -1)
                tl.store(freed_rows_ptr + n_freed, src_row)
                n_freed += 1

    tl.store(free_count_ptr, free_top)
    tl.store(n_freed_ptr, n_freed)
    tl.store(n_h2d_ptr, n_h2d)
    tl.store(n_d2h_ptr, n_d2h)
    tl.store(stats_ptr + 0, tl.load(stats_ptr + 0) + swaps)
    tl.store(stats_ptr + 1, tl.load(stats_ptr + 1) + free_evict)
    tl.store(stats_ptr + 2, tl.load(stats_ptr + 2) + n_d2h)
    tl.store(stats_ptr + 3, tl.load(stats_ptr + 3) + violations)
    tl.store(stats_ptr + 4, tl.load(stats_ptr + 4) + starved)


@triton.jit
def _publish_freed_kernel(
    freed_rows_ptr,
    n_freed_ptr,
    free_rows_ptr,
    free_count_ptr,
    BLOCK: tl.constexpr,
):
    """Append this step's freed rows to the free stack, then clear the staging."""
    n = tl.load(n_freed_ptr)
    top = tl.load(free_count_ptr)
    for i in range(0, n):
        tl.store(free_rows_ptr + top, tl.load(freed_rows_ptr + i))
        top += 1
    tl.store(free_count_ptr, top)
    tl.store(n_freed_ptr, 0)
