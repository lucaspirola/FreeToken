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

``writeback_buffer_occupants`` is the same writeback decision (retained
duplicate needs no copy, sole copy pops the free stack) applied to the
prefill double buffer's own slots instead of a decode admission's. It shares
the free stack and the fault counters with ``_resolve_swaps_kernel`` above,
but nothing is being admitted, so there is no retention choice and no
``freed_rows`` staging: a vacated slot is simply empty until decode or the
next prefill fill claims it.
"""
from __future__ import annotations

import triton
import triton.language as tl

from freetoken.moe.mirror_stats import MirrorStat


# Copy spaces for ``fast_index_copy_kinds_jit`` descriptors (see resolve_swaps).
KIND_CACHE = 0   # GPU slot cache
KIND_POOL = 1    # pinned host pool (mapped)
KIND_STAGE = 2   # VRAM writeback staging ring


def resolve_swaps(cache, layer_id: int) -> None:
    """Translate this step's misses into mirror copy descriptors (device-side).

    Two descriptor groups, each one ``fast_index_copy_kinds_jit`` launch:

    * ``g1`` -- everything that must read a slot BEFORE admissions overwrite
      slots: victim writebacks (slot -> staging ring, or slot -> pool when the
      ring is full or disabled) and slot -> slot relocations.
    * ``g2`` -- admissions: pool -> slot, or staging ring -> slot when the
      admitted expert's pool row is still waiting for its DMA.
    """
    m = cache.residency._mirror
    plan = m["g1"].shape[1]
    # Rows the free stack must keep for this launch's writebacks; retention
    # stops above it. See the kernel's `retain_floor`.
    retain_floor = cache.residency._mirror_pool.reserve_rows
    # Slots below this are the prefill double buffer's own region: an
    # admission landing there is forced to retain its source row (see
    # `buffer_slots` in the kernel) so that region's own eviction, in
    # `_writeback_buffer_kernel`, is always free. 0 when there is no mirror
    # overlap buffer, which makes the forced-retention branch unreachable.
    buffer_slots = cache.residency._mirror_prefill_base()
    _resolve_swaps_kernel[(1,)](
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        cache.victim_ids,
        cache.prior_ids,
        m["prev_slot_of_id"],
        m["pool_row_of_id"],
        m["id_of_pool_row"],
        m["free_rows"],
        m["free_count"],
        m["freed_rows"],
        m["n_freed"],
        m["g1"],
        m["n_g1"],
        m["g2"],
        m["n_g2"],
        m["wb_state"],
        m["wb_pend"],
        m["stats"],
        layer_id,
        cache.num_experts,
        retain_floor,
        buffer_slots,
        plan,
        SWAPS=MirrorStat.SWAPS,
        FREE_EVICTIONS=MirrorStat.FREE_EVICTIONS,
        WRITEBACKS=MirrorStat.WRITEBACKS,
        VIOLATIONS=MirrorStat.VIOLATIONS,
        STARVED=MirrorStat.STARVED,
        RETAINED=MirrorStat.RETAINED,
        STAGED=MirrorStat.STAGED_WRITEBACKS,
        REDIRECTS=MirrorStat.STAGE_REDIRECTS,
        STAGE_ROWS=m["wb_stage_rows"],
        K_CACHE=KIND_CACHE,
        K_POOL=KIND_POOL,
        K_STAGE=KIND_STAGE,
    )


def publish_freed_rows(cache) -> None:
    """Fold rows freed this step into the free stack (after the copies)."""
    m = cache.residency._mirror
    _publish_freed_kernel[(1,)](
        m["freed_rows"],
        m["n_freed"],
        m["free_rows"],
        m["free_count"],
        BLOCK=triton.next_power_of_2(max(m["freed_rows"].numel(), 1)),
    )


@triton.jit
def _emit(desc_ptr, cap, n, dst, src, dst_kind, src_kind):
    """Append one (dst row, src row, dst space, src space) copy descriptor."""
    tl.store(desc_ptr + n, dst)
    tl.store(desc_ptr + cap + n, src)
    tl.store(desc_ptr + 2 * cap + n, dst_kind)
    tl.store(desc_ptr + 3 * cap + n, src_kind)


@triton.jit(do_not_specialize=["layer_id", "num_experts", "retain_floor", "buffer_slots", "cap"])
def _resolve_swaps_kernel(
    src_indices_ptr,      # int32 [plan]  layer-local expert id of each miss
    evict_slots_ptr,      # int32 [plan]  GPU slot each miss lands in
    num_indices_ptr,      # int64 [1]     miss count
    victim_ids_ptr,       # int32 [plan]  expert losing its GPU copy, -1 if none
    prior_ids_ptr,        # int32 [plan]  slot's owner before this step, -1 if empty
    prev_slot_ptr,        # int32 [L*E]   slot of each expert BEFORE this step
    pool_row_of_id_ptr,   # int32 [L*E]   pool row holding each expert, or -1
    id_of_pool_row_ptr,   # int32 [cap]   inverse of the above
    free_rows_ptr,        # int32 [cap]   stack of unowned pool rows
    free_count_ptr,       # int32 [1]     stack depth
    freed_rows_ptr,       # int32 [plan]  rows freed this step (published later)
    n_freed_ptr,          # int32 [1]
    g1_ptr,               # int32 [4, plan] writebacks + relocations (read slots
                          #               before any admission writes one)
    n_g1_ptr,             # int64 [1]
    g2_ptr,               # int32 [4, plan] admissions
    n_g2_ptr,             # int64 [1]
    wb_state_ptr,         # int64 [2 + STAGE_ROWS]: [0] ring head (entries ever
                          #               staged), [1] entries whose DMA has
                          #               completed (written by the host's copy
                          #               stream), [2 + s] pool row of ring slot s
    wb_pend_ptr,          # int64 [pool cap] ring index of the LATEST staged
                          #               writeback into each pool row, -1 if none
    stats_ptr,            # int64 [9]     MirrorStat layout
    layer_id,
    num_experts,
    retain_floor,         # keep this many rows free; retain duplicates above it
    buffer_slots,         # slots < this are the prefill buffer region; an
                          # admission there always retains (see below)
    cap,                  # descriptor capacity (plan)
    SWAPS: tl.constexpr,          # stats_ptr offsets, from MirrorStat
    FREE_EVICTIONS: tl.constexpr,
    WRITEBACKS: tl.constexpr,
    VIOLATIONS: tl.constexpr,
    STARVED: tl.constexpr,
    RETAINED: tl.constexpr,
    STAGED: tl.constexpr,
    REDIRECTS: tl.constexpr,
    STAGE_ROWS: tl.constexpr,     # staging ring rows; 0 = every writeback is an
                                  # SM store into the pool (the pre-DMA path)
    K_CACHE: tl.constexpr,
    K_POOL: tl.constexpr,
    K_STAGE: tl.constexpr,
):
    """One program: the miss count is <= top_k * batch (decode) or num_experts
    (prefill materialize), small either way. Serial within the program so the
    free stack and the ownership updates need no atomics; correctness depends
    on that single-program shape.

    DMA writebacks. SM stores into mapped pinned memory run at 2-14 GB/s
    depending on the host (box numbers: 2.1-2.4 GB/s on ft-dev, 12.2-13.8 on
    ft-ck) against 22-29 GB/s for the copy engine, and a captured graph cannot
    hold a memcpy whose addresses the device decides. So a victim is copied
    slot -> staging ring (VRAM, fast) here, and the host later issues the ring
    -> pool DMA (``MirrorResidency.service_writebacks``). Every DECISION --
    which victims need a writeback, which row each lands in, retention, the
    free stack -- is exactly the pre-DMA one; only where the bytes travel
    through changes. Three rules keep the pool byte-exact:

    * A ring slot is reused only after its DMA completed (``head - done <
      STAGE_ROWS``); when the ring is full the writeback falls back to the SM
      store.
    * An admission whose pool row still has a pending DMA (``wb_pend >=
      done``) reads the ring slot instead of the stale row.
    * An SM-store fallback never targets a row with a pending DMA (that DMA,
      landing later, would overwrite it with older bytes): it takes the
      topmost free row with none pending. Two staged writebacks into one row
      are safe, the host issues DMAs in ring order.
    """
    n = tl.load(num_indices_ptr)
    n_g1 = 0
    n_g2 = 0
    n_freed = 0
    swaps = 0
    free_evict = 0
    violations = 0
    starved = 0
    retained = 0
    writebacks = 0
    staged = 0
    redirects = 0
    free_top = tl.load(free_count_ptr)
    head = tl.load(wb_state_ptr)
    # Read once: a stale (smaller) value is always the safe side of both tests.
    done = tl.load(wb_state_ptr + 1, volatile=True)

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
            # A prefill materialize reinstalls a whole layer; experts of that
            # layer are often already GPU-resident, just in a different slot.
            # Copying them slot -> slot on the device costs no PCIe traffic and,
            # crucially, needs no mirror row -- which is what made a full-layer
            # staging exhaust the pool ("223 rows needed, 169 available").
            here = tl.load(prev_slot_ptr + flat_new)
            src_row = tl.load(pool_row_of_id_ptr + flat_new)
            if here >= 0 and here != slot:
                # Group 1, BEFORE the admissions: a relocation source is a
                # slot that still holds its old expert, and group 2 is about
                # to overwrite exactly such slots. Issuing it afterwards read
                # admitted bytes instead (21K-token completions came out as
                # noise).
                _emit(g1_ptr, cap, n_g1, slot, here, K_CACHE, K_CACHE)
                n_g1 += 1
                swaps += 1
                free_evict += 1
                if victim >= 0:
                    if tl.load(pool_row_of_id_ptr + victim) < 0:
                        # The relocation overwrites `slot`, whose occupant has
                        # no mirror row -- its only copy dies here, and unlike
                        # the else-branch below this path emits no writeback.
                        # Unreachable from today's callers (materialize forces
                        # victim = -1, and a decode miss implies here < 0), but
                        # that is an invariant held by omission in two other
                        # files. Count it so the host check turns a future
                        # caller's mistake into a failure instead of silence.
                        starved += 1
            elif src_row < 0:
                # Coverage invariant broken: not on the GPU, not mirrored.
                violations += 1
            else:
                swaps += 1
                if STAGE_ROWS > 0:
                    pidx = tl.load(wb_pend_ptr + src_row)
                    if pidx >= done:
                        # The row's DMA has not landed: its bytes are still
                        # (only) in the ring.
                        _emit(g2_ptr, cap, n_g2, slot, pidx % STAGE_ROWS, K_CACHE, K_STAGE)
                        redirects += 1
                    else:
                        _emit(g2_ptr, cap, n_g2, slot, src_row, K_CACHE, K_POOL)
                else:
                    _emit(g2_ptr, cap, n_g2, slot, src_row, K_CACHE, K_POOL)
                n_g2 += 1
                writeback = False
                if victim >= 0:
                    if tl.load(pool_row_of_id_ptr + victim) < 0:
                        writeback = True
                if writeback:
                    if free_top > 0:
                        pick = free_top - 1
                        use_stage = False
                        if STAGE_ROWS > 0:
                            if head - done < STAGE_ROWS:
                                use_stage = True
                            else:
                                # Ring full: SM store, into a row no pending
                                # DMA will overwrite later.
                                pick = -1
                                k = free_top - 1
                                while k >= 0:
                                    if tl.load(wb_pend_ptr + tl.load(free_rows_ptr + k)) < done:
                                        pick = k
                                        k = -1
                                    else:
                                        k -= 1
                        if pick >= 0:
                            free_top -= 1
                            dst_row = tl.load(free_rows_ptr + pick)
                            if pick != free_top:
                                tl.store(free_rows_ptr + pick, tl.load(free_rows_ptr + free_top))
                                tl.store(free_rows_ptr + free_top, dst_row)
                            if STAGE_ROWS > 0:
                                if use_stage:
                                    s = head % STAGE_ROWS
                                    _emit(g1_ptr, cap, n_g1, s, slot, K_STAGE, K_CACHE)
                                    tl.store(wb_state_ptr + 2 + s, dst_row.to(tl.int64))
                                    tl.store(wb_pend_ptr + dst_row, head)
                                    head += 1
                                    staged += 1
                                else:
                                    _emit(g1_ptr, cap, n_g1, dst_row, slot, K_POOL, K_CACHE)
                            else:
                                _emit(g1_ptr, cap, n_g1, dst_row, slot, K_POOL, K_CACHE)
                            n_g1 += 1
                            writebacks += 1
                            tl.store(id_of_pool_row_ptr + dst_row, victim)
                            tl.store(pool_row_of_id_ptr + victim, dst_row)
                        else:
                            # Every free row still has a DMA in flight (the
                            # ring holds at most STAGE_ROWS of them, the
                            # reserve is several layers): no safe target.
                            starved += 1
                    else:
                        # Reserve exhausted: the victim's only copy would be
                        # lost. plan_capacity sizes against this; the host
                        # treats a nonzero count as a hard error.
                        starved += 1
                else:
                    # Victim mirrored already (duplicate) or slot never used.
                    free_evict += 1
                # What happens to the row this admission read from decides
                # whether pool capacity buys anything at all.
                #
                # Freeing it unconditionally (what shipped) means every expert
                # that reaches the GPU immediately loses its host copy, so its
                # next eviction must pay a D2H -- forever. Only the duplicates
                # seeded at warm start are ever free to evict, and they are
                # consumed once. Measured on Nemotron: free evictions were
                # 1.4% of swaps at 1800 rows and 16.9% with the WHOLE model
                # mirrored (2944 rows), i.e. host RAM bought nothing, which is
                # why the first capacity sweep came out flat.
                #
                # The row's contents are expert weights: read-only, never
                # written on the GPU. Keeping it therefore stays valid for as
                # long as the expert is resident, and turns that expert's next
                # eviction into a free one. An expert that just arrived from
                # the pool is also the likeliest to go back out, so it is the
                # right row to hold.
                #
                # Retention consumes free rows and nothing returns them, so it
                # must stop while the stack can still absorb this launch's
                # writebacks: `retain_floor` is the pool's reserve, which is
                # three layers' worth and bounds `plan` by construction.
                #
                # Below `buffer_slots`, retention is not optional: a slot
                # there is the prefill double buffer's, and MEASURED (Phase
                # 1 lever 1 record) freeing this row anyway made every
                # buffer-region eviction pay a 5.36 MiB D2H on the very next
                # prefill that reuses this half -- TTFT at 8K went from 0.68s
                # to 1.47s, all of it in pass 2 (after decode has populated
                # the buffer), none in pass 1 (still empty). Forcing
                # retention here means `_writeback_buffer_kernel` always
                # finds a duplicate and takes its free-evict branch instead.
                # This never competes with `retain_floor` for a scarce
                # resource -- it only WITHHOLDS a push onto the free stack,
                # which is always possible -- so it cannot itself starve;
                # what it can do is make the free stack drain faster for
                # everyone else, which is exactly what the existing
                # `starved` counter on the victim-writeback branch above
                # already watches for.
                if slot < buffer_slots or free_top > retain_floor:
                    retained += 1
                else:
                    # At the floor: the admission's old row falls free --
                    # staged, not pushed (hazard 2).
                    tl.store(pool_row_of_id_ptr + flat_new, -1)
                    tl.store(id_of_pool_row_ptr + src_row, -1)
                    tl.store(freed_rows_ptr + n_freed, src_row)
                    n_freed += 1

    tl.store(free_count_ptr, free_top)
    tl.store(n_freed_ptr, n_freed)
    tl.store(n_g1_ptr, n_g1.to(tl.int64))
    tl.store(n_g2_ptr, n_g2.to(tl.int64))
    tl.store(wb_state_ptr, head)
    tl.store(stats_ptr + SWAPS, tl.load(stats_ptr + SWAPS) + swaps)
    tl.store(stats_ptr + FREE_EVICTIONS, tl.load(stats_ptr + FREE_EVICTIONS) + free_evict)
    tl.store(stats_ptr + WRITEBACKS, tl.load(stats_ptr + WRITEBACKS) + writebacks)
    tl.store(stats_ptr + VIOLATIONS, tl.load(stats_ptr + VIOLATIONS) + violations)
    tl.store(stats_ptr + STARVED, tl.load(stats_ptr + STARVED) + starved)
    tl.store(stats_ptr + RETAINED, tl.load(stats_ptr + RETAINED) + retained)
    tl.store(stats_ptr + STAGED, tl.load(stats_ptr + STAGED) + staged)
    tl.store(stats_ptr + REDIRECTS, tl.load(stats_ptr + REDIRECTS) + redirects)


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
    # Clearing the staging is not optional: publish is idempotent ONLY because
    # of this line. Without it a publish that is not preceded by a resolve
    # republishes the previous step's freed rows, putting the same pool row on
    # the free stack twice -- two experts would then own one row and the model
    # would serve the wrong weights with every fault counter still at zero.
    tl.store(n_freed_ptr, 0)


def writeback_buffer_occupants(cache, slot_start: int, count: int) -> None:
    """Write back a prefill buffer half's current occupants (device-side).

    Decode is no longer fenced out of the buffer region (the victim floor
    that did that is gone -- see ``_sync_layer_slot_bounds``), so a slot here
    may hold a real decode resident instead of the permanent empty the buffer
    used to guarantee. Called from ``_invalidate_prefill_buffer`` right before
    the fill overwrites these slots' bytes: a retained duplicate
    (``pool_row_of_id >= 0`` already) needs no copy, and a sole GPU copy is
    written into a free-stack row exactly as a decode eviction would do it.
    One launch per buffer half, not a Python loop over its slots.
    """
    m = cache.residency._mirror
    wb = cache.residency._mirror_writeback
    _writeback_buffer_kernel[(1,)](
        cache.id_of_slot,
        cache.slot_for_id.view(-1),
        m["pool_row_of_id"],
        m["id_of_pool_row"],
        m["free_rows"],
        m["free_count"],
        wb["d2h_src"],
        wb["d2h_dst"],
        wb["n_d2h"],
        wb["ids"],
        wb["rows"],
        wb["n_wb"],
        wb["vacated"],
        wb["n_vacated"],
        m["stats"],
        slot_start,
        count,
        WRITEBACKS=MirrorStat.WRITEBACKS,
        STARVED=MirrorStat.STARVED,
        BUFFER_FREE_EVICTIONS=MirrorStat.BUFFER_FREE_EVICTIONS,
        BLOCK=triton.next_power_of_2(max(count, 1)),
    )


@triton.jit(do_not_specialize=["slot_start", "count"])
def _writeback_buffer_kernel(
    id_of_slot_ptr,        # int32 [cap]  slot -> flat expert id, or -1
    slot_for_id_ptr,       # int32 [L*E]  flat expert id -> slot, or -1
    pool_row_of_id_ptr,    # int32 [L*E]  pool row holding each expert, or -1
    id_of_pool_row_ptr,    # int32 [cap]  inverse of the above
    free_rows_ptr,         # int32 [cap]  stack of unowned pool rows
    free_count_ptr,        # int32 [1]    stack depth
    d2h_src_ptr,           # int32 [count] GPU slot of each occupant written back
    d2h_dst_ptr,           # int32 [count] pool row receiving it
    n_d2h_ptr,             # int64 [1]    writeback count
    wb_id_ptr,             # int32 [count] flat expert id of each writeback
    wb_row_ptr,            # int32 [count] pool row it landed in (== d2h_dst)
    n_wb_ptr,              # int64 [1]
    vacated_ptr,           # int32 [count] flat expert id of every occupant found
                           #               here, written back or not
    n_vacated_ptr,         # int64 [1]
    stats_ptr,             # int64 [7]    shared with resolve_swaps: swaps,
                           #              free_evict, d2h, violations, starved,
                           #              retained, buffer_free_evict. This
                           #              kernel bumps 2, 4 and 6 only.
    slot_start,            # first GPU slot of this buffer half
    count,                 # slots in this buffer half (== num_experts)
    WRITEBACKS: tl.constexpr,          # stats_ptr offsets, from MirrorStat
    STARVED: tl.constexpr,
    BUFFER_FREE_EVICTIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program, serial: ``count`` is one expert layer's worth (<= a few
    thousand), small enough that no atomics are needed for the free stack or
    the ownership maps -- the same shape argument ``_resolve_swaps_kernel``
    relies on.

    Unlike a decode eviction, nothing here is a swap: the slot is not being
    handed to a new admission, only vacated, so there is no retention
    decision and no ``freed_rows`` staging (hazard 2 in this module's
    docstring does not apply -- no row is pushed back to the stack this
    launch, only popped).

    ``vacated_ptr`` exists because ``_prefetch_split_mirror`` classifies a
    layer's experts from a per-chunk HOST snapshot of ``slot_for_id``, not
    the live device tensor -- and a decode resident can now sit in a buffer
    slot that a LATER layer this same chunk still expects to read as a hit
    from that exact slot. Reporting every occupant this call vacates, not
    only the ones written back, is what lets the caller patch that snapshot
    to -1 for all of them, so a later classification never reads a slot this
    call has already invalidated.
    """
    free_top = tl.load(free_count_ptr)
    n_d2h = 0
    n_wb = 0
    n_vacated = 0
    free_evict = 0
    starved = 0

    for i in range(0, count):
        slot = slot_start + i
        old = tl.load(id_of_slot_ptr + slot)
        if old >= 0:
            row = tl.load(pool_row_of_id_ptr + old)
            if row >= 0:
                # Retained duplicate: the pool already has this expert's
                # bytes, so evicting the GPU copy is free (measured: ~28% of
                # buffer-region occupants at the reserve this design was
                # priced against).
                free_evict += 1
            elif free_top > 0:
                free_top -= 1
                dst_row = tl.load(free_rows_ptr + free_top)
                tl.store(d2h_src_ptr + n_d2h, slot)
                tl.store(d2h_dst_ptr + n_d2h, dst_row)
                n_d2h += 1
                tl.store(id_of_pool_row_ptr + dst_row, old)
                tl.store(pool_row_of_id_ptr + old, dst_row)
                tl.store(wb_id_ptr + n_wb, old)
                tl.store(wb_row_ptr + n_wb, dst_row)
                n_wb += 1
            else:
                # Reserve exhausted: this occupant's only copy is about to be
                # destroyed by the fill. Identical to resolve_swaps's starved
                # branch -- counted, not corrected, because the host check
                # (mirror_fault_check) is what turns a nonzero count into a
                # hard failure instead of silently wrong experts.
                starved += 1
            # Vacate ownership either way: the slot is about to hold a
            # different (or no) expert once the fill runs, and an admission
            # elsewhere must see this expert as no longer GPU-resident.
            tl.store(slot_for_id_ptr + old, -1)
            tl.store(id_of_slot_ptr + slot, -1)
            tl.store(vacated_ptr + n_vacated, old)
            n_vacated += 1

    tl.store(free_count_ptr, free_top)
    tl.store(n_d2h_ptr, n_d2h)
    tl.store(n_wb_ptr, n_wb)
    tl.store(n_vacated_ptr, n_vacated)
    # BUFFER_FREE_EVICTIONS, NOT FREE_EVICTIONS: these evictions come from
    # invalidating a prefill buffer half, not from a decode admission, so
    # folding them into FREE_EVICTIONS (whose denominator is `swaps`, bumped
    # only by _resolve_swaps_kernel) makes free_eviction_rate exceed 1.0 and
    # overstates what lever 1 bought.
    tl.store(stats_ptr + BUFFER_FREE_EVICTIONS, tl.load(stats_ptr + BUFFER_FREE_EVICTIONS) + free_evict)
    tl.store(stats_ptr + WRITEBACKS, tl.load(stats_ptr + WRITEBACKS) + n_d2h)
    tl.store(stats_ptr + STARVED, tl.load(stats_ptr + STARVED) + starved)
