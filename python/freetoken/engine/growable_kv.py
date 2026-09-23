"""Growable KV: one session's KV cache grows and shrinks at runtime, funded from the
GPU expert cache (``--kv-grow-step-tokens``).

The KV pool keeps its virtual address range and commits/decommits physical pages in
``kv_grow_step_tokens`` steps; every step is paid for by the MoE offload cache. With the
expert arena (``EngineConfig.expert_arena``), the cache's usable slot count shrinks or
grows in place (``OffloadMoeCache.set_usable_slots``), so bank addresses never move and the
captured decode CUDA graphs stay valid across every resize.

The arena is the only mechanism. Refactor step S10 deleted the legacy path that funded a
step with ``OffloadMoeCache.rebuild`` and tore the decode graphs down for a second capture;
the engine refuses ``--kv-grow-step-tokens`` at startup for a cache without an arena, and
formats the arena does not serve (marlin/b12x tiled NVFP4, GGUF with mixed size classes)
have no growable KV until S12b re-implements them on it.

Refactor step S7 (``tasks/exclusive-expert-ram/reviews/2026-09-22-refactor-plan-final.md``)
MOVED these methods out of ``Engine`` WITHOUT editing their bodies (S10 then deleted the
legacy branches). The S7 rewrites were mechanical:

* ``self.moe_offload_cache`` -> ``self.moe``;
* ``self.<engine attribute>`` -> ``self.engine.<attribute>`` for the engine state the
  transaction reads or writes (``config``, ``device``, ``num_pages``, ``_pool_cls``,
  ``_baseline_free``, ``_weights_bytes``, ``linear_state_pool``, ``_growable_moe_ceiling``,
  ``_growable_moe_prefill_overlap``, ``_sync_get_memory``, ``sync_all_ranks``; also
  ``_pending_graph_bs`` and ``ensure_decode_graphs``, which S10 and S11 deleted with the
  legacy path and the retired resident-capacity resize).

``kv_cache``, ``graph_runner`` and ``attn_backend`` keep their spelling: they are
properties that read the engine's CURRENT object, because the engine may replace
``graph_runner`` and a stored reference would go stale. ``_growable_transition_failed`` (the poison flag a failed rollback sets)
is the controller's own state. ``Engine.grow_runtime_kv`` / ``shrink_runtime_kv`` are
one-line delegations, so the scheduler's call sites are unchanged.
"""

from __future__ import annotations

import math

import torch
from freetoken.kvcache.linear_state_pool import state_pool_bytes
from freetoken.utils import init_logger, mem_GB

logger = init_logger(__name__)

# Until S12b re-implements them on the arena, formats the expert arena does not serve have
# no growable KV. The engine raises this at startup; the controller re-checks it.
# Live VRAM every growable-KV commit must leave free on top of the pages it maps
# (WSL/DXG needs it for cuMemSetAccess; see _plan_growable_kv). One constant,
# because the runtime check, the ceiling plan and the mirror pool's sizing
# estimate (residency._mirror_final_gpu_slots) must all price the same cushion:
# the estimate once omitted it and sized Ornith's pool for an arena 128 slots
# bigger than the ceiling plan allowed -- the 250K request died on the commit.
VMM_COMMIT_CUSHION_BYTES = 256 * 1024 * 1024

GROWABLE_KV_UNSUPPORTED = (
    "growable KV unsupported for this format: --kv-grow-step-tokens funds KV from the "
    "expert arena (--expert-arena, alias FREETOKEN_EXPERT_ARENA=1), which this expert "
    "cache does not have. The rebuild-based fallback was removed (refactor step S10); "
    "marlin/b12x tiled NVFP4 and mixed-size-class GGUF experts regain growable KV when "
    "they move onto the arena (S12b)"
)


class GrowableKvController:
    """The growable-KV transaction: plan, fund, commit, roll back (see module docstring)."""

    def __init__(self, engine) -> None:
        self.engine = engine

    @property
    def kv_cache(self):
        return self.engine.kv_cache

    @property
    def moe(self):
        return self.engine.moe_offload_cache

    @property
    def graph_runner(self):
        return self.engine.graph_runner

    @property
    def attn_backend(self):
        return self.engine.attn_backend

    # ------------------------------------------------------------------
    # Moved verbatim from Engine (rewrites listed in the module docstring).
    # ------------------------------------------------------------------

    def _growable_moe_bytes(self, cache_size: int) -> int:
        """Exact GPU bytes of the expert arena at ``cache_size`` usable slots.

        Byte counts are NOT linear in ``cache_size``: shrinking releases whole 2 MiB
        granules per independent bank/layer allocation, and each allocation's row size
        rounds up to a different granule remainder (see
        ``freetoken.engine.cache_budget.arena_bytes_for_usable``).

        S12b: a mixed-GGUF cache exposes ``class_arena_layouts``/
        ``class_bank_row_bytes`` (one arena per size class) instead of the
        single-class ``arena_layout``/``bank_row_bytes``; ``cache_size`` is then
        the JOINT usable cutoff swept across every class's fixed range (see
        ``cache_budget.joint_arena_bytes_for_usable`` and
        ``OffloadMoeCache.set_class_usable_slots``). A uniform-signature cache
        never populates the class attributes, so this branch is a pure addition:
        the single-class path below is untouched (the N=1 case).
        """
        moe = self.moe
        assert moe is not None, "growable KV requires the MoE offload cache"
        class_layouts = getattr(moe, "class_arena_layouts", None)
        if class_layouts is not None:
            from freetoken.engine.cache_budget import joint_arena_bytes_for_usable

            capacities = [c for c, _ in class_layouts]
            steps = [s for _, s in class_layouts]
            row_bytes = moe.class_bank_row_bytes
            assert row_bytes is not None
            return joint_arena_bytes_for_usable(cache_size, capacities, steps, row_bytes)

        from freetoken.engine.cache_budget import arena_bytes_for_usable

        bank_row_bytes = getattr(moe, "bank_row_bytes", None)
        arena_layout = getattr(moe, "arena_layout", None)
        if bank_row_bytes is None or arena_layout is None:
            raise RuntimeError(GROWABLE_KV_UNSUPPORTED)
        capacity, step_slots = arena_layout
        return arena_bytes_for_usable(cache_size, capacity, step_slots, bank_row_bytes)

    def _arena_transition_bytes(self, cache_size: int) -> int:
        """Exact GPU bytes at ``cache_size`` usable slots, for the arena
        grow/shrink TRANSACTION methods only (``_grow_runtime_kv_arena``/
        ``_shrink_runtime_kv_arena``).

        This duplicates ``_growable_moe_bytes``'s dispatch (single-class
        ``arena_layout``/``bank_row_bytes`` vs. the S12b joint per-class model)
        under a DIFFERENT name on purpose: ``test_growable_kv_transaction.py``'s
        ``_controller`` helper stubs ``_growable_moe_bytes`` to the identity
        function for its own, unrelated planner-adjacent tests, and the
        transaction methods must keep computing real bytes regardless -- that
        is exactly what they read directly off ``cache_budget.arena_bytes_for_usable``
        before this method existed (no behavior change for the N=1 case).
        """
        moe = self.moe
        assert moe is not None
        class_layouts = getattr(moe, "class_arena_layouts", None)
        if class_layouts is not None:
            from freetoken.engine.cache_budget import joint_arena_bytes_for_usable

            capacities = [c for c, _ in class_layouts]
            steps = [s for _, s in class_layouts]
            row_bytes = moe.class_bank_row_bytes
            assert row_bytes is not None
            return joint_arena_bytes_for_usable(cache_size, capacities, steps, row_bytes)

        from freetoken.engine.cache_budget import arena_bytes_for_usable

        bank_row_bytes = moe.bank_row_bytes
        assert bank_row_bytes is not None
        capacity, step_slots = moe.arena_layout
        return arena_bytes_for_usable(cache_size, capacity, step_slots, bank_row_bytes)

    def _growable_moe_class_floor(self) -> "list[int] | None":
        """Per-class decode floor (``num_experts``, or ``2*num_experts`` with
        prefill overlap) for the joint planner's ``joint_arena_floor``/
        ``plan_joint_arena_usable`` (S12b). ``None`` off the multi-class arena."""
        moe = self.moe
        class_layouts = getattr(moe, "class_arena_layouts", None)
        if class_layouts is None:
            return None
        floor = 2 * moe.num_experts if moe.prefill_overlap else moe.num_experts
        return [floor for _ in class_layouts]

    def _growable_arena_boundaries(self) -> "list[int]":
        """Every usable-slot value the arena can be resized to, ascending.

        S12b: dispatches to the joint per-class boundaries
        (``cache_budget.joint_arena_boundaries``) when the cache is a mixed-GGUF
        class arena; otherwise the single-class chunk boundaries from
        ``arena_layout``'s ``(capacity, step_slots)`` -- the N=1 case, unchanged.
        """
        moe = self.moe
        assert moe is not None
        class_layouts = getattr(moe, "class_arena_layouts", None)
        if class_layouts is not None:
            from freetoken.engine.cache_budget import joint_arena_boundaries

            capacities = [c for c, _ in class_layouts]
            steps = [s for _, s in class_layouts]
            return list(joint_arena_boundaries(capacities, steps))

        from freetoken.engine.cache_budget import _arena_chunk_boundaries

        capacity, step_slots = moe.arena_layout
        return list(_arena_chunk_boundaries(capacity, step_slots))

    def _growable_usable_for_target_free_bytes(self, target_free_bytes: int) -> int:
        """Largest arena boundary that frees at least ``target_free_bytes`` when
        shrinking from full capacity down to it.

        S12b: dispatches to ``cache_budget.joint_usable_for_target_free_bytes``
        for a mixed-GGUF class arena, else ``cache_budget.usable_for_target_free_bytes``
        against ``arena_layout``/``bank_row_bytes`` -- the N=1 case, unchanged.
        """
        moe = self.moe
        assert moe is not None
        class_layouts = getattr(moe, "class_arena_layouts", None)
        if class_layouts is not None:
            from freetoken.engine.cache_budget import (
                joint_usable_for_target_free_bytes,
            )

            capacities = [c for c, _ in class_layouts]
            steps = [s for _, s in class_layouts]
            row_bytes = moe.class_bank_row_bytes
            assert row_bytes is not None
            return joint_usable_for_target_free_bytes(
                target_free_bytes, capacities, steps, row_bytes
            )

        from freetoken.engine.cache_budget import usable_for_target_free_bytes

        bank_row_bytes = moe.bank_row_bytes
        assert bank_row_bytes is not None
        capacity, step_slots = moe.arena_layout
        return usable_for_target_free_bytes(
            target_free_bytes, capacity, step_slots, bank_row_bytes
        )

    def _growable_arena_step_down(self, target_moe: int, floor: int) -> int:
        """Next lower usable-slot target for the live-memory top-up loop.

        Single-class: the exact original computation,
        ``max(target_moe - step_slots, floor)`` (``arena_layout``'s fixed step) --
        the N=1 case, unchanged. Class arena: chunk sizes are not uniform across
        the joint range, so step to the next lower boundary at or above ``floor``
        instead of subtracting a flat amount that could skip over, or land short
        of, a real boundary.
        """
        moe = self.moe
        assert moe is not None
        class_layouts = getattr(moe, "class_arena_layouts", None)
        if class_layouts is None:
            _capacity, step_slots = moe.arena_layout
            return max(target_moe - step_slots, floor)
        lower = [b for b in self._growable_arena_boundaries() if floor <= b < target_moe]
        return max(lower) if lower else floor

    def _plan_growable_kv(
        self,
        target_pages: int,
        *,
        extra_vmm_reserve_bytes: int = 0,
    ) -> tuple[int, int]:
        """Return the largest affordable MoE cache and exact mapped KV bytes."""
        pool = self.kv_cache
        moe = self.moe
        assert moe is not None, "growable KV requires the MoE offload cache"
        from freetoken.engine.cache_budget import (
            net_cache_budget_bytes,
        )

        _cache_per_page, fixed_cache_size, _page_tokens, _min_reserve = (
            self.engine._pool_cls.kv_cost(self.engine.config)
        )
        fixed_cache_size += state_pool_bytes(
            self.engine.config,
            (
                self.engine.linear_state_pool.num_slots
                if getattr(self.engine, "linear_state_pool", None) is not None
                else None
            ),
        )
        budget = net_cache_budget_bytes(
            self.engine.config.memory_ratio,
            self.engine._baseline_free,
            self.engine._weights_bytes,
            fixed_cache_size,
        )
        # A VMM growth step must make the new physical allocation resident before it
        # can expose the mapping.  In particular, WSL/DXG needs more live headroom
        # than the final pool-byte identity alone implies; with only memory_ratio's
        # nominal slack, cuMemSetAccess can fail even though MoE released exactly as
        # many bytes as KV is about to consume.  Keep a small, permanent commit
        # cushion instead of discovering that condition by poisoning the CUDA
        # context midway through a long prompt.
        budget -= VMM_COMMIT_CUSHION_BYTES + extra_vmm_reserve_bytes
        if budget <= 0:
            raise RuntimeError(
                "growable KV has no budget after its 256 MiB VMM safety margin"
            )
        kv_bytes = pool.mapped_bytes_for_pages(target_pages)

        # S12b: mixed-GGUF per-class arena. One joint step count across every
        # class's fixed range (see _growable_moe_bytes above), each class's own
        # floor checked simultaneously (cache_budget.joint_arena_floor) --
        # never a per-candidate prefill_overlap toggle, unlike the uniform-arena
        # search below, because a size class's prefill-buffer borrow is fixed at
        # construction (OffloadMoeCache._prefill_borrow_class), not re-derived
        # per candidate. A uniform-signature cache never takes this branch (the
        # N=1 case): it falls through to the untouched code below.
        class_layouts = getattr(moe, "class_arena_layouts", None)
        if class_layouts is not None:
            from freetoken.engine.cache_budget import plan_joint_arena_usable

            capacities = [c for c, _ in class_layouts]
            steps = [s for _, s in class_layouts]
            row_bytes = moe.class_bank_row_bytes
            assert row_bytes is not None
            floors = self._growable_moe_class_floor()
            target_moe = plan_joint_arena_usable(
                budget, kv_bytes, capacities, steps, row_bytes, floors
            )
            return target_moe, kv_bytes

        maximum = self.engine._growable_moe_ceiling
        desired_overlap = self.engine._growable_moe_prefill_overlap

        def overlap_at(size: int) -> bool:
            return desired_overlap and size >= 2 * moe.num_experts

        minimum = moe.num_experts
        while minimum <= maximum:
            current_overlap = moe.prefill_overlap
            try:
                moe.prefill_overlap = overlap_at(minimum)
                moe.validate_rebuild(minimum)
                break
            except ValueError:
                minimum += 1
            finally:
                moe.prefill_overlap = current_overlap
        if minimum > maximum:
            raise RuntimeError(
                f"MoE cache ceiling {maximum} has no valid growable-KV floor"
            )
        if self._growable_moe_bytes(minimum) + kv_bytes > budget:
            need = self._growable_moe_bytes(minimum) + kv_bytes
            raise RuntimeError(
                f"KV growth to {target_pages} tokens cannot fit even with the minimum "
                f"MoE cache ({minimum} slots): need {mem_GB(need)}, budget {mem_GB(budget)}"
            )

        lo, hi = minimum, maximum
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._growable_moe_bytes(mid) + kv_bytes <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo, kv_bytes

    def _rollback_growable_kv_transition(
        self,
        *,
        old_pages: int,
        old_moe: int,
        old_overlap: bool,
    ) -> None:
        """Best-effort rollback for an interrupted MoE/KV ownership transfer.

        Ordinary allocation failures are recoverable: MHA VMM commits are reversible and
        the expert arena's usable count moves back with ``set_usable_slots`` (bank
        addresses never moved, so the decode graphs are untouched). A CUDA context error is
        not recoverable; mark the engine poisoned and re-raise so the scheduler cannot
        resume against a half-built cache.
        """
        pool = self.kv_cache
        moe = self.moe
        assert moe is not None
        try:
            current_pages = int(pool.committed_pages)
            # When growth got as far as the KV commit, return those mappings before asking
            # the old, larger expert geometry to fit again.
            if current_pages > old_pages:
                pool.decommit_pages(old_pages)
            if moe.cache_size != old_moe:
                moe.prefill_overlap = old_overlap
                # The failed transition only ever called set_usable_slots; undo it the
                # same way.
                moe.set_usable_slots(old_moe)
            else:
                moe.prefill_overlap = old_overlap
            # Shrink frees KV before growing experts. Restore the (smaller) old expert cache
            # first, then recommit the exact recorded VMM suffix.
            if int(pool.committed_pages) < old_pages:
                pool.commit_pages(old_pages)
            object.__setattr__(self.engine.config, "moe_cache_size", old_moe)
        except Exception:
            self._growable_transition_failed = True
            logger.exception(
                "Growable KV rollback failed; engine is not safe to resume"
            )
            raise

    def _refuse_if_growable_transition_failed(self) -> None:
        if getattr(self, "_growable_transition_failed", False):
            raise RuntimeError(
                "growable KV/MoE rollback failed; engine restart is required"
            )

    def _grow_runtime_kv_arena(
        self,
        *,
        old_pages: int,
        target_pages: int,
        old_moe: int,
        old_overlap: bool,
        kv_bytes: int,
    ) -> tuple[int, int]:
        """``grow_runtime_kv``'s transaction (design step 5).

        Funds the KV commit by calling ``OffloadMoeCache.set_usable_slots``: bank
        buffer addresses never move, so the captured decode CUDA graphs stay valid
        and the engine's graph runner is never replaced -- asserted below.

        Caller's responsibility (documented, not enforced beyond the sync
        already done by the caller): this must run at a no-forward-in-flight
        scheduler boundary, because
        ``set_usable_slots`` mutates slot bookkeeping with plain (non-graph)
        ops on the current stream and a shrink physically unmaps pages.

        S12b: all arena byte/boundary math below goes through
        ``_growable_moe_bytes``/``_growable_usable_for_target_free_bytes``/
        ``_growable_arena_step_down``, which dispatch to the joint per-class
        model for a mixed-GGUF cache and to the untouched single-class formulas
        (``arena_layout``/``bank_row_bytes``) otherwise -- the N=1 case is byte-
        identical to before this method stopped taking ``arena_layout`` as a
        parameter (its old local ``capacity`` is now the arena's top boundary,
        ``self._growable_arena_boundaries()[-1]`` -- the same value
        ``arena_layout[0]`` gave for a single-class cache, no new attribute
        read).
        """
        pool = self.kv_cache
        moe = self.moe
        assert moe is not None
        floor = 2 * moe.num_experts if moe.prefill_overlap else moe.num_experts
        # A bounded host mirror puts a second, higher floor under the arena:
        # every expert the GPU drops must have a pool row outside the pool's
        # writeback/staging reserve. The pool derives that slot count itself
        # (min_gpu_slots); round it UP to chunk granularity, because a partial
        # chunk is not releasable and a floor below a chunk boundary would let
        # the shrink land under it.
        # Whole-model residency's min_gpu_slots() is 0, so its cov_floor is 0.
        mirror_pool = getattr(moe.residency, "pool", None)
        # No prefill_buffer_slots term: those slots are candidates for
        # decode residents now (_invalidate_prefill_buffer writes one
        # back before a prefill fill overwrites it), so they are no
        # longer dead space the floor must additionally protect --
        # min_gpu_slots alone already counts every resident the pool's
        # capacity guarantees coverage for.
        need = moe.residency.min_gpu_slots()
        # S12b: a class arena has no single ``step_slots`` (each class owns its
        # own fixed range); round to ``num_experts`` instead, matching the
        # uniform per-class chunk granularity ``_set_gguf_size_class_sources``
        # already builds each class's arena with. Single-class: unchanged,
        # rounds to the real ``arena_layout`` step.
        class_layouts = getattr(moe, "class_arena_layouts", None)
        cov_step = moe.num_experts if class_layouts is not None else moe.arena_layout[1]
        cov_floor = -(-need // cov_step) * cov_step
        floor = max(floor, cov_floor)
        runner_before = self.engine.graph_runner
        target_moe = old_moe
        # Assigned only by the shrink branch below; the ledger add at the
        # pre-commit log must stay valid on the no-shrink path too (that path
        # runs whenever free VRAM already covers the commit -- i.e. the whole-
        # model-in-RAM profile, which never shrinks the arena).
        released_bytes = 0
        try:
            commit_bytes = kv_bytes - pool.mapped_bytes_for_pages(old_pages)
            required_free = commit_bytes + VMM_COMMIT_CUSHION_BYTES
            desired_free = required_free + 128 * 1024 * 1024
            live_free_before = self.engine._sync_get_memory()[0]
            if live_free_before < desired_free:
                extra_needed = desired_free - live_free_before
                # usable_for_target_free_bytes prices a shrink FROM full capacity;
                # old_moe (the cache's current usable count) may already be below
                # capacity from an earlier growth step, so add back the bytes
                # already given up between capacity and old_moe (the "deficit")
                # to translate "extra_needed more, from here" into "this much,
                # from capacity" before calling it.
                capacity_bytes = self._arena_transition_bytes(
                    self._growable_arena_boundaries()[-1]
                )
                old_bytes = self._arena_transition_bytes(old_moe)
                deficit_from_capacity = capacity_bytes - old_bytes
                target_moe = self._growable_usable_for_target_free_bytes(
                    extra_needed + deficit_from_capacity
                )
                target_moe = max(target_moe, floor)
                if target_moe >= old_moe:
                    pool_rows = getattr(mirror_pool, "capacity", None)
                    hint = (
                        " (bounded expert mirror cannot cover the complement: "
                        f"raise --moe-mirror-host-rows above {pool_rows})"
                        if mirror_pool is not None else ""
                    )
                    raise RuntimeError(
                        "growable KV live-memory guard could not fund the next "
                        f"VMM commit from the expert arena{hint}"
                    )
                # Byte accounting, not the driver reading: on this WSL2 host
                # cudaMemGetInfo can pin at 0 while the arena's VMM unmaps are
                # real (EXCLUSIVE-DIAG showed set_usable_slots 2175->2056, i.e.
                # 0.58 GiB actually uncommitted, with mem_get_info still 0.00
                # in-process). set_usable_slots returns the EXACT bytes it
                # uncommitted and commit_pages maps the exact KV suffix, so
                # trust the ledger; keep mem_get_info as an advisory log line.
                released_bytes = moe.set_usable_slots(target_moe)
                object.__setattr__(self.engine.config, "moe_cache_size", target_moe)
                logger.info_rank0(
                    "Growable-KV arena shrink: %d -> %d slots, %s uncommitted "
                    "(driver-reported free %s)",
                    old_moe, target_moe, mem_GB(released_bytes),
                    mem_GB(torch.cuda.mem_get_info(self.engine.device)[0]),
                )
            live_free = self.engine._sync_get_memory()[0]
            live_free = max(live_free, live_free_before + released_bytes)
            logger.info_rank0(
                "Growable-KV pre-commit (arena): %s free (driver %s), %s commit, "
                "%s required (allocator %s allocated / %s reserved)",
                mem_GB(live_free),
                mem_GB(torch.cuda.mem_get_info(self.engine.device)[0]),
                mem_GB(commit_bytes),
                mem_GB(required_free),
                mem_GB(torch.cuda.memory_allocated(self.engine.device)),
                mem_GB(torch.cuda.memory_reserved(self.engine.device)),
            )
            if live_free < required_free:
                # The release estimate rounds conservatively: four 600K-class
                # runs died with "need 0.46 GiB, have 0.42/0.37/0.17/0.08"
                # where "have" was exactly what the estimate produced, while
                # the arena still held releasable rows above the coverage
                # floor. Top up: shrink a chunk further (mirroring the real
                # release into the ledger), re-sync, and re-check -- instead
                # of refusing the commit with releasable rows still sitting
                # above the floor.
                while live_free < required_free and target_moe > floor:
                    target_moe = self._growable_arena_step_down(target_moe, floor)
                    released_bytes += moe.set_usable_slots(target_moe)
                    object.__setattr__(self.engine.config, "moe_cache_size", target_moe)
                    live_free = max(live_free, live_free_before + released_bytes)
                    logger.info_rank0(
                        "Growable-KV arena top-up: %d slots, %s released total "
                        "(need %s, have %s)",
                        target_moe, mem_GB(released_bytes),
                        mem_GB(required_free), mem_GB(live_free),
                    )
                if live_free < required_free:
                    # Say WHY there is nothing left to release. When a mirror
                    # is attached and the arena has been driven onto its floor,
                    # the binding constraint is the pool's coverage floor, not
                    # VRAM -- and the fix is a bigger pool, which this message
                    # is the only place the operator will hear about.
                    at_floor = mirror_pool is not None and target_moe <= floor
                    hint = (
                        f" (expert arena is at its coverage floor of {floor} "
                        f"slots for a {mirror_pool.capacity}-row mirror; it "
                        f"released {mem_GB(released_bytes)} and has no more to "
                        f"give: raise --moe-mirror-host-rows, or serve a "
                        f"shorter context)"
                        if at_floor else ""
                    )
                    raise RuntimeError(
                        "growable KV refused an unsafe VMM commit: "
                        f"need {mem_GB(required_free)} free, have "
                        f"{mem_GB(live_free)}{hint}"
                    )
            pool.commit_pages(target_pages)
            if self.engine.config.tp_info.size > 1:
                self.engine.sync_all_ranks()
        except Exception:
            self._rollback_growable_kv_transition(
                old_pages=old_pages,
                old_moe=old_moe,
                old_overlap=old_overlap,
            )
            raise
        assert self.engine.graph_runner is runner_before, (
            "expert-arena resize must never replace the decode graphs"
        )
        logger.info_rank0(
            "Committed growable KV through %d tokens (%s physical); MoE slots %d -> %d",
            target_pages,
            mem_GB(kv_bytes),
            old_moe,
            target_moe,
        )
        return old_pages, target_pages

    def _shrink_runtime_kv_arena(
        self,
        *,
        old_pages: int,
        target_pages: int,
        old_moe: int,
        old_overlap: bool,
        kv_bytes: int,
        old_kv_bytes: int,
    ) -> tuple[int, int]:
        """``shrink_runtime_kv``'s expert-arena branch (design step 5): regrow
        experts with ``set_usable_slots`` up to the largest chunk boundary the
        released KV bytes fund. Bank addresses never move (see
        ``_grow_runtime_kv_arena`` for the shared reasoning).

        S12b: boundary enumeration and byte pricing go through
        ``_growable_arena_boundaries``/``_growable_moe_bytes``, which dispatch to
        the joint per-class model for a mixed-GGUF cache and to the untouched
        single-class formulas otherwise -- the N=1 case is byte-identical to
        before this method stopped taking ``arena_layout`` as a parameter.
        """
        pool = self.kv_cache
        moe = self.moe
        assert moe is not None
        runner_before = self.engine.graph_runner
        target_moe = old_moe
        try:
            pool.decommit_pages(target_pages)
            released = old_kv_bytes - kv_bytes
            old_bytes = self._arena_transition_bytes(old_moe)
            budget_bytes = old_bytes + released
            for boundary in self._growable_arena_boundaries():
                if boundary < old_moe:
                    continue
                if self._arena_transition_bytes(boundary) <= budget_bytes:
                    target_moe = boundary
            if target_moe != old_moe:
                moe.set_usable_slots(target_moe)
                object.__setattr__(self.engine.config, "moe_cache_size", target_moe)
            if self.engine.config.tp_info.size > 1:
                self.engine.sync_all_ranks()
        except Exception:
            self._rollback_growable_kv_transition(
                old_pages=old_pages,
                old_moe=old_moe,
                old_overlap=old_overlap,
            )
            raise
        assert self.engine.graph_runner is runner_before, (
            "expert-arena resize must never replace the decode graphs"
        )
        logger.info_rank0(
            "Released growable KV %d -> %d tokens (%s returned); MoE slots %d -> %d",
            old_pages,
            target_pages,
            mem_GB(old_kv_bytes - kv_bytes),
            old_moe,
            target_moe,
        )
        return old_pages, target_pages

    @torch.inference_mode()
    def grow_runtime_kv(self, required_pages: int) -> tuple[int, int]:
        """Commit the next KV suffix at a safe batch boundary and fund it from MoE slots.

        The KV tensors keep their virtual addresses, so existing K/V remains valid, and the
        expert arena shrinks in place, so the decode graphs do too. The caller (the
        scheduler) guarantees a no-forward-in-flight boundary.
        """
        self._refuse_if_growable_transition_failed()
        pool = self.kv_cache
        old_pages = int(getattr(pool, "committed_pages", self.engine.num_pages))
        if required_pages <= old_pages:
            return old_pages, old_pages
        step = self.engine.config.kv_grow_step_tokens // self.engine.config.page_size
        target_pages = min(self.engine.num_pages, math.ceil(required_pages / step) * step)
        if target_pages <= old_pages:
            return old_pages, old_pages

        moe = self.moe
        assert moe is not None, "growable KV requires the MoE offload cache"
        # S12b: a mixed-GGUF cache has no single arena_layout, only
        # class_arena_layouts (one arena per size class) -- accept either.
        if (
            getattr(moe, "arena_layout", None) is None
            and getattr(moe, "class_arena_layouts", None) is None
        ):
            raise RuntimeError(GROWABLE_KV_UNSUPPORTED)
        old_moe = moe.cache_size
        old_overlap = moe.prefill_overlap
        # The planner also refuses a target no expert-cache size can fund; its MoE size is
        # not used here -- _grow_runtime_kv_arena shrinks only as far as live VRAM needs.
        _planned_moe, kv_bytes = self._plan_growable_kv(target_pages)

        torch.cuda.synchronize(self.engine.device)
        if self.engine.config.tp_info.size > 1:
            self.engine.sync_all_ranks()
        return self._grow_runtime_kv_arena(
            old_pages=old_pages,
            target_pages=target_pages,
            old_moe=old_moe,
            old_overlap=old_overlap,
            kv_bytes=kv_bytes,
        )

    @torch.inference_mode()
    def shrink_runtime_kv(self, target_pages: int) -> tuple[int, int]:
        """Decommit a free KV suffix and regrow the expert arena from the released VRAM."""
        self._refuse_if_growable_transition_failed()
        pool = self.kv_cache
        old_pages = int(getattr(pool, "committed_pages", self.engine.num_pages))
        if target_pages >= old_pages:
            return old_pages, old_pages
        step = self.engine.config.kv_grow_step_tokens // self.engine.config.page_size
        initial = min(self.engine.num_pages, step)
        target_pages = max(initial, math.ceil(target_pages / step) * step)
        if target_pages >= old_pages:
            return old_pages, old_pages

        moe = self.moe
        assert moe is not None, "growable KV requires the MoE offload cache"
        # S12b: accept either the single-class arena_layout or the mixed-GGUF
        # class_arena_layouts (one arena per size class).
        if (
            getattr(moe, "arena_layout", None) is None
            and getattr(moe, "class_arena_layouts", None) is None
        ):
            raise RuntimeError(GROWABLE_KV_UNSUPPORTED)
        old_moe = moe.cache_size
        old_overlap = moe.prefill_overlap
        _planned_moe, kv_bytes = self._plan_growable_kv(target_pages)
        old_kv_bytes = pool.mapped_bytes_for_pages(old_pages)

        torch.cuda.synchronize(self.engine.device)
        if self.engine.config.tp_info.size > 1:
            self.engine.sync_all_ranks()
        return self._shrink_runtime_kv_arena(
            old_pages=old_pages,
            target_pages=target_pages,
            old_moe=old_moe,
            old_overlap=old_overlap,
            kv_bytes=kv_bytes,
            old_kv_bytes=old_kv_bytes,
        )
