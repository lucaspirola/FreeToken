"""Expert residency: where an expert's bytes live while it is not on the GPU.

``OffloadMoeCache`` owns the GPU side -- the slot arena, ``slot_for_id`` /
``id_of_slot``, LRU/LFU admission, the prefill double buffer. What sits behind
a GPU miss is a separate question with two answers, and this module is the seam
between them (refactor plan S6,
``tasks/exclusive-expert-ram/reviews/2026-09-22-refactor-plan-final.md``):

``WholeModelResidency`` (``--expert-residency whole``, the default)
    Every expert row is pinned in host RAM (``set_bank_sources``). A miss is a
    plain host->device copy of a row that always exists, so every hook below
    is a no-op and ``min_gpu_slots() == 0``.

``MirrorResidency`` (``--expert-residency mirror``, sized by
``--moe-mirror-host-rows``)
    A bounded pinned pool (``moe/mirror_pool.py``) holds ``capacity`` rows, and
    every expert is on the GPU, in the pool, or both (the coverage invariant).
    A miss admits expert ``new`` (pool row r) into the slot of victim ``v``;
    since ``new`` becomes GPU-resident, r falls free and receives ``v``: the
    swap is a permutation of pool rows, never an allocation. When ``v`` already
    has a pool row (a duplicate, which fits whenever capacity + gpu_slots >
    L*E) the writeback is skipped entirely. The swap kernels live in
    ``moe/mirror_kernels.py``.

The cache calls only the hooks of ``ExpertResidency``. The ``MirrorResidency``
methods below the hooks were moved out of ``OffloadMoeCache`` (and
``_mirror_final_gpu_slots`` out of ``Engine``) WITHOUT editing their bodies;
the only rewrites are mechanical: ``self.<cache attribute>`` became
``self.cache.<cache attribute>``, the three kernel launches receive
``self.cache``, and ``Engine._mirror_final_gpu_slots``'s ``self`` is the
explicit ``engine`` argument. A stale free-row publish once served wrong
experts with every counter at zero; the only defence against re-introducing
that class of bug in a move is that nothing but those rewrites changed.

This module must not import ``moe.mirror_pool`` or ``moe.mirror_kernels`` at
import time: the whole-model path never loads the pool.
"""

from __future__ import annotations

import math
import os
from typing import Protocol

import torch
from freetoken.moe.mirror_stats import (
    MIRROR_STAT_COUNT,
    mirror_fault_counts_from_vector,
    mirror_stats_from_vector,
)
from freetoken.utils import init_logger

logger = init_logger(__name__)


class ExpertResidency(Protocol):
    """What ``OffloadMoeCache`` asks of the host side of its expert cache.

    The bool-returning hooks answer "did the residency do this step itself?";
    False means the cache runs its own default path.
    """

    kind: str      # "whole" | "mirror" -- the --expert-residency value
    bounded: bool  # False: every expert has a host row; True: bounded pool
    cache: object | None  # the OffloadMoeCache it is attached to

    def min_gpu_slots(self) -> int:
        """Arena slots the GPU must keep so every expert stays covered."""

    def before_ensure(self, layer_id: int) -> None:
        """Before a decode step's admission kernel for ``layer_id``."""

    def before_buffer_fill(self, buffer_id: int) -> bool:
        """Before a prefill buffer half is overwritten; True: already vacated."""

    def prefetch_layer(self, layer_id: int, buffer_id: int) -> bool:
        """Assemble prefill layer ``layer_id`` into the buffer half; True: done."""

    def before_shrink(self, n: int, current: int) -> None:
        """Before arena slots ``[n, current)`` are invalidated and unmapped."""

    def host_copy_mask(self):
        """Per flat expert id: does it have a host copy (dropping its GPU copy is
        free)? ``None`` means every expert does (``moe/arena_compaction.py``)."""

    def fault_check(self) -> None:
        """Raise if coverage was lost (host idle boundary, once per batch)."""

    def stats(self) -> dict:
        """Counters for /v1/stats (empty when there is nothing to count)."""

    def init_prefill_buffers(self) -> bool:
        """Allocate the residency's own prefill buffers; True: allocated."""

    def begin_prefill(self) -> bool:
        """Per-chunk prefill setup; True: the cache's own setup is replaced."""

    def prefill_begin_blocks_host(self) -> bool:
        """Does this residency's prefill setup/prefetch wait on the host for the copy stream?"""

    def copy_missing(self) -> bool:
        """Issue this step's copies; True: the cache's own copy is replaced."""

    def after_reset(self) -> None:
        """After ``reset_cache`` dropped every GPU resident."""

    def service_writebacks(self) -> None:
        """Host step boundary, before a forward is enqueued: move completed
        device-side work that needs a host hand (the mirror's DMA writebacks)."""

    def issue_writebacks(self) -> None:
        """After the previous step's results were drained (that step is
        complete): hand its finished device work to the host, no snapshot."""

    def attention_gate_hook(self):
        """A ``gate(layer_id)`` callable the engine runs before each decode
        attention call (``DecodeGatedBackend``), or None for no gating."""


class WholeModelResidency:
    """Whole model pinned in host RAM: nothing to do behind a miss."""

    kind = "whole"
    bounded = False

    def __init__(self) -> None:
        self.cache = None

    def min_gpu_slots(self) -> int:
        return 0

    def before_ensure(self, layer_id: int) -> None:
        return None

    def before_buffer_fill(self, buffer_id: int) -> bool:
        return False

    def prefetch_layer(self, layer_id: int, buffer_id: int) -> bool:
        return False

    def before_shrink(self, n: int, current: int) -> None:
        return None

    def host_copy_mask(self):
        return None  # the whole model is in host RAM

    def fault_check(self) -> None:
        return None

    def stats(self) -> dict:
        return {}

    def init_prefill_buffers(self) -> bool:
        return False

    def begin_prefill(self) -> bool:
        return False

    def prefill_begin_blocks_host(self) -> bool:
        return False

    def copy_missing(self) -> bool:
        return False

    def after_reset(self) -> None:
        return None

    def service_writebacks(self) -> None:
        return None

    def issue_writebacks(self) -> None:
        return None

    def attention_gate_hook(self):
        return None

    def attach(self, cache, banks) -> None:
        """Engine entry point: bind to ``cache`` and register the host banks."""
        cache.attach_residency(self)
        cache.set_bank_sources(banks.sources, layer_residency=banks.layer_residency)
        cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)


class MirrorResidency:
    """Bounded host mirror; see the module docstring for the invariant."""

    kind = "mirror"
    bounded = True

    def __init__(self, pool, cache=None) -> None:
        self._mirror_pool = pool
        # Bound by OffloadMoeCache.attach_residency, before _build_mirror_plan.
        self.cache = cache
        # Requests decoded per step (--max-running-requests), which scales the
        # writebacks one step can stage (wb_stage_rows); set by build_residency.
        self._decode_batch = 1
        # Device-resident residency maps + copy descriptors (_build_mirror_plan).
        self._mirror: dict | None = None
        # Host half of the DMA writebacks (_build_mirror_plan); None when the
        # staging ring is disabled (FREETOKEN_MIRROR_WB_STAGE_MB=0) or on CPU.
        self._wb: dict | None = None
        self._snap: dict | None = None
        # Mirror-only prefill assembly (see _prefetch_split_mirror), allocated
        # by init_prefill_buffers when the cache has prefill overlap.
        self._mirror_writeback: dict | None = None
        self._mirror_prefill_pool_snapshot: torch.Tensor | None = None
        self._mirror_prefill_pool_np = None
        self._mirror_prefill_idx: torch.Tensor | None = None
        self._mirror_prefill_idx_host: torch.Tensor | None = None
        self._mirror_prefill_idx_np = None
        self._mirror_prefill_hit_count: torch.Tensor | None = None
        self._mirror_prefill_miss_count: torch.Tensor | None = None

    @property
    def pool(self):
        return self._mirror_pool

    # ------------------------------------------------------------------
    # The seam: the only calls OffloadMoeCache makes into the mirror.
    # ------------------------------------------------------------------

    def min_gpu_slots(self) -> int:
        return self._mirror_pool.min_gpu_slots

    def before_ensure(self, layer_id: int) -> None:
        """Nothing to do before the LRU kernel.

        Decode coverage is maintained by construction under the mirror: prefill
        runs exclusively through the overlap path (prefetch_prefill_layer ->
        _prefetch_split_mirror), which never empties the mirror the way
        materialize_layer's whole-layer invalidation would -- so there is no
        batch-boundary restore to run here. (materialize_layer raises under
        the mirror; see its docstring.)

        The swap kernel's per-layer bookkeeping (clear victim_ids/prior_ids,
        whose entries linger from a longer previous step, and snapshot the
        layer's pre-step slot map) is done by the ensure launch itself:
        inside the v2 LRU kernel, or by ``mirror_kernels.begin_layer`` right
        before the other variants (``offload_kernels.ensure_experts``). A
        separate launch here was a graph node per MoE layer.
        """
        return None

    def before_buffer_fill(self, buffer_id: int) -> bool:
        """Moved from ``OffloadMoeCache._invalidate_prefill_buffer``'s mirror branch."""
        if self._mirror_prefill_base():
            # Under the mirror these slots CAN hold a decode resident now (the
            # victim floor that used to fence decode out is gone), so the old
            # empties-only assumption would drop an expert's only copy when
            # the next layer overwrites the bytes. Write it back first.
            self._mirror_writeback_buffer(buffer_id)
            return True
        return False

    def prefetch_layer(self, layer_id: int, buffer_id: int) -> bool:
        self._prefetch_split_mirror(layer_id, buffer_id)
        return True

    def before_shrink(self, n: int, current: int) -> None:
        """Moved from ``OffloadMoeCache._arena_shrink``'s mirror branch."""
        if getattr(self, "_mirror", None) is not None:
            # Slots [n, current) go to the KV arena; any expert held only there
            # must come back to the mirror or coverage breaks.
            self._mirror_refill_uncovered(n, current, from_gpu=True)

    def host_copy_mask(self):
        """Duplicates: experts that own a pool row (device map, the kernel's truth)."""
        if getattr(self, "_mirror", None) is None:
            return None
        return [row >= 0 for row in self._mirror["pool_row_of_id"].tolist()]

    def fault_check(self) -> None:
        self.mirror_fault_check()

    def stats(self) -> dict:
        return self.mirror_stats()

    def init_prefill_buffers(self) -> bool:
        """Moved from ``OffloadMoeCache._init_prefill_overlap_buffers``' mirror branch."""
        if getattr(self, "_mirror", None) is not None and self.cache.device.type == "cuda":
            # The mirror assembles a prefill layer from the only two places
            # coverage allows -- a resident's own slot, or the expert's pool row
            # -- so it needs a pinned view of both maps and one index pair per
            # layer. The slot-map snapshot is taken once per chunk (begin_prefill)
            # and never goes stale: the only chunk-internal writer
            # (_invalidate_prefill_buffer) always rewrites a slot already below
            # 2E, and slots < 2E (including -1) classify as a miss on both sides
            # (see _prefetch_split_mirror). The pool-row snapshot is NOT immune
            # the same way -- a buffer occupant's writeback changes its pool row
            # mid-chunk -- so _mirror_writeback_buffer patches this numpy array
            # in place for every row it writes, instead of leaving it frozen.
            E = self.cache.num_experts
            self.cache._prefill_slot_snapshot = torch.empty(
                (self.cache.num_layers, E), dtype=torch.int32, pin_memory=True
            )
            self.cache._prefill_snapshot_np = self.cache._prefill_slot_snapshot.numpy()
            self._mirror_prefill_pool_snapshot = torch.empty(
                (self.cache.num_layers * E,), dtype=torch.int32, pin_memory=True
            )
            self._mirror_prefill_pool_np = self._mirror_prefill_pool_snapshot.numpy()
            # Rows: 0 hit dst, 1 hit src, 2 miss dst, 3 miss src.
            self._mirror_prefill_idx_host = torch.empty(
                (2, 4, E), dtype=torch.int32, pin_memory=True
            )
            self._mirror_prefill_idx_np = self._mirror_prefill_idx_host.numpy()
            self._mirror_prefill_idx = torch.empty(
                (2, 4, E), dtype=torch.int32, device=self.cache.device
            )
            self._mirror_prefill_hit_count = torch.zeros(
                (1,), dtype=torch.int64, device=self.cache.device
            )
            self._mirror_prefill_miss_count = torch.zeros(
                (1,), dtype=torch.int64, device=self.cache.device
            )
            # Dedicated writeback descriptors for _mirror_writeback_buffer, sized
            # for a whole buffer half (<= E occupants) rather than reusing the
            # decode step's g1/g2 descriptors (sized for one admission batch,
            # which can be far smaller): the two run at different points in the
            # request lifecycle (prefill vs decode) but on the same device
            # state, so aliasing their descriptor buffers would let one
            # overwrite the other's in-flight plan.
            self._mirror_writeback = {
                "d2h_src": torch.zeros((E,), dtype=torch.int32, device=self.cache.device),
                "d2h_dst": torch.zeros((E,), dtype=torch.int32, device=self.cache.device),
                "n_d2h": torch.zeros((1,), dtype=torch.int64, device=self.cache.device),
                "ids": torch.zeros((E,), dtype=torch.int32, device=self.cache.device),
                "rows": torch.zeros((E,), dtype=torch.int32, device=self.cache.device),
                "n_wb": torch.zeros((1,), dtype=torch.int64, device=self.cache.device),
                # Every occupant this call finds, written back or not (a
                # retained duplicate is vacated too) -- see
                # _writeback_buffer_kernel's docstring for why the slot
                # snapshot needs all of them, not just the writebacks.
                "vacated": torch.zeros((E,), dtype=torch.int32, device=self.cache.device),
                "n_vacated": torch.zeros((1,), dtype=torch.int64, device=self.cache.device),
                # n_d2h and n_vacated read together as one pinned host copy
                # per invalidate call (see _mirror_writeback_buffer), rather
                # than each forcing its own device sync via a separate
                # .item().
                "n_d2h_vacated_host": torch.zeros((2,), dtype=torch.int64, pin_memory=True),
            }
            return True
        return False

    def begin_prefill(self) -> bool:
        """Moved from ``OffloadMoeCache.begin_prefill``'s mirror branch."""
        if getattr(self, "_mirror", None) is not None:
            # The mirror runs its own hit/miss split (_prefetch_split_mirror),
            # whose sources are the two residency maps rather than the host
            # banks. One sync per chunk buys pure host math for every layer.
            self.cache._prefill_hit_d2d_active = False
            # Prefill reads pool rows (_prefetch_split_mirror) and SM-stores
            # buffer occupants into free rows (_mirror_writeback_buffer); both
            # need every staged writeback landed first. begin_prefill already
            # waits for the preceding decode (the synchronize below), so this
            # adds only the DMA tail.
            self.drain_writebacks()
            with torch.cuda.stream(self.cache.prefill_copy_stream):
                self.cache._prefill_slot_snapshot.copy_(self.cache.slot_for_id, non_blocking=True)
                self._mirror_prefill_pool_snapshot.copy_(
                    self._mirror["pool_row_of_id"], non_blocking=True
                )
            self.cache.prefill_copy_stream.synchronize()
            return True
        return False

    def prefill_begin_blocks_host(self) -> bool:
        # begin_prefill above and _mirror_writeback_buffer both synchronize the copy stream
        return getattr(self, "_mirror", None) is not None

    def copy_missing(self) -> bool:
        self.copy_missing_mirror()
        return True

    def service_writebacks(self) -> None:
        """Issue the ring -> pool DMAs of every step that has finished, and
        snapshot the step just enqueued.

        Called by the scheduler right after each forward is launched (and by
        any driver of ``copy_missing`` that wants its writebacks to land). Two
        halves, both non-blocking in the steady state:

        1. For each earlier snapshot whose event has completed, the ring
           entries it names are complete in VRAM: copy each to its pool row on
           the writeback stream, then publish the new completed count to the
           device (``wb_state[1]``) behind those copies, which is what lets
           the resolve kernel reuse the ring slots and read the pool rows.
        2. Snapshot the device ring state and the fault counters behind
           everything enqueued so far (i.e. after the step just launched)
           into pinned memory, with an event, for a later call to consume.

        The snapshot copies run on a side stream that waits for the compute
        stream, never the other way round: a small D2H on the compute stream
        queues behind the writeback DMAs on the copy engine and put the whole
        DMA on the decode critical path (ft-g5 nsys, 2026-09-24: ~370 us of a
        6149 us step). The compute stream never waits on either side stream.

        The step's entries are issued by the next call (or by
        ``issue_writebacks``) once the step is complete, and copied while the
        next step computes. The ring must
        hold what is staged meanwhile; when it cannot, the kernel falls back
        to SM stores.
        """
        m = getattr(self, "_mirror", None)
        if m is None or self._snap is None:
            return
        wb = self._wb
        if wb is not None:
            self._wb_issue_completed()
            pending = wb["pending"]
            if len(pending) == len(wb["snaps"]):
                pending[0][0].synchronize()
                self._wb_issue_completed()
        side = self._snap["stream"]
        after_step = self._snap["after_step"]
        after_step.record()
        side.wait_event(after_step)
        with torch.cuda.stream(side):
            # Monotone counters: the host check may read one that lags.
            m["stats_host"].copy_(m["stats"], non_blocking=True)
            if wb is not None:
                slot = wb["next_snap"]
                wb["next_snap"] = (slot + 1) % len(wb["snaps"])
                snap = wb["snaps"][slot]
                event = wb["events"][slot]
                snap.copy_(m["wb_state"], non_blocking=True)
                event.record()
                wb["pending"].append((event, snap))

    def issue_writebacks(self) -> None:
        """Issue the DMAs of every snapshot whose step has completed.

        The scheduler calls this right after draining the previous step's
        results: the snapshot taken before the current step was launched is
        complete then, so its entries are copied while the current step
        computes (one step of lag instead of two under overlap scheduling).
        """
        if self._wb is not None:
            self._wb_issue_completed()

    def attention_gate_hook(self):
        """Schedule the write-back DMA behind decode attention.

        The ring -> pool DMA is issued by the host right after a step is
        launched and used to start at once, beside that step's in-graph expert
        fetches, which read the same PCIe link and host memory the other way
        (nsys at 1M, exp/reorg-next 2dc6757 results/nsys1m-local: fetches
        176 vs 139 us at 70 MB/step of write-back). Decode attention uses no
        PCIe (3.6 ms/step at 1M on Nemotron), so the DMA is split into one
        chunk per decode attention layer and each chunk waits for an event
        recorded just before that layer's attention (``attention_gate``):
        inside the CUDA graph an external-event record node, so one capture
        serves every replay; eager decode records it the same way. The events
        are the latest records when the host issues the DMA (after the
        launch), so the chunks follow the step just launched. With no decode
        attention recorded yet (or after a prefill), the waits are satisfied
        and the DMA starts at once, as before.

        ``FREETOKEN_MIRROR_WB_AT_ATTENTION=0`` restores the ungated DMA.
        """
        if getattr(self, "_wb", None) is None:
            return None
        if os.environ.get("FREETOKEN_MIRROR_WB_AT_ATTENTION", "1").strip() == "0":
            logger.info_rank0("mirror DMA writebacks: not gated on attention "
                              "(FREETOKEN_MIRROR_WB_AT_ATTENTION=0)")
            return None
        self._attn_gates = {}
        self._attn_gate_list = []
        logger.info_rank0("mirror DMA writebacks: one chunk per decode attention layer, "
                          "issued behind that layer's attention start event")
        return self.attention_gate

    def attention_gate(self, layer_id: int) -> None:
        """Record layer ``layer_id``'s gate event on the current stream (the
        decode stream, or the graph being captured)."""
        ev = self._attn_gates.get(layer_id)
        if ev is None:
            ev = self._attn_gates[layer_id] = torch.cuda.Event(external=True)
            self._attn_gate_list = [self._attn_gates[k] for k in sorted(self._attn_gates)]
        ev.record()

    def _wb_issue_completed(self) -> None:
        wb = self._wb
        pending = wb["pending"]
        while pending and pending[0][0].query():
            _event, snap = pending.pop(0)
            state = snap.tolist()
            if wb["trace"] is not None:     # ring head at each step boundary
                wb["trace"].write(f"{int(state[0])}\n")
            self._wb_issue(state)

    def _wb_issue(self, state: list) -> None:
        """DMA ring entries [issued, state[0]) to their pool rows, in ring order."""
        wb = self._wb
        head = int(state[0])
        issued = wb["issued"]
        if head <= issued:
            return
        rows = wb["rows"]
        assert head - issued <= rows, (head, issued, rows)
        wb["peak_pending"] = max(wb["peak_pending"], head - issued)
        gates = getattr(self, "_attn_gate_list", None) or [None]
        per = -(-(head - issued) // len(gates))
        stream = wb["stream"]
        with torch.cuda.stream(stream):
            for j, gate in enumerate(gates):
                lo = issued + j * per
                hi = min(head, lo + per)
                if lo >= hi:
                    break
                if gate is not None:        # see attention_gate_hook
                    stream.wait_event(gate)
                for idx in range(lo, hi):
                    s = idx % rows
                    row = int(state[2 + s])
                    for pool_view, stage_view in wb["views"]:
                        pool_view[row].copy_(stage_view[s], non_blocking=True)
                if gate is not None and hi < head:
                    # Publish each chunk as it lands, not only the last one:
                    # the ring (wb_stage_rows) holds about one step of
                    # write-backs because the previous step's slots free up
                    # early in the next step. With a single publish behind the
                    # last attention layer they freed near the END of the step,
                    # the ring stayed full and the next step's victims fell
                    # back to in-graph SM stores (1M box nsys, f98c7c5: ~10 MB
                    # per step moved from DMA to SM stores, fetch kernels
                    # without any DMA beside them 1070 vs 529 us mean).
                    self._mirror["wb_state"][1:2].fill_(hi)
            # Stream-ordered behind the copies: only now may the resolve
            # kernel reuse these ring slots or read these pool rows.
            self._mirror["wb_state"][1:2].fill_(head)
        wb["issued"] = head
        wb["dma_rows"] += head - issued

    def drain_writebacks(self) -> None:
        """Land every staged writeback in the pool (host sync).

        Required before anything reads or writes pool rows outside the decode
        kernels: prefill, arena shrink/refill, warm start, tests that inspect
        the pool. After it returns, no ring entry is pending.
        """
        wb = self._wb
        if wb is None:
            return
        torch.cuda.synchronize(self.cache.device)
        state = self._mirror["wb_state"].tolist()
        if wb["trace"] is not None:         # a drain: prefill / refill boundary
            wb["trace"].write(f"D {int(state[0])}\n")
            wb["trace"].flush()
        self._wb_issue(state)
        wb["pending"].clear()
        wb["stream"].synchronize()

    def after_reset(self) -> None:
        """Moved from ``OffloadMoeCache.reset``'s mirror branch.

        A cold start, not data loss: reset_cache drops every GPU resident,
        which would strand the experts the mirror does not hold. Re-run the
        warm start instead -- it refills the GPU from the checkpoint and
        rebuilds the mirror's complement, restoring coverage by
        construction. Host idle boundary only (graph capture, engine
        teardown), never a decode step.
        """
        self.mirror_warm_start()

    def placeholder_banks(self, mc):
        """Metadata-only banks for the engine: the pool, not the banks, holds the bytes.

        Moved from ``Engine._init_offload_moe_cache``'s mirror branch.
        """
        from freetoken.moe.expert_banks import ExpertBanks

        mirror_pool = self._mirror_pool
        banks = ExpertBanks(mirror_pool.quant_format, {
            name: [torch.empty((mc.num_experts, *tail), dtype=dtype, device="meta")]
            * mc.num_moe_layers
            for name, (tail, dtype) in mirror_pool.shapes.items()
        })
        return banks

    def attach(self, cache, banks=None) -> None:
        """Engine entry point: bind to ``cache``, warm-start, log the budget.

        Moved from ``Engine._init_offload_moe_cache``'s ``if mirror_pool is not
        None`` branch; ``banks`` (the placeholders) is unused -- the pool holds
        the bytes.
        """
        mirror_pool = self._mirror_pool
        cache.attach_residency(self)
        # Coverage must hold before the first forward: fill the GPU cache
        # and give the mirror the complement.
        cache.mirror_warm_start()
        # The pinned pool IS the host RAM this profile costs, and how
        # much of it is duplicates is what decides the writeback rate,
        # so both belong in the log rather than in a benchmark's notes.
        # mirror_warm_start seats residents from slot 0 now (the
        # buffer region is no longer excluded), so every arena slot
        # counts as a resident here.
        _residents = cache.cache_size
        _complement = max(mirror_pool.total - _residents, 0)
        _dupes = max(mirror_pool.capacity - _complement
                     - mirror_pool.reserve_rows, 0)
        logger.info_rank0(
            "Mirror pool: %d rows pinned (%.2f GiB), %d cover the "
            "complement of %d GPU residents, %d reserved, up to %d "
            "duplicates (a duplicate makes its expert's next eviction "
            "free of any host traffic)",
            mirror_pool.capacity,
            getattr(mirror_pool, "pool_bytes", 0) / 2**30,
            _complement, _residents, mirror_pool.reserve_rows, _dupes,
        )
        # How much of the expert arena the growable KV may still take.
        # The coverage floor is the mirror's, so a pool too small for
        # the configured context ceiling shows up HERE -- at startup,
        # in slots and GiB -- instead of 30 s into the request that
        # cannot be funded. Measured on Nemotron at 1700 rows: floor
        # 1888 against a 1923-slot arena, i.e. 35 slots = 0.18 GiB of
        # slack, and an 80K prompt needs 0.46 GiB. That server died
        # mid-request with "growable KV refused an unsafe VMM commit"
        # and could not be restarted.
        # Both of these were wrong the first time and the except
        # below swallowed it, so the line never printed: the arena step
        # is the RESOLVED ``_arena_step_slots`` (the dataclass field of
        # the same name is None until __post_init__ resolves it, which
        # silently made the floor a 1-slot rounding), and
        # ``bank_row_bytes`` is a LIST of per-bank row bytes, one entry
        # per arena VMM allocation -- multiplying a list by the slot
        # count repeats the list and then raises on the division.
        # ``arena_layout`` is the public accessor for the pair.
        try:
            _layout = cache.arena_layout
            _step = max(int(_layout[1]) if _layout else 1, 1)
            # No prefill_buffer_slots term: the double buffer's slots
            # are candidates for decode residents now (a resident
            # there is written back before a prefill fill overwrites
            # it), so they no longer need to be priced out of the
            # floor as dead space. Measured on Nemotron: this drops
            # the floor by 256 of 2173 arena slots.
            _need = mirror_pool.min_gpu_slots
            _cov_floor = -(-_need // _step) * _step
            _slack = cache.cache_size - _cov_floor
            _row = sum(cache.bank_row_bytes or ())
            logger.info_rank0(
                "Mirror pool: the coverage floor is %d of %d arena "
                "slots, leaving %d slots (%.2f GiB) the growable KV "
                "may still take%s",
                _cov_floor, cache.cache_size, _slack,
                max(_slack, 0) * _row / 2**30,
                "" if _slack > 0 else
                " -- the arena is AT its floor, so KV cannot grow at "
                "all: raise --moe-mirror-host-rows",
            )
        except Exception:             # diagnosis must never fail a load
            # WARNING, not debug: this handler silently hid two real
            # bugs in the block above for a whole measurement round.
            logger.warning_rank0("mirror arena-slack log skipped",
                                 exc_info=True)

    # ------------------------------------------------------------------
    # Moved verbatim from OffloadMoeCache (self.<cache attr> -> self.cache.<attr>).
    # ------------------------------------------------------------------

    def _mirror_prefill_base(self) -> int:
        """First cache slot a decode resident may occupy.

        Under the mirror the prefill double buffer owns the head of the cache
        outright (``mirror_pool.prefill_buffer_slots``): everything below this
        index is buffer, everything at or above it is a decode resident. The
        baseline shares those slots between the two uses, which is safe only
        because it holds the whole model in host RAM; here the buffer's own
        invalidation would drop an expert's only copy.

        Returns 0 when there is no mirror or no overlap -- the shared-slot
        behaviour the rest of the cache already implements.
        """
        if getattr(self, "_mirror", None) is None or not self.cache.prefill_overlap:
            return 0
        from freetoken.moe.mirror_pool import prefill_buffer_slots

        return prefill_buffer_slots(self.cache.num_experts)

    def _build_mirror_plan(self, pool) -> None:
        """Device-resident residency maps + the fused copy descriptors.

        Both directions reuse ``fast_index_copy_multi_jit``: it copies row
        ``src_indices[i]`` of each source bank into row ``dst_indices[i]`` of
        each destination bank, and is agnostic to which side is host memory (the
        mirror is pinned and device-visible, so its rows are addressable from
        the GPU exactly like the baseline's whole-model banks).
        """
        from freetoken.kernel.pinned import device_ptr

        dev = self.cache.device
        plan = self.cache.evict_slots.numel()
        host_rows = [-1] * pool.capacity
        fwd = [-1] * pool.total
        for row, flat in enumerate(pool.id_of_pool_row):
            host_rows[row] = flat
        for flat, row in enumerate(pool.pool_row_of_id):
            fwd[flat] = row
        pool_ptrs, cache_ptrs, feat_bytes = [], [], []
        for name in self.cache.bank_schema:
            host = pool.banks[name]
            cache = self.cache.bank_caches[name]
            row_bytes = math.prod(host.shape[1:]) * host.element_size()
            if row_bytes % 16 or device_ptr(host) % 16 or cache.data_ptr() % 16:
                raise RuntimeError("mirror banks must be 16B aligned for the fused copy")
            pool_ptrs.append(device_ptr(host))
            cache_ptrs.append(cache.data_ptr())
            feat_bytes.append(row_bytes)
        self._mirror = {
            "pool_row_of_id": torch.tensor(fwd, dtype=torch.int32, device=dev),
            "id_of_pool_row": torch.tensor(host_rows, dtype=torch.int32, device=dev),
            # Copy descriptors, (dst row, src row, dst space, src space) per
            # entry (mirror_kernels.KIND_*), one fast_index_copy_kinds launch
            # per group. g1: victim writebacks (slot -> staging ring, or slot
            # -> pool) and slot -> slot relocations of experts a materialize
            # reinstalls that are already GPU-resident (no PCIe traffic, no
            # mirror row). g2: admissions (pool or ring -> slot), compacted:
            # an expert already in its target slot emits nothing.
            "g1": torch.zeros((4, plan), dtype=torch.int32, device=dev),
            "n_g1": torch.zeros((1,), dtype=torch.int64, device=dev),
            "g2": torch.zeros((4, plan), dtype=torch.int32, device=dev),
            "n_g2": torch.zeros((1,), dtype=torch.int64, device=dev),
            # Pre-step slot of every expert: the LRU/materialize kernels rewrite
            # slot_for_id before the swap kernel runs, so a "where was it?"
            # question must be asked of this snapshot.
            "prev_slot_of_id": torch.zeros((pool.total,), dtype=torch.int32, device=dev),
            # Rows owned by nobody, usable as writeback targets. A swap pops
            # at most one per miss; it pushes one back only once retention has
            # filled the pool down to the reserve (see mirror_kernels), so the
            # depth falls to that floor and then oscillates around it.
            "free_rows": torch.zeros((pool.capacity,), dtype=torch.int32, device=dev),
            "free_count": torch.zeros((1,), dtype=torch.int32, device=dev),
            # Rows freed this step, folded into the stack only after every copy
            # is issued (recycling one mid-step would corrupt a pending upload).
            "freed_rows": torch.zeros((plan,), dtype=torch.int32, device=dev),
            "n_freed": torch.zeros((1,), dtype=torch.int32, device=dev),
            "stats": torch.zeros((MIRROR_STAT_COUNT,), dtype=torch.int64, device=dev),
            # Host-visible copy of the same counters, refreshed by a
            # non-blocking D2H after every swap. Reading `stats` directly costs
            # a device sync per step, which is why nothing read it and the
            # faults stayed silent; a pinned buffer costs nothing and may lag a
            # few steps, which is harmless for monotone counters.
            "stats_host": torch.zeros((MIRROR_STAT_COUNT,), dtype=torch.int64, pin_memory=True),
            "pool_ptrs": torch.tensor(pool_ptrs, dtype=torch.int64, device=dev),
            "cache_ptrs": torch.tensor(cache_ptrs, dtype=torch.int64, device=dev),
            "feat_bytes": torch.tensor(feat_bytes, dtype=torch.int64, device=dev),
        }
        self._build_writeback_ring(pool, pool_ptrs, cache_ptrs, feat_bytes)
        self.cache._copy_fused_ok = False  # the mirror path drives the copies itself

    def _build_writeback_ring(self, pool, pool_ptrs, cache_ptrs, feat_bytes) -> None:
        """VRAM staging ring for DMA writebacks (see _resolve_swaps_kernel).

        Sized by wb_stage_rows from the routing geometry and the warm-start
        arena (``FREETOKEN_MIRROR_WB_STAGE_MB=0`` disables it and every
        writeback is an SM store into the pool, the pre-DMA path). Allocated
        here, with the other mirror state and before the first forward, so the
        arena and KV budgets see it.
        """
        m = self._mirror
        dev = self.cache.device
        row_total = sum(feat_bytes)
        # top_k from the model config; a pool built without one (unit tests)
        # gets the one-layer ceiling, i.e. the reserve cap.
        top_k = getattr(getattr(pool, "_config", None), "num_experts_per_tok", None)
        # The cap, a third of the pool's reserve: a ring-full fallback needs a
        # free row with no DMA pending; pending rows are <= ring rows, and the
        # free stack stays near the reserve minus one launch's misses (the
        # reserve is sized for those), so this keeps a clean row available.
        # Measured without the cap: a 32-row ring over an 8-row reserve,
        # never serviced, starved writebacks in tests/moe/test_mirror_retention.
        rows = wb_stage_rows(
            top_k=int(top_k or pool.num_experts), moe_layers=pool.num_layers,
            experts=pool.num_experts, gpu_slots=self.cache.cache_size,
            reserve_rows=pool.reserve_rows, batch=self._decode_batch,
        ) if dev.type == "cuda" else 0
        m["wb_stage_rows"] = rows
        # [0] entries ever staged, [1] entries whose DMA has landed, [2 + s]
        # pool row of ring slot s.
        m["wb_state"] = torch.zeros((2 + rows,), dtype=torch.int64, device=dev)
        # Ring index of the latest staged writeback into each pool row.
        m["wb_pend"] = torch.full((pool.capacity,), -1, dtype=torch.int64, device=dev)
        stage_ptrs = list(cache_ptrs)   # placeholders when the ring is off
        views = []
        if rows:
            for name, row_bytes in zip(self.cache.bank_schema, feat_bytes):
                stage = torch.empty((rows, row_bytes), dtype=torch.uint8, device=dev)
                host = pool.banks[name]
                views.append((host.view(-1).view(torch.uint8).view(host.shape[0], row_bytes),
                              stage))
            stage_ptrs = [stage.data_ptr() for _pool_view, stage in views]
        # Space-major pointer table for fast_index_copy_kinds_jit:
        # KIND_CACHE, KIND_POOL, KIND_STAGE.
        m["kind_ptrs"] = torch.tensor(cache_ptrs + pool_ptrs + stage_ptrs,
                                      dtype=torch.int64, device=dev)
        # Side stream for the per-step host snapshots (service_writebacks).
        self._snap = ({"stream": torch.cuda.Stream(device=dev), "after_step": torch.cuda.Event()}
                      if dev.type == "cuda" else None)
        if not rows:
            self._wb = None
            return
        n_snaps = 4
        self._wb = {
            "rows": rows,
            "views": views,
            "stream": torch.cuda.Stream(device=dev),
            "snaps": [torch.zeros((2 + rows,), dtype=torch.int64, pin_memory=True)
                      for _ in range(n_snaps)],
            "events": [torch.cuda.Event() for _ in range(n_snaps)],
            "next_snap": 0,
            "pending": [],
            "issued": 0,
            "dma_rows": 0,
            "peak_pending": 0,
            # Measurement only: FREETOKEN_MIRROR_WB_TRACE=<file> appends the
            # ring head at every step boundary ("D <head>" at drains), from
            # which the rows staged per step, and so the ring any size would
            # have needed, can be replayed offline (tasks/.../ring_replay.py).
            "trace": (open(os.environ["FREETOKEN_MIRROR_WB_TRACE"], "a", buffering=1 << 16)
                      if os.environ.get("FREETOKEN_MIRROR_WB_TRACE") else None),
        }
        logger.info_rank0(
            "mirror DMA writebacks: %d-row VRAM staging ring (%.1f MiB); "
            "FREETOKEN_MIRROR_WB_STAGE_MB=0 restores SM-store writebacks",
            rows, rows * row_total / 2**20,
        )

    def mirror_stats(self) -> dict:
        """Swap counters (swaps, free evictions, writebacks, coverage faults).

        Unpacks the shared ``stats`` vector BY NAME through ``MirrorStat``
        (``mirror_stats.py``), not by position -- see that module's docstring
        for the two defects the positional form caused.
        """
        if getattr(self, "_mirror", None) is None:
            return {}
        stats = self._mirror["stats"]
        out = mirror_stats_from_vector(stats.tolist())
        # Host-side: ring size and rows DMA'd so far (staged_writebacks minus
        # this is what is still in flight or not yet issued).
        out["wb_stage_rows"] = self._mirror.get("wb_stage_rows", 0)
        out["dma_writebacks"] = self._wb["dma_rows"] if self._wb else 0
        # Most ring entries ever waiting for their DMA at once (a lower bound
        # on the demand when ring_full_fallbacks > 0).
        out["wb_peak_pending"] = self._wb["peak_pending"] if self._wb else 0
        # Rows idle_seed gave to the next eviction victims (host idle boundaries).
        out["idle_seed_rows"] = getattr(self, "idle_seed_totals", {}).get("rows", 0)
        return out

    def mirror_warm_start(self) -> dict:
        """Establish coverage at startup: fill the GPU, then mirror the rest.

        A cold cache holds nothing, so coverage (on_gpu or in_pool) would demand
        a mirror row for all ``L*E`` experts -- exactly the whole-model residency
        this pool exists to avoid. Instead the GPU cache is warm-filled here and
        the mirror takes the complement, which is what the bound
        ``capacity >= L*E - gpu_slots`` was derived from.

        Experts are spread evenly across layers rather than filled in flat id
        order: every token routes through all ``L`` layers, so a GPU holding
        layers 0..k entirely and nothing of the rest would miss constantly.
        """
        # Host writes pool rows below: no staged DMA may land on them later.
        self.drain_writebacks()
        pool = self._mirror_pool
        m = self._mirror
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        # Idempotent in any context: at startup the cache is fresh, but this also
        # runs at the prefill->decode boundary, where stale slot_for_id/id_of_slot
        # claims from before the sweep would otherwise point decode at slots
        # that now hold someone else's bytes.
        self.cache.id_of_slot.fill_(-1)
        self.cache.slot_for_id.fill_(-1)
        self.cache.usage.zero_()

        # Residents are seated from slot 0: the prefill double buffer no
        # longer needs the head of the cache reserved for its exclusive use.
        # _invalidate_prefill_buffer now writes an occupant back to the pool
        # before the fill overwrites it, so a decode resident there is exactly
        # as safe as one anywhere else in the cache.
        base_slot = 0
        resident_slots = self.cache.cache_size

        per_layer = resident_slots // self.cache.num_layers
        extra = resident_slots - per_layer * self.cache.num_layers
        gpu_plan: list[int] = []
        for layer in range(self.cache.num_layers):
            count = per_layer + (1 if layer < extra else 0)
            base = layer * self.cache.num_experts
            gpu_plan.extend(base + e for e in range(min(count, self.cache.num_experts)))
        gpu_plan = gpu_plan[:resident_slots]
        # Stage through the mirror's own pinned rows (overwritten below), a
        # chunk at a time, so no extra host buffer is needed.
        chunk = min(pool.capacity, len(gpu_plan))
        dst_rows = torch.empty((chunk,), dtype=torch.int32, device=self.cache.device)
        src_rows = torch.arange(chunk, dtype=torch.int32, device=self.cache.device)
        count_t = torch.zeros((1,), dtype=torch.int64, device=self.cache.device)
        for begin in range(0, len(gpu_plan), chunk):
            batch = gpu_plan[begin:begin + chunk]
            for i, flat in enumerate(batch):
                pool._read_row(flat, i)
            dst_rows[: len(batch)].copy_(
                torch.tensor([base_slot + begin + i for i in range(len(batch))],
                             dtype=torch.int32)
            )
            count_t.fill_(len(batch))
            fast_index_copy_multi_jit(
                m["cache_ptrs"], m["pool_ptrs"], m["feat_bytes"],
                dst_rows[: len(batch)], src_rows[: len(batch)], count_t,
            )
            torch.cuda.synchronize(self.cache.device)
        ids = torch.tensor(gpu_plan, dtype=torch.int32, device=self.cache.device)
        self.cache.id_of_slot[base_slot: base_slot + len(gpu_plan)] = ids
        self.cache.slot_for_id.view(-1)[ids.long()] = torch.arange(
            base_slot, base_slot + len(gpu_plan), dtype=torch.int32, device=self.cache.device
        )
        # Now the mirror takes everything the GPU does not hold, then spends the
        # slack on duplicates of the GPU's coldest rows -- the ones LFU evicts
        # first, so their writeback is the one worth skipping.
        pool.pool_row_of_id = [-1] * pool.total
        pool.id_of_pool_row = [-1] * pool.capacity
        filled = pool.load_initial(gpu_plan)
        # No literal here: seed_duplicates defaults to the pool's own
        # reserve_rows, which is the number the arena floor was priced
        # against. Passing 3 * num_experts re-stated that constant in a second
        # place, free to drift from the one min_gpu_slots is derived from.
        seeded = pool.seed_duplicates(list(reversed(gpu_plan)))
        m["pool_row_of_id"].copy_(
            torch.tensor(pool.pool_row_of_id, dtype=torch.int32)
        )
        m["id_of_pool_row"].copy_(
            torch.tensor(pool.id_of_pool_row, dtype=torch.int32)
        )
        self._mirror_publish_free_rows()
        logger.info_rank0(
            "mirror warm start: %d experts on GPU (slots %d..%d), %d mirrored, "
            "%d duplicated, %d rows in reserve (%.1f%% of evictions skip the "
            "writeback)",
            len(gpu_plan), base_slot, base_slot + len(gpu_plan),
            filled, seeded, int(m["free_count"].item()),
            100.0 * seeded / max(len(gpu_plan), 1),
        )
        return {"gpu": len(gpu_plan), "mirrored": filled, "duplicates": seeded,
                "reserve": int(m["free_count"].item())}

    def mirror_fault_check(self, fresh: bool = False) -> None:
        """Raise if the swap kernel ever lost an expert's only copy.

        ``resolve_swaps`` cannot raise from inside a Triton kernel, so it counts
        two fatal conditions instead: ``violations`` (an admission whose expert
        was in neither the GPU nor the pool -- the slot keeps whatever bytes it
        held, so the GEMM computes with an unrelated expert) and ``starved`` (no
        free row for a writeback, so a GPU-only victim was dropped). Both mean
        the coverage invariant is gone and every later token in the request is
        suspect.

        This used to be checked nowhere in the server: ``mirror_stats()`` fed
        the /v1/stats document inside a ``try/except`` that swallowed
        everything, and only the offline swap_smoke script compared the counts
        against zero. The kernel comment claiming "the host treats a nonzero
        count as a hard error" was aspirational.

        Not a complete detector -- a stale free-row publish once served wrong
        experts with ``violations`` still at 0 (see _mirror_publish_free_rows) --
        so it is a backstop for the sizing, not a substitute for it.

        The scheduler's check reads the pinned snapshot ``service_writebacks``
        takes once per step (it may lag a step). ``fresh`` syncs and reads the
        device counters now, for callers that drive the kernels directly.
        """
        m = getattr(self, "_mirror", None)
        if m is None:
            return
        if fresh:
            if m["stats"].is_cuda:
                torch.cuda.current_stream(m["stats"].device).synchronize()
            m["stats_host"].copy_(m["stats"])
        violations, starved = mirror_fault_counts_from_vector(m["stats_host"].tolist())
        if violations or starved:
            raise RuntimeError(
                f"bounded expert mirror lost coverage: {violations} admissions "
                f"with no host copy, {starved} dropped writebacks. Output from "
                f"this point is wrong, not merely slow. The pool is too small "
                f"for this KV ceiling: raise --moe-mirror-host-rows (or leave "
                f"it at -1 to auto-size)."
            )

    def _mirror_publish_free_rows(self) -> None:
        """Rebuild the device free stack from the DEVICE ownership map.

        The swap kernel owns ``id_of_pool_row`` and mutates it every step; the
        host-side list in the pool is only the startup snapshot and goes stale
        immediately. Reading the host copy here published rows that were in fact
        owned, so a later swap handed an expert a row holding someone else's
        weights -- decode then served wrong experts with coverage_faults at 0.
        """
        m = self._mirror
        pool = self._mirror_pool
        inv = m["id_of_pool_row"].cpu().tolist()
        # Keep the host mirror of the map in step, so every host-side path
        # (staging, coverage restore) sees what the kernel actually did.
        pool.id_of_pool_row = list(inv)
        pool.pool_row_of_id = m["pool_row_of_id"].cpu().tolist()
        free = [row for row, owner in enumerate(inv) if owner < 0]
        if free:
            m["free_rows"][: len(free)].copy_(
                torch.tensor(free, dtype=torch.int32)
            )
        m["free_count"].fill_(len(free))

    def _mirror_refill_uncovered(self, begin: int = 0, end: int | None = None,
                                 *, from_gpu: bool = False) -> int:
        """Restore coverage for GPU rows about to be dropped.

        Invalidating GPU slots (``reset``, or an arena shrink handing slots to
        the KV cache) destroys the only copy of any expert the mirror does not
        already hold. This runs only at host idle boundaries -- never inside a
        captured graph or a decode step. ``from_gpu`` (the arena shrink: the
        doomed slots are still mapped and hold their experts' bytes) writes those
        rows back device -> pool in one launch, the swap path's D2H; otherwise
        they are re-read from the checkpoint.

        Returns the number of rows re-read.
        """
        # Host writes pool rows below: no staged DMA may land on them later.
        self.drain_writebacks()
        pool = self._mirror_pool
        m = self._mirror
        if end is None:
            end = self.cache.cache_size
        doomed = self.cache.id_of_slot[begin:end]
        ids = [flat for flat in doomed.cpu().tolist() if flat >= 0]
        if not ids:
            return 0
        # Device maps are the source of truth; the pool's host lists lag behind
        # whatever the swap kernel did since the last publish.
        fwd = m["pool_row_of_id"].cpu().tolist()
        inv = m["id_of_pool_row"].cpu().tolist()
        # Rows the mirror already covers need nothing; the rest must be read
        # back into rows that are free (owned by nobody) or hold an expert that
        # stays GPU-resident (a duplicate we can safely overwrite).
        uncovered = [flat for flat in ids if fwd[flat] < 0]
        if not uncovered:
            return 0
        survivors = set(self.cache.id_of_slot.cpu().tolist()) - set(ids)
        spare = [row for row, owner in enumerate(inv)
                 if owner < 0 or (owner in survivors and fwd[owner] == row)]
        if len(spare) < len(uncovered):
            raise RuntimeError(
                f"mirror cannot restore coverage for {len(uncovered)} experts: "
                f"only {len(spare)} rows are free or duplicated"
            )
        rows = spare[: len(uncovered)]
        gpu = from_gpu and self.cache.device.type == "cuda"
        if gpu:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            slot_of = {flat: begin + i for i, flat in enumerate(doomed.tolist()) if flat >= 0}
            dev = self.cache.device
            fast_index_copy_multi_jit(
                m["pool_ptrs"], m["cache_ptrs"], m["feat_bytes"],
                torch.tensor(rows, dtype=torch.int32, device=dev),
                torch.tensor([slot_of[f] for f in uncovered], dtype=torch.int32, device=dev),
                torch.tensor([len(rows)], dtype=torch.int64, device=dev),
            )
            torch.cuda.synchronize(dev)
        for flat, row in zip(uncovered, rows):
            old = inv[row]
            if old >= 0:
                fwd[old] = -1
            if not gpu:
                pool._read_row(flat, row)
            fwd[flat] = row
            inv[row] = flat
        m["pool_row_of_id"].copy_(torch.tensor(fwd, dtype=torch.int32))
        m["id_of_pool_row"].copy_(torch.tensor(inv, dtype=torch.int32))
        # Refresh the host mirror of the maps and the free stack from the device
        # (the kernel owns them between calls); writing pool.* directly above
        # would have been overwritten by the next publish.
        self._mirror_publish_free_rows()
        self.refill_totals = getattr(self, "refill_totals", {"rows": 0, "gpu_rows": 0})
        self.refill_totals["rows"] += len(uncovered)
        self.refill_totals["gpu_rows"] += len(uncovered) if gpu else 0
        logger.info_rank0("mirror restored coverage for %d experts (%s)", len(uncovered),
                          "written back from the GPU" if gpu else "re-read from the checkpoint")
        return len(uncovered)

    def idle_seed(self, should_stop=None, chunk: int = 16) -> dict:
        """Give the residents LFU will evict next a pool row, at a host idle boundary.

        A duplicate (an expert with both a GPU slot and a pool row) makes its
        eviction free; without one the eviction pays a writeback. The swap kernel
        only makes duplicates of experts it has just admitted (retention), and
        those are the ones the next steps route to again: the next victims --
        the coldest residents by the LRU kernel's own key (LFU count, then last
        use) -- mostly have no host copy. Here, while nothing runs, each of them
        gets one: into a free row above the reserve first, then into the row of
        a duplicate held by a strictly hotter resident (which stays on the GPU,
        so coverage is untouched). The bytes are the resident's own slot copied
        GPU -> pool, the arena shrink's refill path (``_mirror_refill_uncovered``),
        so outputs cannot change; the pool keeps its capacity and its reserve.

        Chunks of ``chunk`` rows; ``should_stop()`` (a request is waiting) is
        checked before any work and between chunks, so a request waits at most
        the plan (host lists, ~2-3 ms at 10K experts) or one chunk (16 rows,
        ~30 MiB of D2H on Ornith, ~1 ms on the RTX 5080, where 1218 rows took
        79-91 ms). Slots of the prefill double buffer are left alone on both
        sides: their occupants always retain their rows (see the swap kernel) and are
        vacated by every prefill anyway. ``FREETOKEN_MIRROR_IDLE_SEED=0``
        disables it.
        """
        m = getattr(self, "_mirror", None)
        out = {"seeded": 0, "from_free": 0, "from_hot": 0, "stopped": False}
        if m is None or self.cache.device.type != "cuda":
            return out
        if should_stop is not None and should_stop():
            out["stopped"] = True
            return out
        # Host reads and writes pool rows below: no staged DMA may land later.
        self.drain_writebacks()
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import _lfu_recency_config

        cache = self.cache
        pool = self._mirror_pool
        dev = cache.device
        base = self._mirror_prefill_base()
        ids = cache.id_of_slot.cpu().tolist()
        usage = cache.usage.cpu().tolist()
        fwd = m["pool_row_of_id"].cpu().tolist()
        inv = m["id_of_pool_row"].cpu().tolist()
        lfu = cache.cache_policy_id == 1
        if lfu:
            freq = cache.expert_frequency.view(-1).cpu().tolist()
            recency_tokens, bonus = _lfu_recency_config(cache)
            step = int(cache.step.item())
            window = recency_tokens * cache.num_layers

        def key(slot):
            if not lfu:
                return (usage[slot],)
            recent = bonus if recency_tokens and step - usage[slot] <= window else 0
            return (freq[ids[slot]] + recent, usage[slot])

        live = sorted((s for s in range(base, len(ids)) if ids[s] >= 0), key=key)
        # Next victims without a host copy, coldest first.
        cold = [s for s in live if fwd[ids[s]] < 0]
        # Rows available: free rows above the reserve, then duplicates of the
        # hottest residents (only ever handed to a strictly colder one).
        free_rows = [r for r, owner in enumerate(inv) if owner < 0]
        spare_free = free_rows[: max(0, len(free_rows) - pool.reserve_rows)]
        hot_dups = [s for s in reversed(live) if fwd[ids[s]] >= 0]
        plan = []  # (dst row, src slot, hot expert losing the row or -1)
        hi = 0
        for s in cold:
            if spare_free:
                plan.append((spare_free.pop(), s, -1))
                continue
            if hi >= len(hot_dups) or key(hot_dups[hi]) <= key(s):
                break
            h = hot_dups[hi]
            hi += 1
            plan.append((fwd[ids[h]], s, ids[h]))
        for begin in range(0, len(plan), chunk):
            if should_stop is not None and should_stop():
                out["stopped"] = True
                break
            part = plan[begin:begin + chunk]
            fast_index_copy_multi_jit(
                m["pool_ptrs"], m["cache_ptrs"], m["feat_bytes"],
                torch.tensor([r for r, _, _ in part], dtype=torch.int32, device=dev),
                torch.tensor([s for _, s, _ in part], dtype=torch.int32, device=dev),
                torch.tensor([len(part)], dtype=torch.int64, device=dev),
            )
            torch.cuda.synchronize(dev)
            for row, slot, hot in part:
                if hot >= 0:
                    fwd[hot] = -1
                    out["from_hot"] += 1
                else:
                    out["from_free"] += 1
                fwd[ids[slot]] = row
                inv[row] = ids[slot]
            out["seeded"] += len(part)
            m["pool_row_of_id"].copy_(torch.tensor(fwd, dtype=torch.int32))
            m["id_of_pool_row"].copy_(torch.tensor(inv, dtype=torch.int32))
        if out["seeded"]:
            self._mirror_publish_free_rows()
        totals = getattr(self, "idle_seed_totals", None)
        if totals is None:
            totals = self.idle_seed_totals = {"calls": 0, "rows": 0, "from_hot": 0}
        totals["calls"] += 1
        totals["rows"] += out["seeded"]
        totals["from_hot"] += out["from_hot"]
        return out

    def _mirror_stage_layer(self, layer_id: int) -> int:
        """Make a prefill materialize's sources available, dropping what it evicts.

        ``materialize_layer`` reinstalls a whole layer into slots ``[begin,
        begin+E)`` and invalidates every other resident -- the baseline behaves
        the same way, so during prefill the GPU cache holds one layer, not the
        full working set.

        That has a sharp consequence for a bounded mirror: insisting on the
        decode coverage invariant (every expert on the GPU or mirrored) through
        prefill would force ``L*E - E`` mirror rows -- 2816 rows, 14.74 GiB for
        this model, i.e. the whole model minus one layer. The bound would buy
        nothing.

        So prefill does NOT preserve displaced experts. Their bytes are not lost:
        the checkpoint is immutable, and a later layer that needs one gets it
        re-read here, off the decode hot path. Coverage is re-established for
        decode by ``mirror_warm_start`` / ``_mirror_refill_uncovered``.

        Only the layer's own experts are staged, and only those neither mirrored
        nor GPU-resident (a resident one is relocated slot -> slot by the swap
        kernel, needing no row at all).

        Returns the number of rows read from the checkpoint.
        """
        # Host writes pool rows below: no staged DMA may land on them later.
        self.drain_writebacks()
        pool = self._mirror_pool
        m = self._mirror
        base = layer_id * self.cache.num_experts
        fwd = m["pool_row_of_id"].cpu().tolist()
        inv = m["id_of_pool_row"].cpu().tolist()
        slot_of = self.cache.slot_for_id.view(-1).cpu().tolist()
        need = [base + e for e in range(self.cache.num_experts)
                if fwd[base + e] < 0 and slot_of[base + e] < 0]
        if not need:
            return 0
        # Rows free, then rows whose owner keeps a GPU copy after this call
        # (this layer's experts included: the materialize makes them resident,
        # so their mirror rows become duplicates).
        resident_after = set(range(base, base + self.cache.num_experts))
        free = [r for r, owner in enumerate(inv) if owner < 0]
        if len(free) < len(need):
            spare = [r for r, owner in enumerate(inv)
                     if owner >= 0 and owner in resident_after]
            free.extend(spare[: len(need) - len(free)])
        if len(free) < len(need):
            # Last resort: any row, since prefill does not owe coverage. Keep
            # the mandatory complement (experts the GPU will not hold) last.
            others = [r for r, owner in enumerate(inv)
                      if owner >= 0 and owner not in resident_after]
            free.extend(others[: len(need) - len(free)])
        if len(free) < len(need):
            raise RuntimeError(
                f"mirror cannot stage layer {layer_id}: {len(need)} rows "
                f"needed, {len(free)} available"
            )
        for flat, row in zip(need, free):
            old = inv[row]
            if old >= 0:
                fwd[old] = -1
                pool.pool_row_of_id[old] = -1
            pool._read_row(flat, row)
            fwd[flat] = row
            inv[row] = flat
            pool.pool_row_of_id[flat] = row
            pool.id_of_pool_row[row] = flat
        m["pool_row_of_id"].copy_(torch.tensor(fwd, dtype=torch.int32))
        m["id_of_pool_row"].copy_(torch.tensor(inv, dtype=torch.int32))
        self._mirror_publish_free_rows()
        return len(need)

    def copy_missing_mirror(self) -> None:
        """Issue this step's writebacks and admissions, in that order.

        Ordering is load-bearing and stream-ordered, not synchronized:

          1. g1: gpu[slot] -> ring/pool   reads the slots' *old* (victim)
             bytes, and moves already-resident experts slot -> slot
          2. g2: pool/ring -> gpu[slot]   overwrites the slots with admissions

        The writeback target is never the row an upload reads (``resolve_swaps``
        takes it from the free stack), and rows freed this step are only
        recycled in step 3, after both copies are issued. The GPU slot's own
        read-after-write is resolved by stream order. A staged writeback
        reaches its pool row later, by DMA (``service_writebacks``).
        """
        from freetoken.kernel.fast_index_copy import fast_index_copy_kinds_jit
        from freetoken.moe.mirror_kernels import publish_freed_rows, resolve_swaps

        layer_id = self.cache._pending_src_layer
        assert layer_id is not None, (
            "no staged misses (ensure_experts/materialize_layer first)"
        )
        m = self._mirror
        resolve_swaps(self.cache, layer_id)
        # 1. victims leave their slots and relocations move, BEFORE the
        # uploads: a relocation source is a slot that still holds its old
        # expert, and step 2 is about to overwrite exactly such slots.
        # Issuing the relocation afterwards read admitted bytes instead of
        # the expert being relocated (21K-token completions came out as noise).
        fast_index_copy_kinds_jit(m["kind_ptrs"], m["feat_bytes"], m["g1"], m["n_g1"])
        # 2. admissions enter the GPU, from the compacted descriptor.
        fast_index_copy_kinds_jit(m["kind_ptrs"], m["feat_bytes"], m["g2"], m["n_g2"])
        # 3. only now may this step's vacated rows be reused
        publish_freed_rows(self.cache)
        # The fault counters reach the host once per step, off this stream
        # (service_writebacks): a D2H here was a copy-engine node per layer
        # inside the decode graph, queued behind the writeback DMAs.
        self.cache._pending_src_layer = None
        self.cache._pending_whole_layer = False

    def _mirror_writeback_buffer(self, buffer_id: int) -> None:
        """Preserve a prefill buffer half's occupants before the fill overwrites them.

        Runs on the copy stream, ahead of the fill launches
        ``_prefetch_split_mirror`` issues on that same stream, so ordering
        between "vacate this slot" and "write new bytes into it" is plain
        stream order -- the same discipline ``copy_missing_mirror`` uses for a
        decode swap. Safe against a concurrently running decode step for the
        same reason the frozen prefill snapshots are (see
        ``_init_prefill_overlap_buffers``): under single-lane scheduling this
        session is the one prefilling, so nothing else is admitting into the
        shared free stack / ownership maps while this runs.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.mirror_kernels import writeback_buffer_occupants

        E = self.cache.num_experts
        slot_start = buffer_id * E
        m = self._mirror
        wb = self._mirror_writeback
        with torch.cuda.stream(self.cache.prefill_copy_stream):
            if self.cache._prefill_buffer_has_release_event[buffer_id]:
                self.cache.prefill_copy_stream.wait_event(
                    self.cache.prefill_release_events[buffer_id]
                )
            writeback_buffer_occupants(self.cache, slot_start, E)
            # Forced retention (resolve_swaps) makes every buffer-region
            # occupant a duplicate, so in steady state this launch finds
            # nothing to D2H -- only the vacated-slot list is ever
            # non-empty. Issue the D2H unconditionally and let the
            # device-side count drive how many rows it actually moves
            # (the same pattern copy_missing_mirror uses for m["d2h_*"]),
            # instead of reading the count on the host first just to
            # decide whether to launch: that read is itself a sync, so
            # gating on it bought nothing but a second one below.
            fast_index_copy_multi_jit(
                m["pool_ptrs"], m["cache_ptrs"], m["feat_bytes"],
                wb["d2h_dst"], wb["d2h_src"], wb["n_d2h"],
            )
            m["stats_host"].copy_(m["stats"], non_blocking=True)
            # One host sync per invalidate call, not two: n_d2h and
            # n_vacated are read together off one small pinned copy rather
            # than each forcing its own device sync. The vacated-snapshot
            # patch below still needs a host round trip for correctness
            # (_prefetch_split_mirror's classification reads the frozen
            # numpy snapshot, not the live device tensor -- see
            # _init_prefill_overlap_buffers), so this sync itself cannot be
            # dropped, only shared between the two counts it gates.
            wb["n_d2h_vacated_host"].copy_(
                torch.stack((wb["n_d2h"][0], wb["n_vacated"][0])),
                non_blocking=True,
            )
        self.cache.prefill_copy_stream.synchronize()
        n_d2h = int(wb["n_d2h_vacated_host"][0].item())
        n_vacated = int(wb["n_d2h_vacated_host"][1].item())
        if n_d2h:
            # _prefetch_split_mirror's miss lookup reads this frozen
            # snapshot, not the live tensor (see
            # _init_prefill_overlap_buffers), so a row this call just wrote
            # must land here too -- otherwise a later (or this very) layer
            # would see the pre-writeback -1 and raise a false coverage-lost
            # error for an expert that is, in fact, covered as of this line.
            ids = wb["ids"][:n_d2h].cpu().numpy().astype(np.int64)
            rows = wb["rows"][:n_d2h].cpu().numpy()
            self._mirror_prefill_pool_np[ids] = rows
        if n_vacated:
            # This buffer half's occupants -- ALL of them, duplicate or sole
            # copy -- are no longer GPU-resident once this call returns, but
            # decode may have seated any of them anywhere in the arena
            # (there is no floor keeping them out of the buffer region any
            # more), including a slot a LATER layer this same chunk still
            # expects to read as a hit. Patching the frozen slot snapshot to
            # -1 for every one of them is what keeps that later
            # classification from reading a slot this call already
            # invalidated -- see _writeback_buffer_kernel's docstring.
            vacated = wb["vacated"][:n_vacated].cpu().numpy().astype(np.int64)
            self.cache._prefill_snapshot_np.reshape(-1)[vacated] = -1

    def _prefetch_split_mirror(self, layer_id: int, buffer_id: int) -> None:
        """Assemble one prefill expert layer in the double buffer, without disk.

        The coverage invariant -- every expert on the GPU or in the pool -- is
        exactly the statement that this is always possible. Each of the layer's
        experts is either a decode resident, copied slot -> buffer on the device
        with no PCIe traffic, or it has a pool row, copied host -> buffer over
        PCIe. Each half is one ``fast_index_copy_multi_jit`` launch across all
        banks, on the copy stream under the existing release/ready discipline.
        The two row sets are disjoint and the buffer region is disjoint from the
        residents, so neither half can alias the other.

        This replaces what the branch inherited, where a mirrored prefill drove
        ``materialize_layer``: that reinstalls the layer into the LRU slots and
        invalidates every other resident, which emptied the mirror and made a
        full checkpoint re-read mandatory at every prefill->decode transition
        (1923 + 1021 + 739 rows, 19.3 GiB, measured as 8.0 s of TTFT on an 8K
        prompt whose baseline prefill is 0.34 s). Here the LRU is never
        touched, so coverage survives the sweep and the boundary restore that
        cost never runs.

        Called first, before any of the classification below: this buffer
        half may currently hold a decode resident (the victim floor that used
        to prevent that is gone), and ``_invalidate_prefill_buffer`` must
        write it back -- and patch the pool-row and slot snapshots the
        lookups below read -- before those lookups run. Without this
        ordering, an expert of THIS layer sitting in one of this half's
        slots would still show the pre-writeback -1 a few lines down and
        trip the coverage-lost error for coverage that, by then, actually
        holds.

        The hit test below is ``slots >= 0``, not a threshold against the
        buffer region: a decode resident can now sit ANYWHERE in the arena,
        including the OTHER buffer half (not this call's own), and reading
        that live, not-yet-invalidated slot as a hit is exactly as valid as
        reading one from the ordinary decode region -- the two buffer halves
        are invalidated one at a time, in program order on this same stream,
        so a half not yet touched this call is still exactly what its
        snapshot says. What makes the frozen, once-per-chunk snapshot safe to
        keep reading after the first invalidate is that every invalidate
        patches it in place for every occupant it vacates (see
        ``_mirror_writeback_buffer``); without that patch a later layer could
        read a slot THIS call already handed to someone else.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        E = self.cache.num_experts
        base_flat = layer_id * E
        m = self._mirror
        self.cache._invalidate_prefill_buffer(buffer_id)
        idx_np = self._mirror_prefill_idx_np[buffer_id]

        slots = self.cache._prefill_snapshot_np[layer_id]
        resident = slots >= 0
        pos = np.arange(E, dtype=np.int32)
        dst_base = np.int32(buffer_id * E)

        hit_pos = pos[resident]
        n_hit = int(hit_pos.size)
        idx_np[0, :n_hit] = dst_base + hit_pos
        idx_np[1, :n_hit] = slots[resident]

        miss_pos = pos[~resident]
        n_miss = int(miss_pos.size)
        rows = self._mirror_prefill_pool_np[base_flat + miss_pos]
        if n_miss and int(rows.min()) < 0:
            flat = int(base_flat) + int(miss_pos[int(rows.argmin())])
            raise RuntimeError(
                f"bounded expert mirror lost coverage before prefill: expert "
                f"{flat} (layer {layer_id}) is neither a GPU resident nor in "
                f"the pool, so this layer cannot be assembled. The pool is too "
                f"small for this KV ceiling: raise --moe-mirror-host-rows (or "
                f"leave it at -1 to auto-size)."
            )
        idx_np[2, :n_miss] = dst_base + miss_pos
        idx_np[3, :n_miss] = rows

        self.cache.prefill_hit_rows += n_hit
        self.cache.prefill_total_rows += E

        dev = self._mirror_prefill_idx[buffer_id]
        with torch.cuda.stream(self.cache.prefill_copy_stream):
            if self.cache._prefill_buffer_has_release_event[buffer_id]:
                self.cache.prefill_copy_stream.wait_event(
                    self.cache.prefill_release_events[buffer_id]
                )
            dev.copy_(self._mirror_prefill_idx_host[buffer_id], non_blocking=True)
            if n_hit:
                self._mirror_prefill_hit_count.fill_(n_hit)
                fast_index_copy_multi_jit(
                    m["cache_ptrs"], m["cache_ptrs"], m["feat_bytes"],
                    dev[0, :n_hit], dev[1, :n_hit],
                    self._mirror_prefill_hit_count,
                )
            if n_miss:
                self._mirror_prefill_miss_count.fill_(n_miss)
                fast_index_copy_multi_jit(
                    m["cache_ptrs"], m["pool_ptrs"], m["feat_bytes"],
                    dev[2, :n_miss], dev[3, :n_miss],
                    self._mirror_prefill_miss_count,
                )
            self.cache.prefill_ready_events[buffer_id].record(self.cache.prefill_copy_stream)

    # ------------------------------------------------------------------
    # Moved verbatim from Engine (self -> engine).
    # ------------------------------------------------------------------

    @staticmethod
    def _mirror_final_gpu_slots(engine, config) -> int:
        """GPU slots left for experts once the KV arena reaches its ceiling.

        ``_plan_growable_kv`` answers this exactly but needs the MoE cache to
        exist, and the mirror must be sized before that. This reproduces the
        same budget arithmetic from config alone: total budget minus the KV
        ceiling, divided by the bytes one expert slot costs.
        """
        from freetoken.engine.cache_budget import (
            expert_bytes_per_slot,
            net_cache_budget_bytes,
        )

        mc = config.model_config
        if mc.expert_quant in SOURCED_MIRROR_FORMATS:
            # Shapes from the source (S12c): metadata-only, no file I/O -- see
            # gguf_mirror_bank_shapes. The NVFP4 branch below is unchanged.
            _extents_fn, bank_shapes_fn = sourced_mirror_hooks(mc)
            shapes = bank_shapes_fn(mc)
        else:
            from freetoken.moe.mirror_pool import nvfp4_bank_shapes

            from freetoken.models.nvfp4_banks import expert_source_spec

            spec = expert_source_spec(mc)
            hidden = (getattr(mc, spec.hidden_size_attr) if spec and spec.hidden_size_attr
                      else mc.hidden_size)
            shapes = nvfp4_bank_shapes(
                hidden, mc.moe_intermediate_size,
                gated=bool(getattr(spec, "gated", False)),
            )
        sources = {
            name: [torch.empty((mc.num_experts, *tail), dtype=dtype, device="meta")]
            for name, (tail, dtype) in shapes.items()
        }
        per_slot = expert_bytes_per_slot(sources)
        from freetoken.engine.growable_kv import (
            VMM_COMMIT_CUSHION_BYTES,
            growable_headroom_bytes,
        )
        from freetoken.kvcache.linear_state_pool import state_pool_bytes

        # The SAME budget arithmetic as GrowableKV._plan_growable_kv, which
        # decides the arena's real size at the ceiling once the cache exists:
        # the KV family's fixed cost PLUS the sibling linear-state pool, minus
        # the growable headroom (growable_headroom_bytes: the VMM commit cushion
        # or one prefill chunk's transient, whichever is larger -- priced here
        # from the engine's pre-measurement estimate, since the pool must exist
        # before a forward can measure it; Engine._validate_growable_ceiling
        # re-checks the pool against the measured value). This estimate once
        # omitted the state pool and the cushion and
        # a conservative prefill-buffer term happened to cover them on
        # Nemotron (floor 1440 vs plan 1552); on Ornith (30 GatedDeltaNet
        # layers x 13 state slots = 0.80 GiB of state) it did not: the pool
        # was sized for a 4848-slot arena, the ceiling plan said 4720, and the
        # 250K request died on "growable KV refused an unsafe VMM commit".
        cache_per_page, fixed_cache_size, _tok, _res = engine._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)
        budget = net_cache_budget_bytes(
            config.memory_ratio,
            engine._baseline_free,
            engine._weights_bytes,
            fixed_cache_size,
            getattr(config, "runtime_reserve_bytes", 0),
        ) - growable_headroom_bytes(getattr(engine, "prefill_transient_bytes", 0))
        # The mirror is built before the KV pool exists, so take the ceiling
        # from config (--num-tokens / --num-pages, the growable KV target)
        # rather than self.num_pages, which is set later.
        ceiling_tokens = config.num_token_override or (
            (config.num_page_override or 0) * config.page_size
        )
        kv_ceiling = cache_per_page * (ceiling_tokens // config.page_size)
        # The DMA-writeback staging ring (_build_writeback_ring) is VRAM the
        # live arena fill will see; price it here too, or the pool is sized
        # for an arena that many slots larger than the one it gets. Priced at
        # the ceiling's arena, the smallest one, so the rows priced are never
        # fewer than the rows built from the (larger) warm-start arena.
        total = mc.num_moe_layers * mc.num_experts
        from freetoken.moe.mirror_pool import resolve_reserve_rows

        stage = wb_stage_rows(
            top_k=int(getattr(mc, "num_experts_per_tok", 0) or mc.num_experts),
            moe_layers=mc.num_moe_layers, experts=mc.num_experts,
            gpu_slots=int(max(budget - kv_ceiling, 0) // per_slot),
            reserve_rows=resolve_reserve_rows(mc.num_experts),
            batch=getattr(config, "max_running_req", 1) or 1,
        ) * per_slot
        slots = int(max(budget - kv_ceiling - stage, 0) // per_slot)
        # One margin below the plan, the larger of:
        #   * four arena chunks of release granularity (FREETOKEN_ARENA_STEP_
        #     SLOTS, default 8) so chunk-boundary overshoot stays covered, and
        #   * one more commit cushion in BYTES: the plan's budget is read at
        #     startup, and what is allocated after it (graph pools, allocator
        #     slack) is not priced -- on Ornith the live free VRAM at the
        #     ceiling step ran 0.03 GiB below the plan. Bytes, not slots, so a
        #     model with small experts gets as much slack as one with large.
        # The prefill double buffer is NOT subtracted any more: it no longer
        # owns the head of the arena under the mirror (a resident there is
        # written back before a fill), and the arena's coverage floor already
        # dropped that term (MirrorResidency.attach).
        # Getting this wrong is not a slow path but a dead request, so the
        # engine also checks the built pool's floor against the real ceiling
        # plan at startup (Engine, "Growable-KV ceiling validated") and fails
        # the load there instead of mid-request.
        from freetoken.engine.engine import _arena_step_slots

        step = _arena_step_slots()
        margin = max(4 * step, -(-VMM_COMMIT_CUSHION_BYTES // per_slot))
        return max(min(slots - margin, total), mc.num_experts)


def wb_stage_rows(*, top_k: int, moe_layers: int, experts: int, gpu_slots: int,
                  reserve_rows: int, batch: int = 1) -> int:
    """Rows of the mirror's DMA-writeback staging ring.

    The ring must hold what the resolve kernels stage between two services.
    Under lag-1 servicing (issue_writebacks) the previous step's DMAs land
    early in the next step and every MoE layer's resolve re-reads the landed
    count, so what the ring really has to hold is about ONE step's
    writebacks (replay of the box traces: a ring of one step's demand at a
    given percentile stages about that share of it, ring_replay.py).

    A step's writebacks are its routed misses whose victim has no pool copy.
    Their ceiling is ``top_k * moe_layers * batch`` (every routed expert
    missing); what a step actually stages grows with how much of the model
    is off the GPU, ``uncovered = 1 - gpu_slots / (moe_layers * experts)``.
    Measured per-step writebacks on the box (FREETOKEN_MIRROR_WB_TRACE,
    2026-09-24, both at the 99th percentile): EXL3 Ornith 75 (ceiling 320,
    uncovered 0.41 at the warm-start arena), Nemotron 20 (ceiling 138,
    uncovered 0.22); half the ceiling times the uncovered share predicts
    65 and 15. So the ring is

        rows = ceil(top_k * moe_layers * batch * uncovered / 2)

    with no model constant in it. Bigger is not better: every ring row is an
    arena slot the experts lose for the whole decode (Nemotron 85 rows vs 17:
    swaps +10%, decode -4%; EXL3 256 vs 32: swaps +11%).

    At least 4 rows, at most a third of the pool's reserve (a ring-full
    fallback needs a free pool row with no DMA pending; see
    _build_writeback_ring). ``FREETOKEN_MIRROR_WB_STAGE_ROWS`` overrides the
    rule (measurement), ``FREETOKEN_MIRROR_WB_STAGE_MB=0`` disables the ring
    (SM-store writebacks).
    """
    off = os.environ.get("FREETOKEN_MIRROR_WB_STAGE_MB", "").strip()
    if off and float(off) <= 0:
        return 0
    cap = max(int(reserve_rows) // 3, 0)
    rows_env = os.environ.get("FREETOKEN_MIRROR_WB_STAGE_ROWS", "").strip()
    if rows_env:                      # measurement override (still capped)
        return min(max(int(rows_env), 0), cap)
    total = max(int(moe_layers) * int(experts), 1)
    uncovered = 1.0 - min(max(int(gpu_slots), 0), total) / total
    per_step = int(top_k) * int(moe_layers) * max(int(batch), 1)
    rows = math.ceil(per_step * uncovered / 2)
    return min(max(rows, 4), cap)


# Expert formats whose pool rows come from a model-supplied ``source`` (S12c's
# MirrorExpertPool protocol) instead of an NVFP4 expert source spec.
SOURCED_MIRROR_FORMATS = ("gguf", "exl3")


def sourced_mirror_hooks(mc):
    """``(row_extents_fn(model_path, config), bank_shapes_fn(config))`` for a sourced format;
    refuses a model whose row layout for that format was never verified."""
    if mc.expert_quant == "gguf":
        from freetoken.models.gguf.reader import gguf_mirror_hooks

        hooks = gguf_mirror_hooks(mc)
        export = "gguf_expert_row_extents and gguf_mirror_bank_shapes from that model's gguf module"
    elif mc.expert_quant == "exl3":
        from freetoken.models.exl3_banks import exl3_mirror_hooks

        hooks = exl3_mirror_hooks(mc)
        export = "exl3_expert_spec from that model's package"
    else:
        raise ValueError(f"mirror expert RAM: {mc.expert_quant!r} experts have no sourced pool rows")
    if hooks is None:
        raise ValueError(
            f"mirror expert RAM has no {mc.expert_quant} expert-row hook for model "
            f"type {mc.model_type!r}: it cannot locate expert rows in this "
            f"checkpoint. Export {export} once its layout is verified."
        )
    return hooks


def build_residency(config, mc, engine) -> ExpertResidency:
    """Choose and build the expert residency, before any expert bank is loaded.

    ``config.expert_residency`` is ``"whole"`` or ``"mirror"``;
    ``config.moe_mirror_host_rows`` sizes the mirror (``-1``/``0`` auto-size,
    ``>0`` explicit rows). Environment aliases (``FREETOKEN_MIRROR_EXPERT_RAM``,
    ``FREETOKEN_MIRROR_HOST_ROWS``) are resolved in ``server/args.py`` only.
    The returned residency is not yet bound to a cache: the engine calls
    ``residency.attach(cache, banks)`` once the cache exists.

    Moved from ``Engine._init_offload_moe_cache``'s ``if mirror`` branches
    (validation, reserve resolution, capacity planning, pool construction).
    """
    if config.expert_residency == "whole":
        return WholeModelResidency()
    if config.expert_residency != "mirror":
        raise ValueError(
            f"expert_residency must be 'whole' or 'mirror', got "
            f"{config.expert_residency!r}"
        )
    mirror_rows = config.moe_mirror_host_rows or 0
    cache_factory = getattr(engine.model, "make_offload_moe_cache", None)
    is_sourced = mc.expert_quant in SOURCED_MIRROR_FORMATS

    if (
        cache_factory is not None or config.moe_strategy != "offload"
        or config.moe_pageable_gpu or config.moe_cpu_layers is not None
        or config.use_dummy_weight or config.tp_info.size != 1
        or mc.expert_quant not in ("nvfp4", *SOURCED_MIRROR_FORMATS)
        or (mc.expert_quant == "nvfp4" and config.nvfp4_backend != "triton")
    ):
        raise ValueError(
            "mirror expert RAM requires native NVFP4 experts (triton backend), "
            "GGUF or EXL3 experts, and single-rank GPU offload"
        )
    mirror_spec = None
    sourced_row_extents_fn = None
    if is_sourced:
        sourced_row_extents_fn, _bank_shapes_fn = sourced_mirror_hooks(mc)
    else:
        from freetoken.models.nvfp4_banks import expert_source_spec

        mirror_spec = expert_source_spec(mc)
        if mirror_spec is None:
            raise ValueError(
                f"mirror expert RAM has no expert source spec for model type "
                f"{mc.model_type!r}: it cannot locate expert rows in this "
                f"checkpoint. Export NVFP4_EXPERT_SOURCE_SPEC from that "
                f"model's weight module once its layout is verified."
            )
        if mirror_spec.gated != bool(mc.expert_gated):
            raise ValueError(
                f"mirror expert RAM: spec says gated={mirror_spec.gated} but "
                f"the config says expert_gated={mc.expert_gated}; the row "
                f"layout would be half the size it should be"
            )
    # Decode CUDA graphs stay ON -- that is the whole point: a mirror
    # miss is a plain H2D, exactly what the baseline captures.
    #
    # Prefill overlap stays ON too, and must. Turning it off (as this
    # branch first did, on the grounds that the double buffer streams a
    # layer "straight from the host banks", which the mirror does not
    # have) sends prefill through materialize_layer, which reinstalls
    # the layer into the LRU slots and invalidates every other
    # resident. That empties the mirror, so coverage has to be rebuilt
    # from the checkpoint at every prefill->decode transition: 19.3 GiB
    # of disk per request, measured as 8.0 s of TTFT on an 8K prompt
    # whose baseline prefill is 0.34 s, and 74.7 s at 80K against 9.8 s.
    # The premise was wrong in the first place: a layer's rows do not
    # have to come from the host banks. Coverage says every expert is a
    # GPU resident or has a pool row, so the buffer is assembled from
    # those two places with no disk at all
    # (OffloadMoeCache._prefetch_split_mirror). The buffer region no
    # longer needs to be exclusive to prefill either -- a decode
    # resident there is written back before a fill overwrites it
    # (_invalidate_prefill_buffer) -- so neither the arena's coverage
    # floor nor the pool-capacity estimate (_mirror_final_gpu_slots)
    # prices prefill_buffer_slots(num_experts) as unavailable to decode.
    # Graphs stay ON. The graphs-mode corruption is not a race: a
    # pure replay never runs host code, so the prefill->decode boundary
    # warm start (host + disk, in ensure_experts) could never fire
    # after any post-capture prefill, leaving decode against an empty
    # mirror. The restore is now driven by the scheduler's batch
    # boundary (Scheduler._forward), which is always host-visible,
    # before the replay is admitted.
    logger.info_rank0(
        "Mirror expert RAM: bounded host pool, GPU<->RAM swap, decode "
        "graphs enabled, prefill overlap on (layers assembled from "
        "resident slots + pool rows, no disk)"
    )
    from freetoken.moe.mirror_pool import (
        MirrorExpertPool, plan_capacity, resolve_reserve_rows,
    )
    # Resolved ONCE here and passed to BOTH plan_capacity and
    # MirrorExpertPool below: they must agree on the reserve or
    # the planner sizes the arena floor against a different
    # number than the pool actually withholds (see
    # resolve_reserve_rows / default_reserve_rows docstrings).
    mirror_reserve_rows = resolve_reserve_rows(mc.num_experts)
    logger.info_rank0(
        "Mirror pool: reserve resolved to %d rows%s",
        mirror_reserve_rows,
        (f" (FREETOKEN_MIRROR_RESERVE_ROWS="
         f"{os.environ['FREETOKEN_MIRROR_RESERVE_ROWS']!r})")
        if os.environ.get("FREETOKEN_MIRROR_RESERVE_ROWS", "").strip()
        else " (default: 3 * num_experts)",
    )
    # Size for the KV ceiling, where the GPU cache is smallest and
    # the host side must be largest. Growing a pinned pool later
    # costs ~762 ms/GiB (measured), a stall no request should pay.
    capacity = mirror_rows if mirror_rows > 0 else plan_capacity(
        mc.num_moe_layers, mc.num_experts,
        MirrorResidency._mirror_final_gpu_slots(engine, config),
        reserve=mirror_reserve_rows,
    )
    if is_sourced:
        source = sourced_row_extents_fn(config.model_path, mc)
        _check_mirror_host_ram(capacity, source.shapes)
        mirror_pool = MirrorExpertPool(
            config.model_path, mc.num_moe_layers, mc.num_experts, capacity,
            hidden_size=mc.hidden_size, intermediate_size=mc.moe_intermediate_size,
            source=source, config=mc,
            device=engine.device,
            reserve_rows=mirror_reserve_rows,
        )
    else:
        from freetoken.moe.mirror_pool import nvfp4_bank_shapes

        hidden = (getattr(mc, mirror_spec.hidden_size_attr)
                  if mirror_spec.hidden_size_attr else mc.hidden_size)
        _check_mirror_host_ram(
            capacity,
            nvfp4_bank_shapes(hidden, mc.moe_intermediate_size, gated=mirror_spec.gated),
        )
        mirror_pool = MirrorExpertPool(
            config.model_path, mc.num_moe_layers, mc.num_experts, capacity,
            hidden_size=hidden,
            intermediate_size=mc.moe_intermediate_size,
            spec=mirror_spec, config=mc,
            device=engine.device,
            reserve_rows=mirror_reserve_rows,
        )
    residency = MirrorResidency(mirror_pool)
    residency._decode_batch = max(int(getattr(config, "max_running_req", 1) or 1), 1)
    return residency


# Startup RAM guard (S12c): applies to both NVFP4 and GGUF mirror pools alike.
# Measured on this host at a 256K ceiling, the Ornith Q6_K pool is ~18-20 GiB
# against ~25 GiB MemAvailable -- inside the banks+4GiB rule by only 1-3 GiB,
# not "well under" it -- so a checkpoint whose pool does not actually fit must
# refuse loudly at startup instead of the process getting OOM-killed (or
# worse, partially initialized) minutes into loading.
_MIRROR_HOST_RAM_GUARD_GIB = 4


def _mem_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("mirror expert RAM: /proc/meminfo has no MemAvailable line")


def _check_mirror_host_ram(capacity: int, shapes: dict) -> None:
    from freetoken.moe.mirror_pool import _row_bytes

    pool_bytes = capacity * sum(_row_bytes(tail, dtype) for tail, dtype in shapes.values())
    guard_bytes = _MIRROR_HOST_RAM_GUARD_GIB * (1 << 30)
    available = _mem_available_bytes()
    if pool_bytes + guard_bytes > available:
        raise ValueError(
            f"mirror expert RAM: the pool needs {pool_bytes / 2**30:.2f} GiB "
            f"plus a {_MIRROR_HOST_RAM_GUARD_GIB} GiB headroom, but "
            f"MemAvailable is only {available / 2**30:.2f} GiB (/proc/meminfo). "
            "Reduce --moe-mirror-host-rows or the KV ceiling, or free host RAM."
        )
