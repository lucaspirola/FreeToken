from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    CloseSessionBackendMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    SessionClosedResultMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput
    from .session_spill import SessionSpillRecord


logger = init_logger(__name__)

_ELASTIC_INTERMEDIATE_SHRINK_GRACE_SECONDS = 2.0

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


def _resolve_max_prefill_seqs(config: SchedulerConfig) -> int:
    if config.max_prefill_seqs is not None:
        return config.max_prefill_seqs
    return int(
        config.max_running_req > 1
        and bool(config.kv_grow_step_tokens)
        and config.model_config.gguf_expert_types is not None
    )


def _elastic_target_capacity(initial: int, maximum: int, demand: int) -> int:
    """Smallest enabled request tier that can admit the current live demand."""
    return max(initial, min(maximum, demand))


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


@dataclass
class SessionLease:
    handle: object | None
    ttl_seconds: float
    expires_at: float | None = None
    active_uid: int | None = None
    reclaimable: bool = False
    protected_until: float | None = None
    last_used_at: float = 0.0
    token_ids: torch.Tensor | None = None
    spill: SessionSpillRecord | None = None


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(
            config.max_running_req, self.engine.page_table
        )
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        growable_kv = getattr(self.engine.kv_cache, "growable", False)
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (
                    g.sliding_window
                    for g in config.model_config.kv_cache_group_specs()
                    if g.is_swa
                ),
                None,
            )
            or getattr(self.engine.kv_cache, "sliding_window_size", None),
            committed_pages=(
                self.engine.kv_cache.committed_pages if growable_kv else None
            ),
            page_index_offset=(1 if growable_kv else 0),
        )
        # Second-currency demand signal. ``ensure_mamba_slots`` can only reach UNLOCKED radix
        # snapshots, and an idle automatic session lease holds its node locked for as long as
        # the conversation stays resident -- so without this hook a pool whose whole snapshot
        # cache has become leases reports zero evictable slots and every donation/restore fails.
        self.cache_manager.mamba_reclaim_hook = self._reclaim_soft_sessions_for_state_slot
        self.decode_manager = DecodeManager(config.page_size)
        max_prefill_seqs = _resolve_max_prefill_seqs(config)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            interleave_chunks=bool(
                config.kv_grow_step_tokens and config.max_running_req > 1
            ),
            max_batch_seqs=max_prefill_seqs,
            small_prompt_group_tokens=(
                1280
                if config.max_prefill_seqs is None and max_prefill_seqs == 1
                else 0
            ),
        )
        if max_prefill_seqs:
            logger.info_rank0(
                "Prefill sequence limit: %d (decode remains continuously batched)",
                max_prefill_seqs,
            )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        self._pending_session_close_acks: dict[int, tuple[str, str]] = {}
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = (
            0  # global forward counter; drives the SWA proactive-eviction cadence
        )
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        # Set when a request releases pages. Growable mode checks this at the next no-forward-
        # in-flight boundary, compacts surviving private pages, decommits a free suffix, and
        # spends the returned VRAM on MoE expert slots.
        self._growable_shrink_pending = False
        self._elastic_capacity = (
            config.elastic_initial_requests or config.max_running_req
        )
        self._elastic_resize_pending = False
        self._elastic_shrink_candidate: tuple[int, float] | None = None
        # Chunked prefill and decode use different kernels, so a truly mixed batch is not yet
        # available. Time-slice a short decode burst between helper-prefill chunks: this bounds
        # an existing agent's stream latency while keeping the large prefill kernels efficient.
        self._growable_decode_burst = 32
        self._growable_decode_steps = 0
        # Work-conserving controller for the only scheduler path that cannot mix prefill and
        # decode kernels. Forward timing is sampled after its CUDA completion barrier, so no
        # synchronize is added to the hot path. Keep enough tokens per waiting lane for the
        # efficient grouped-GEMM prefill path while targeting bounded wall-clock slices.
        self._scheduler_prefill_tps_ewma: float | None = None
        self._scheduler_prefill_key: tuple[int, ...] | None = None
        self._scheduler_decode_seconds_ewma: float | None = None
        self._scheduler_prefill_slice_seconds = 8.0
        self._scheduler_decode_slice_seconds = 0.25
        self._scheduler_min_prefill_tokens_per_lane = 2048
        # Opt-in, client-named conversations. A completed turn leaves a radix handle locked;
        # the lease is released only by explicit close, abort/disconnect, or idle expiry.
        self._sessions: dict[str, SessionLease] = {}
        from .session_spill import SessionSpillStore

        self._session_spill_store = SessionSpillStore.create_if_supported(
            self.engine, config
        )
        self._session_spill_last_pressure_check = 0.0
        if self._session_spill_store is not None:
            logger.info_rank0(
                "Cold session tier enabled: RAM %.2f GiB, disk %.2f GiB, retained "
                "%.2f GiB total (%s), host reserve %.2f GiB",
                config.session_spill_ram_gb,
                config.session_spill_disk_gb,
                config.session_spill_limit_gb,
                "persistent" if config.session_spill_persist else "wiped on exit",
                config.host_ram_reserve_gb,
            )
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap)
            if _chunk_cap
            else config.max_extend_tokens
        )
        self.config = config
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )
        self._last_moe_stats_calls = 0
        self._pageable_cost_history: dict[int, float] = {}
        self._pageable_trial: tuple[int, int] | None = None
        self._pageable_rejected: set[tuple[int, int]] = set()
        self._pageable_retune_disabled = False

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        moe = self.engine.moe_offload_cache
        if self.config.moe_collect_stats and moe is not None:
            stats = moe.decode_miss_stats()
            calls = int(stats["layer_calls"])
            if calls > self._last_moe_stats_calls:
                logger.info_rank0("MoE decode miss stats: %s", stats)
                per_layer = moe.decode_miss_stats_per_layer()["per_layer"]
                # One compact machine-readable line for every layer, pageable or not, so
                # cache-sizing runs can scrape the whole profile: json.dumps, not repr.
                logger.info_rank0(
                    "MoE decode miss stats per layer: %s",
                    json.dumps(per_layer, default=float),
                )
                pageable = [row for row in per_layer if row["pageable_stage_calls"]]
                if pageable:
                    hottest = sorted(
                        pageable,
                        key=lambda row: (
                            row["pageable_plan_wait_seconds"]
                            + row["pageable_gather_seconds"]
                        ),
                        reverse=True,
                    )[:8]
                    logger.info_rank0("MoE pageable layer stalls (top 8): %s", hottest)
                    candidates = sorted(
                        per_layer,
                        key=lambda row: (row["missing_per_step"], row["miss_rate"]),
                    )[: len(pageable)]
                    logger.info_rank0(
                        "MoE lowest-miss pageable candidates (%d): %s",
                        len(candidates),
                        candidates,
                    )
                    self._maybe_retune_pageable_layers(per_layer)
                self._last_moe_stats_calls = int(moe.decode_miss_stats()["layer_calls"])
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def _maybe_retune_pageable_layers(self, per_layer: list[dict]) -> None:
        """Explore one measured host-residency swap at a full idle boundary."""
        moe = self.engine.moe_offload_cache
        if (
            getattr(self.config, "moe_pageable_profile", "off") != "train"
            or self._pageable_retune_disabled
            or moe is None
            or not moe.pageable_gpu
            or self.config.tp_info.size > 1
            or not moe._unpinned_layers
        ):
            return
        if min((int(row["steps"]) for row in per_layer), default=0) < 128:
            return

        pageable = set(moe._unpinned_layers)
        row_seconds = [
            row["pageable_gather_seconds"] / row["pageable_rows"]
            for row in per_layer
            if row["pageable_rows"] and row["pageable_gather_seconds"] > 0
        ]
        if not row_seconds:
            return
        row_seconds.sort()
        median_row_seconds = row_seconds[len(row_seconds) // 2]
        for row in per_layer:
            layer = int(row["layer"])
            if layer in pageable and row["steps"]:
                measured = row["pageable_gather_seconds"] / row["steps"]
                prior = self._pageable_cost_history.get(layer)
                self._pageable_cost_history[layer] = (
                    measured if prior is None else 0.5 * prior + 0.5 * measured
                )

        def cost(layer: int) -> float:
            if layer in self._pageable_cost_history:
                return self._pageable_cost_history[layer]
            return float(per_layer[layer]["missing_per_step"]) * median_row_seconds

        # Save a complete ranking, rather than only today's pageable subset, so
        # startup can select any count dictated by the next run's RAM/pin budget.
        # This is the safe default on WSL: measurement happens at an idle boundary,
        # while host registration and graph capture happen on the next clean start.
        ranking = sorted(range(moe.num_layers), key=cost)
        from freetoken.moe.placement import save_pageable_ranking

        try:
            profile = save_pageable_ranking(
                self.config.model_path,
                ranking,
                [cost(layer) for layer in ranking],
            )
            recommended = sorted(ranking[: len(pageable)])
            if set(recommended) != pageable:
                logger.info_rank0(
                    "Measured pageable placement for next startup: %s (current %s, "
                    "profile %s)",
                    recommended,
                    sorted(pageable),
                    profile,
                )
        except (OSError, ValueError) as exc:
            logger.warning_rank0("Could not save pageable placement profile: %s", exc)

        # Live pin/unpin remains an expert-only diagnostic. cudaHostRegister can
        # leave WSL's CUDA context unusable when the driver rejects a swap.
        if os.getenv("FREETOKEN_MOE_LIVE_RETUNE", "0") != "1":
            return

        if self._pageable_trial is not None:
            old_layer, new_layer = self._pageable_trial
            old_cost = self._pageable_cost_history.get(old_layer)
            new_cost = self._pageable_cost_history.get(new_layer)
            if (
                old_cost is not None
                and new_cost is not None
                and new_cost > old_cost * 1.05
            ):
                target = frozenset((pageable - {new_layer}) | {old_layer})
                logger.info_rank0(
                    "Reverting pageable placement trial %d -> %d: measured %.3f ms/layer "
                    "versus %.3f ms/layer",
                    old_layer,
                    new_layer,
                    new_cost * 1e3,
                    old_cost * 1e3,
                )
                try:
                    self.engine.retune_pageable_layers(target)
                except Exception as exc:  # keep serving on registration rejection
                    logger.warning_rank0("Disabling pageable retuning: %s", exc)
                    self._pageable_retune_disabled = True
                self._pageable_rejected.add((old_layer, new_layer))
                self._pageable_trial = None
                self._last_moe_stats_calls = 0
                return
            self._pageable_trial = None

        worst = max(pageable, key=cost)
        candidates = [
            layer
            for layer in range(moe.num_layers)
            if layer not in pageable and (worst, layer) not in self._pageable_rejected
        ]
        if not candidates:
            return
        best = min(candidates, key=cost)
        if cost(best) >= cost(worst) * 0.85:
            return
        target = frozenset((pageable - {worst}) | {best})
        logger.info_rank0(
            "Pageable placement trial %d -> %d at idle: predicted %.3f -> %.3f "
            "ms/layer from measured gather cost",
            worst,
            best,
            cost(worst) * 1e3,
            cost(best) * 1e3,
        )
        try:
            self.engine.retune_pageable_layers(target)
        except Exception as exc:
            logger.warning_rank0("Disabling pageable retuning: %s", exc)
            self._pageable_retune_disabled = True
            return
        self._pageable_trial = (worst, best)
        self._last_moe_stats_calls = 0

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size,
            num_pages=num_pages,
            num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if (
            num_pages is not None
            or num_mamba_slots is not None
            or num_swa_pages is not None
        ):
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap
            else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        self._expire_sessions()
        self._release_due_soft_sessions()
        self._enforce_session_host_reserve()
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild
            is not None  # a queued rebuild to drain toward + execute
            or getattr(self, "_growable_shrink_pending", False)
            or self._sessions_need_service()
        )
        messages = self.receive_msg(blocking=blocking)
        if not messages and not blocking and self._only_idle_sessions(last_data):
            time.sleep(0.01)
        for msg in messages:
            self._process_one_msg(msg)

        self._maybe_shrink_growable_kv()
        self._maybe_resize_elastic_capacity()

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if (
            self._pending_rebuild is not None
            and last_data is None
            and not (self.prefill_manager.runnable or self.decode_manager.runnable)
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        self._expire_sessions()
        self._release_due_soft_sessions()
        self._enforce_session_host_reserve()
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
            or getattr(self, "_growable_shrink_pending", False)
            or self._sessions_need_service()
        )
        messages = self.receive_msg(blocking=blocking)
        if not messages and not blocking and self._only_idle_sessions(None):
            time.sleep(0.01)
        for msg in messages:
            self._process_one_msg(msg)

        self._maybe_shrink_growable_kv()
        self._maybe_resize_elastic_capacity()

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING or self.config.kv_grow_step_tokens:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        spill_store = getattr(self, "_session_spill_store", None)
        if spill_store is not None:
            spill_store.shutdown()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        # Several low-level drain tests intentionally invoke this method with a
        # minimal scheduler-shaped stub. Runtime schedulers always expose the
        # observer; keeping it optional here preserves that narrow test seam.
        observe_batch = getattr(self, "_observe_scheduler_batch", None)
        if observe_batch is not None:
            observe_batch(batch)
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                # EOS / stop-string -> "stop", output budget exhausted -> "length";
                # EOS and stop strings win over length.
                hit_length = not req.can_decode
                hit_eos = (
                    not req.sampling_params.ignore_eos
                    and next_token in self.eos_token_ids
                )
                matched_stop = (
                    self._match_stop_str(req)
                    if not hit_eos and req.sampling_params.stop_strs
                    else None
                )
                finished = hit_length or hit_eos or matched_stop is not None
                finish_reason = (
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    if getattr(req, "session_id", None) is not None:
                        self._free_req_resources(req, retain_session=True)
                    else:
                        self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        self.send_result(reply)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission",
                    msg.uid,
                )
                return
            if msg.session_id is not None:
                if not msg.session_id or len(msg.session_id) > 256:
                    self.send_result(
                        [ErrorReplyMsg(uid=msg.uid, error="invalid session_id")]
                    )
                    return
                if not hasattr(self, "_sessions"):
                    self._sessions = {}
                session = self._sessions.get(msg.session_id)
                if session is not None and session.active_uid is not None:
                    self.send_result(
                        [
                            ErrorReplyMsg(
                                uid=msg.uid, error=f"session {msg.session_id!r} is busy"
                            )
                        ]
                    )
                    return
                self._reclaim_soft_sessions_for_admission(msg)
                ttl = float(msg.session_ttl_seconds or 300.0)
                if not math.isfinite(ttl):
                    ttl = 300.0
                ttl = min(max(ttl, 1.0), 86_400.0)
                if session is None:
                    session = self._sessions[msg.session_id] = SessionLease(
                        None, ttl, reclaimable=msg.session_reclaimable
                    )
                session.ttl_seconds = ttl
                session.reclaimable = msg.session_reclaimable
                session.expires_at = None
                session.protected_until = None
                session.last_used_at = time.monotonic()
                session.active_uid = msg.uid
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            # "maximum context length" is OpenAI's phrasing, which Switchyard and
                            # the OpenAI SDKs match on — both spellings ride in the one message.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(this model's maximum context length, prompt + generation); "
                                f"shorten the prompt or increase the KV cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                if msg.session_id is not None:
                    session = self._sessions.get(msg.session_id)
                    if session is not None:
                        session.active_uid = None
                        session.expires_at = time.monotonic() + session.ttl_seconds
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            if msg.session_id is not None:
                self._restore_cold_session(msg.session_id, msg.input_ids)
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            if msg.session_id is not None:
                # A disconnect ends the lease; the checkpoint stays for a reconnect.
                self._close_session(msg.session_id, discard_state=False)
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CloseSessionBackendMsg):
            existed, active_uid = self._close_session(msg.session_id)
            if active_uid is None:
                self.send_result(
                    [
                        SessionClosedResultMsg(
                            session_id=msg.session_id,
                            request_id=msg.request_id,
                            status="closed" if existed else "not_found",
                        )
                    ]
                )
            else:
                self._pending_session_close_acks[active_uid] = (
                    msg.request_id,
                    msg.session_id,
                )
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id,
                    "unsupported",
                    "this model's cache does not support runtime rebuild",
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id,
                    "unsupported",
                    f"mode {msg.mode!r} unsupported (use if_idle)",
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id,
                    "unsupported",
                    "runtime rebuild unsupported under TP > 1",
                )
            elif (
                self.prefill_manager.runnable
                or self.decode_manager.runnable
                or getattr(self, "_sessions", {})
            ):
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req, *, retain_session: bool = False) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        if retain_session and req.session_id is not None and req.mm_embeds is None:
            session = self._sessions.get(req.session_id)
            if session is not None and session.active_uid == req.uid:
                new_handle = self.cache_manager.retain_prefix(
                    req.input_ids, req.cached_len
                )
                old_handle = session.handle
                self._discard_session_spill(session)
                session.handle = new_handle
                retained_len = int(getattr(new_handle, "cached_len", req.cached_len))
                session.token_ids = torch.as_tensor(
                    req.input_ids[:retained_len], device="cpu", dtype=torch.int32
                )
                session.active_uid = None
                now = time.monotonic()
                # A reclaimable lease that still holds GPU state does NOT age out: its
                # residency ends on demand (an admission that needs the space), never on a
                # clock. The idle TTL starts only once the state has been checkpointed --
                # from then on it bounds the empty lease object, not the computation.
                session.expires_at = (
                    None if session.reclaimable else now + session.ttl_seconds
                )
                session.last_used_at = now
                grace = max(0.0, float(self.config.auto_session_grace_seconds))
                # Grace 0 (the default) means "resident until something needs the slot":
                # the timer is a safety net, not the normal release path.
                session.protected_until = (
                    now + grace if session.reclaimable and grace > 0 else None
                )
                if old_handle is not None:
                    self.cache_manager.unlock(old_handle)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1
        if getattr(getattr(self, "config", None), "kv_grow_step_tokens", 0):
            self._growable_shrink_pending = True
        if getattr(getattr(self, "config", None), "elastic_initial_requests", None):
            self._elastic_resize_pending = True

    def _close_session(
        self, session_id: str, *, discard_state: bool = True
    ) -> tuple[bool, int | None]:
        """Release a lease. ``discard_state`` also destroys its cold checkpoint.

        Only an explicit client DELETE discards. Idle expiry and disconnect end the lease
        but keep any existing checkpoint under the spill store's own capacity/age policy,
        so a later request with the same prefix restores it instead of recomputing. Closing
        never *creates* a checkpoint: spilling is demand-driven, and a cancel/disconnect
        must not pay a full device-to-host copy.
        """
        session = getattr(self, "_sessions", {}).pop(session_id, None)
        store = getattr(self, "_session_spill_store", None)
        if store is not None:
            # Abort, disconnect or idle expiry: nobody is waiting on this promotion.
            store.cancel_prefetch(session_id)
        if session is None:
            return False, None
        if session.handle is not None:
            self.cache_manager.unlock(session.handle)
        if discard_state:
            self._discard_session_spill(session)
        else:
            session.spill = None  # the store keeps it, keyed by session id
        if session.active_uid is not None:
            req = self.prefill_manager.abort_req(session.active_uid)
            req = req or self.decode_manager.abort_req(session.active_uid)
            if req is not None:
                inflight = (
                    self._last_data is not None and req in self._last_data[0].batch.reqs
                )
                if inflight:
                    req.aborted = True
                else:
                    self._free_req_resources(req)
            self._pending_abort_acks.add(session.active_uid)
        self._growable_shrink_pending = True
        logger.info_rank0(
            "Closed session %s; retained KV is now reclaimable", session_id
        )
        return True, session.active_uid

    def _expire_sessions(self) -> None:
        now = time.monotonic()
        expired = [
            sid
            for sid, lease in getattr(self, "_sessions", {}).items()
            if lease.active_uid is None
            and lease.expires_at is not None
            and lease.expires_at <= now
            # Resident automatic sessions are demand-evicted, never time-evicted.
            and not (lease.reclaimable and lease.handle is not None)
        ]
        for sid in expired:
            logger.info_rank0("Session %s expired after idle timeout", sid)
            self._close_session(sid, discard_state=False)

    def _release_soft_session_handle(self, session_id: str, reason: str) -> bool:
        session = getattr(self, "_sessions", {}).get(session_id)
        if (
            session is None
            or not session.reclaimable
            or session.active_uid is not None
            or session.handle is None
        ):
            return False
        self._spill_soft_session(session_id, session)
        self.cache_manager.unlock(session.handle)
        session.handle = None
        session.protected_until = None
        # The lease no longer pins GPU state, so the idle TTL may now reap the identity.
        # Its checkpoint outlives it under the spill store's capacity/age policy.
        session.expires_at = time.monotonic() + session.ttl_seconds
        self._growable_shrink_pending = True
        logger.info_rank0(
            "Released soft session %s KV protection (%s); cached prefix is now evictable",
            session_id,
            reason,
        )
        return True

    def _discard_session_spill(self, session: SessionLease) -> None:
        store = getattr(self, "_session_spill_store", None)
        if store is not None and session.spill is not None:
            store.discard(session.spill)
        session.spill = None

    def _enforce_session_host_reserve(self) -> None:
        store = getattr(self, "_session_spill_store", None)
        if store is None:
            return
        now = time.monotonic()
        if now - getattr(self, "_session_spill_last_pressure_check", 0.0) < 1.0:
            return
        self._session_spill_last_pressure_check = now
        demoted, dropped = store.enforce_host_reserve()
        if dropped:
            for session in self._sessions.values():
                if session.spill is not None and not session.spill.valid:
                    session.spill = None
        if demoted or dropped:
            logger.warning(
                "Cold-session host pressure: moved %d checkpoint(s) RAM -> disk, "
                "dropped %d; %.2f GiB MemAvailable reserve remains mandatory",
                demoted,
                dropped,
                self.config.host_ram_reserve_gb,
            )

    def _spill_soft_session(self, session_id: str, session: SessionLease) -> None:
        store = getattr(self, "_session_spill_store", None)
        handle = session.handle
        if store is None or handle is None or session.token_ids is None:
            return
        node = getattr(handle, "node", None)
        linear_slot = getattr(node, "mamba_value", None)
        page_indices = handle.get_matched_indices()
        if linear_slot is None or len(page_indices) != len(session.token_ids):
            return
        self._discard_session_spill(session)
        started = time.perf_counter()
        record = store.spill(session_id, session.token_ids, page_indices, linear_slot)
        elapsed = time.perf_counter() - started
        if record is None:
            logger.warning(
                "Cold checkpoint for session %s did not fit RAM/disk budgets; "
                "resume will recompute if its GPU prefix is evicted",
                session_id,
            )
            return
        session.spill = record
        logger.info_rank0(
            "Spilled soft session %s: %d tokens, %s, %.2f GiB in %.3f s (%.2f GiB/s)",
            session_id,
            record.num_pages,
            record.tier,
            record.byte_size / (1 << 30),
            elapsed,
            record.byte_size / (1 << 30) / max(elapsed, 1e-9),
        )

    @torch.inference_mode()
    def _restore_cold_session(self, session_id: str, input_ids: torch.Tensor) -> bool:
        session = getattr(self, "_sessions", {}).get(session_id)
        store = getattr(self, "_session_spill_store", None)
        if session is None or store is None:
            return False
        # The store is the owner: a checkpoint outlives its lease (idle expiry, disconnect,
        # or a server restart), so look it up by id rather than through the lease object.
        record = session.spill or store.get(session_id)
        if record is None:
            return False
        # A look-ahead promotion of this very record is worth waiting out: it is reading
        # the same bytes this restore needs, only into RAM.
        store.collect_prefetch(session_id, wait=True)
        session.spill = record
        store.touch(record)
        # The final prompt token must still run through prefill. Never install a checkpoint
        # that reaches beyond the client's exact reusable prefix.
        if record.num_pages > max(0, len(input_ids) - 1) or not torch.equal(
            record.token_ids,
            input_ids[: record.num_pages].to(device="cpu", dtype=torch.int32),
        ):
            self._discard_session_spill(session)
            logger.info_rank0(
                "Discarded cold session %s: client token prefix changed", session_id
            )
            return False

        cm = self.cache_manager
        try:
            missing, allocatable = cm.hybrid_session_restore_geometry(record.token_ids)
            if missing > allocatable:
                required = cm.committed_pages + missing - allocatable
                old_pages, new_pages = self.engine.grow_runtime_kv(required)
                if new_pages > old_pages:
                    cm.add_committed_pages(new_pages)
                    logger.info_rank0(
                        "KV grew %d -> %d tokens to restore cold session %s",
                        old_pages,
                        new_pages,
                        session_id,
                    )
            tier, tokens, nbytes = record.tier, record.num_pages, record.byte_size
            started = time.perf_counter()
            session.handle = cm.restore_hybrid_session_prefix(record, store)
            elapsed = time.perf_counter() - started
            self._discard_session_spill(session)
            logger.info_rank0(
                "Restored cold session %s: %d tokens from %s, %.2f GiB in %.3f s (%.2f GiB/s)",
                session_id,
                tokens,
                tier,
                nbytes / (1 << 30),
                elapsed,
                nbytes / (1 << 30) / max(elapsed, 1e-9),
            )
            return True
        except Exception as exc:  # reuse is an optimization, never an admission gate
            # Keep the checkpoint: the usual failure is a pool that the resident session
            # still owns, and the retry after its release (_reclaim_for_blocked_prefill)
            # is exactly what makes a queued 1M session restore instead of recompute.
            logger.warning(
                "Cold restore for session %s failed (%r); will retry before admission",
                session_id,
                exc,
            )
            session.handle = None
            return False

    def _session_resource_pressure(self) -> bool:
        """True when the live pools cannot seat anything without reclaiming a lease."""
        cm = getattr(self, "cache_manager", None)
        if cm is None:
            return False
        try:
            if getattr(cm, "is_hybrid", False) and cm.mamba_available_size < 3:
                return True
            return cm.available_size <= 0
        except Exception:  # pressure probing must never break the loop
            return False

    def _release_due_soft_sessions(self) -> None:
        """Safety net only: a resident session is never spilled while nobody needs it.

        ``--auto-session-grace-seconds 0`` (the default) disables the timer entirely; with a
        timer configured, an expired grace still releases only under real demand -- a queued
        request or exhausted pools. Admission failure is the normal trigger
        (:meth:`_reclaim_for_blocked_prefill`).
        """
        config = getattr(self, "config", None)
        grace = max(0.0, float(getattr(config, "auto_session_grace_seconds", 0.0) or 0.0))
        if grace <= 0:
            return
        if not (
            getattr(getattr(self, "prefill_manager", None), "runnable", False)
            or self._session_resource_pressure()
        ):
            return
        now = time.monotonic()
        due = [
            sid
            for sid, lease in getattr(self, "_sessions", {}).items()
            if lease.reclaimable
            and lease.active_uid is None
            and lease.handle is not None
            and lease.protected_until is not None
            and lease.protected_until <= now
        ]
        for sid in due:
            self._release_soft_session_handle(sid, "grace expired")

    def _reclaim_soft_sessions_for_admission(self, msg: UserMsg) -> bool:
        """Release oldest idle automatic leases only when they block this admission."""
        from .utils import PendingReq

        pending = PendingReq(msg.uid, msg.input_ids, msg.sampling_params)
        return self._reclaim_soft_sessions_for_pending(
            pending, getattr(msg, "session_id", None)
        )

    def _reclaim_soft_sessions_for_pending(self, pending, session_id: str | None) -> bool:
        """Checkpoint the least-recently-used idle soft leases blocking ``pending``."""
        candidates = sorted(
            (
                (lease.last_used_at, sid)
                for sid, lease in getattr(self, "_sessions", {}).items()
                if sid != session_id
                and lease.reclaimable
                and lease.active_uid is None
                and lease.handle is not None
            ),
            key=lambda item: item[0],
        )
        if not candidates:  # cheap gate: skip the prefix match when nothing can be freed
            return False
        cm = self.cache_manager
        try:
            cached_len = cm.match_req(pending).cuda_handle.cached_len
        except (
            Exception
        ):  # matching is repeated by admission; stay conservative on failure
            cached_len = 0
        needed = max(0, pending.input_len - cached_len) + pending.output_len

        def pressured() -> bool:
            kv_short = needed > cm.available_size
            state_short = cm.is_hybrid and cm.mamba_available_size < 3
            return kv_short or state_short

        released = False
        for _last_used, sid in candidates:
            if not pressured():
                break
            released |= self._release_soft_session_handle(sid, "admission pressure")
        return released

    def _reclaim_soft_sessions_for_state_slot(self, n: int = 1) -> bool:
        """Checkpoint LRU idle automatic leases until ``n`` GDN state slots are reachable.

        The KV-shaped reclaim above is driven by a *queued* request. A recurrent-state slot is
        also needed at moments no admission covers -- when a prefill chunk commits its snapshot,
        or a cold session is restored -- and an idle lease pins its snapshot node against
        ``evict_mamba``, so without this nothing would ever release it. Spilling the idle
        conversation rather than failing the live one is the 3E residency policy: no spill while
        nobody needs the slot, an on-demand checkpoint the moment somebody does.

        Stops as soon as the demand is met -- a per-allocation signal, not a bulk drain.
        """
        cm = getattr(self, "cache_manager", None)
        if cm is None or not getattr(cm, "is_hybrid", False):
            return False
        candidates = sorted(
            (
                (lease.last_used_at, sid)
                for sid, lease in getattr(self, "_sessions", {}).items()
                if lease.reclaimable
                and lease.active_uid is None
                and lease.handle is not None
            ),
            key=lambda item: item[0],
        )
        released = False
        for _last_used, sid in candidates:
            if cm.mamba_available_size >= n:
                break
            released |= self._release_soft_session_handle(sid, "GDN state-slot pressure")
        return released

    def _reclaim_for_blocked_prefill(self) -> bool:
        """A queued request just failed admission: free the oldest idle session slot.

        This is the demand signal that replaces the idle grace timer. A session that was
        mid-turn when its competitor arrived becomes a candidate the instant its turn ends,
        so the queued request is admitted on the next scheduler iteration.
        """
        self._prefetch_queued_session()
        for pending in getattr(self.prefill_manager, "pending_list", ()):
            if pending.chunked_req is not None:
                continue  # a continuation already owns its resources
            released = self._reclaim_soft_sessions_for_pending(pending, pending.session_id)
            if released and pending.session_id:
                # The competitor's KV and state slot are free now, so a checkpoint that
                # could not be installed at message receipt gets its second chance -- from
                # RAM if the look-ahead above finished in time.
                self._restore_cold_session(pending.session_id, pending.input_ids)
            return released
        return False

    def _prefetch_queued_session(self) -> str | None:
        """Promote the next queued session's disk checkpoint to RAM while another runs.

        NVMe restore of a 1M checkpoint costs ~2.5 s against ~0.13 s from RAM, and a queued
        request has the whole of the resident session's turn to wait. Best effort: a refused
        promotion (budget, host reserve, one already in flight) just leaves it on disk.
        """
        store = getattr(self, "_session_spill_store", None)
        if store is None:
            return None
        promoted = store.collect_prefetch()
        sessions = getattr(self, "_sessions", {})
        resident = {sid for sid, lease in sessions.items() if lease.handle is not None}
        for pending in getattr(self.prefill_manager, "pending_list", ()):
            session_id = getattr(pending, "session_id", None)
            lease = sessions.get(session_id) if session_id else None
            if not session_id or session_id in resident:
                continue
            if lease is not None and not lease.reclaimable:
                continue  # explicit leases are never spilled: there is nothing to promote
            if store.start_prefetch(session_id, protect=resident):
                break
        return promoted

    def _sessions_need_service(self) -> bool:
        """True while a lease deadline or a live checkpoint still needs periodic polling.

        A resident automatic session has no deadline (it is released on demand), so it must
        not keep the receive loop spinning: without this the scheduler would poll at 100 Hz
        for the whole life of an idle conversation.
        """
        for lease in getattr(self, "_sessions", {}).values():
            if lease.expires_at is not None or lease.protected_until is not None:
                return True
        store = getattr(self, "_session_spill_store", None)
        return bool(store is not None and store.num_records)

    def _only_idle_sessions(self, last_data: ForwardData | None) -> bool:
        return (
            last_data is None
            and not self.prefill_manager.runnable
            and not self.decode_manager.runnable
            and self._pending_rebuild is None
            and not getattr(self, "_growable_shrink_pending", False)
        )

    @torch.inference_mode()
    def _maybe_shrink_growable_kv(self) -> None:
        """Physically release unused KV steps and restore expert residency after teardown."""
        grow_step_tokens = getattr(
            getattr(self, "config", None), "kv_grow_step_tokens", 0
        )
        if not grow_step_tokens or not getattr(self, "_growable_shrink_pending", False):
            return
        # A queued helper is about to consume the space again. Deferring avoids a costly
        # decommit/rebuild/recapture immediately followed by the inverse operation.
        if self.prefill_manager.runnable:
            return
        self._growable_shrink_pending = False
        cm = self.cache_manager
        step = grow_step_tokens // self.config.page_size
        initial = min(cm.num_pages, step)
        if cm.committed_pages <= initial:
            return

        evicted = cm.evict_all_unlocked_prefixes()
        occupied_pages = cm.committed_pages - len(cm.free_slots)
        compacted_target = max(initial, math.ceil(occupied_pages / step) * step)
        compacted_target = cm.compact_active_pages(
            list(self.decode_manager.running_reqs),
            compacted_target,
            self.engine.kv_cache.copy_pages,
        )
        target = max(initial, math.ceil(compacted_target / step) * step)
        if target >= cm.committed_pages:
            logger.info_rank0(
                "Growable KV teardown evicted %d prefix pages; protected/live pages keep "
                "%d tokens committed",
                evicted,
                cm.committed_pages,
            )
            return

        old_pages, new_pages = self.engine.shrink_runtime_kv(target)
        if new_pages < old_pages:
            cm.remove_committed_pages(new_pages)
            logger.info_rank0(
                "KV shrank %d -> %d tokens after agent teardown; MoE cache restored to "
                "%d slots",
                old_pages,
                new_pages,
                self.engine.moe_offload_cache.cache_size,
            )

    def _elastic_live_requests(self) -> list[Req]:
        """Every request object that currently owns GDN slots (deduplicated)."""
        reqs = list(self.decode_manager.running_reqs)
        reqs.extend(
            pending.chunked_req
            for pending in self.prefill_manager.pending_list
            if pending.chunked_req is not None
        )
        return list({id(req): req for req in reqs}.values())

    def _elastic_demand(self) -> int:
        # Every pending item is one independent agent. A chunked continuation is not
        # also in decode, so decode + pending is the exact admission demand here.
        return len(self.decode_manager.running_reqs) + len(
            self.prefill_manager.pending_list
        )

    def _remap_req_mamba_slots(self, req: Req, remap: dict[int, int]) -> None:
        if req.linear_slot_idx is not None:
            req.linear_slot_idx = remap[req.linear_slot_idx]
        if req.mamba_ping_pong is not None:
            req.mamba_ping_pong = tuple(remap[slot] for slot in req.mamba_ping_pong)
        if req.mamba_restore_src is not None:
            req.mamba_restore_src = remap[req.mamba_restore_src]

    @torch.inference_mode()
    def _maybe_resize_elastic_capacity(self) -> None:
        initial = getattr(
            getattr(self, "config", None), "elastic_initial_requests", None
        )
        if initial is None:
            return
        demand = self._elastic_demand()
        target = _elastic_target_capacity(
            initial, self.config.max_running_req, demand
        )
        if target == self._elastic_capacity:
            self._elastic_resize_pending = False
            self._elastic_shrink_candidate = None
            return

        # Finishing requests are commonly staggered by a second or two. Avoid an
        # expensive graph/state recapture for every transient intermediate tier;
        # returning to the compact initial tier remains immediate.
        if initial < target < self._elastic_capacity:
            now = time.monotonic()
            candidate = self._elastic_shrink_candidate
            if candidate is None or candidate[0] != target:
                self._elastic_shrink_candidate = (
                    target,
                    now + _ELASTIC_INTERMEDIATE_SHRINK_GRACE_SECONDS,
                )
                self._elastic_resize_pending = True
                return
            if now < candidate[1]:
                self._elastic_resize_pending = True
                return
        else:
            self._elastic_shrink_candidate = None

        pool = self.engine.linear_state_pool
        assert pool is not None
        from freetoken.kvcache.linear_state_pool import linear_pool_slots_for_capacity

        target_slots = linear_pool_slots_for_capacity(self.config, target)
        if target < self._elastic_capacity:
            # Unlocked snapshots are cache, not live agent state. Evict just enough
            # to fit the compact pool; protected session snapshots postpone shrink.
            overflow = max(0, len(pool.occupied_slots) - (target_slots - 1))
            if overflow:
                self.cache_manager.ensure_mamba_slots(pool.num_free_slots + overflow)
            if len(pool.occupied_slots) > target_slots - 1:
                self._elastic_resize_pending = True
                logger.info_rank0(
                    "Elastic shrink deferred: %d protected/live GDN slots exceed the "
                    "%d-slot compact capacity",
                    len(pool.occupied_slots),
                    target_slots - 1,
                )
                return

        occupied = sorted(pool.occupied_slots)
        remap = {slot: i + 1 for i, slot in enumerate(occupied)}
        self.engine.resize_elastic_capacity(target, remap)
        self.cache_manager.remap_mamba_slots(remap)
        for req in self._elastic_live_requests():
            self._remap_req_mamba_slots(req, remap)
        self._elastic_capacity = target
        self._elastic_resize_pending = False
        self._elastic_shrink_candidate = None
        # The exact page-conservation check is idle-only: live requests own pages
        # that are intentionally in neither the free list nor the radix tree.
        # Elastic growth normally happens with active requests, so limit the
        # full check to the truly idle shrink boundary.
        if not self.prefill_manager.runnable and not self.decode_manager.runnable:
            self.cache_manager.check_integrity()

    def _reply_rebuild(
        self, request_id: str, status: str, error: str | None = None
    ) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(
                    f"cache rebuild failed before teardown: {e!r} — old cache intact"
                )
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(
                f"cache rebuild failed: {e!r} — rolling back to the previous geometry"
            )
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(
                    f"cache rebuild rollback failed: {e2!r} — server latched failed"
                )
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning(
                "cache rebuild rolled back to the previous geometry — still serving"
            )
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id,
                "rejected",
                error=f"rebuild failed and was rolled back: {e!r}",
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if (
                sizes is not None
            ):  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(
                0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1
            )
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size
            if eng.moe_offload_cache is not None
            else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1)
            if eng.linear_state_pool is not None
            else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import (
            compute_cache_pools,
            compute_cache_unit_bytes,
        )

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        if self.config.kv_grow_step_tokens:
            required = self.cache_manager.committed_pages_required(batch.reqs)
            old_pages, new_pages = self.engine.grow_runtime_kv(required)
            if new_pages > old_pages:
                self.cache_manager.add_committed_pages(new_pages)
                logger.info_rank0(
                    f"KV grew {old_pages} -> {new_pages} tokens; "
                    f"MoE cache now {self.engine.moe_offload_cache.cache_size} slots"
                )
            # A growth event reallocates the expert cache and destroys every decode graph.
            # Recapture only after the final geometry for this batch is known; doing this
            # before aggregate growth would pad/replay through stale expert pointers.
            if batch.is_decode:
                self.engine.ensure_decode_graphs()
        if (
            batch.is_prefill
            and self.config.kv_grow_step_tokens
            and not any(isinstance(req, ChunkedReq) for req in batch.reqs)
            and not self.prefill_manager.runnable
        ):
            # No queued prefill can immediately grow again. Rebuild once here so the first
            # streamed token-to-token interval contains inference, not graph recapture.
            self.engine.ensure_decode_graphs()
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter
            )
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [
                        r.linear_slot_idx
                        if r.linear_slot_idx is not None
                        else pool.padding_slot
                        for r in batch.padded_reqs
                    ]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        if (
            getattr(getattr(self, "config", None), "kv_grow_step_tokens", 0)
            and self.prefill_manager.runnable
            and self.decode_manager.runnable
        ):
            decode_burst = self._adaptive_decode_burst()
            if self._growable_decode_steps < decode_burst:
                batch = self.decode_manager.schedule_next_batch()
                self._growable_decode_steps += 1
            else:
                batch = self.prefill_manager.schedule_next_batch(
                    self._adaptive_prefill_budget()
                )
                if batch is None and self.prefill_manager.runnable:
                    self._reclaim_for_blocked_prefill()
                self._growable_decode_steps = 0
        else:
            batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
            if batch is not None:
                self._growable_decode_steps = 0
            else:
                if self.prefill_manager.runnable:
                    self._reclaim_for_blocked_prefill()
                batch = self.decode_manager.schedule_next_batch()
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        if getattr(getattr(self, "config", None), "adaptive_scheduler", False):
            batch.scheduler_started_at = time.perf_counter()
        self._report_prompt_admissions(batch)
        return forward_input

    def _adaptive_decode_burst(self) -> int:
        """Decode forwards that approximate one short wall-clock service slice."""
        if not getattr(getattr(self, "config", None), "adaptive_scheduler", False):
            return self._growable_decode_burst
        elapsed = getattr(self, "_scheduler_decode_seconds_ewma", None)
        if not elapsed or elapsed <= 0:
            return self._growable_decode_burst
        target = getattr(self, "_scheduler_decode_slice_seconds", 0.25)
        return min(64, max(8, round(target / elapsed)))

    def _adaptive_prefill_budget(self) -> int:
        """Size the aggregate prefill lane by measured throughput and waiting lanes."""
        if not getattr(getattr(self, "config", None), "adaptive_scheduler", False):
            return self.prefill_budget
        tps = getattr(self, "_scheduler_prefill_tps_ewma", None)
        if not tps or tps <= 0:
            return self.prefill_budget
        pending = len(getattr(self.prefill_manager, "pending_list", ()))
        per_lane = getattr(self, "_scheduler_min_prefill_tokens_per_lane", 2048)
        floor = min(self.prefill_budget, max(1, pending) * per_lane)
        target = int(tps * getattr(self, "_scheduler_prefill_slice_seconds", 8.0))
        page_size = max(1, int(getattr(self.config, "page_size", 1)))
        target = max(page_size, target // page_size * page_size)
        return min(self.prefill_budget, max(floor, target))

    def _observe_scheduler_batch(self, batch: Batch) -> None:
        """Update phase EWMAs using a forward's existing completion barrier."""
        started = getattr(batch, "scheduler_started_at", None)
        if started is None:
            return
        elapsed = max(time.perf_counter() - started, 1e-6)
        alpha = 0.25
        if batch.is_prefill:
            tokens = int(getattr(batch, "log_new_tokens", 0))
            if tokens <= 0:
                return
            sample = tokens / elapsed
            key = tuple(sorted(int(getattr(req, "uid", id(req))) for req in batch.reqs))
            old = getattr(self, "_scheduler_prefill_tps_ewma", None)
            old_key = getattr(self, "_scheduler_prefill_key", None)
            self._scheduler_prefill_key = key
            self._scheduler_prefill_tps_ewma = (
                sample
                if old is None or key != old_key
                else (alpha * sample + (1.0 - alpha) * old)
            )
        elif batch.is_decode:
            old = getattr(self, "_scheduler_decode_seconds_ewma", None)
            self._scheduler_decode_seconds_ewma = (
                elapsed if old is None else (alpha * elapsed + (1.0 - alpha) * old)
            )

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(
                    uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens
                )
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        replies = [ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids]
        close_acks = getattr(self, "_pending_session_close_acks", {})
        for uid in uids:
            close = close_acks.pop(uid, None)
            if close is not None:
                request_id, session_id = close
                replies.append(
                    SessionClosedResultMsg(
                        session_id=session_id,
                        request_id=request_id,
                        status="closed",
                    )
                )
        self.send_result(replies)

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(
        device, non_blocking=True
    )
