from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from freetoken.core import Batch, Req
from freetoken.utils import align_down, div_ceil, init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from freetoken.kvcache import BaseCacheHandle
    from freetoken.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


def _maybe_pinned(t: torch.Tensor) -> torch.Tensor:
    """Pinning only buys the async H2D copy below; without a device it just raises."""
    return t.pin_memory() if torch.cuda.is_available() else t


class ChunkedReq(Req):
    def _alloc_ids_buf(self) -> None:
        pass  # never sampled; keep input_ids a view of the pending prompt

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    # SWA-pool tokens charged to reqs admitted so far this pass. Mirrors reserved_size: swa is
    # allocated only in allocate_paged (after the pass), so swa_available_size does not decrement
    # across the admission loop -- without this, successive admits all see the full pool.
    reserved_swa: int = 0

    def _try_allocate_one(self, req: PendingReq):
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        mr = self.cache_manager.match_req(req)
        handle = mr.cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # Second currency (hybrid GDN): reserve 1 live + 2 ping-pong state slots; evict tree
        # snapshots if the pool is short, fail admission if still short (mirrors the KV gate).
        if self.cache_manager.is_hybrid:
            pool = self.cache_manager.linear_state_pool
            if pool.num_free_slots < 3:
                self.cache_manager.ensure_mamba_slots(3)
            if pool.num_free_slots < 3:
                return self.cache_manager.unlock(handle)

        # Third currency (SWA): refuse admission unless the swa pool can seat this request's first
        # chunk / one window (the per-chunk charge is in _add_one_req; the reclaim -- radix
        # evict_swa -- happens in allocate_paged, so no ensure here; swa_available_size already
        # folds the evictable tree). For naive (no tree) this can only refuse, which is correct.
        if self.cache_manager.swa_paged:
            ps = self.cache_manager.page_size
            # swa is charged per WHOLE page (allocate_paged -> alloc_swa), so the seat check is
            # in page units too; identical at page_size==1.
            need_swa = div_ceil(
                min(max(extend_len, 1), self.cache_manager.sliding_window_size) + 1, ps
            ) * ps
            if self.cache_manager.swa_available_size - self.reserved_swa < need_swa:
                return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            device_ids.copy_(_maybe_pinned(req.input_ids[:cached_len]), non_blocking=True)
            # Write the matched indices into the TAIL of the page_entry: a cache may return
            # fewer matched indices than cached_len, in which case only the trailing n slots are
            # known-live. Today both the generic radix and the SWA radix match a prefix whose
            # full-loc row is entirely live (n == cached_len), so the tail IS the whole prefix.
            # (DSV4 reads this table too: its pool's full_loc_map is attached to it.)
            matched = handle.get_matched_indices()
            n = int(matched.numel())
            self.table_manager.page_table[table_idx][cached_len - n : cached_len].copy_(matched)

        linear_slot_idx = ping_pong = None
        if self.cache_manager.is_hybrid:
            pool = self.cache_manager.linear_state_pool
            linear_slot_idx = pool.alloc(1)[0]
            ping_pong = tuple(pool.alloc(2))

        return handle, table_idx, linear_slot_idx, ping_pong, mr.mamba_value

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        linear_slot_idx: int | None = None,
        ping_pong: tuple | None = None,
        next_track_idx: int = 0,
        restore_src: int | None = None,
        swa_evicted_seqlen: int = 0,
        chunk_limit: int | None = None,
    ) -> Req | None:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        if chunk_limit is not None:
            chunk_size = min(chunk_size, chunk_limit)
        if self.cache_manager.swa_paged:
            # Cap this chunk by the swa the pool can back this pass. swa is allocated per token in
            # allocate_paged, and token_budget (max_extend_tokens, default 8192) won't chunk a
            # shorter prompt -- so this cap is what forces a prompt whose swa footprint exceeds the
            # pool to chunk. Credit the slots THIS request's own extend-free (in _prepare_batch,
            # which runs AFTER this sizing) will release this batch, else a continuation sees a
            # drained pool and stalls at chunk_size 0.
            cm = self.cache_manager
            window, ps = cm.sliding_window_size, cm.page_size
            floor = cache_handle.cached_len
            new_evicted = align_down(cached_len - window - ps, ps)
            self_reclaim = max(0, new_evicted - max(swa_evicted_seqlen, floor))
            swa_budget = cm.swa_available_size + self_reclaim - self.reserved_swa
            # swa is charged per WHOLE page: cap the chunk so its PAGE-SPAN cost fits the budget
            # (the extend [cached_len, cached_len+chunk) pulls div_ceil(end,ps)-div_ceil(start,ps)
            # fresh pages -- the partial head page was charged by the previous chunk), and reserve
            # that cost, not the raw token count. Degenerates to the token math at page_size==1.
            max_end = (div_ceil(cached_len, ps) + max(swa_budget, 0) // ps) * ps
            chunk_size = min(chunk_size, max(max_end - cached_len, 0))
            # A continuation resumes the compressor carry at its boundary, which must be
            # page-aligned; the token_budget leftover (unlike max_end) is not. Align the end
            # down when the chunk mints a continuation; no whole page -> retry next pass.
            # 0 <: a chunk the swa cap collapsed to 0 must NOT bail (undersized pool --
            # bailing would livelock; the floor tests pin the loud failure).
            if 0 < chunk_size < remain_len:
                aligned = align_down(cached_len + chunk_size, ps) - cached_len
                if aligned <= 0:
                    return None
                chunk_size = aligned
            self.reserved_swa += (
                div_ceil(cached_len + chunk_size, ps) - div_ceil(cached_len, ps)
            ) * ps
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(_maybe_pinned(pending_req.input_ids[_slice]), non_blocking=True)
        if is_chunked and pending_req.mm_embeds is not None:
            raise NotImplementedError(
                "Multimodal prompts must fit in a single prefill chunk; increase "
                "--max-extend-tokens or shrink the prompt."
            )
        req = CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            mm_embeds=pending_req.mm_embeds,
            session_id=pending_req.session_id,
            session_ttl_seconds=pending_req.session_ttl_seconds,
        )
        # Hybrid GDN per-request state slots (None for non-hybrid). On a fresh admit these are
        # freshly allocated; on a chunked continuation they are inherited from the prior chunk.
        req.linear_slot_idx = linear_slot_idx
        req.mamba_ping_pong = ping_pong
        req.mamba_next_track_idx = next_track_idx
        req.mamba_restore_src = restore_src
        req.swa_evicted_seqlen = swa_evicted_seqlen  # carry the extend-free watermark across chunks
        return req

    def try_add_one(self, pending_req: PendingReq, chunk_limit: int | None = None) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
                linear_slot_idx=chunked_req.linear_slot_idx,
                ping_pong=chunked_req.mamba_ping_pong,
                next_track_idx=chunked_req.mamba_next_track_idx,
                restore_src=None,  # continuation chunk already has live state
                swa_evicted_seqlen=chunked_req.swa_evicted_seqlen,  # extend-free watermark so far
                chunk_limit=chunk_limit,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx, linear_slot_idx, ping_pong, restore_src = resource
            req = self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
                linear_slot_idx=linear_slot_idx,
                ping_pong=ping_pong,
                next_track_idx=0,
                restore_src=restore_src,
                chunk_limit=chunk_limit,
            )
            if req is None:
                # no aligned chunk this pass: undo the admission (a continuation keeps its
                # resources -- they belong to the prior chunk's Req)
                self.cache_manager.unlock(cache_handle)
                self.table_manager.free(table_idx)
                if linear_slot_idx is not None:
                    self.cache_manager.linear_state_pool.free([linear_slot_idx, *ping_pong])
            return req

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
    # Growable multi-agent mode shares one aggregate token budget across waiting agents. A
    # max_batch_seqs cap can serialize inefficient grouped GGUF prefills while rotation keeps
    # unfinished long prompts fair between chunks.
    interleave_chunks: bool = False
    max_batch_seqs: int = 0
    # Auto-selected GGUF serialization has a measured small-prompt exception:
    # grouping fresh prompts through this length amortizes full-layer setup.
    # Zero keeps ``max_batch_seqs`` a hard cap (including explicit CLI values).
    small_prompt_group_tokens: int = 0

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(
            PendingReq(
                req.uid,
                req.input_ids,
                req.sampling_params,
                mm_embeds=req.mm_embeds,
                session_id=req.session_id,
                session_ttl_seconds=req.session_ttl_seconds,
            )
        )

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        lane_cap = self.max_batch_seqs
        if (
            lane_cap == 1
            and self.small_prompt_group_tokens > 0
            and len(self.pending_list) > 1
            and all(
                req.chunked_req is None
                and req.input_len <= self.small_prompt_group_tokens
                for req in self.pending_list
            )
            and sum(req.input_len for req in self.pending_list) <= prefill_budget
        ):
            lane_cap = 0

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        prompt_admissions: List[Tuple[int, int, int]] = []
        # Snapshot here, before the forward's complete_one() advances cached_len: the tokens
        # forwarded this batch (extend_len) and the prefix-cache hit. SGLang counts the hit
        # once at admission, so continuation chunks (already-chunked reqs) contribute 0.
        log_new_tokens = 0
        log_cached_tokens = 0
        admitted_items = 0
        stopped_for_lane_cap = False
        for index, pending_req in enumerate(self.pending_list):
            is_continuation = pending_req.chunked_req is not None
            chunk_limit = None
            if self.interleave_chunks:
                waiting = len(self.pending_list) - index
                available_lanes = (
                    max(lane_cap - len(reqs), 1)
                    if lane_cap
                    else waiting
                )
                chunk_limit = max(1, adder.token_budget // min(waiting, available_lanes))
            if req := adder.try_add_one(pending_req, chunk_limit=chunk_limit):
                admitted_items += 1
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
                if not is_continuation:
                    # Record the COMPLETE prompt length and the prefix-cache hit on the
                    # first chunk. The scheduler publishes them only after _prepare_batch
                    # succeeds; continuation chunks must never publish them again.
                    prompt_admissions.append(
                        (req.uid, pending_req.input_len, req.cache_handle.cached_len)
                    )
                log_new_tokens += req.extend_len
                if not is_continuation:
                    log_cached_tokens += req.cache_handle.cached_len
                if lane_cap and len(reqs) >= lane_cap:
                    stopped_for_lane_cap = index + 1 < len(self.pending_list)
                    break
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        remaining = self.pending_list[admitted_items:]
        # Interleaved mode rotates unfinished lanes behind requests that did not run this pass.
        # The default preserves the original strict chunked-prefill ordering.
        # If admission stopped on a resource-constrained request, keep the active continuations
        # first. Putting the blocked request at the head would make the next pass return no batch
        # forever while a runnable continuation sat behind it.
        self.pending_list = (
            remaining + chunked_list
            if self.interleave_chunks and stopped_for_lane_cap
            else chunked_list + remaining
        )
        batch = Batch(reqs=reqs, phase="prefill")
        batch.log_new_tokens = log_new_tokens
        batch.log_cached_tokens = log_cached_tokens
        batch.prompt_admissions = prompt_admissions
        return batch

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
