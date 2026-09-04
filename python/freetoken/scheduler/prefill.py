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
    # KV tokens this prompt pins once it is complete: ``input_len + output_len``, carried
    # unchanged across its continuations. The admission gate sums it over every queued
    # continuation (see ``PrefillAdder.inflight_prefill_size``).
    #
    # The prefix match is deliberately NOT subtracted. Matched pages are evictable radix
    # pages until this request locks them, and locking is what makes them unreclaimable for
    # the rest of the prompt's life -- so a prompt that reuses 75% of a family prefix still
    # pins the whole 100% of its own length. Netting the match out undercounts exactly
    # that, and the stage replay livelocks in 6,600 forwards when it does. Two unfinished
    # prompts sharing a live prefix are therefore double-charged; that is the conservative
    # direction, and the only one that is safe.
    prefill_footprint: int = 0

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
    # KV pages this pass will actually ALLOCATE: in-flight decode growth, one chunk per
    # admitted req, and the output of every req whose prompt FINISHES in this pass. It exists
    # only to keep ``committed_pages_required`` satisfiable, and that check demands exactly
    # the batch's per-chunk page deltas -- so it must never be charged a whole remaining
    # prompt. It was, once: one long in-flight prompt then reserved the pool away from every
    # other continuation in the pass (6 lanes -> 2 with 700 free pages and a 600-page batch).
    # Negative means "start from reserved_size".
    reserved_pages: int = -1
    # Sum of ``ChunkedReq.prefill_footprint`` over the prompts already mid-prefill, seeded
    # by the manager from the pending list. Charged against ``cache_manager.max_size`` by
    # the finishability half of the fresh-admit gate.
    #
    # It is each prompt's WHOLE footprint and not its unforwarded tail because the
    # forwarded part is exactly what is already pinned: a ChunkedReq holds a locked handle,
    # so its pages are neither free nor evictable until it completes. It is charged against
    # the pool MAXIMUM and not against ``available_size`` because the rest of that
    # difference -- pages held by *decoding* requests -- is guaranteed to come back: a
    # decode has a bounded output and always terminates, whereas a half-forwarded prompt
    # releases nothing until it reaches its last chunk. PrefillAdder is rebuilt every pass,
    # so ``reserved_size`` cannot see any of this; charging the shrinking tail instead
    # livelocks the stage profile within 3,700 forwards (each prompt frees room for the
    # next one to be admitted, and none of them ever finishes). The old
    # whole-prompt-against-``available_size`` gate prevented that by accident, at the cost
    # of 2.5x the prefill throughput.
    inflight_prefill_size: int = 0

    def __post_init__(self) -> None:
        if self.reserved_pages < 0:
            self.reserved_pages = self.reserved_size

    def _page_span(self, start: int, end: int) -> int:
        """Fresh pages the extend ``[start, end)`` pulls, charged per WHOLE page."""
        ps = self.cache_manager.page_size
        return (div_ceil(end, ps) - div_ceil(start, ps)) * ps

    def _seat_len(self, extend_len: int, output_len: int, chunk_limit: int | None) -> int:
        """Tokens the admitting pass will actually write for this request, plus its decode.

        This is the charge the *availability* half of the gate uses. Only the first chunk
        is forwarded now; every later chunk is re-tested against the pool at admission time
        by ``_add_one_req``'s page cap and back-pressured (deferred) when it does not fit,
        with ``Scheduler._reclaim_for_blocked_prefill`` raising the demand signal for the
        head of the queue. ``output_len`` stays in the charge: a prompt that fits in one
        chunk goes straight to decode, and that growth is allocated with no further gate.
        """
        chunk = min(extend_len, self.token_budget)
        if chunk_limit is not None:
            chunk = min(chunk, chunk_limit)
        return max(chunk, 0) + output_len

    def _try_allocate_one(self, req: PendingReq, chunk_limit: int | None = None):
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        mr = self.cache_manager.match_req(req)
        handle = mr.cuda_handle
        cached_len = handle.cached_len
        extend_len = req.input_len - cached_len

        # Two questions, two currencies.
        #
        # (1) Can this prompt EVER finish? Charge its whole resident footprint -- plus
        #     that of every prompt already mid-prefill -- against the pool at its MAXIMUM
        #     size (see ``inflight_prefill_size``). Satisfying this keeps every admitted
        #     prompt simultaneously residentable, so each one reaches its last chunk once
        #     the decodes ahead of it drain. It is the invariant that keeps a chunked
        #     prompt out of a pool that cannot finish it.
        #
        # (2) What will THIS PASS write? Only that -- one chunk plus the request's own
        #     decode reservation -- is charged against ``available_size``, because that is
        #     all the forward allocates, and every later chunk is re-gated by
        #     ``_add_one_req``'s page cap (deferring, not failing) with
        #     ``Scheduler._reclaim_for_blocked_prefill`` behind it. Charging the whole
        #     remainder here instead pinned long-context concurrency at two lanes with 23K
        #     free pages and a 512-token chunk, and ended 99.8% of prefill passes at the
        #     first refusal: the same "reserve the whole prompt" mistake the chunk cap in
        #     ``_add_one_req`` already had one layer down.
        footprint = req.input_len + req.output_len
        if footprint + self.inflight_prefill_size > self.cache_manager.max_size:
            return None
        seat_len = self._seat_len(extend_len, req.output_len, chunk_limit)
        if seat_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if seat_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # Second currency (hybrid GDN): reserve 1 live + 2 ping-pong state slots; evict tree
        # snapshots if the pool is short, fail admission if still short (mirrors the KV gate).
        if self.cache_manager.is_hybrid:
            # reserve_mamba_slots escalates past eviction into the session-lease spill: an idle
            # automatic lease pins its snapshot node, so without that tier a small pool (the 1M
            # profile runs five slots: padding + live + 2 ping-pong + one lease) admits nothing
            # once a second conversation arrives.
            if not self.cache_manager.reserve_mamba_slots(3):
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
        # First currency (full KV pages), charged per WHOLE page like the swa cap below. A
        # CONTINUATION never passes through _try_allocate_one, so until here nothing had checked
        # that the pool can back its NEXT chunk: it owns its table slot, its GDN slots and its
        # already-forwarded pages, but the pages this chunk writes are allocated later, in
        # allocate_paged -- after committed_pages_required has already found the batch
        # unbackable and killed the scheduler. ``available_size`` (evictable prefix + free
        # slots + the not-yet-committed growable suffix) is exactly the ceiling
        # committed_pages_required tests against, and ``reserved_pages`` is the page demand the
        # reqs admitted earlier in this pass have already placed on it. A fresh admit passes
        # through here too and is NOT covered by its own admission gate: that gate charges
        # ``available_size``, which the pass has not spent yet, so this cap is what actually
        # keeps the batch backable.
        kv_ps = self.cache_manager.page_size
        kv_pages = max(0, self.cache_manager.available_size - self.reserved_pages) // kv_ps
        max_kv_end = (div_ceil(cached_len, kv_ps) + kv_pages) * kv_ps
        chunk_size = min(chunk_size, max(max_kv_end - cached_len, 0))
        # A chunk that COMPLETES the prompt hands the request to the decode manager, and its
        # ``output_len`` pages are then allocated one per forward with no gate of their own.
        # They have to come out of the same budget as this chunk, or the pass leaves
        # ``available_size`` below ``inflight_tokens`` and the next decode batch is
        # unbackable -- reproduced as "batch needs 5 pages but only 2 are physically
        # allocatable" on a 131,072-page pool. When both do not fit, forward less and stay
        # chunked: the request finishes in a later pass, when its decode IS backable.
        # (Charging ``output_len`` unconditionally instead would make every continuation
        # reserve against its own next chunk and starve its peers.)
        if chunk_size >= remain_len:
            chunk_size = min(
                chunk_size, max(max_kv_end - cached_len - pending_req.output_len, 0)
            )
        if chunk_size <= 0:
            # The pool cannot back one page for this request right now. Defer instead of
            # raising: the request keeps its place at the head of the pending list and the
            # scheduler back-pressures (_reclaim_for_blocked_prefill) until pages return.
            return None
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
        # Charge ``available_size`` only what this pass writes plus this request's own decode
        # growth -- the same currency ``_try_allocate_one``'s availability half gates on. The
        # prompt's whole resident footprint goes to ``inflight_prefill_size`` instead, against
        # the pool MAXIMUM, and only for a fresh admit that stays chunked: a continuation's
        # footprint was already counted when the manager seeded the adder, and a prompt that
        # finishes in this chunk is a decode from here on.
        prior = pending_req.chunked_req
        # Only a FRESH admit adds decode demand the adder has not already been seeded with;
        # a continuation's output_len is in ``reserved_size`` from the start of the pass.
        self.reserved_size += chunk_size + (pending_req.output_len if prior is None else 0)
        footprint = (
            pending_req.input_len + pending_req.output_len
            if prior is None
            else prior.prefill_footprint
        )
        if is_chunked and prior is None:
            self.inflight_prefill_size += footprint
        self.reserved_pages += self._page_span(cached_len, cached_len + chunk_size)
        if not is_chunked:
            # This request's prompt is complete, so it decodes on the very next forward and
            # those pages are allocated with no further gate. Book them now -- after its own
            # chunk is sized, so it never reserves against itself -- and the lanes behind it
            # in this same pass can no longer spend them. That is the window the pressure
            # profile died in ("batch needs 2 pages but only 1 are physically allocatable")
            # once the whole-prompt reservation stopped covering it by accident.
            self.reserved_pages += pending_req.output_len
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
            hidden_states=pending_req.hidden_states,
            no_prefix_cache=pending_req.no_prefix_cache,
        )
        # Hybrid GDN per-request state slots (None for non-hybrid). On a fresh admit these are
        # freshly allocated; on a chunked continuation they are inherited from the prior chunk.
        req.linear_slot_idx = linear_slot_idx
        req.mamba_ping_pong = ping_pong
        req.mamba_next_track_idx = next_track_idx
        req.mamba_restore_src = restore_src
        req.swa_evicted_seqlen = swa_evicted_seqlen  # carry the extend-free watermark across chunks
        if is_chunked:
            req.prefill_footprint = footprint  # admission-time cost, carried across chunks
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

        if resource := self._try_allocate_one(pending_req, chunk_limit):
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
    # A refused request no longer ends the pass (see schedule_next_batch), so the scan needs
    # its own bound or a long queue of unseatable requests would be walked -- and prefix-
    # matched -- on every pass. Sized above the usual --max-running-requests so a full queue
    # is still scanned once; the pass also stops as soon as the token budget is spent.
    max_admission_refusals: int = 32

    def _seatable_lanes(self, adder: PrefillAdder) -> int:
        """Lanes this pass can plausibly seat -- the divisor the interleave share needs.

        ``token_budget // waiting`` divides by the QUEUE DEPTH: sixteen queued requests buy a
        512-token chunk even in a pass whose pools can seat two lanes, which is 12.5% of the
        budget and the whole of the interleave's throughput regression. The seating limit is
        knowable, though, and cheaply: after the fresh-admit gate stopped reserving whole
        prompts against ``available_size``, what bounds a long-prompt pass is the
        finishability sum against ``cache_manager.max_size`` (plus the table slots), and both
        are pure arithmetic over the pending list -- no prefix-cache walk, no allocation.

        Deliberately an estimate, and deliberately an over-estimate where it is wrong: a lane
        counted here that the pass then cannot seat only makes the surviving chunks smaller,
        never unbackable. ``available_size`` is not modelled (the per-chunk cap in
        ``_add_one_req`` is what enforces it, and it enforces it exactly).
        """
        budget = self.cache_manager.max_size - adder.inflight_prefill_size
        slots = self.table_manager.available_size
        lanes = 0
        blocked_fresh = False
        for req in self.pending_list:
            if req.chunked_req is not None:
                lanes += 1  # already admitted and already charged; the gate does not re-run
                continue
            if blocked_fresh:
                continue
            cost = req.input_len + req.output_len
            if slots <= 0 or cost > budget:
                # Mirror the admission loop's strict FIFO: nothing fresh behind a fresh
                # request the pools turned away is seatable this pass either. Counting them
                # anyway is what made the estimate read 4+ lanes on the stage profile in a
                # pass that seats the two continuations at its head and nothing else.
                blocked_fresh = True
                continue
            budget -= cost
            slots -= 1
            lanes += 1
        return max(lanes, 1)

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(
            PendingReq(
                req.uid,
                req.input_ids,
                req.sampling_params,
                mm_embeds=req.mm_embeds,
                session_id=req.session_id,
                session_ttl_seconds=req.session_ttl_seconds,
                hidden_states=req.hidden_states,
                no_prefix_cache=req.no_prefix_cache,
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
            # A queued continuation is not in ``running_reqs``, so ``inflight_tokens`` misses
            # the decode it is guaranteed to need the moment its last chunk lands. Nothing
            # else books it across passes -- the whole-prompt reservation used to, by
            # accident -- and without it a prefill pass spends the pool to zero and the next
            # decode batch dies in ``committed_pages_required`` (reproduced on the pressure
            # profile: "batch needs 2 pages but only 1 are physically allocatable").
            reserved_size=self.decode_manager.inflight_tokens
            + sum(
                req.output_len
                for req in self.pending_list
                if req.chunked_req is not None
            ),
            # NOT the sum above. ``reserved_pages`` is the per-chunk cap's budget, and it is
            # spent in the pass it is computed for, so it books only decode pages that pass
            # can actually be asked for: the requests already decoding, plus (charged in
            # ``_add_one_req``, after each one's own chunk is sized) the requests whose
            # prefill FINISHES here and that therefore decode on the very next forward.
            # A continuation with chunks still to go decodes many passes from now, against a
            # pool that will have moved; reserving for it here only starves its peers.
            reserved_pages=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            # Prompts already mid-prefill: the adder is rebuilt every pass, so their resident
            # footprint is invisible to it unless it is handed over here.
            inflight_prefill_size=sum(
                req.chunked_req.prefill_footprint
                for req in self.pending_list
                if req.chunked_req is not None
            ),
        )
        seatable_lanes = self._seatable_lanes(adder) if self.interleave_chunks else 0
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        prompt_admissions: List[Tuple[int, int, int]] = []
        # Snapshot here, before the forward's complete_one() advances cached_len: the tokens
        # forwarded this batch (extend_len) and the prefix-cache hit. SGLang counts the hit
        # once at admission, so continuation chunks (already-chunked reqs) contribute 0.
        log_new_tokens = 0
        log_cached_tokens = 0
        admitted_index: set[int] = set()
        refusals = 0
        # FIFO fairness among FRESH admits: STRICT, no overtaking. A refusal no longer ends
        # the pass, so without this a later fresh prompt would jump the queue past one the
        # pools just turned away -- and since the refusals land on the long prompts, "later
        # and cheaper" means "every short prompt, forever".
        #
        # The permissive rule was measured, not assumed: letting a fresh admit overtake a
        # refused one when its ``input_len + output_len`` is strictly smaller costs 25% of
        # the prefill tokens on the stage profile (4.76 M vs 6.35 M) and triples
        # wait-to-first-chunk p95 (1,146 s vs 379 s), because the long-context prompts that
        # the soak timed out on are exactly the ones that never win that comparison. Strict
        # order also keeps the wait distribution comparable with upstream's (p50 153 s /
        # p95 379 s here against 155 s / 395 s at the merge base).
        #
        # Continuations are a different class and are never gated by this rule: they were
        # admitted in an earlier pass, so they are already ahead in FIFO order. Letting them
        # through a refused fresh admit is the whole point of not breaking -- the pass used
        # to abandon a median of 13 queued requests, 11 of them seatable.
        blocked_fresh = False
        stopped_for_lane_cap = False
        for index, pending_req in enumerate(self.pending_list):
            if adder.token_budget <= 0:
                break
            is_continuation = pending_req.chunked_req is not None
            if not is_continuation:
                if pending_req.input_len + pending_req.output_len > self.cache_manager.max_size:
                    # Unsatisfiable at EVERY pool state, not just this one: the pool could
                    # be empty and this prompt still would not fit. Skipping it without
                    # setting ``blocked_fresh`` is what keeps it from wedging the queue --
                    # FIFO fairness towards a request that can never be served is just a
                    # stalled scheduler. (It is also never admitted into a chunked lane it
                    # could not leave; the same ceiling is re-checked in _try_allocate_one.)
                    if not pending_req.oversize_warned:
                        pending_req.oversize_warned = True  # every pass re-skips it
                        logger.warning_rank0(
                            "Request %d needs %d KV tokens but the pool holds at most %d; "
                            "it can never be admitted and is being skipped",
                            pending_req.uid,
                            pending_req.input_len + pending_req.output_len,
                            self.cache_manager.max_size,
                        )
                    continue
                if blocked_fresh:
                    continue
            chunk_limit = None
            if self.interleave_chunks:
                waiting = len(self.pending_list) - index
                available_lanes = max(
                    (lane_cap if lane_cap else seatable_lanes) - len(reqs), 1
                )
                chunk_limit = max(1, adder.token_budget // min(waiting, available_lanes))
            if req := adder.try_add_one(pending_req, chunk_limit=chunk_limit):
                admitted_index.add(index)
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
                # Skip, do not stop.
                refusals += 1
                blocked_fresh = blocked_fresh or not is_continuation
                if refusals >= self.max_admission_refusals:
                    break
        if len(reqs) == 0:
            return None
        remaining = [
            req for i, req in enumerate(self.pending_list) if i not in admitted_index
        ]
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
