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
    # KV pages this pass will actually ALLOCATE: in-flight decode growth plus one chunk per
    # admitted req. The chunk cap below must be charged against this and NOT against
    # ``reserved_size``, which books every admitted req's WHOLE remaining prompt. That
    # reservation is an admission policy (it keeps a fresh prompt from being admitted into a
    # pool that cannot finish it); the cap exists only to keep ``committed_pages_required``
    # satisfiable, and that check demands exactly the batch's per-chunk page deltas. Charging
    # the whole remainder let one long in-flight prompt reserve the pool away from every other
    # continuation in the pass (6 lanes -> 2 with 700 free pages and a 600-page batch).
    # Negative means "start from reserved_size". The manager passes it explicitly instead,
    # because reserved_size now also carries the unforwarded tail of every prompt already
    # mid-prefill -- a cross-pass admission claim, not a page demand this pass can be asked
    # for. Inheriting it here would re-create exactly the starvation the note above records.
    reserved_pages: int = -1

    def __post_init__(self) -> None:
        if self.reserved_pages < 0:
            self.reserved_pages = self.reserved_size

    def _page_span(self, start: int, end: int) -> int:
        """Fresh pages the extend ``[start, end)`` pulls, charged per WHOLE page."""
        ps = self.cache_manager.page_size
        return (div_ceil(end, ps) - div_ceil(start, ps)) * ps

    def _try_allocate_one(self, req: PendingReq):
        if self.table_manager.available_size == 0:
            return None

        # Cheap pre-check on the memoised match, BEFORE the radix walk. While the tree is
        # unchanged a re-walk returns the same ``cached_len``, so this arithmetic is exact
        # and a refusal here is the refusal the full path would have reached. It exists
        # because the walk is O(prompt): during the soak's stalls the scheduler re-matched
        # 118K-token prompts on every pass and forwarded nothing, and py-spy found it inside
        # ``fast_compare_key`` on four of five samples (soak report S5).
        fingerprint = self.cache_manager.prefix_fingerprint()
        budget = self.cache_manager.admissible_size
        if req.match_fp == fingerprint:
            estimated = req.input_len - req.match_cached_len + req.output_len
            if estimated + self.reserved_size > budget:
                return None

        # TODO: consider host cache match case
        mr = self.cache_manager.match_req(req)
        handle = mr.cuda_handle
        cached_len = handle.cached_len
        req.match_fp = fingerprint
        req.match_cached_len = cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # Charge the whole remaining footprint against the pages the pool can actually
        # OBTAIN. ``available_size`` is evictable prefix + free list + uncommitted suffix:
        # by construction it already excludes the KV held by decoding requests and the
        # locked/retained session prefixes that eviction cannot touch, which is exactly what
        # a gate against ``max_size`` could not see. ``reserved_size`` carries the rest of
        # the claim -- in-flight decode growth, the unforwarded tail of every prompt already
        # mid-prefill, and whatever this pass has admitted so far.
        #
        # Charging the *remaining* tail rather than the whole prompt is what makes this
        # stable as a prefill advances: forwarding a chunk drops ``available_size`` and the
        # charge by the same number of tokens, so an in-flight prompt never frees admission
        # room for a new one by making progress. (Charging the shrinking tail against
        # ``max_size`` does exactly that, and livelocks.)
        if estimated_len + self.reserved_size > budget:
            return None
        self.cache_manager.lock(handle)
        # Re-read: lock() moved the matched prefix out of ``evictable``, so the budget shrank.
        if estimated_len + self.reserved_size > self.cache_manager.admissible_size:
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
        # reqs admitted earlier in this pass have already placed on it. On a fresh admit this
        # is a no-op: _try_allocate_one just reserved the WHOLE remainder plus output_len
        # against the same budget, which is never smaller than this chunk.
        kv_ps = self.cache_manager.page_size
        kv_pages = max(0, self.cache_manager.available_size - self.reserved_pages) // kv_ps
        max_kv_end = (div_ceil(cached_len, kv_ps) + kv_pages) * kv_ps
        chunk_size = min(chunk_size, max(max_kv_end - cached_len, 0))
        # A chunk that COMPLETES the prompt hands the request straight to the decode
        # manager, and its ``output_len`` pages are then allocated one per forward with no
        # gate of their own. They have to come out of the same budget as this chunk, or the
        # pass leaves ``available_size`` below ``inflight_tokens`` and the next decode batch
        # is unbackable -- "batch needs 5 pages but only 2 are physically allocatable".
        # When both do not fit, forward less and stay chunked: the request finishes in a
        # later pass, when its decode IS backable. (Charging ``output_len`` unconditionally
        # would instead make every continuation reserve against its own next chunk and
        # starve its peers.)
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
        # Only a FRESH admit adds a new claim on ``available_size``: a continuation's
        # remaining footprint is seeded into ``reserved_size`` by the manager at the top of
        # every pass (see schedule_next_batch), because the adder is rebuilt each pass and
        # would otherwise be blind to prompts admitted earlier. Charging it twice was
        # harmless while the pass stopped at the first refusal; now that it continues past
        # one, a continuation can be reached after a fresh admit and the duplicate would
        # refuse its own peers.
        if pending_req.chunked_req is None:
            self.reserved_size += remain_len + pending_req.output_len
        self.reserved_pages += self._page_span(cached_len, cached_len + chunk_size)
        if not is_chunked:
            # This request's prompt is complete, so it decodes on the very next forward and
            # those pages are allocated with no further gate. Book them now -- after its own
            # chunk was sized, so it never reserves against itself -- so the lanes behind it
            # in this same pass can no longer spend them.
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
    # A refused request no longer ends the pass (see schedule_next_batch), so the scan needs
    # its own bound or a long queue of unseatable requests would be walked on every pass.
    # Sized above the usual --max-running-requests so a full queue is still scanned once;
    # the pass also stops as soon as the token budget is spent.
    max_admission_refusals: int = 32
    # Passes a refused FRESH prompt tolerates being overtaken before it reserves the queue.
    #
    # Strict FIFO among fresh admits (never overtake a refused prompt) protects long prompts
    # -- without it the long-context requests the soak timed out on never win a size
    # comparison against a short one, costing 25% of the prefill tokens on the stage profile.
    # But applied unconditionally it converts ONE temporarily unaffordable prompt into a dead
    # scheduler: the pass admits nothing, so nothing completes, so nothing is freed, so the
    # prompt stays unaffordable -- observed as a pass refusing a 118K head while a median of
    # 8 admissible requests sat behind it, until the 600 s client timeout collected them all.
    #
    # Aging gets both. While a refusal is young its queue-mates go first, and the requests
    # they complete are what release the KV (and idle the sessions whose leases reclaim can
    # then buy) that the blocked prompt is waiting for. Once it has been passed over this
    # many times it blocks the fresh queue behind it and the pool drains toward it.
    #
    # The value is the knob between throughput and long-prompt latency, swept on the replay
    # (seed 7, 20,000 forwards; switchyard-stage error rate / stage wait-to-first-chunk p95):
    #     patience   2      4      8     16     32
    #     err rate  .296   .230   .210   .185   .092
    #     wait p95  355 s  360 s  379 s  427 s  466 s
    # Larger is better for goodput and worse for the long-context prompts the soak timed out
    # on. 8 is the largest value whose wait p95 still sits under the upstream merge-base
    # baseline (395 s), so the fairness the strict rule was protecting is preserved.
    admission_patience: int = 8

    def _pending_prefill_size(self) -> int:
        """Unforwarded footprint of every prompt already mid-prefill.

        The adder is rebuilt every pass, so without this hand-over it cannot see the claim
        that prompts admitted in EARLIER passes still have on the pool. Each contributes the
        tail it has yet to forward plus its decode, and not its whole prompt: the part it
        already forwarded is allocated and locked, so it has left ``available_size``
        already, and charging it twice would refuse admissions the pool can afford.
        """
        total = 0
        for req in self.pending_list:
            chunked = req.chunked_req
            if chunked is not None:
                total += max(0, req.input_len - chunked.cached_len) + req.output_len
        return total

    def _seatable_lanes(self, reserved: int) -> int:
        """Lanes this pass can plausibly seat -- the divisor the interleave share needs.

        ``token_budget // waiting`` divides by the QUEUE DEPTH: sixteen queued requests buy
        a 512-token chunk even in a pass whose pool can seat two lanes, which is 12.5% of
        the budget and the whole of the interleave's throughput regression (soak report R5).
        The seating limit is knowable and cheap -- pure arithmetic over the pending list,
        no prefix walk and no allocation.

        Deliberately an estimate, and deliberately an over-estimate where it is wrong: a
        lane counted here that the pass then cannot seat only makes the surviving chunks
        smaller, never unbackable. The per-chunk cap in ``_add_one_req`` is what enforces
        the pool exactly.
        """
        budget = self.cache_manager.admissible_size - reserved
        slots = self.table_manager.available_size
        lanes = 0
        blocked_fresh = False
        for req in self.pending_list:
            if req.chunked_req is not None:
                lanes += 1  # already admitted and already charged; the gate does not re-run
                continue
            if blocked_fresh:
                continue
            # The cached part is not modelled (a prefix hit can only make a lane cheaper),
            # so this reads the prompt at full price -- the conservative direction for a
            # divisor, since over-counting lanes is what shrinks chunks.
            cost = req.input_len + req.output_len
            if slots <= 0 or cost > budget:
                # Mirror the admission loop's strict FIFO: nothing fresh behind a fresh
                # request the pool turned away is seatable this pass either.
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
        inflight_prefill = self._pending_prefill_size()
        adder = PrefillAdder(
            token_budget=prefill_budget,
            # Every claim on ``available_size`` that this pass did not create: the growth
            # the running decodes will still allocate, plus the unforwarded tail of every
            # prompt already mid-prefill. The second term is what keeps a fresh admit from
            # being let into a pool that its own in-flight peers have already spoken for --
            # the anti-livelock invariant. It is charged against ``available_size`` and not
            # against the pool maximum precisely because the maximum still counts the KV
            # held by decoding requests and by locked/retained session prefixes, which
            # admission cannot spend (soak report S5).
            reserved_size=self.decode_manager.inflight_tokens + inflight_prefill,
            # NOT the sum above. ``reserved_pages`` is the per-chunk cap's budget and it is
            # spent in the pass it is computed for, so it books only the decode pages THIS
            # pass can be asked for: the requests already decoding, plus (charged in
            # ``_add_one_req``, after each one's own chunk is sized) the requests whose
            # prefill finishes here. A continuation with chunks still to go decodes many
            # passes from now, against a pool that will have moved; reserving for it here
            # only starves its peers -- the 6-lanes-to-2 regression of soak report R6.
            reserved_pages=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        seatable_lanes = (
            self._seatable_lanes(adder.reserved_size) if self.interleave_chunks else 0
        )
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
        # pool just turned away -- and since the refusals land on the long prompts, "later
        # and cheaper" means "every short prompt, forever".
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
                if (pending_req.input_len + pending_req.output_len
                        > self.cache_manager.max_size):
                    # Unsatisfiable at EVERY pool state, not just this one: the pool could
                    # be empty and this prompt still would not fit. Skipping it WITHOUT
                    # setting ``blocked_fresh`` is what keeps it from wedging the queue --
                    # FIFO fairness towards a request that can never be served is just a
                    # stalled scheduler.
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
                pending_req.refused_passes = 0
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
                # Skip, do not stop. The pass used to end here, abandoning every queued
                # request behind the refused one -- a median of 13, of which 11 the pools
                # could have seated. Continuations in particular are never gated by the
                # fresh-admit rule, so a blocked fresh prompt must not block them.
                refusals += 1
                if not is_continuation:
                    pending_req.refused_passes += 1
                    # Reserve the queue only once this prompt has been patient enough; see
                    # ``admission_patience``.
                    blocked_fresh = blocked_fresh or (
                        pending_req.refused_passes >= self.admission_patience
                    )
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
