from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, List, Tuple

import torch
from freetoken.core import Batch, Req
from freetoken.utils import align_down, div_ceil, init_logger

from .counters import PrefillCounters
from .utils import PendingReq

if TYPE_CHECKING:
    from freetoken.kvcache import BaseCacheHandle
    from freetoken.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)

_INVARIANT_OFF = ("", "0", "off", "false", "no")


def _invariant_mode() -> str:
    """Debug assertion on the finishability invariant, off unless asked for.

      FREETOKEN_SCHEDULER_INVARIANT=warn   log every violation (safe for a live soak)
      FREETOKEN_SCHEDULER_INVARIANT=raise  raise on the first one (CI / a bisect)

    The invariant is stated on ``PrefillManager._check_finishability``. Read per pass
    rather than once at import, so a soak can be told to start checking without a restart
    and a test can set it without reloading this module out from under everyone who has
    already imported ``PrefillManager``; one env lookup per prefill pass is free next to
    the radix walk the same pass runs.
    """
    mode = os.environ.get("FREETOKEN_SCHEDULER_INVARIANT", "").strip().lower()
    return "" if mode in _INVARIANT_OFF else mode


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
    # Negative means "start from reserved_size" (i.e. the decode in-flight tokens) -- which
    # is only correct while ``reserved_size`` carries nothing else. The manager passes it
    # explicitly, because ``reserved_size`` now also carries the STANDING RESERVATION of
    # every prompt already mid-prefill: a cross-pass admission claim, not a page demand this
    # pass can be asked for. Inheriting that here would re-create the starvation above.
    reserved_pages: int = -1

    def __post_init__(self) -> None:
        if self.reserved_pages < 0:
            self.reserved_pages = self.reserved_size

    def _page_span(self, start: int, end: int) -> int:
        """Fresh pages the extend ``[start, end)`` pulls, charged per WHOLE page."""
        ps = self.cache_manager.page_size
        return (div_ceil(end, ps) - div_ceil(start, ps)) * ps

    # ---- the three admission gates -----------------------------------------------------
    # Factored out so the seat scan (:meth:`would_seat`, driven by
    # ``PrefillManager._seatable_lanes``) asks the pools exactly the questions the real
    # admission below asks them, off the same reservation counters. A second, drifting copy
    # of this arithmetic is precisely how the previous chunk-share attempt became a no-op:
    # it modelled a lane as costing one page and a table slot, which every pass could
    # afford, so its count was always the queue depth it was meant to replace.

    def _kv_gate_ok(self, estimated_len: int) -> bool:
        """Whole-footprint admission gate for a FRESH prompt.

        ``owed(admitted set) + owed(this prompt) <= available_size`` -- see the long note in
        :meth:`_try_allocate_one`, which is the only caller that may act on a True.
        """
        return estimated_len + self.reserved_size <= self.cache_manager.available_size

    def _swa_seat_ok(self, extend_len: int) -> bool:
        """Can the swa pool seat this fresh request's first chunk / one window?"""
        cm = self.cache_manager
        if not cm.swa_paged:
            return True
        ps = cm.page_size
        # swa is charged per WHOLE page (allocate_paged -> alloc_swa), so the seat check is
        # in page units too; identical at page_size==1.
        need = div_ceil(min(max(extend_len, 1), cm.sliding_window_size) + 1, ps) * ps
        return cm.swa_available_size - self.reserved_swa >= need

    def _kv_chunk_end(self, cached_len: int) -> int:
        """Highest token offset the pool's remaining pages can back for this lane THIS pass.

        ``available_size`` (evictable prefix + free slots + the not-yet-committed growable
        suffix) is exactly the ceiling ``committed_pages_required`` tests against, and
        ``reserved_pages`` is the page demand the reqs admitted earlier in this pass have
        already placed on it.
        """
        ps = self.cache_manager.page_size
        pages = max(0, self.cache_manager.available_size - self.reserved_pages) // ps
        return (div_ceil(cached_len, ps) + pages) * ps

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

        # Charge the whole remaining footprint against what the pool can actually give.
        # ``reserved_size`` is every claim already standing on it: the growth the running
        # decodes will still allocate, the unforwarded tail (plus decode) of every prompt
        # ALREADY mid-prefill, and whatever this pass has admitted so far. Together with the
        # per-admit charge below, passing this check is exactly the statement
        #
        #     owed(admitted set) + owed(this prompt)  <=  available_size
        #
        # i.e. the admitted SET stays finishable, not merely each arrival at the instant it
        # arrives. That distinction is the whole of soak report T5: 14 prefills each passed
        # a gate that measured only itself, and between them owed 1.76x the pool.
        if not self._kv_gate_ok(estimated_len):
            return None
        self.cache_manager.lock(handle)
        # Re-read: lock() moved the matched prefix out of ``evictable``, so the budget shrank.
        if not self._kv_gate_ok(estimated_len):
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
        if not self._swa_seat_ok(extend_len):
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
        # unbackable and killed the scheduler. See :meth:`_kv_chunk_end`. On a fresh admit
        # this is a no-op: _try_allocate_one just reserved the WHOLE remainder plus
        # output_len against the same budget, which is never smaller than this chunk.
        chunk_size = min(chunk_size, max(self._kv_chunk_end(cached_len) - cached_len, 0))
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
        # Only a FRESH admit adds a new claim. A continuation's remaining footprint is
        # already in ``reserved_size``: the manager seeds it at the top of every pass from
        # the in-flight prefills (see PrefillManager._standing_reservation), because the
        # adder is rebuilt per pass and would otherwise be blind to prompts admitted
        # earlier. Charging it a second time here would refuse the continuation's own peers.
        if pending_req.chunked_req is None:
            self.reserved_size += remain_len + pending_req.output_len
        self.reserved_pages += self._page_span(cached_len, cached_len + chunk_size)
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

    def would_seat(self, pending_req: PendingReq, table_free: int, mamba_free: int) -> bool:
        """Non-mutating mirror of :meth:`try_add_one`: does this lane get a page this pass?

        Run on a throwaway copy of the adder (``dataclasses.replace``), so the reservations
        it charges accumulate exactly as the real admission loop's would -- which is the
        whole point: the constraint that decides how many lanes a pass seats is the FRESH
        gate's ``reserved_size``, and it only bites once the earlier admits of the same pass
        have been charged to it.

        Each seated lane is charged ONE page, not a chunk: the question this scan answers is
        whether the lane appears in the batch at all, and its chunk size is what the count
        being computed decides. That makes the scan optimistic about pages, so the real pass
        may seat fewer lanes than it predicted -- the loop recomputes the share against the
        lanes it has actually seated, so the budget the missing lanes would have taken goes
        to the ones behind them instead of being lost. It can never make a pass UNSAFE:
        ``chunk_limit`` is only ever an upper bound handed to :meth:`_add_one_req`, and every
        hard cap (the whole-footprint gate, the per-lane KV page cap, the swa cap, the
        finishability invariant) is re-applied there against the real pools.

        ``table_free`` / ``mamba_free`` are the caller's simulated slot counts; the pools
        themselves cannot be decremented without allocating.
        """
        if chunked := pending_req.chunked_req:
            # A continuation owns its table slot and GDN slots already; the only thing that
            # can keep it out of the batch is the pool being unable to back one more page.
            cached_len = chunked.cached_len
            if self._kv_chunk_end(cached_len) - cached_len <= 0:
                return False
            self.reserved_pages += self.cache_manager.page_size
            return True

        if table_free <= 0:
            return False
        # Same radix walk the real admit will run a moment later, on the same tokens: it
        # bumps the same LRU timestamps and takes the same node splits, so the scan cannot
        # reorder eviction. The loop below stops at the first refusal exactly as the real
        # one does, so this doubles the pass's match calls over the SEATED prefix only --
        # in the measured stage regime that is one extra walk per pass.
        handle = self.cache_manager.match_req(pending_req).cuda_handle
        cached_len = handle.cached_len
        extend_len = pending_req.input_len - cached_len
        if not self._kv_gate_ok(extend_len + pending_req.output_len):
            return False
        # 1 live + 2 ping-pong, as _try_allocate_one reserves. The scan does not escalate
        # into eviction or the session-lease spill the way reserve_mamba_slots does, so it
        # is conservative here; being conservative only shrinks the divisor, which the
        # per-iteration recompute then corrects upward as lanes are actually seated.
        if self.cache_manager.is_hybrid and mamba_free < 3:
            return False
        if not self._swa_seat_ok(extend_len):
            return False
        if self._kv_chunk_end(cached_len) - cached_len <= 0:
            return False
        self.reserved_size += extend_len + pending_req.output_len
        self.reserved_pages += self.cache_manager.page_size
        if self.cache_manager.swa_paged:
            self.reserved_swa += self.cache_manager.page_size
        return True

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
    # Hard ceiling on how many prompts may be mid-prefill at once, enforced against FRESH
    # admits only. Belt and braces: the standing reservation below is the invariant that
    # actually keeps the admitted set finishable, and this is the bound that survives an
    # arithmetic mistake in it. A chunked prefill holds its forwarded pages until it
    # completes and cannot be evicted, aborted or reclaimed, so N of them are N unbounded
    # claims on the pool; soak report T5 deadlocked a 262,144-token pool with fourteen.
    # Set well above the 2.4 mean lanes the passing tree seats, so in ordinary operation it
    # never binds -- if it starts binding, the reservation arithmetic is wrong.
    max_chunked_prefills: int = 8
    # Cumulative record of what each pass decided, published on ``/v1/stats``. See
    # :mod:`freetoken.scheduler.counters` -- every increment sits on a branch this
    # scheduler already takes.
    counters: PrefillCounters = field(default_factory=PrefillCounters)

    def _standing_reservation(self) -> int:
        """Unforwarded footprint of every prompt already mid-prefill.

        The adder is rebuilt every pass, so without this hand-over it cannot see the claim
        that prompts admitted in EARLIER passes still have on the pool -- and an admission
        gate that cannot see them re-sells the same capacity once per pass. Each contributes
        the tail it has yet to forward plus its decode, and NOT its whole prompt: the part
        it already forwarded is allocated and locked, so it has left ``available_size``
        already, and charging it twice would refuse admissions the pool can afford.

        Held for the life of the prefill and released implicitly as it advances: forwarding
        a chunk drops both ``available_size`` and this figure by the same number of tokens,
        so an in-flight prompt never buys admission room for a new one by making progress,
        and the reservation is gone the moment the request leaves ``pending_list``.
        """
        total = 0
        for req in self.pending_list:
            chunked = req.chunked_req
            if chunked is not None:
                total += max(0, req.input_len - chunked.cached_len) + req.output_len
        return total

    def finishability_reservation(self) -> int:
        """Everything the pool has already promised, in tokens.

        The exact left-hand side of :meth:`_check_finishability`: the standing reservation
        of the prompts mid-prefill plus the growth the running decodes will still allocate.
        Exported because that proof is only worth what it is checked against -- anything
        else that spends pool pages BETWEEN two prefill passes (a cold session restore,
        which materialises committed pages straight out of the message path) has to charge
        itself against this figure, or it retroactively invalidates a finishability the
        admission gate had already established for requests it cannot see.

        Soak §W6 is that failure: four cold restores, one of 79,104 tokens, landed in the
        two seconds before nine ``finishability invariant`` warnings whose shortfall was a
        constant 1,401 tokens -- the pool over-promised exactly once, by a restore, and
        then tracked the admitted set in lockstep as it drained.
        """
        return self._standing_reservation() + self.decode_manager.inflight_tokens

    def _seatable_lanes(
        self, adder: PrefillAdder, lane_cap: int, chunked_inflight: int
    ) -> int:
        """How many lanes will this pass actually seat? -- the interleave share's divisor.

        The 8,192-token prefill budget has to be split between the lanes that end up in the
        batch, and the admission loop below is FIFO with a single stopping rule, so that set
        is a prefix of the queue: continuations (seatable whenever the pool can back a page)
        and fresh prompts that clear the standing-reservation gate, minus the fresh prompts
        the ``max_chunked_prefills`` cap skips, cut short by the first refusal and by
        ``lane_cap``. This walks that prefix on a copy of the adder and counts it.

        Dividing by the QUEUE DEPTH instead -- what ``f3c3ac4`` shipped and what soak report
        §R7 ticket 1 / §U8 measured -- is the stage route's starvation: sixteen queued
        requests cost a 16x smaller chunk in a pass whose pools will seat one lane, so a
        118 K-token prompt advanced 512 tokens per pass (1,278 of 2,091 stage passes carried
        the ``1 lane / <=512 new tokens / >=8 queued`` signature) and prefill budget
        utilisation collapsed to 14 %. The divisor has to be the seat count, and the seat
        count is dominated by the fresh gate's whole-prompt reservation -- which is why a
        cheaper approximation of "admissible lanes" that models a lane as one page and a
        table slot returned the queue depth again and changed nothing.
        """
        scan = replace(adder)
        table_free = self.table_manager.available_size
        mamba_free = (
            self.cache_manager.mamba_available_size if self.cache_manager.is_hybrid else 0
        )
        inflight = chunked_inflight
        seatable = 0
        for pending_req in self.pending_list:
            is_fresh = pending_req.chunked_req is None
            if is_fresh and inflight >= self.max_chunked_prefills:
                continue  # skipped, not stopped on -- mirrors the loop below
            if not scan.would_seat(pending_req, table_free, mamba_free):
                break
            seatable += 1
            if is_fresh:
                table_free -= 1
                mamba_free -= 3
                inflight += 1
            if lane_cap and seatable >= lane_cap:
                break
        return seatable

    def _check_finishability(self, standing: int, mode: str = "") -> None:
        """Debug assertion: is the set already admitted still finishable?

            owed = standing reservation of the in-flight prefills
                 + DecodeManager.inflight_tokens
            owed <= CacheManager.available_size

        Free-or-evictable is everything admission can spend, and every token in ``owed``
        has to be bought out of it before any of those requests can complete and hand a
        page back. A violation is therefore the deadlock PRECONDITION, not a symptom of one
        (the pool can still be nowhere near full when it first goes negative): it says the
        pool has promised more than it holds, and the chunked-prefill scheduler advances
        every lane together, so nothing will complete to break the tie.

        Soak report T5 is what this catches: 14 chunked prefills that had forwarded 237,819
        tokens of a 262,144-token pool and still owed 222,538, with #running-req 0 and
        ``_reclaim_for_blocked_prefill`` returning False forever.

        Off by default -- ``FREETOKEN_SCHEDULER_INVARIANT=warn`` for a live soak,
        ``=raise`` to fail fast.
        """
        owed = standing + self.decode_manager.inflight_tokens
        budget = self.cache_manager.available_size
        # Counted on EVERY pass, whatever the mode: the comparison is three attribute reads
        # (the same ``available_size`` the adder built two lines later reads anyway), so
        # there is nothing to save by gating it -- and a violation that only a soak with the
        # env var set can see is a violation nobody sees. ``mode`` still decides whether it
        # is also logged or raised.
        self.counters.note_invariant(owed - budget)
        if owed <= budget:
            return
        if not mode:
            return
        chunked = sum(1 for r in self.pending_list if r.chunked_req is not None)
        msg = (
            "prefill finishability invariant violated: %d in-flight chunked prefills plus "
            "%d decode tokens owe %d, but only %d tokens are obtainable (short by %d). "
            "This is the deadlock precondition of soak report T5."
            % (chunked, self.decode_manager.inflight_tokens, owed, budget, owed - budget)
        )
        if mode == "raise":
            raise AssertionError(msg)
        logger.warning_rank0("%s", msg)

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
        standing = self._standing_reservation()
        # Checked BEFORE this pass admits anything: the gate below makes the invariant
        # true by construction for what it admits, so the only interesting question is
        # whether it still holds for the set admitted in EARLIER passes, against a pool
        # that has moved since. Always evaluated (and counted); FREETOKEN_SCHEDULER_INVARIANT
        # only decides whether a violation is additionally logged (``warn``) or raised.
        self._check_finishability(standing, _invariant_mode())
        adder = PrefillAdder(
            token_budget=prefill_budget,
            # Every claim on ``available_size`` that this pass did not create: the growth the
            # running decodes will still allocate, plus the standing reservation of every
            # prompt already mid-prefill.
            reserved_size=self.decode_manager.inflight_tokens + standing,
            # NOT the sum above -- see PrefillAdder.reserved_pages. This is the per-chunk
            # cap's budget and it is spent in the pass it is computed for, so it books only
            # the pages THIS pass can be asked for.
            reserved_pages=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        chunked_inflight = sum(
            1 for req in self.pending_list if req.chunked_req is not None
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
        stopped_for_lane_cap = False
        # Divisor for the interleave share: the lanes this pass will actually seat, not the
        # queue depth. Computed once, then spent down as lanes are seated (below), so a lane
        # that takes less than its share -- a short remainder, or a cap the pools imposed --
        # hands the difference to the lanes behind it instead of leaving the budget unspent.
        seatable = (
            self._seatable_lanes(adder, lane_cap, chunked_inflight)
            if self.interleave_chunks and len(self.pending_list) > 1
            else 0
        )
        self.counters.note_pass(seatable=seatable, chunked_inflight=chunked_inflight)
        for index, pending_req in enumerate(self.pending_list):
            is_continuation = pending_req.chunked_req is not None
            if not is_continuation and chunked_inflight >= self.max_chunked_prefills:
                # Belt and braces on top of the standing reservation; see the knob. Skipped
                # rather than breaking, so the continuations behind it -- which are what
                # brings the count back down -- still get their chunk this pass.
                self.counters.fresh_admits_blocked_by_cap += 1
                continue
            chunk_limit = None
            if seatable:
                # ``seatable - len(reqs)`` are the seats left to fill. Floored at 1: the scan
                # is optimistic about pages and blind to the eviction tiers a real admit can
                # escalate through, so the pass can outrun it -- and the lane that does is by
                # construction the last one, which may as well have what is left.
                chunk_limit = max(1, adder.token_budget // max(1, seatable - len(reqs)))
            if req := adder.try_add_one(pending_req, chunk_limit=chunk_limit):
                admitted_index.add(index)
                was_chunked = pending_req.chunked_req is not None
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                    # This lane took a chunk and still owes a remainder: one deferral.
                    self.counters.deferred_chunks += 1
                    if not was_chunked:
                        chunked_inflight += 1
                elif was_chunked:
                    chunked_inflight -= 1
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
                    # The queue tail was never REFUSED, only never reached: it is safe (and
                    # fair) to rotate it in front of the lanes that just ran. See the
                    # re-queue below -- this is the only stop for which that holds.
                    stopped_for_lane_cap = index + 1 < len(self.pending_list)
                    break
            else:
                # Refused for pool / table / budget, not for lanes: the queue tail behind it
                # goes unserved this pass. Distinguished from the lane-cap stop above, which
                # is a fair rotation rather than back-pressure.
                self.counters.refusals += 1
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        remaining = [
            req for i, req in enumerate(self.pending_list) if i not in admitted_index
        ]
        # Re-queue order. Unfinished lanes go back to the HEAD unless the pass stopped
        # because it ran out of LANES: any other stop means the pass was refused, and putting
        # the refused request in front of a live continuation makes the next pass break on it
        # and return no batch while a runnable lane sits behind it (pinned by
        # ``test_interleaved_prefill_does_not_queue_blocked_agent_before_active_lane``).
        #
        # ``stopped_for_lane_cap`` is therefore narrower than the sentence below used to
        # claim: ``lane_cap`` is ``max_batch_seqs``, which ``_resolve_max_prefill_seqs``
        # leaves at 0 for every non-GGUF model, so this rotation does NOT run on the soaked
        # Nemotron tree and never has -- bisect §R5 called it dead on that evidence. It is not
        # dead in general (a GGUF lane cap reaches it; see
        # ``test_single_lane_prefill_rotates_long_prompts_without_grouping_them``), so it
        # stays, with the scope stated rather than implied. Generalising it to "rotate
        # whenever the pass stopped for capacity, budget included" was tried and the blocked-
        # agent test above rejects it: the token budget runs out with a request at the head
        # that cannot reserve KV, and the admitted lane never gets its second chunk.
        #
        # What makes chunked lanes fair on the non-GGUF path is the chunk SHARE, not this
        # ordering: every lane the pass seats gets ``token_budget / seatable`` (see
        # :meth:`_seatable_lanes`).
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
