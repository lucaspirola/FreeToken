"""Prompt-lookup (n-gram) speculative decoding, greedy-only, single-stream.

Background: ``benchmarks/results/nemotron35_lightning_5080_ngram_spec_2026-09-05.md``
measured the acceptance side offline (an n = 8 drafter reaches lambda = 3.6 accepted tokens
per step at 93 % per-token acceptance on copy-heavy agent traffic, and stays within 0.5 % of
neutral on code and prose), and ``..._extend_moe_2026-09-05.md`` removed the blocker: an
extend forward of <= 64 tokens now costs ~30 ms instead of ~290. Measured end to end
(``..._ngram_spec_impl_2026-09-05.md``) a verify step is ~7x a graphed decode step at short
context and ~10x at 131K -- so it pays on copy-heavy traffic and not at long context, which
is what the break-even gate in :meth:`SpecNgramDecoder._pays_off` is for.

Shape of a step
---------------
After an ordinary decode step a request is in a fixed shape: ``cached_len == L`` (the
recurrent + KV state covers ``tokens[:L]``), ``device_len == L + 1``, and ``tokens[L]`` is
the freshly sampled token, not yet forwarded. A verify step:

1. drafts ``k`` tokens from the request's own prompt + output (most-recent occurrence of the
   trailing n-gram) and stages them into ``token_pool`` at positions ``L+1 .. L+k``;
2. runs ONE extend forward over the ``m = k + 1`` positions ``L .. L+k``, keeping every
   logits row, with the Mamba-2 state pointed at a private scratch slot so the live state is
   not advanced (see models/nemotron_h/spec_scan.py);
3. accepts the longest prefix of the draft that the greedy argmax agrees with, plus the
   bonus token the first disagreeing row predicts -- so a step emits ``accepted + 1`` tokens
   and is always at least as productive as a plain decode step;
4. commits the accepted prefix into the live recurrent state, frees the KV pages the
   rejected positions allocated, and emits one ``DetokenizeMsg`` per accepted token.

Rejected tokens never reach the host token list, so they cannot reach the prefix cache
either -- the radix insert boundary is ``req.cached_len``, which this module only ever
advances by the accepted count.

Why it runs drained
-------------------
A drafter needs every emitted token before it can index the next n-gram, so a verify step
cannot overlap with its own successor. Running the whole decode loop drained would cost
~30 % on the (overwhelmingly common) steps that never draft, so the engagement decision is
made *before* the drain, with the one-token-stale token list: ``peek()`` is a single dict
lookup, and only a hit pays for the drain. On code and prose the drafter fires on ~0.5 % of
steps, so the overlapped path is what runs; the price is that a burst is entered one step
late, which costs a factor of ~4 in draft rate on copy-heavy traffic (write-up §7).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from freetoken.core import Batch, Req
from freetoken.message import DetokenizeMsg
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Weight of a new sample in the accepted-length EWMA, and how often a closed break-even gate
# lets one verify step through anyway so its own estimate can be refuted.
_EWMA_ALPHA = 0.25
_REPROBE_EVERY = 16
# The gate is deliberately hard to close and easy to leave open. A false close costs a real
# win on the traffic the feature exists for (measured: a too-eager gate turned +11 % on the
# copy class into -0.3 %); a false open costs one verify step per _REPROBE_EVERY. So it takes
# at least two timed verify steps and a clear margin before it will refuse.
_GATE_MARGIN = 1.25
_GATE_MIN_SAMPLES = 2


def _ewma(old: float | None, sample: float) -> float:
    return sample if old is None else _EWMA_ALPHA * sample + (1.0 - _EWMA_ALPHA) * old


def _floor(old: float | None, sample: float) -> float:
    """Running minimum -- the steady-state cost of a VERIFY step.

    The two sides of the ratio need different estimators, and getting this wrong closes the
    gate on traffic that should pass (measured, both ways):

    * **verify: a floor.** A verify step is sampled a few dozen times at most, and the FIRST
      one pays Triton autotuning for a shape nothing else uses -- that one-off made an EWMA
      read 11.35 where the steady state is 4-5, which closed the gate and then starved it of
      the samples that would have reopened it.
    * **decode: an EWMA, NOT a floor.** A decode step is sampled hundreds of times and the
      scheduler loop's gap is not uniform, so its *minimum* is far below its typical value --
      a floor there inflated the ratio and gated out 264 of 278 copy-class drafts, turning
      +11 % into -1.5 %. With hundreds of samples an average is the robust estimator.
    """
    return sample if old is None else min(old, sample)


# --------------------------------------------------------------------------- drafter


class NgramDrafter:
    """Most-recent-occurrence n-gram lookup over one request's token stream.

    ``n`` is 8 by default, not the literature's 3: when verification is expensive you draft
    for precision, not recall. A 3-gram fires on 12 % of code steps and is right 23 % of the
    time, which costs more in wasted verify steps than the accepted tokens buy; n = 8 keeps
    93 % of the copy-class acceptance and drops the code/prose draft rate 30x (see §2 of the
    n-gram write-up).
    """

    __slots__ = ("n", "index", "cursor")

    def __init__(self, n: int) -> None:
        assert n >= 1
        self.n = n
        self.index: Dict[Tuple[int, ...], int] = {}
        self.cursor = 0

    def observe(self, tokens: Sequence[int]) -> None:
        """Index every n-gram of ``tokens`` that ends strictly before the last position.

        The n-gram ending at the very end is the query itself; indexing it would overwrite
        the most recent real occurrence with a self-match and the drafter would never fire.
        """
        n = self.n
        limit = len(tokens) - 1
        i = self.cursor
        idx = self.index
        while i + n <= limit:
            idx[tuple(tokens[i : i + n])] = i + n
            i += 1
        self.cursor = i

    def _hit(self, tokens: Sequence[int]) -> int | None:
        pos = len(tokens)
        if pos < self.n:
            return None
        hit = self.index.get(tuple(tokens[pos - self.n : pos]))
        # A match at the very end predicts nothing (it is the query).
        return None if hit is None or hit >= pos else hit

    def has_match(self, tokens: Sequence[int]) -> bool:
        return self._hit(tokens) is not None

    def draft(self, tokens: Sequence[int], k: int) -> List[int]:
        """Up to ``k`` tokens proposed to follow ``tokens``. Empty list = no match."""
        if k <= 0:
            return []
        hit = self._hit(tokens)
        if hit is None:
            return []
        return list(tokens[hit : hit + k])


def accepted_count(draft: Sequence[int], greedy: Sequence[int]) -> int:
    """Length of the longest prefix of ``draft`` the greedy continuation agrees with.

    ``greedy[i]`` is the model's own token for the position ``draft[i]`` occupies, so the
    step emits ``greedy[: accepted + 1]`` -- the accepted drafts (identical to the greedy
    tokens by construction) plus the bonus token the first disagreement predicts.
    """
    j = 0
    while j < len(draft) and j < len(greedy) - 1 and greedy[j] == draft[j]:
        j += 1
    return j


# --------------------------------------------------------------------------- per-request state


@dataclass
class SpecStats:
    verify_steps: int = 0
    plain_peeks: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    # Why a peek that matched did not become a verify step. A drafter that fires and then
    # never runs is otherwise indistinguishable from one that never fires -- which is exactly
    # how the first measured run hid a starved state-pool scratch slot.
    declined_shape: int = 0
    declined_no_slot: int = 0
    declined_budget: int = 0
    declined_stale_match: int = 0
    declined_uneconomic: int = 0
    # Accepted-token histogram, one entry per verify step, keyed by the accepted count as a
    # string (the wire document is JSON). The mean alone hides the shape that decides whether
    # speculation pays: lambda 3.6 from "half the steps accept 0 and half accept 7" is a
    # different engine than lambda 3.6 from "every step accepts 3-4".
    accepted_hist: dict = field(default_factory=dict)

    @property
    def accept_rate(self) -> float:
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0

    @property
    def tokens_per_verify(self) -> float:
        return self.emitted_tokens / self.verify_steps if self.verify_steps else 0.0

    @property
    def declines(self) -> dict:
        """Per-reason decline counts. A drafter that fires and then never runs is otherwise
        indistinguishable from one that never fires."""
        return {
            "shape": self.declined_shape,
            "no_slot": self.declined_no_slot,
            "budget": self.declined_budget,
            "stale_match": self.declined_stale_match,
            "uneconomic": self.declined_uneconomic,
        }

    def note_accepted(self, accepted: int) -> None:
        key = str(accepted)
        self.accepted_hist[key] = self.accepted_hist.get(key, 0) + 1

    def as_dict(self) -> dict:
        """The ``/v1/stats["scheduler"]["spec"]`` block."""
        return {
            "verify_steps": self.verify_steps,
            "plain_peeks": self.plain_peeks,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "emitted_tokens": self.emitted_tokens,
            "accept_rate": round(self.accept_rate, 4),
            "tokens_per_verify": round(self.tokens_per_verify, 4),
            "declined": self.declines,
            "accepted_hist": dict(self.accepted_hist),
        }


@dataclass
class _SpecState:
    """Drafter + adaptive draft length for the one request speculation is engaged on."""

    req: Req
    drafter: NgramDrafter
    max_k: int
    adaptive: bool
    tokens: List[int] = field(default_factory=list)
    k: int = 0
    # Break-even estimator, per request because the verify/decode cost ratio is a function
    # of THIS request's context length (see SpecNgramDecoder._pays_off). Carrying it across
    # requests would let a 131K session close the gate on the next short one.
    decode_ms: float | None = None
    verify_ms: float | None = None
    verify_samples: int = 0
    emit: float = 0.0
    gated: int = 0

    def __post_init__(self) -> None:
        self.k = self.max_k
        self.emit = float(self.max_k + 1)

    def sync(self, input_ids: torch.Tensor) -> None:
        """Extend the host token list to match ``req.input_ids`` and index the new n-grams."""
        have = len(self.tokens)
        want = int(input_ids.numel())
        if want > have:
            self.tokens.extend(input_ids[have:want].tolist())
        elif want < have:  # a stop-string truncation rewound the request
            del self.tokens[want:]
            self.drafter.cursor = min(self.drafter.cursor, max(0, want - self.drafter.n))
        self.drafter.observe(self.tokens)

    def note(self, drafted: int, accepted: int) -> None:
        """Adaptive draft length: halve after a rejection at position 0 (the drafter is on a
        span that does not continue), restore in full after a complete acceptance. The
        offline sweep found no gain from a finer ladder at n >= 8 -- the precision gate has
        already removed the regressions adaptivity exists to prevent -- so this exists only
        to bound the cost of a run of dead drafts."""
        if not self.adaptive:
            return
        if accepted == drafted:
            self.k = self.max_k
        elif accepted == 0:
            self.k = max(1, self.k // 2)


# --------------------------------------------------------------------------- the decoder


class SpecNgramDecoder:
    """Drives speculative verify steps for a single greedy request.

    v1 scope: exactly one running request, greedy sampling, hybrid (mamba2) or stateless
    attention, no SWA, no multimodal, no hidden-state probe. Anything else falls through to
    the ordinary decode path, which is why this class never has to be correct for a shape it
    does not recognise -- it declines.
    """

    def __init__(self, scheduler, *, n: int, draft_len: int, adaptive: bool = True) -> None:
        self.sch = scheduler
        self.n = n
        self.draft_len = draft_len
        self.adaptive = adaptive
        self.stats = SpecStats()
        self._state: _SpecState | None = None
        self._last_peek_at: float | None = None
        self._last_peek_hit = False
        self._check_commit = int(os.environ.get("FREETOKEN_SPEC_CHECK_COMMIT", "0"))
        self._spare_slot: int | None = None
        self.commit_error: tuple[float, float] = (0.0, 0.0)
        self.enabled, self.disabled_reason = self._supported()
        if not self.enabled:
            logger.warning(
                f"--speculative ngram disabled: {self.disabled_reason}"
            )

    # ------------------------------------------------------------------ support

    def _supported(self) -> tuple[bool, str]:
        cm = self.sch.cache_manager
        pool = self.sch.engine.linear_state_pool
        if getattr(cm, "is_swa", False):
            return False, "sliding-window attention has no per-step KV rollback"
        if pool is not None:
            if pool.state_layout != "mamba2":
                return False, f"recurrent state layout {pool.state_layout!r} has no verify commit"
            if not getattr(cm, "is_hybrid", False):
                # Non-hybrid keys the state slot on table_idx, which is also the page-table
                # row; there is no free-list the scratch slot could safely come from.
                return False, "recurrent models need radix caching (--cache-type naive has no state free-list)"
        return True, ""

    # ------------------------------------------------------------------ engagement

    def candidate(self) -> Req | None:
        """The request speculation may run on this iteration, or None."""
        if not self.enabled:
            return None
        sch = self.sch
        if sch.prefill_manager.runnable:
            return None
        running = sch.decode_manager.running_reqs
        if len(running) != 1:
            return None
        req = next(iter(running))
        params = req.sampling_params
        if (
            req.table_idx == -1
            or req.aborted
            or req in sch.finished_reqs
            or not req.can_decode
            or not params.is_greedy
            or req.mm_embeds is not None
            or getattr(req, "hidden_states", None) is not None
        ):
            return None
        return req

    def peek(self) -> Req | None:
        """The candidate whose trailing n-gram already has a match, or None.

        Called BEFORE the previous step is drained, so the token list is one token stale.
        That is exactly the right predictor: a copy burst that matched at ``L-1`` matches at
        ``L`` too, and a miss here costs one dict lookup.

        Doubles as the clock for the break-even estimator: the gap between two consecutive
        peeks that both took the ordinary path IS one overlapped decode step.
        """
        now = time.perf_counter()
        prev, self._last_peek_at = self._last_peek_at, None
        req = self.candidate()
        if req is None:
            # No candidate: this iteration is a prefill, an idle wait or a multi-request
            # step, none of which is a decode step. Drop the clock rather than poison the
            # estimator with a gap that measures something else.
            self._last_peek_hit = False
            return None
        self._last_peek_at = now
        state = self._state_for(req)
        if prev is not None and not self._last_peek_hit:
            sample = (now - prev) * 1e3
            # Outlier guard on both ends: a gap this short did not contain a forward, and
            # one this long was an idle wait, not a decode step. A floor is only as good as
            # its smallest sample, so the lower bound is the one that matters.
            if 0.2 < sample < 1000.0:
                state.decode_ms = _ewma(state.decode_ms, sample)
        self._last_peek_hit = False
        state.sync(req.input_ids)
        if state.k <= 0 or not state.drafter.has_match(state.tokens):
            self.stats.plain_peeks += 1
            return None
        if not self._pays_off(state):
            self.stats.declined_uneconomic += 1
            state.gated += 1
            # Re-probe periodically: the ratio is a function of context length and
            # acceptance, both of which move during a generation, so a closed gate has to
            # stay falsifiable.
            if state.gated % _REPROBE_EVERY:
                return None
        self._last_peek_hit = True
        return req

    def _pays_off(self, state: _SpecState) -> bool:
        """Is a verify step still cheaper than the decode steps it replaces?

        A verify step emits ``accepted + 1`` tokens for the price of ``verify_ms``, against
        ``decode_ms`` per token on the ordinary path -- so it pays exactly when
        ``emitted > verify_ms / decode_ms``. That ratio is NOT a constant: the verify
        forward's extend attention scales with ``m x context`` where a decode step is
        ``1 x context``, so the measured ratio is ~4.4x at short context and ~10x at 131K,
        where ``k + 1 = 9`` can no longer reach it (measured, §5 of the write-up). Estimating
        both terms online is what keeps a long-context session from paying for a drafter that
        cannot win, without a context-length threshold anyone has to tune.
        """
        if state.decode_ms is None or state.verify_samples < _GATE_MIN_SAMPLES:
            return True  # not measured yet: the first drafts ARE the measurement
        return state.emit * _GATE_MARGIN > state.verify_ms / max(state.decode_ms, 1e-6)

    def _state_for(self, req: Req) -> _SpecState:
        state = self._state
        if state is None or state.req is not req:
            state = _SpecState(
                req=req,
                drafter=NgramDrafter(self.n),
                max_k=self.draft_len,
                adaptive=self.adaptive,
            )
            self._state = state
        return state

    # ------------------------------------------------------------------ the step

    def run_step(self, req: Req) -> bool:
        """Run one verify step. Returns False if the request was not in a draftable shape,
        in which case nothing was mutated and the caller should take the ordinary path."""
        sch = self.sch
        state = self._state_for(req)
        state.sync(req.input_ids)
        L = req.cached_len
        if req.device_len != L + 1 or len(state.tokens) != req.device_len:
            self.stats.declined_shape += 1
            return False
        k = self._budget(req, state)
        if k <= 0:
            return False
        draft = state.drafter.draft(state.tokens, k)
        if not draft:
            # peek() matched on the one-token-stale prefix and this token broke the span.
            self.stats.declined_stale_match += 1
            return False

        started = time.perf_counter()
        with sch.engine_stream_ctx:
            sch.engine.stream.wait_stream(sch.stream)
            greedy, capture, scratch = self._verify(req, draft)
            greedy_ids = greedy.tolist()  # syncs the stream
            accepted = accepted_count(draft, greedy_ids)
            emitted = greedy_ids[: accepted + 1]
            self._commit(req, state, L, len(draft), emitted, capture, scratch)
            reply = self._finish(req, state, L, emitted)

        state.verify_ms = _floor(state.verify_ms, (time.perf_counter() - started) * 1e3)
        state.verify_samples += 1
        state.emit = _ewma(state.emit, float(len(emitted)))
        state.note(len(draft), accepted)
        self.stats.verify_steps += 1
        self.stats.drafted_tokens += len(draft)
        self.stats.accepted_tokens += accepted
        self.stats.emitted_tokens += len(emitted)
        self.stats.note_accepted(accepted)
        self._report(req, reply, len(emitted))
        return True

    def _budget(self, req: Req, state: _SpecState) -> int:
        """Draft length for this step, clamped by every hard bound.

        A full acceptance leaves ``device_len == cached_len + k + 2``, which must fit the
        request's output budget (``_ids_buf`` is sized exactly ``max_device_len``). A pending
        tool-call anchor freeze needs ``cached_len`` to land *exactly* on the anchor
        (``snapshot_toolcall_anchor``), so the draft is cut short of stepping over it.
        """
        k = min(state.k, self.draft_len, req.max_device_len - req.cached_len - 2)
        anchor = req.toolcall_anchor_len
        if (
            anchor is not None
            and req.mamba_last_track_seqlen is None
            and req.cached_len < anchor
        ):
            k = min(k, anchor - req.cached_len - 1)
        if k <= 0:
            self.stats.declined_budget += 1
            return 0
        pool = self.sch.engine.linear_state_pool
        if pool is not None and req.spec_scratch_slot is None and pool.num_free_slots < 1:
            # The pool's free-list is normally EMPTY during steady-state decode: a request
            # holds live + 2 ping-pong and the radix tree owns every donated snapshot, so
            # "no free slot" is the common case, not the exceptional one. Escalate to tier 2
            # (LRU eviction of unlocked tree snapshots) -- but never tier 3
            # (``reserve_mamba_slots``, which spills a session lease): checkpointing an idle
            # conversation to fund an optimisation is not a trade this feature gets to make.
            ensure = getattr(self.sch.cache_manager, "ensure_mamba_slots", None)
            if ensure is not None:
                ensure(1)
            if pool.num_free_slots < 1:
                self.stats.declined_no_slot += 1
                return 0  # the scratch slot is not optional; decline rather than corrupt state
        return k

    # -- forward ---------------------------------------------------------------

    def _verify(self, req: Req, draft: List[int]):
        """Stage the draft, run the extend forward, return (greedy ids, capture, scratch)."""
        sch = self.sch
        pool = sch.engine.linear_state_pool
        L = req.cached_len
        m = len(draft) + 1

        # The forward reads its ids from token_pool, not from the host tensor.
        # Blocking on purpose: <= 64 bytes, and a non_blocking copy from a host buffer this
        # function is about to drop would be a use-after-free race with the forward.
        sch.token_pool[req.table_idx, L + 1 : L + m] = torch.tensor(draft, dtype=torch.int32)

        # A decode step gets this from Scheduler._forward; a verify batch is prefill-phase,
        # so freeze the tool-call anchor here instead, before the state moves.
        if sch.toolcall_anchor_id is not None:
            sch.cache_manager.snapshot_toolcall_anchor([req])

        scratch = None
        if pool is not None:
            scratch = self._ensure_scratch(req, pool)
            pool.copy_from(req.linear_slot_idx, scratch)

        req.device_len = L + m
        live_slot = req.linear_slot_idx
        capture = None
        try:
            if scratch is not None:
                req.linear_slot_idx = scratch
            batch = Batch(reqs=[req], phase="prefill")
            forward_input = sch._prepare_batch(batch)
            batch.logits_indices = torch.arange(m, dtype=torch.int64, device=sch.device)
            if scratch is not None:
                capture = _make_capture(m)
                batch.spec_capture = capture
            batch.input_ids = sch.token_pool[forward_input.input_tuple]
            greedy = sch.engine.spec_verify_forward(batch)
        finally:
            req.linear_slot_idx = live_slot
        return greedy, capture, scratch

    @staticmethod
    def _ensure_scratch(req: Req, pool) -> int:
        if req.spec_scratch_slot is None:
            req.spec_scratch_slot = pool.alloc(1)[0]
        return req.spec_scratch_slot

    # -- commit ----------------------------------------------------------------

    def _commit(self, req, state, L: int, drafted: int, emitted: List[int], capture, scratch) -> None:
        """Adopt the accepted tokens and undo everything the rejected ones did."""
        sch = self.sch
        alloc_len = L + drafted + 1
        req.append_host(torch.tensor(emitted, dtype=req.input_ids.dtype))
        # tokens[L .. L+accepted] were forwarded and are now committed; the bonus token at
        # L+accepted+1 is the next step's single unforwarded token, exactly the shape an
        # ordinary decode step leaves behind.
        req.cached_len = L + len(emitted)
        req.device_len = req.cached_len + 1
        assert req.device_len == int(req.input_ids.numel())
        state.tokens.extend(emitted)

        # The bonus token has no device copy yet (the drafted ones were staged pre-forward).
        sch.token_pool[req.table_idx, req.cached_len] = emitted[-1]

        # KV: rejected positions keep their pages only up to page_ceil(cached_len); the rest
        # must go back, or the next step re-allocates the same positions and they leak.
        sch.cache_manager.free_spec_tail(req, keep_len=req.cached_len, alloc_len=alloc_len)

        if capture is not None:
            pool = sch.engine.linear_state_pool
            if self._check_commit:
                self._verify_commit(capture, pool, req.linear_slot_idx, scratch)
            capture.commit(pool, req.linear_slot_idx, scratch, req.cached_len - L)

    def _verify_commit(self, capture, pool, live: int, scratch: int) -> None:
        """FREETOKEN_SPEC_CHECK_COMMIT: does the commit replay reproduce the forward?

        Debug-only and destructive of nothing: it replays all m positions into a spare slot
        and compares with what the verify forward left in the scratch slot. A greedy-diff
        against the non-speculative arm cannot separate a commit bug from the extend-path
        kernels simply being a different reduction order; this can.
        """
        # The spare slot is held for the life of the process, not per request: this path is
        # env-gated debug, and re-allocating it per verify step would evict a tree snapshot
        # every time.
        if self._spare_slot is None:
            if pool.num_free_slots < 1:
                ensure = getattr(self.sch.cache_manager, "ensure_mamba_slots", None)
                if ensure is not None:
                    ensure(1)
            if pool.num_free_slots < 1:
                return
            self._spare_slot = pool.alloc(1)[0]
        rec, conv = capture.replay_error(pool, live, scratch, self._spare_slot)
        self._check_commit -= 1
        self.commit_error = (max(self.commit_error[0], rec), max(self.commit_error[1], conv))
        logger.info_rank0(
            f"spec commit self-check: recurrent |d|max={rec:.3e} conv |d|max={conv:.3e} "
            f"(m={capture.num_tokens})"
        )

    # -- emit ------------------------------------------------------------------

    def _finish(self, req: Req, state: _SpecState, L: int, emitted: List[int]) -> List[DetokenizeMsg]:
        """One DetokenizeMsg per emitted token, truncated at the first EOS / stop string."""
        sch = self.sch
        reply: List[DetokenizeMsg] = []
        buf = req._ids_buf
        stopped = False
        for e, token in enumerate(emitted):
            pos_end = L + 1 + e + 1  # input_ids length if generation ends after this token
            req.input_ids = buf[:pos_end]
            hit_length = pos_end >= req.max_device_len
            hit_eos = not req.sampling_params.ignore_eos and token in sch.eos_token_ids
            matched_stop = (
                sch._match_stop_str(req)
                if not hit_eos and req.sampling_params.stop_strs
                else None
            )
            finished = hit_length or hit_eos or matched_stop is not None
            if (
                token == sch.toolcall_anchor_id
                and req.toolcall_anchor_len is None
                and not finished
            ):
                req.toolcall_anchor_len = pos_end
            reply.append(
                DetokenizeMsg(
                    uid=req.uid,
                    next_token=token,
                    finished=finished,
                    finish_reason=(
                        ("stop" if (hit_eos or matched_stop is not None) else "length")
                        if finished
                        else None
                    ),
                    matched_stop=matched_stop,
                    stop_strs=req.sampling_params.stop_strs or None,
                )
            )
            if finished:
                stopped = True
                break

        pos_end = int(req.input_ids.numel())
        if stopped:
            # Drop the tokens after the stop from the request and from the drafter, and
            # rewind the lengths to the ordinary finish shape (cached_len == len - 1) so the
            # prefix-cache insert and the page free below cover exactly what survived.
            del state.tokens[pos_end:]
            alloc_len = req.cached_len
            req.cached_len = pos_end - 1
            req.device_len = pos_end
            sch.cache_manager.free_spec_tail(
                req, keep_len=req.cached_len, alloc_len=alloc_len
            )
            sch.decode_manager.remove_req(req)
            with sch.cache_manager.lazy_free_region():
                if req.session_id is not None:
                    sch._free_req_resources(req, retain_session=True)
                else:
                    sch._free_req_resources(req)
            sch.finished_reqs = set(sch.finished_reqs) | {req}
            self._state = None
        return reply

    def _report(self, req: Req, reply: List[DetokenizeMsg], emitted: int) -> None:
        sch = self.sch
        used, total = sch._kv_usage_pages()
        mamba_slots = sch._mamba_slot_usage()
        swa_tokens = sch._swa_token_usage()
        if reply:
            mem = sch._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for msg in reply:
                msg.kv_used_pages = used
                msg.kv_total_pages = total
                msg.mamba_used_slots = mamba_used
                msg.mamba_total_slots = mamba_total
                msg.swa_used_tokens = swa_used
                msg.swa_total_tokens = swa_total
                msg.gpu_mem_bytes = mem
        sch.status_reporter.report_batch(
            Batch(reqs=[req], phase="decode"),
            running_reqs=len(sch.decode_manager.running_reqs),
            queue_reqs=len(sch.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=sch.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
            generated_tokens=emitted,
        )
        sch.send_result(reply)


def _make_capture(num_tokens: int):
    from freetoken.models.nemotron_h.spec_scan import SpecScanCapture

    return SpecScanCapture(num_tokens)


__all__ = [
    "NgramDrafter",
    "SpecNgramDecoder",
    "SpecStats",
    "accepted_count",
]
