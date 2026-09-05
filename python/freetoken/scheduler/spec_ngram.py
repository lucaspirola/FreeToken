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

# Seeded gate -- FREETOKEN_SPEC_GATE_SEED=1, OFF by default until it has a GPU measurement.
#
# _GATE_MIN_SAMPLES full-width verify steps is a real price at long context: 2 x ~82 ms at 131K
# against a 10.4 ms decode step, which on the 79-token needle generation IS the whole -10 %
# regression (misc_tickets_2026-09-05.md §4). A verify step's wall clock is well described by
# ``t(m) = a + b*m`` -- a width-independent term (weights over PCIe, the eager launch path) plus
# the extend attention, which reads the whole KV history once per query token. So a couple of
# NARROW probes can price the full-width step without ever running one: at 131K, m = 2 and m = 4
# cost ~88 ms together against ~163 ms for two full-width probes, and they measure the same thing.
#
# TWO probes, not one, and that is the whole design. With a single point ``a`` is unidentifiable,
# and the obvious closed form -- scale ``t`` linearly through the origin -- over-estimates the
# full-width step by ``(m_full/m_seed - 1) * a``. At 131K ``a`` is small next to ``b*m`` so that is
# survivable; at short context ``a`` IS the step (b ~ 0.85 ms/token against a ~28 ms fixed term),
# the estimate comes out ~4x high, and the gate closes on exactly the copy-class traffic the
# feature exists for -- the failure that turned +11 % into -0.3 % once already. Two points fit both
# terms, cost less than one full-width probe, and need no compensating margin.
_SEED_WIDTHS = (1, 3)   # draft lengths k of the probe steps, i.e. m = 2 then m = 4


def _fit_verify_ms(samples: Sequence[Tuple[int, float]], m_full: int) -> float | None:
    """Price a full-width verify step from narrow probes, fitting ``t(m) = a + b*m``.

    Least squares over the probes with both terms clamped non-negative (a fit that says a wider
    extend is cheaper is noise, not a measurement) and the result floored at the largest sample --
    extrapolating below a cost already paid is never right. Returns None when the probes do not
    span at least two widths, in which case the caller falls back to timing real full-width steps
    and the gate behaves exactly as it does with seeding off.
    """
    if len(samples) < 2:
        return None
    ms = [float(m) for m, _ in samples]
    ts = [t for _, t in samples]
    mean_m, mean_t = sum(ms) / len(ms), sum(ts) / len(ts)
    var = sum((m - mean_m) ** 2 for m in ms)
    if var <= 0.0:  # every probe landed on the same width: the slope is unidentifiable
        return None
    b = sum((m - mean_m) * (t - mean_t) for m, t in zip(ms, ts)) / var
    if b < 0.0:
        b, a = 0.0, mean_t
    else:
        a = mean_t - b * mean_m
        if a < 0.0:  # refit through the origin rather than credit the step a negative fixed cost
            a = 0.0
            b = sum(m * t for m, t in zip(ms, ts)) / sum(m * m for m in ms)
    return max(a + b * float(m_full), max(ts))


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

    __slots__ = ("n", "index", "prefix", "cursor")

    def __init__(self, n: int) -> None:
        assert n >= 1
        self.n = n
        self.index: Dict[Tuple[int, ...], int] = {}
        # Hashes of the (n-1)-prefixes of every indexed n-gram: the *post-drain* predictor.
        # A burst that begins at position p is only visible in the n-gram key once token p
        # is known, and the scheduler only knows it after the drain -- but the key's first
        # n-1 entries are already in the pre-drain token list. Membership here therefore
        # answers "could the drafter fire once this step's token lands?", which is a strict
        # superset of "will it": no burst entry can be missed, and the exact test still runs
        # post-drain in ``draft``. Hashes, not tuples, so a 131K prompt costs ~8 MB and one
        # C-level hash per token rather than a second tuple table.
        self.prefix: set[int] = set()
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
        pre = self.prefix
        while i + n <= limit:
            key = tuple(tokens[i : i + n])
            idx[key] = i + n
            pre.add(hash(key[:-1]))
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

    def could_match(self, tokens: Sequence[int]) -> bool:
        """Would some continuation of ``tokens`` be draftable once one more token lands?

        The pre-drain predictor. ``tokens`` is one token short of the list the verify step
        will actually query, so the key it will use is ``(tokens[-(n-1):], next_token)``;
        this asks whether ANY n-gram with that (n-1)-prefix was ever indexed. False here
        means the drafter provably cannot fire on the next step, which is what keeps the
        drain off the common path; True costs a drain and an exact re-test.
        """
        n1 = self.n - 1
        pos = len(tokens)
        if pos < n1:
            return False
        return hash(tuple(tokens[pos - n1 : pos])) in self.prefix

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
    # Seeded gate: narrow probe steps run, and requests whose gate was primed from them. A seeded
    # request declines after ``probes`` narrow steps instead of _GATE_MIN_SAMPLES full-width ones,
    # so these two next to ``declined.uneconomic`` are what the A/B reads.
    seed_probe_steps: int = 0
    seed_fits: int = 0
    # Wall-clock breakdown of a verify step, in ms, summed over ``verify_steps``. A verify
    # step is ~7x a decode step end to end against a ~30 ms forward, and "the rest" was one
    # undivided number until this existed -- so it is reported, not inferred: draft+stage,
    # batch preparation, the (eager) forward launch, the argmax sync, the state commit and
    # the emission. ``drain_ms`` is charged by the scheduler, which owns the drain.
    t_draft: float = 0.0
    t_prep: float = 0.0
    t_launch: float = 0.0
    t_sync: float = 0.0
    t_commit: float = 0.0
    t_finish: float = 0.0
    t_total: float = 0.0
    t_drain: float = 0.0
    drains: int = 0
    # GPU-side (CUDA event) time of the forward and of the commit, so the host-launch share
    # of an eager extend forward is a measurement rather than a subtraction.
    g_forward: float = 0.0
    g_commit: float = 0.0
    g_commit_n: int = 0
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
    def cost_ms(self) -> dict:
        """Mean per-verify-step wall clock, by phase."""
        v = self.verify_steps or 1
        d = self.drains or 1
        return {
            "draft": round(self.t_draft / v, 3),
            "prep": round(self.t_prep / v, 3),
            "launch": round(self.t_launch / v, 3),
            "sync": round(self.t_sync / v, 3),
            "commit": round(self.t_commit / v, 3),
            "finish": round(self.t_finish / v, 3),
            "total": round(self.t_total / v, 3),
            "gpu_forward": round(self.g_forward / v, 3),
            "gpu_commit": round(self.g_commit / max(self.g_commit_n, 1), 3),
            "drain": round(self.t_drain / d, 3),
            "drains": self.drains,
        }

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
            "seed": {"probes": self.seed_probe_steps, "fits": self.seed_fits},
            "cost_ms": self.cost_ms,
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
    # Seeded gate (_SEED_WIDTHS). ``seeding`` is set at construction from the decoder's flag and
    # cleared once the probes are in; ``verify_seeded`` says the current ``verify_ms`` is an
    # extrapolation, so the first real full-width step must REPLACE it rather than _floor()
    # against it.
    seeding: bool = False
    seed_probes: List[Tuple[int, float]] = field(default_factory=list)
    verify_seeded: bool = False

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


@dataclass
class _VerifyBuffers:
    """Persistent device buffers for the verify batch's fixed geometry."""

    ar32: torch.Tensor
    ar64: torch.Tensor
    pos32: torch.Tensor
    pos64: torch.Tensor
    rows: torch.Tensor
    table_idx: int = -1


# --------------------------------------------------------------------------- the decoder


class SpecNgramDecoder:
    """Drives speculative verify steps for a single greedy request.

    v1 scope: exactly one running request, greedy sampling, hybrid (mamba2) or stateless
    attention, no SWA, no multimodal, no hidden-state probe. Anything else falls through to
    the ordinary decode path, which is why this class never has to be correct for a shape it
    does not recognise -- it declines.
    """

    # Class-level defaults so an instance built without __init__ (the CPU tests drive this
    # class against a scheduler-shaped stub) still has the full attribute surface.
    post_drain = True
    fused_commit = True
    fast_prep = True
    gate_seed = False
    _prep = None
    _fla_cache: dict | None = None
    _ev = None
    _ev_pending = False

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
        # Three optimisations, each independently switchable so one GPU session can measure
        # the before and the after (see benchmarks/probe_spec_ngram_impl.py --variants).
        #   post_drain    engagement decided from the post-drain token list (§ NgramDrafter
        #                 .could_match): catches a burst on the step it starts, not one later
        #   fused_commit  one SSD scan for all layers instead of one per layer
        #   fast_prep     the verify batch built from the request's own fixed shape instead
        #                 of through the general _prepare_batch
        self.post_drain = os.environ.get("FREETOKEN_SPEC_POST_DRAIN", "1") == "1"
        self.fused_commit = os.environ.get("FREETOKEN_SPEC_FUSED_COMMIT", "1") == "1"
        self.fast_prep = os.environ.get("FREETOKEN_SPEC_FAST_PREP", "1") == "1"
        #   gate_seed     price the break-even gate from two NARROW verify steps instead of
        #                 _GATE_MIN_SAMPLES full-width ones (see _SEED_WIDTHS). OFF by default:
        #                 it changes what the gate decides, and it has no GPU measurement yet.
        self.gate_seed = os.environ.get("FREETOKEN_SPEC_GATE_SEED", "0") == "1"
        self._prep: _VerifyBuffers | None = None
        self._fla_cache: dict = {}
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

    def peek(self, *, stale: bool = True) -> Req | None:
        """The candidate the drafter could fire on this iteration, or None.

        With ``stale`` (the overlapped loop, where the previous step's token has not been
        drained yet) the token list is one token short of the one the verify step will
        query, so the test is :meth:`NgramDrafter.could_match` -- "is there an indexed
        n-gram whose first n-1 tokens are the ones I already have?". That is a strict
        superset of the exact test, so a burst is entered on the step it begins rather than
        the step after (the old exact-on-stale-tokens test cost a factor of ~4 in draft rate,
        write-up §7); the exact test then runs post-drain inside :meth:`run_step`, which
        declines and falls back to the ordinary path when the prediction does not hold.

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
        hit = (
            state.drafter.could_match(state.tokens)
            if (stale and self.post_drain)
            else state.drafter.has_match(state.tokens)
        )
        if state.k <= 0 or not hit:
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

        With ``FREETOKEN_SPEC_GATE_SEED=1`` the "measurement" the first drafts pay for is two
        narrow probes fitted to ``t(m) = a + b*m`` (:func:`_fit_verify_ms`) rather than
        ``_GATE_MIN_SAMPLES`` full-width steps. The fit targets the same quantity, so the test
        below and its margin are unchanged; only the price of learning it moves.
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
                # Per request, like every other term of the gate: the verify/decode ratio is a
                # function of THIS request's context length, so the probes must be too. A draft
                # length that does not leave room for two distinct narrow widths cannot be seeded.
                seeding=self.gate_seed and self.draft_len > max(_SEED_WIDTHS),
            )
            self._state = state
        return state

    # ------------------------------------------------------------------ the step

    def run_step(self, req: Req) -> bool:
        """Run one verify step. Returns False if the request was not in a draftable shape,
        in which case nothing was mutated and the caller should take the ordinary path."""
        sch = self.sch
        t_draft0 = time.perf_counter()
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

        st = self.stats
        started = t = time.perf_counter()
        st.t_draft += (t - t_draft0) * 1e3
        with sch.engine_stream_ctx:
            sch.engine.stream.wait_stream(sch.stream)
            ev = self._events()
            greedy, capture, scratch, prep_ms = self._verify(req, draft, ev)
            st.t_prep += prep_ms
            t, prev = time.perf_counter(), t
            st.t_launch += (t - prev) * 1e3 - prep_ms
            greedy_ids = greedy.tolist()  # syncs the stream
            t, prev = time.perf_counter(), t
            st.t_sync += (t - prev) * 1e3
            if ev is not None:
                st.g_forward += ev[0].elapsed_time(ev[1])
            accepted = accepted_count(draft, greedy_ids)
            emitted = greedy_ids[: accepted + 1]
            if ev is not None:
                ev[2].record()
            self._commit(req, state, L, len(draft), emitted, capture, scratch)
            if ev is not None:
                ev[3].record()
                self._ev_pending = True
            t, prev = time.perf_counter(), t
            st.t_commit += (t - prev) * 1e3
            reply = self._finish(req, state, L, emitted)
            t, prev = time.perf_counter(), t
            st.t_finish += (t - prev) * 1e3

        elapsed = (time.perf_counter() - started) * 1e3
        st.t_total += elapsed
        if state.seeding:
            # A narrow probe. Its wall clock prices the full-width step (below), but its emitted
            # count describes a k=1/k=3 draft, so `emit` may only see it when the probe says
            # something a full-width step would have said too.
            #
            # Under greedy decoding it sometimes does, exactly: the drafted tokens are a prefix of
            # the same continuation whatever k is, and the model's greedy token at each position
            # does not depend on how many more were drafted -- so a probe that REJECTS at position
            # j has found the divergence a full-width draft would have found at the same j, and its
            # accepted count IS the full-width accepted count. A probe that accepts everything it
            # drafted has only run out of width and bounds nothing, so it is dropped rather than
            # allowed to drag the EWMA down towards closing the gate it exists to inform.
            state.seed_probes.append((len(draft) + 1, elapsed))
            st.seed_probe_steps += 1
            if accepted < len(draft):
                state.emit = _ewma(state.emit, float(len(emitted)))
                state.note(self.draft_len, accepted)   # a full-width-equivalent rejection
            if len(state.seed_probes) >= len(_SEED_WIDTHS):
                state.seeding = False
                fit = _fit_verify_ms(state.seed_probes, self.draft_len + 1)
                if fit is not None:
                    state.verify_ms = fit
                    state.verify_seeded = True
                    state.verify_samples = max(state.verify_samples, _GATE_MIN_SAMPLES)
                    st.seed_fits += 1
        else:
            if state.verify_seeded:
                # The first real full-width step REPLACES the extrapolation. A _floor() against a
                # seed that came out high would pin verify_ms there for the life of the request
                # and starve the gate of the samples that reopen it -- the same trap the EWMA fell
                # into on the first, autotuning verify step.
                state.verify_ms, state.verify_seeded = elapsed, False
            else:
                state.verify_ms = _floor(state.verify_ms, elapsed)
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

    def _events(self):
        """Four reusable CUDA events: forward start/end, commit start/end.

        The commit pair is read on the NEXT verify step, and only if it has completed, so
        the breakdown costs no synchronisation of its own (the forward pair is already
        covered by the argmax sync that the step pays anyway). ``g_commit`` is averaged over
        the samples actually taken, not over every verify step.
        """
        if self._ev is None:
            if getattr(self.sch.device, "type", None) != "cuda":
                return None
            self._ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            self._ev_pending = False
        if self._ev_pending:
            # Inside a tight burst the previous commit can still be queued. Reading an
            # incomplete event raises, and this is a statistic -- drop the sample rather
            # than synchronise the step that is about to run.
            if self._ev[3].query():
                self.stats.g_commit += self._ev[2].elapsed_time(self._ev[3])
                self.stats.g_commit_n += 1
            self._ev_pending = False
        return self._ev

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
        if state.seeding and len(state.seed_probes) < len(_SEED_WIDTHS):
            # Price the step with a narrow probe before ever paying for a full-width one. The
            # hard bounds above still win -- a clamped probe is a valid sample, it just may land
            # on a width a previous probe already took, which _fit_verify_ms then declines.
            k = min(k, _SEED_WIDTHS[len(state.seed_probes)])
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

    def _verify(self, req: Req, draft: List[int], ev=None):
        """Stage the draft, run the extend forward.

        Returns ``(greedy ids, capture, scratch, prep_ms)``."""
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
            t0 = time.perf_counter()
            if self.fast_prep and not sch.config.kv_grow_step_tokens:
                input_tuple = self._prepare_verify(batch, req, L, m)
            else:
                input_tuple = sch._prepare_batch(batch).input_tuple
            batch.logits_indices = self._rows(m)
            if scratch is not None:
                capture = _make_capture(m, fused=self.fused_commit)
                batch.spec_capture = capture
            batch.input_ids = sch.token_pool[input_tuple]
            prep_ms = (time.perf_counter() - t0) * 1e3
            if ev is not None:
                ev[0].record()
            greedy = sch.engine.spec_verify_forward(batch)
            if ev is not None:
                ev[1].record()
        finally:
            req.linear_slot_idx = live_slot
        return greedy, capture, scratch, prep_ms

    # -- batch preparation ------------------------------------------------------

    def _prepare_verify(self, batch: Batch, req: Req, L: int, m: int):
        """The subset of ``Scheduler._prepare_batch`` a verify batch actually needs.

        A verify batch is the most predictable batch this engine ever builds: exactly one
        request, prefill phase, extend length ``k + 1``, never multimodal, never chunked,
        never SWA (the decoder refuses all of those), and the same shape on every step of a
        burst. Almost everything ``_prepare_batch`` does for it is either a branch that
        cannot apply or a pinned staging tensor rebuilt at an identical shape -- including a
        full ``Sampler.prepare``, whose per-request parameter rows this path never reads
        (the verify forward is greedy by construction and runs no sampler at all).

        What is left is the page allocation, the positions, the page-table gather and the
        two metadata builders. The positions and the per-token request row come off
        persistent device buffers, so the whole preparation is four kernel launches and no
        host->device staging.
        """
        sch = self.sch
        batch.padded_reqs = batch.reqs
        sch._forward_iter += 1
        sch.cache_manager.allocate_paged(batch.reqs)
        buf = self._buffers(m)
        torch.add(buf.ar32[:m], L, out=buf.pos32[:m])
        torch.add(buf.ar64[:m], L, out=buf.pos64[:m])
        if buf.table_idx != req.table_idx:
            buf.rows.fill_(req.table_idx)
            buf.table_idx = req.table_idx
        batch.positions = buf.pos32[:m]
        input_tuple = (buf.rows[:m], buf.pos64[:m])
        batch.out_loc = sch.engine.page_table[input_tuple]
        if sch.engine.linear_state_pool is not None:
            batch.fla_metadata = self._fla_metadata(batch, req, m)
        sch.engine.attn_backend.prepare_metadata(batch)
        return input_tuple

    def _buffers(self, m: int) -> "_VerifyBuffers":
        buf = self._prep
        if buf is None or buf.ar32.numel() < m:
            size = max(m, self.draft_len + 1)
            dev = self.sch.device
            buf = _VerifyBuffers(
                ar32=torch.arange(size, dtype=torch.int32, device=dev),
                ar64=torch.arange(size, dtype=torch.int64, device=dev),
                pos32=torch.empty(size, dtype=torch.int32, device=dev),
                pos64=torch.empty(size, dtype=torch.int64, device=dev),
                rows=torch.empty(size, dtype=torch.int64, device=dev),
            )
            self._prep = buf
        return buf

    def _rows(self, m: int) -> torch.Tensor:
        """``arange(m)`` -- the verify batch keeps every logits row."""
        return self._buffers(m).ar64[:m]

    def _fla_metadata(self, batch: Batch, req: Req, m: int):
        """Cached recurrent metadata for a one-request extend of ``m`` tokens.

        Every field is a function of ``(m, state slot)`` here: ``cu_seqlens`` is ``[0, m]``,
        ``cache_indices`` is the scratch slot, ``has_initial_state`` is True (a verify step
        only ever runs on a request that already has state), and the mid-chunk snapshot
        metadata is None because ``m`` never reaches a chunk boundary. That last one is an
        invariant, not an observation, so it is asserted.
        """
        from freetoken.attention.linear import build_fla_metadata

        pool = self.sch.engine.linear_state_pool
        assert m <= pool.track_chunk_size, (
            f"a verify extend of {m} tokens can cross a {pool.track_chunk_size}-token "
            "snapshot boundary; the cached metadata would drop the snapshot"
        )
        # (m, slot) is the whole key: every tensor in the result is built from those two
        # integers, so a pool rebuild -- which moves the state tensors but not the slot
        # numbering -- does not invalidate it.
        key = (m, req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx)
        if self._fla_cache is None:
            self._fla_cache = {}
        meta = self._fla_cache.get(key)
        if meta is None:
            meta = build_fla_metadata(batch, self.sch.device)
            assert meta.track_dst is None, "verify batch unexpectedly wants a state snapshot"
            if len(self._fla_cache) > 64:
                self._fla_cache.clear()
            self._fla_cache[key] = meta
        return meta

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


def _make_capture(num_tokens: int, *, fused: bool = True):
    from freetoken.models.nemotron_h.spec_scan import SpecScanCapture

    return SpecScanCapture(num_tokens, fused=fused)


__all__ = [
    "NgramDrafter",
    "SpecNgramDecoder",
    "SpecStats",
    "accepted_count",
]
