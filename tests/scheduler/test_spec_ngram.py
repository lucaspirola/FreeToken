"""Prompt-lookup (n-gram) speculative decoding: drafter, acceptance, and rollback.

The verify step is the part with no natural test seam on real hardware -- what it does is
mutate a request's lengths, its host token list, the device token pool, the KV page
allocation and the recurrent state, all from one forward's argmax. So the scheduler
collaborators are faked here and the *arithmetic* is the thing under test: after a step with
``j`` of ``k`` drafts accepted, exactly ``j + 1`` tokens are emitted, ``cached_len`` advances
by exactly ``j + 1``, ``device_len`` is one past it, the rejected pages are handed back, and
nothing rejected is ever visible to the host token list (which is what the prefix cache
inserts from).
"""

from __future__ import annotations

import contextlib
from typing import List, NamedTuple

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler import spec_ngram
from freetoken.scheduler.spec_ngram import (
    NgramDrafter,
    SpecNgramDecoder,
    _SpecState,
    accepted_count,
)


# --------------------------------------------------------------------------- drafter


def test_drafter_holds_back_the_query_ngram():
    """The n-gram ending at the cursor is the query; indexing it would self-match."""
    d = NgramDrafter(3)
    tokens = [1, 2, 3, 4]
    d.observe(tokens)
    # only [1,2,3] ends strictly before the last position
    assert set(d.index) == {(1, 2, 3)}
    assert d.draft(tokens, 4) == []  # (2,3,4) was never indexed


def test_drafter_returns_the_most_recent_continuation():
    d = NgramDrafter(3)
    tokens = [1, 2, 3, 9, 9, 1, 2, 3]
    d.observe(tokens)
    # (1,2,3) at 0 continues with 9,9,1,...; the occurrence at 5 is the query itself.
    assert d.draft(tokens, 4) == [9, 9, 1, 2]


def test_drafter_truncates_the_draft_at_the_end_of_the_stream():
    d = NgramDrafter(2)
    tokens = [7, 8, 5, 7, 8]
    d.observe(tokens)
    assert d.draft(tokens, 8) == [5, 7, 8]  # only three tokens follow the hit


def test_drafter_is_incremental_and_matches_a_full_rebuild():
    tokens = [(i * 7919) % 23 for i in range(400)]
    incremental = NgramDrafter(4)
    for end in range(1, len(tokens) + 1):
        incremental.observe(tokens[:end])
    whole = NgramDrafter(4)
    whole.observe(tokens)
    assert incremental.index == whole.index


def test_drafter_declines_short_streams_and_zero_k():
    d = NgramDrafter(8)
    d.observe([1, 2, 3])
    assert d.draft([1, 2, 3], 4) == []
    assert not d.has_match([1, 2, 3])
    d2 = NgramDrafter(2)
    d2.observe([1, 2, 1, 2])
    assert d2.has_match([1, 2, 1, 2])
    assert d2.draft([1, 2, 1, 2], 0) == []


# --------------------------------------------------------------------------- acceptance


@pytest.mark.parametrize(
    "draft, greedy, expected",
    [
        ([5, 6, 7], [5, 6, 7, 8], 3),  # full accept -> emits 4 tokens
        ([5, 6, 7], [5, 9, 7, 8], 1),  # first disagreement stops it
        ([5, 6, 7], [9, 6, 7, 8], 0),  # nothing accepted -> still emits the bonus token
        ([], [4], 0),
    ],
)
def test_accepted_count(draft, greedy, expected):
    assert accepted_count(draft, greedy) == expected


def test_the_two_sides_of_the_ratio_use_different_estimators():
    """Verify: a floor (few samples, the first pays Triton autotune). Decode: an EWMA
    (hundreds of samples, and the loop-gap minimum is far below a real decode step -- a
    floor there gated out 264 of 278 copy-class drafts)."""
    assert spec_ngram._floor(None, 118.0) == 118.0
    assert spec_ngram._floor(118.0, 27.0) == 27.0
    assert spec_ngram._floor(27.0, 118.0) == 27.0
    assert spec_ngram._ewma(None, 7.4) == 7.4
    assert 6.0 < spec_ngram._ewma(7.4, 2.0) < 7.4      # one fast loop barely moves it


def test_adaptive_draft_length():
    state = _SpecState(req=None, drafter=NgramDrafter(8), max_k=8, adaptive=True)
    assert state.k == 8
    state.note(drafted=8, accepted=0)
    assert state.k == 4
    state.note(drafted=4, accepted=0)
    assert state.k == 2
    state.note(drafted=2, accepted=1)  # partial: neither halve nor restore
    assert state.k == 2
    state.note(drafted=2, accepted=2)  # full accept restores
    assert state.k == 8


def test_adaptive_can_be_switched_off():
    state = _SpecState(req=None, drafter=NgramDrafter(8), max_k=8, adaptive=False)
    state.note(drafted=8, accepted=0)
    assert state.k == 8


# --------------------------------------------------------------------------- fakes


class _FakeForwardInput(NamedTuple):
    batch: object
    input_tuple: tuple


class _FakePool:
    """LinearStatePool stand-in: only the calls a verify step makes."""

    state_layout = "mamba2"

    def __init__(self, slots: int = 3) -> None:
        # 4.. so a scratch slot is never confused with the live slot (3) the fakes use.
        self._free = list(range(4, 4 + slots))
        self.copies: List[tuple] = []

    @property
    def num_free_slots(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> List[int]:
        return [self._free.pop() for _ in range(n)]

    def free(self, slots) -> None:
        self._free.extend(slots)

    def copy_from(self, src: int, dst: int) -> None:
        self.copies.append((src, dst))


class _FakeCacheManager:
    is_hybrid = True
    is_swa = False

    def __init__(self) -> None:
        self.tail_frees: List[tuple] = []
        self.anchor_calls = 0
        self.ensure_calls = 0
        self.evictable = 0
        self.freed_reqs: List[Req] = []

    def free_spec_tail(self, req, keep_len, alloc_len) -> None:
        self.tail_frees.append((keep_len, alloc_len))

    def snapshot_toolcall_anchor(self, reqs) -> None:
        self.anchor_calls += 1

    def attach_pool(self, pool) -> None:
        self._pool = pool

    def ensure_mamba_slots(self, n: int) -> None:
        """Tier 2: evict unlocked tree snapshots. The fake tree holds ``self.evictable``."""
        self.ensure_calls += 1
        pool = getattr(self, "_pool", None)
        while pool is not None and pool.num_free_slots < n and self.evictable > 0:
            self.evictable -= 1
            pool.free([100 + self.evictable])

    @contextlib.contextmanager
    def lazy_free_region(self):
        yield


class _FakeCapture:
    instances: List["_FakeCapture"] = []

    def __init__(self, num_tokens: int) -> None:
        self.num_tokens = num_tokens
        self.commits: List[tuple] = []
        _FakeCapture.instances.append(self)

    def commit(self, pool, live, scratch, n) -> None:
        self.commits.append((live, scratch, n))


class _NullStream:
    def wait_stream(self, other) -> None:
        pass


class _FakeEngine:
    def __init__(self, pool, greedy_script: List[List[int]]) -> None:
        self.linear_state_pool = pool
        self.stream = _NullStream()
        self._script = greedy_script
        self.forward_widths: List[int] = []

    def spec_verify_forward(self, batch) -> torch.Tensor:
        assert batch.logits_indices is not None
        self.forward_widths.append(int(batch.logits_indices.numel()))
        return torch.tensor(self._script.pop(0), dtype=torch.int32)


class _FakeDecodeManager:
    def __init__(self, reqs) -> None:
        self.running_reqs = set(reqs)

    def remove_req(self, req) -> None:
        self.running_reqs.discard(req)


class _FakePrefillManager:
    runnable = False
    pending_list: list = []


class _FakeStatusReporter:
    def __init__(self) -> None:
        self.generated: List[int] = []

    def report_batch(self, batch, *, generated_tokens=None, **kw) -> None:
        self.generated.append(generated_tokens)


class _FakeScheduler:
    """The exact surface SpecNgramDecoder touches -- nothing more."""

    def __init__(self, req: Req, greedy_script, *, pool=None, max_len: int = 256) -> None:
        self.cache_manager = _FakeCacheManager()
        self.cache_manager.attach_pool(pool)
        self.engine = _FakeEngine(pool, greedy_script)
        self.decode_manager = _FakeDecodeManager([req])
        self.prefill_manager = _FakePrefillManager()
        self.status_reporter = _FakeStatusReporter()
        self.finished_reqs: set = set()
        self.eos_token_ids = {2}
        self.toolcall_anchor_id = None
        self.device = torch.device("cpu")
        self.stream = _NullStream()
        self.engine_stream_ctx = contextlib.nullcontext()
        self.token_pool = torch.zeros(2, max_len, dtype=torch.int32)
        self.config = type("cfg", (), {"page_size": 1})()
        self.replies: list = []
        self.freed: List[Req] = []
        self.prepared_widths: List[int] = []

    # -- collaborators the decoder calls into ------------------------------

    def _prepare_batch(self, batch):
        req = batch.reqs[0]
        self.prepared_widths.append(req.extend_len)
        rows = torch.full((req.extend_len,), req.table_idx, dtype=torch.int64)
        cols = torch.arange(req.cached_len, req.device_len, dtype=torch.int64)
        return _FakeForwardInput(batch=batch, input_tuple=(rows, cols))

    def _match_stop_str(self, req):
        return None

    def _kv_usage_pages(self):
        return (1, 2)

    def _mamba_slot_usage(self):
        return None

    def _swa_token_usage(self):
        return None

    def _gpu_mem_bytes(self):
        return 0

    def send_result(self, reply):
        self.replies.extend(reply)

    def _free_req_resources(self, req, *, retain_session: bool = False):
        self.freed.append(req)


def _make_req(tokens: List[int], *, output_len: int = 64, table_idx: int = 1) -> Req:
    req = Req(
        input_ids=torch.tensor(tokens, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=len(tokens) - 1,
        output_len=output_len,
        uid=0,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
        cache_handle=object(),
    )
    req.linear_slot_idx = 3
    return req


def _decoder(sch, *, n: int = 3, draft_len: int = 4, adaptive: bool = False):
    dec = SpecNgramDecoder.__new__(SpecNgramDecoder)
    dec.sch = sch
    dec.n = n
    dec.draft_len = draft_len
    dec.adaptive = adaptive
    dec.stats = spec_ngram.SpecStats()
    dec._state = None
    dec._last_peek_at = None
    dec._last_peek_hit = False
    dec._check_commit = 0
    dec._spare_slot = None
    dec.commit_error = (0.0, 0.0)
    dec.enabled = True
    dec.disabled_reason = ""
    return dec


def _seed_token_pool(sch, req):
    sch.token_pool[req.table_idx, : req.device_len] = req.input_ids


# --------------------------------------------------------------------------- verify step


def test_partial_rejection_commits_only_the_accepted_prefix():
    # [1,2,3] recurs, so the 3-gram drafter proposes what followed it last time.
    tokens = [1, 2, 3, 7, 8, 9, 1, 2, 3]
    req = _make_req(tokens)
    L = req.cached_len
    # draft is [7, 8, 9, 1]; the model agrees on 7, 8 then says 55, and the bonus is 55.
    sch = _FakeScheduler(req, [[7, 8, 55, 0, 0]])
    _seed_token_pool(sch, req)
    dec = _decoder(sch)

    assert dec.run_step(req) is True

    assert sch.prepared_widths == [5]  # k + 1 positions in one forward
    assert sch.engine.forward_widths == [5]
    # accepted = 2 -> emits [7, 8, 55]
    emitted = [m.next_token for m in sch.replies]
    assert emitted == [7, 8, 55]
    assert req.input_ids.tolist() == tokens + [7, 8, 55]
    assert req.cached_len == L + 3        # tokens[L .. L+2] were forwarded and kept
    assert req.device_len == req.cached_len + 1
    assert req.device_len == int(req.input_ids.numel())
    # the KV pages of the two rejected positions go back
    assert sch.cache_manager.tail_frees == [(L + 3, L + 5)]
    # the device token pool carries the accepted drafts and the bonus token
    assert sch.token_pool[req.table_idx, L : L + 4].tolist() == [3, 7, 8, 55]
    assert sch.status_reporter.generated == [3]
    assert dec.stats.drafted_tokens == 4 and dec.stats.accepted_tokens == 2


def test_full_acceptance_emits_k_plus_one_tokens():
    tokens = [1, 2, 3, 7, 8, 9, 4, 1, 2, 3]
    req = _make_req(tokens)
    L = req.cached_len
    sch = _FakeScheduler(req, [[7, 8, 9, 4, 61]])
    _seed_token_pool(sch, req)
    dec = _decoder(sch)

    assert dec.run_step(req) is True
    assert [m.next_token for m in sch.replies] == [7, 8, 9, 4, 61]
    assert req.cached_len == L + 5
    assert req.device_len == req.cached_len + 1
    # nothing rejected -> nothing to free
    assert sch.cache_manager.tail_frees == [(L + 5, L + 5)]


def test_total_rejection_still_emits_the_bonus_token():
    tokens = [1, 2, 3, 7, 8, 9, 1, 2, 3]
    req = _make_req(tokens)
    L = req.cached_len
    sch = _FakeScheduler(req, [[42, 0, 0, 0, 0]])
    _seed_token_pool(sch, req)
    dec = _decoder(sch)

    assert dec.run_step(req) is True
    assert [m.next_token for m in sch.replies] == [42]
    assert req.cached_len == L + 1        # exactly what a plain decode step would have done
    assert req.device_len == L + 2
    assert req.input_ids.tolist() == tokens + [42]
    assert sch.cache_manager.tail_frees == [(L + 1, L + 5)]


def test_eos_inside_an_accepted_run_truncates_and_finishes():
    tokens = [1, 2, 3, 7, 2, 9, 1, 2, 3]
    req = _make_req(tokens)
    L = req.cached_len
    # draft [7, 2, 9, 1] fully agreed with; token id 2 is EOS, at emitted index 1.
    sch = _FakeScheduler(req, [[7, 2, 9, 1, 5]])
    _seed_token_pool(sch, req)
    dec = _decoder(sch)

    assert dec.run_step(req) is True
    assert [m.next_token for m in sch.replies] == [7, 2]
    assert sch.replies[-1].finished and sch.replies[-1].finish_reason == "stop"
    assert not sch.replies[0].finished
    # the tokens after the EOS are gone from the request -- the prefix cache inserts
    # input_ids[:cached_len], so they can never be published.
    assert req.input_ids.tolist() == tokens + [7, 2]
    assert req.cached_len == int(req.input_ids.numel()) - 1
    assert req.device_len == int(req.input_ids.numel())
    assert sch.cache_manager.tail_frees[-1] == (req.cached_len, L + 5)
    assert sch.freed == [req] and req not in sch.decode_manager.running_reqs
    assert dec._state is None


def test_draft_is_clamped_by_the_output_budget():
    tokens = [1, 2, 3, 7, 8, 9, 4, 1, 2, 3]
    # only 3 tokens of budget left: a full accept must still fit _ids_buf.
    req = _make_req(tokens, output_len=3)
    sch = _FakeScheduler(req, [[7, 8, 61]])
    _seed_token_pool(sch, req)
    dec = _decoder(sch, draft_len=8)

    assert dec.run_step(req) is True
    assert sch.prepared_widths == [3]  # k clamped to max_device_len - cached_len - 2
    assert int(req.input_ids.numel()) <= req.max_device_len


def test_no_match_declines_without_mutating_anything():
    req = _make_req([1, 2, 3, 4, 5])
    before = (req.cached_len, req.device_len, req.input_ids.tolist())
    sch = _FakeScheduler(req, [])
    _seed_token_pool(sch, req)
    dec = _decoder(sch)

    assert dec.run_step(req) is False
    assert (req.cached_len, req.device_len, req.input_ids.tolist()) == before
    assert sch.replies == [] and sch.cache_manager.tail_frees == []


def test_recurrent_state_is_verified_into_a_scratch_slot(monkeypatch):
    _FakeCapture.instances.clear()
    monkeypatch.setattr(spec_ngram, "_make_capture", _FakeCapture)
    tokens = [1, 2, 3, 7, 8, 9, 1, 2, 3]
    req = _make_req(tokens)
    L = req.cached_len
    pool = _FakePool()
    sch = _FakeScheduler(req, [[7, 8, 55, 0, 0]], pool=pool)
    _seed_token_pool(sch, req)
    dec = _decoder(sch)

    assert dec.run_step(req) is True
    scratch = req.spec_scratch_slot
    assert scratch is not None and scratch != 3
    # the live slot is copied INTO the scratch slot, never advanced by the forward
    assert pool.copies == [(3, scratch)]
    assert req.linear_slot_idx == 3  # restored after the forward
    capture = _FakeCapture.instances[-1]
    assert capture.num_tokens == 5
    # accepted 2 -> commit 3 of the 5 verified positions into the live slot
    assert capture.commits == [(3, scratch, req.cached_len - L)]
    assert capture.commits[0][2] == 3
    # a second step reuses the same scratch slot instead of allocating another
    sch.engine._script.append([7, 8, 9, 1, 5])
    dec.run_step(req)
    assert req.spec_scratch_slot == scratch
    assert pool.num_free_slots == _FakePool().num_free_slots - 1


def test_declines_when_the_state_pool_has_no_free_slot():
    """An empty free-list with nothing evictable declines; it must not corrupt live state."""
    tokens = [1, 2, 3, 7, 8, 9, 1, 2, 3]
    req = _make_req(tokens)
    pool = _FakePool(slots=0)  # free-list empty
    assert pool.num_free_slots == 0
    sch = _FakeScheduler(req, [], pool=pool)
    _seed_token_pool(sch, req)
    dec = _decoder(sch)

    assert dec.run_step(req) is False
    assert req.spec_scratch_slot is None
    assert sch.cache_manager.ensure_calls == 1  # tier 2 was tried
    assert dec.stats.declined_no_slot == 1
    assert pool.copies == []


def test_evicts_a_tree_snapshot_for_the_scratch_slot():
    """The free-list is EMPTY in steady-state decode -- every slot is live or tree-owned --
    so speculation has to escalate to tier-2 eviction or it silently never fires."""
    tokens = [1, 2, 3, 7, 8, 9, 1, 2, 3]
    req = _make_req(tokens)
    pool = _FakePool(slots=0)
    sch = _FakeScheduler(req, [[7, 8, 55, 0, 0]], pool=pool)
    sch.cache_manager.evictable = 2
    _seed_token_pool(sch, req)
    dec = _decoder(sch)

    assert dec.run_step(req) is True
    assert sch.cache_manager.ensure_calls == 1
    assert sch.cache_manager.evictable == 1  # exactly one snapshot evicted
    assert req.spec_scratch_slot is not None
    assert dec.stats.declined_no_slot == 0


def test_draft_never_steps_over_a_pending_toolcall_anchor():
    tokens = [1, 2, 3, 7, 8, 9, 4, 5, 1, 2, 3]
    req = _make_req(tokens)
    req.toolcall_anchor_len = req.cached_len + 2   # the freeze needs cached_len == anchor
    sch = _FakeScheduler(req, [[7, 61]])
    _seed_token_pool(sch, req)
    dec = _decoder(sch, draft_len=8)

    assert dec.run_step(req) is True
    assert sch.prepared_widths == [2]              # k clamped to anchor - cached_len - 1
    assert req.cached_len <= req.toolcall_anchor_len


# --------------------------------------------------------------------------- eligibility


def test_candidate_declines_non_greedy_and_crowded_steps():
    req = _make_req([1, 2, 3, 4, 5])
    sch = _FakeScheduler(req, [])
    dec = _decoder(sch)
    assert dec.candidate() is req

    req.sampling_params = SamplingParams(temperature=0.7)
    assert dec.candidate() is None
    req.sampling_params = SamplingParams(temperature=0.0)

    other = _make_req([9, 9, 9], table_idx=0)
    sch.decode_manager.running_reqs.add(other)
    assert dec.candidate() is None
    sch.decode_manager.running_reqs.discard(other)

    sch.prefill_manager.runnable = True
    assert dec.candidate() is None
    sch.prefill_manager.runnable = False

    req.aborted = True
    assert dec.candidate() is None
    req.aborted = False

    req.mm_embeds = torch.zeros(1)
    assert dec.candidate() is None


def test_break_even_gate_closes_when_a_verify_step_stops_paying():
    """The verify/decode cost ratio is a function of context length, so the gate is measured
    rather than thresholded: at 131K a verify step costs ~10x a decode step and k + 1 = 9
    cannot reach it."""
    tokens = [1, 2, 3, 7, 8, 9, 1, 2, 3]
    req = _make_req(tokens)
    sch = _FakeScheduler(req, [])
    dec = _decoder(sch)

    state = dec._state_for(req)
    state.decode_ms, state.verify_ms, state.emit = 11.7, 118.0, 6.0  # measured at 131K
    state.verify_samples = 1
    assert dec._pays_off(state) is True   # one sample is not evidence
    state.verify_samples = spec_ngram._GATE_MIN_SAMPLES
    assert dec._pays_off(state) is False
    dec._last_peek_at = None       # don't let the test's own call gap become a sample
    assert dec.peek() is None
    assert dec.stats.declined_uneconomic == 1

    state.decode_ms, state.verify_ms = 6.9, 30.2   # measured at short context
    assert dec._pays_off(state) is True
    dec._last_peek_at = None
    assert dec.peek() is req


def test_a_closed_gate_still_re_probes():
    """A closed gate has to stay falsifiable: acceptance and context both move."""
    tokens = [1, 2, 3, 7, 8, 9, 1, 2, 3]
    req = _make_req(tokens)
    sch = _FakeScheduler(req, [])
    dec = _decoder(sch)
    state = dec._state_for(req)
    state.decode_ms, state.verify_ms, state.emit = 10.0, 100.0, 2.0
    state.verify_samples = spec_ngram._GATE_MIN_SAMPLES
    hits = 0
    for _ in range(spec_ngram._REPROBE_EVERY):
        dec._last_peek_at = None
        hits += dec.peek() is not None
    assert hits == 1
    assert dec.stats.declined_uneconomic == spec_ngram._REPROBE_EVERY


def test_peek_is_the_hysteresis_gate():
    tokens = [1, 2, 3, 7, 8, 9, 1, 2, 3]
    req = _make_req(tokens)
    sch = _FakeScheduler(req, [])
    dec = _decoder(sch)
    assert dec.peek() is req

    fresh = _make_req([4, 5, 6, 7, 8])
    sch2 = _FakeScheduler(fresh, [])
    dec2 = _decoder(sch2)
    assert dec2.peek() is None
    assert dec2.stats.plain_peeks == 1
