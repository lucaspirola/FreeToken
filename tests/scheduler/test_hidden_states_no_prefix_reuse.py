"""A hidden-state probe must recompute every prompt token, cached prefix or not.

Switchyard's prefill router mean-pools the residual stream of *each* prompt position, so
a request that opted into the export cannot be served partly from the radix tree: the
positions covered by a prefix hit would never enter a forward and would be missing from
the artifact (a short file, or -- worse -- a plausible one built from fewer tokens).

``Req.no_prefix_cache`` is the knob, read in ``CacheManager.match_req`` next to the
multimodal bypass; ``submit_generation`` sets it for every request carrying a
``HiddenStateSpec``. These tests drive the real radix cache and the real ``PrefillAdder``
on CPU.
"""

from __future__ import annotations

import torch

WIDTH = 64
MAX_RUNNING = 4
PROMPT = list(range(16))


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build_managers(num_pages: int = 256):
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    page_table = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    cache_manager = CacheManager(
        num_pages=num_pages, page_size=1, page_table=page_table, type="radix"
    )
    table_manager = TableManager(max_running_reqs=MAX_RUNNING, page_table=page_table)
    return cache_manager, table_manager, PrefillManager(
        cache_manager, table_manager, DecodeManager(page_size=1)
    )


def _pending(uid: int, *, hidden_states=None, no_prefix_cache: bool = False):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(
        uid=uid,
        input_ids=torch.tensor(PROMPT, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        hidden_states=hidden_states,
        no_prefix_cache=no_prefix_cache,
    )


def _run_one(cache_manager, prefill_manager, pending):
    """Admit one request, forward its single chunk, commit its prefix.

    Returns ``(req, admitted_cached_len)``: the forward's ``complete_one`` advances
    ``cached_len`` to the whole prompt, so the admission-time value must be read first.
    """
    prefill_manager.pending_list = [pending]
    batch = prefill_manager.schedule_next_batch(len(PROMPT))
    assert batch is not None, "the request was not admitted"
    (req,) = batch.reqs
    admitted = req.cached_len
    cache_manager.allocate_paged(batch.reqs)
    req.complete_one()
    cache_manager.cache_req(req, finished=True)
    return req, admitted


def _probe_spec(tmp_path):
    from freetoken.hidden_states import HiddenStateSpec

    return HiddenStateSpec(directory=str(tmp_path), layer_ids=[0, 1])


def test_second_ordinary_request_reuses_the_cached_prefix():
    """The control: without the knob, the radix cache does its job."""
    cache_manager, _, prefill_manager = _build_managers()
    _, first_cached = _run_one(cache_manager, prefill_manager, _pending(1))
    assert first_cached == 0

    _, second_cached = _run_one(cache_manager, prefill_manager, _pending(2))
    assert second_cached > 0


def test_probe_request_is_admitted_with_cached_len_zero(tmp_path):
    cache_manager, _, prefill_manager = _build_managers()
    _, warm_cached = _run_one(cache_manager, prefill_manager, _pending(1))
    assert warm_cached == 0
    # The prefix really is in the tree now: an ordinary sibling would hit it.
    assert cache_manager.match_req(_pending(9)).cuda_handle.cached_len > 0

    probe, probe_cached = _run_one(
        cache_manager,
        prefill_manager,
        _pending(2, hidden_states=_probe_spec(tmp_path), no_prefix_cache=True),
    )
    assert probe_cached == 0
    assert probe.hidden_states is not None
    assert probe.no_prefix_cache is True


def test_match_req_bypasses_the_tree_for_a_no_prefix_cache_request():
    """The knob is read where the multimodal bypass is, so it survives every cache kind."""
    cache_manager, _, prefill_manager = _build_managers()
    _run_one(cache_manager, prefill_manager, _pending(1))

    assert cache_manager.match_req(_pending(2)).cuda_handle.cached_len > 0
    assert cache_manager.match_req(
        _pending(3, no_prefix_cache=True)
    ).cuda_handle.cached_len == 0


def test_a_probe_still_leaves_a_reusable_prefix_behind(tmp_path):
    """Bypassing the *match* must not stop the commit: the next ordinary turn still hits."""
    cache_manager, _, prefill_manager = _build_managers()
    _run_one(
        cache_manager,
        prefill_manager,
        _pending(1, hidden_states=_probe_spec(tmp_path), no_prefix_cache=True),
    )
    _, following_cached = _run_one(cache_manager, prefill_manager, _pending(2))
    assert following_cached > 0


def test_probe_fields_survive_chunked_prefill(tmp_path):
    """Each continuation builds a fresh Req; both fields must be carried onto every chunk."""
    cache_manager, _, prefill_manager = _build_managers()
    spec = _probe_spec(tmp_path)
    prefill_manager.pending_list = [
        _pending(1, hidden_states=spec, no_prefix_cache=True)
    ]
    seen = []
    while prefill_manager.runnable:
        batch = prefill_manager.schedule_next_batch(4)
        assert batch is not None
        (req,) = batch.reqs
        seen.append((req.cached_len, req.extend_len, req.hidden_states, req.no_prefix_cache))
        cache_manager.allocate_paged(batch.reqs)
        req.complete_one()

    assert len(seen) == len(PROMPT) // 4
    assert [c for c, _, _, _ in seen] == [0, 4, 8, 12]
    assert all(spec_seen is spec for _, _, spec_seen, _ in seen)
    assert all(flag for _, _, _, flag in seen)


def _user_msg(uid: int, **kwargs):
    from freetoken.core import SamplingParams
    from freetoken.message import UserMsg

    return UserMsg(
        uid=uid,
        input_ids=torch.tensor(PROMPT, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        **kwargs,
    )


def test_prefix_notes_carry_the_lease_free_pin_key():
    """The auto-pin's cross-session rule (CacheManager.note_prompt_admitted, session_key)
    is fed from ``batch.prefix_notes``; the key is ``Req.pin_key`` when set (a pooled probe
    binds no lease, so its ``session_id`` is None) and falls back to ``session_id``."""
    from freetoken.hidden_states import HiddenStateSpec

    cache_manager, _, prefill_manager = _build_managers()
    pooled = HiddenStateSpec(directory=None, layer_ids=[0, 1], pooling=("mean",))

    # A pooled probe: no lease, pin key from the client session header.
    prefill_manager.add_one_req(
        _user_msg(1, hidden_states=pooled, pin_key="switchyard:conv-7")
    )
    (pending,) = prefill_manager.pending_list
    assert pending.pin_key == "switchyard:conv-7"
    assert pending.session_id is None
    batch = prefill_manager.schedule_next_batch(len(PROMPT))
    assert batch is not None
    (req,) = batch.reqs
    assert req.pin_key == "switchyard:conv-7"
    assert req.session_id is None
    (note,) = batch.prefix_notes
    handle, prompt_tokens, is_pooled, session_key = note
    assert handle is req.cache_handle
    assert prompt_tokens == len(PROMPT)
    assert is_pooled is True
    assert session_key == "switchyard:conv-7"
    cache_manager.allocate_paged(batch.reqs)
    req.complete_one()
    cache_manager.cache_req(req, finished=True)

    # A plain leased turn carries the same key on both fields; pin_key is preferred.
    prefill_manager.add_one_req(_user_msg(2, session_id="lease-a", pin_key="lease-a"))
    batch = prefill_manager.schedule_next_batch(len(PROMPT))
    assert batch is not None
    (_, _, is_pooled, session_key) = batch.prefix_notes[0]
    assert is_pooled is False
    assert session_key == "lease-a"
    (req,) = batch.reqs
    cache_manager.allocate_paged(batch.reqs)
    req.complete_one()
    cache_manager.cache_req(req, finished=True)

    # No pin key (an offline/legacy producer): session_id is the fallback.
    prefill_manager.add_one_req(_user_msg(3, session_id="lease-b"))
    batch = prefill_manager.schedule_next_batch(len(PROMPT))
    assert batch is not None
    assert batch.prefix_notes[0][3] == "lease-b"

    # Neither: no key, and the note says so.
    (req,) = batch.reqs
    cache_manager.allocate_paged(batch.reqs)
    req.complete_one()
    cache_manager.cache_req(req, finished=True)
    prefill_manager.add_one_req(_user_msg(4))
    batch = prefill_manager.schedule_next_batch(len(PROMPT))
    assert batch is not None
    assert batch.prefix_notes[0][3] is None


def test_pin_key_survives_chunked_prefill():
    """Each continuation builds a fresh Req; the pin key must ride every chunk."""
    cache_manager, _, prefill_manager = _build_managers()
    prefill_manager.pending_list = [_pending(1)]
    prefill_manager.pending_list[0].pin_key = "switchyard:conv-7"
    seen = []
    while prefill_manager.runnable:
        batch = prefill_manager.schedule_next_batch(4)
        assert batch is not None
        (req,) = batch.reqs
        seen.append(req.pin_key)
        cache_manager.allocate_paged(batch.reqs)
        req.complete_one()
    assert len(seen) == len(PROMPT) // 4
    assert seen == ["switchyard:conv-7"] * len(seen)


# --------------------------------------------------------------------------- #
# What the pooled gate costs: pooled_sumless_misses / pooled_sumless_miss_tokens
#
# A pooled-only probe DOES reuse prefixes, but only nodes whose snapshot also carries
# ``pooled_sums`` (its mean over the skipped positions comes from those). A response with
# ``prefix_tokens: 0`` therefore hides two very different stories, and only one of them is
# a cost the pooled design introduces:
#   (a) no reuse point existed -- the prompt would have missed anyway;
#   (b) a live snapshot WAS there and carried no sums, so the gate forced a full prefill.
# ``match_prefix`` reports the depth it would have reached without the sums requirement on
# the same walk, the handle carries it out through ``prefix_notes``, and only (b) is
# charged. These drive the real hybrid CacheManager on CPU.
# --------------------------------------------------------------------------- #
PREFIX = [1, 2, 3, 4, 5, 6, 7, 8]


def _build_hybrid(num_pages: int = 128, num_slots: int = 16):
    """A hybrid (GDN snapshot) CacheManager -- the only cache that keeps pooled sums, so
    the only one where the pooled gate can refuse a live reuse point."""
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.scheduler.cache import CacheManager

    _setup_context()
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    pool = LinearStatePool(
        group=group, num_slots=num_slots, dtype=torch.bfloat16,
        device=torch.device("cpu"), tp_size=1,
    )
    page_table = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32)
    return CacheManager(
        num_pages, 1, page_table, "hybrid_radix", linear_state_pool=pool
    )


def _pooled_spec():
    from freetoken.hidden_states import HiddenStateSpec

    return HiddenStateSpec(directory=None, layer_ids=[0, 1], pooling=("mean",))


def _hybrid_pending(uid: int, tokens: list, spec=None):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(
        uid=uid, input_ids=torch.tensor(tokens, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1), hidden_states=spec,
    )


def _seed_snapshot(cache_manager, tokens: list, slot: int, *, with_sums: bool) -> None:
    """Put a live GDN snapshot at ``tokens``' end boundary, with or without pooled sums --
    what a pooled producer (with) or an ordinary turn (without) leaves behind."""
    sums = torch.full((2, 4), float(slot), dtype=torch.float32) if with_sums else None
    base = 100 + slot * len(PREFIX) * 2
    cache_manager.prefix_cache.insert(
        torch.tensor(tokens, dtype=torch.int32),
        torch.arange(base, base + len(tokens), dtype=torch.int32),
        slot,
        pooled_sums=sums,
    )


def test_a_pooled_miss_with_no_prefix_at_all_is_not_charged_to_the_sums_gate():
    """Cause (a): nothing was refused, so nothing is charged -- the counter must not
    simply mirror ``pooled`` misses."""
    cache_manager = _build_hybrid()
    pending = _hybrid_pending(1, [90, 91, 92, 93], _pooled_spec())
    handle = cache_manager.match_req(pending).cuda_handle
    assert (handle.cached_len, handle.sumless_len) == (0, 0)
    cache_manager.note_prompt_admitted(handle, 4, pooled=True)
    counters = cache_manager.prefix_counters
    assert (counters.misses, counters.miss_tokens) == (1, 4)
    assert (counters.pooled_sumless_misses, counters.pooled_sumless_miss_tokens) == (0, 0)


def test_a_pooled_miss_past_a_sumless_snapshot_is_charged_the_token_delta():
    """Cause (b): an ordinary turn's snapshot is a reuse point for everyone but a pooled
    probe, which pays for the whole prefix again."""
    cache_manager = _build_hybrid()
    _seed_snapshot(cache_manager, PREFIX, 7, with_sums=False)
    prompt = PREFIX + [9, 10]
    # An ordinary request hits that node; the pooled one walks past it to a full miss.
    assert cache_manager.match_req(_hybrid_pending(1, prompt)).cuda_handle.cached_len == 8
    handle = cache_manager.match_req(
        _hybrid_pending(2, prompt, _pooled_spec())
    ).cuda_handle
    assert (handle.cached_len, handle.sumless_len) == (0, 8)
    cache_manager.note_prompt_admitted(handle, len(prompt), pooled=True)
    counters = cache_manager.prefix_counters
    assert (counters.misses, counters.miss_tokens) == (1, 10)
    assert (counters.pooled_sumless_misses, counters.pooled_sumless_miss_tokens) == (1, 8)


def test_a_pooled_hit_on_a_sums_bearing_node_is_charged_nothing():
    cache_manager = _build_hybrid()
    _seed_snapshot(cache_manager, PREFIX, 7, with_sums=True)
    prompt = PREFIX + [9, 10]
    handle = cache_manager.match_req(
        _hybrid_pending(1, prompt, _pooled_spec())
    ).cuda_handle
    assert (handle.cached_len, handle.sumless_len) == (8, 8)
    cache_manager.note_prompt_admitted(handle, len(prompt), pooled=True)
    counters = cache_manager.prefix_counters
    assert (counters.pooled_hits, counters.pooled_hit_tokens) == (1, 8)
    assert (counters.pooled_sumless_misses, counters.pooled_sumless_miss_tokens) == (0, 0)


def test_a_shortened_pooled_match_is_charged_only_the_difference():
    """Sums at 4, a deeper bare snapshot at 8: the probe keeps 4 and pays for the other 4.
    It is a hit AND a partial cost of the gate at the same time."""
    cache_manager = _build_hybrid()
    _seed_snapshot(cache_manager, PREFIX[:4], 3, with_sums=True)
    _seed_snapshot(cache_manager, PREFIX, 7, with_sums=False)
    prompt = PREFIX + [9, 10]
    handle = cache_manager.match_req(
        _hybrid_pending(1, prompt, _pooled_spec())
    ).cuda_handle
    assert (handle.cached_len, handle.sumless_len) == (4, 8)
    cache_manager.note_prompt_admitted(handle, len(prompt), pooled=True)
    counters = cache_manager.prefix_counters
    assert (counters.hits, counters.pooled_hits, counters.pooled_hit_tokens) == (1, 1, 4)
    assert (counters.misses, counters.pooled_sumless_misses) == (0, 1)
    assert counters.pooled_sumless_miss_tokens == 4


def test_a_plain_request_never_touches_the_pooled_counters():
    cache_manager = _build_hybrid()
    _seed_snapshot(cache_manager, PREFIX, 7, with_sums=False)
    prompt = PREFIX + [9, 10]
    handle = cache_manager.match_req(_hybrid_pending(1, prompt)).cuda_handle
    assert (handle.cached_len, handle.sumless_len) == (8, 8)
    cache_manager.note_prompt_admitted(handle, len(prompt))
    counters = cache_manager.prefix_counters
    assert (counters.hits, counters.hit_tokens) == (1, 8)
    assert counters.pooled_hits == 0
    assert (counters.pooled_sumless_misses, counters.pooled_sumless_miss_tokens) == (0, 0)


def test_matching_twice_cannot_double_count_the_sums_gate():
    """Matching counts nothing: the charge is made once, from ``batch.prefix_notes`` ->
    ``note_prompt_admitted``, on the pass that admits the prompt. The pressure path
    (``Scheduler._release_soft_session_for_admission``, which calls ``match_req`` for a
    queued prompt on every stalled iteration) therefore cannot inflate it."""
    cache_manager = _build_hybrid()
    _seed_snapshot(cache_manager, PREFIX, 7, with_sums=False)
    prompt = PREFIX + [9, 10]
    pending = _hybrid_pending(1, prompt, _pooled_spec())
    pressure = cache_manager.match_req(pending).cuda_handle    # the pressure path's walk
    admission = cache_manager.match_req(pending).cuda_handle   # admission's own walk
    counters = cache_manager.prefix_counters
    assert (counters.misses, counters.pooled_sumless_misses) == (0, 0)
    assert (pressure.cached_len, pressure.sumless_len) == (0, 8)
    assert (admission.cached_len, admission.sumless_len) == (0, 8)
    cache_manager.note_prompt_admitted(admission, len(prompt), pooled=True)
    assert (counters.pooled_sumless_misses, counters.pooled_sumless_miss_tokens) == (1, 8)


def test_the_pass_memo_cannot_share_a_match_between_a_pooled_and_a_plain_request():
    """``PrefillAdder._match`` memoizes on ``req.uid``, so the pooled gate's answer is
    never handed to a request that did not ask for it (and vice versa): one match, one
    request, one charge. Replaying the memo charges nothing either."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager

    cache_manager = _build_hybrid()
    _seed_snapshot(cache_manager, PREFIX, 7, with_sums=False)
    memo: dict = {}
    adder = PrefillAdder(
        token_budget=0, reserved_size=0, cache_manager=cache_manager,
        table_manager=TableManager(
            max_running_reqs=MAX_RUNNING, page_table=cache_manager.page_table
        ),
        match_memo=memo,
    )
    prompt = PREFIX + [9, 10]
    plain = _hybrid_pending(1, prompt)
    pooled = _hybrid_pending(2, prompt, _pooled_spec())
    assert adder._match(plain).cuda_handle.cached_len == 8
    assert adder._match(pooled).cuda_handle.cached_len == 0
    assert adder._match(plain).cuda_handle.cached_len == 8      # replay: still plain's answer
    assert adder._match(pooled).cuda_handle.sumless_len == 8    # replay: still pooled's
    assert set(memo) == {1, 2}
    counters = cache_manager.prefix_counters
    assert (counters.pooled_sumless_misses, counters.misses, counters.hits) == (0, 0, 0)
