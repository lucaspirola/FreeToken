"""Cross-conversation prefix reuse on hybrid (GDN/Mamba) models (exp/prefix-reuse).

* Intermediate prefill chunks donate their boundary snapshot to the shared radix tree
  (``CacheManager.commit_chunk_snapshot``), so a prompt sharing only the head of an earlier
  one -- a system prompt plus tools after a client's /clear -- resumes at the last chunk
  boundary it covers.
* ``evict_mamba`` takes unlocked chunk boundaries first, then chunk boundaries on a locked
  path, then request end states.
* The idle growable-KV shrink releases idle session leases older than the newest unleased
  prefix before it evicts that prefix (one LRU), and only with a checkpoint.
* /v1/messages drops Claude Code's per-conversation billing-header system block.
* /v1/messages keeps a system message that arrives mid-conversation in place (kept as a
  system turn, or folded into the adjacent user turn for a strict template), so turn N's
  prompt is a token prefix of turn N+1's.
* A prefill chunk ends exactly at the leading system+tools segment, whose state becomes a
  "segment" snapshot (``commit_prefix_boundary``), copied on the engine stream first.
CPU only: real LinearStatePool + page_table, hand-built Reqs, stub schedulers.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.scheduler import Scheduler


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def _chunked_req(cm, pool, ids, *, cached_len, boundary, table_idx=0, uid=0):
    """A chunked prompt whose first ``cached_len`` tokens were forwarded, the forward having
    written its track snapshot at ``boundary`` into ping-pong slot 0."""
    mr = cm.match_req(_pend(ids))
    cm.lock(mr.cuda_handle)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    cm.page_table[table_idx, :cached_len] = torch.arange(
        1000 + 100 * table_idx, 1000 + 100 * table_idx + cached_len, dtype=torch.int32)
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx,
              cached_len=cached_len, output_len=1, uid=uid, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1             # the forward wrote pp[0] and flipped
    req.mamba_last_track_seqlen = boundary
    return req, pp


def _cm(pool, rows=4):
    page_table = torch.zeros(rows, 64, dtype=torch.int32)
    return CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)


# ---------------------------------------------------------------- chunk snapshots
def test_intermediate_chunk_snapshot_serves_a_shared_head():
    pool = _pool()
    cm = _cm(pool)
    cm.chunk_snapshots = True
    shared = list(range(1, 9))                       # "system prompt + tools"
    req, pp = _chunked_req(cm, pool, shared + [50, 51, 52, 53], cached_len=10, boundary=8)

    cm.commit_chunk_snapshot(req)

    # the frozen slot is tree-owned now, replaced in the pair; the pending mark is consumed
    assert req.mamba_ping_pong[0] != pp[0] and req.mamba_ping_pong[1] == pp[1]
    assert req.mamba_last_track_seqlen is None
    assert req.cache_handle.cached_len == 8           # the continuation copies this handle
    # a different conversation sharing only the head resumes at the chunk boundary
    other = cm.match_req(_pend(shared + [70, 71, 72]))
    assert other.cuda_handle.cached_len == 8
    assert other.mamba_value == pp[0]
    # one commit per chunk: a second admission attempt of the same continuation is a no-op
    free = pool.num_free_slots
    cm.commit_chunk_snapshot(req)
    assert pool.num_free_slots == free


def test_chunk_snapshots_off_leaves_the_old_behavior():
    pool = _pool()
    cm = _cm(pool)
    cm.chunk_snapshots = False
    req, pp = _chunked_req(cm, pool, list(range(1, 13)), cached_len=10, boundary=8)
    cm.commit_chunk_snapshot(req)
    assert req.mamba_last_track_seqlen == 8 and req.mamba_ping_pong == pp
    assert cm.match_req(_pend(list(range(1, 9)) + [99, 98])).cuda_handle.cached_len == 0


def test_hidden_state_probe_is_not_chunk_committed():
    pool = _pool()
    cm = _cm(pool)
    cm.chunk_snapshots = True
    req, pp = _chunked_req(cm, pool, list(range(1, 13)), cached_len=10, boundary=8)
    req.hidden_states = object()
    cm.commit_chunk_snapshot(req)
    assert req.mamba_last_track_seqlen == 8 and req.mamba_ping_pong == pp


def test_chunk_commit_never_escalates_to_session_reclaim():
    """An optional chunk snapshot must not checkpoint an idle conversation: with the pool
    empty and nothing evictable, the commit is skipped and the reclaim hook never runs."""
    pool = _pool(num_slots=4)
    cm = _cm(pool)
    cm.chunk_snapshots = True
    req, pp = _chunked_req(cm, pool, list(range(1, 13)), cached_len=10, boundary=8)
    pool.alloc(pool.num_free_slots)                  # exhaust the free list
    calls = []
    cm.mamba_reclaim_hook = lambda n: calls.append(n) or False
    cm.commit_chunk_snapshot(req)
    assert calls == []
    assert req.mamba_ping_pong == pp                 # kept: the request loses nothing
    assert req.mamba_last_track_seqlen is None       # skipped, not retried at this boundary


# ---------------------------------------------------------------- snapshot eviction tiers
def _tree():
    return HybridRadixCache(torch.device("cpu"), 1, track_chunk_size=4)


def _ids(n, base=0):
    return torch.arange(base + 1, base + n + 1, dtype=torch.int32)


def test_evict_mamba_takes_chunk_boundaries_before_end_states():
    t = _tree()
    ids = _ids(12)
    kv = torch.arange(100, 112, dtype=torch.int32)
    t.insert(ids[:4], kv[:4], 1)                      # chunk boundary 4
    t.insert(ids[:8], kv[:8], 2)                      # chunk boundary 8
    t.insert(ids[:12], kv[:12], 3, kind="end")          # end state 12
    ev = t.evict_mamba(1)
    assert ev.mamba_slots == [2]                      # deeper chunk boundary first, same age
    ev = t.evict_mamba(1)
    assert ev.mamba_slots == [1]
    ev = t.evict_mamba(1)
    assert ev.mamba_slots == [3]                      # the end state goes last


def test_evict_mamba_spares_chunk_boundaries_on_a_locked_path():
    t = _tree()
    a, b = _ids(8), torch.cat([_ids(4), _ids(4, base=50)])
    t.insert(a[:4], torch.arange(100, 104, dtype=torch.int32), 1)
    t.insert(a, torch.arange(100, 108, dtype=torch.int32), 2, kind="end")
    t.insert(b, torch.cat([torch.arange(100, 104), torch.arange(200, 204)]).int(), 3)
    lease = t.match_prefix(a).node                   # a session lease holds a's end state
    t.inc_lock(lease)
    # candidates: 4 (chunk, on the locked path), 8' (chunk of b, unlocked); 8 is locked
    ev = t.evict_mamba(1)
    assert ev.mamba_slots == [3]
    ev = t.evict_mamba(1)
    assert ev.mamba_slots == [1]


def test_end_state_survives_a_dedup_by_a_chunk_insert():
    t = _tree()
    ids = _ids(8)
    kv = torch.arange(100, 108, dtype=torch.int32)
    t.insert(ids, kv, 1, kind="end")
    _, exist = t.insert(ids, kv, 2)                  # a later chunk boundary at the same length
    assert exist
    t.insert(_ids(4), kv[:4], 3)
    assert t.evict_mamba(1).mamba_slots == [3]


# ---------------------------------------------------------------- idle shrink, one LRU
class _Node:
    def __init__(self, ts, ref=0, children=()):
        self.timestamp, self.ref_count = ts, ref
        self.children = {i: c for i, c in enumerate(children)}


class _ShrinkCM:
    page_size = 1
    is_hybrid = True

    def __init__(self, *, committed, used, occupied, newest_leaf_ns):
        self.committed_pages = committed
        self.used = used
        self.free_slots = list(range(committed - occupied))
        self.prefix_cache = SimpleNamespace(
            root=_Node(0, 1, [_Node(newest_leaf_ns), _Node(1, 1)]),
            full_evictable_size=occupied - used,
            _trace=None,
        )

    def page_usage(self):
        return self.used, self.committed_pages


def _shrink_obj(cm, sessions, release):
    obj = SimpleNamespace(cache_manager=cm, _sessions=sessions,
                          _release_soft_session_handle=release, engine=None)
    obj._newest_evictable_prefix_leaf = Scheduler._newest_evictable_prefix_leaf.__get__(obj)
    obj._newest_evictable_prefix_ns = Scheduler._newest_evictable_prefix_ns.__get__(obj)
    obj._newest_evictable_prefix_pages = (
        Scheduler._newest_evictable_prefix_pages.__get__(obj))
    obj._kv_bytes_per_token = lambda: 0
    return obj


def _lease(last_used, tokens):
    return SimpleNamespace(last_used_at=last_used, reclaimable=True, active_uid=None,
                           handle=SimpleNamespace(cached_len=tokens))


def test_idle_shrink_releases_older_leases_before_the_newest_prefix():
    # base step 64; leases hold 60 pages, a stateless prompt left 20 evictable: 80 occupied
    cm = _ShrinkCM(committed=128, used=60, occupied=80, newest_leaf_ns=int(50e9))
    sessions = {"old": _lease(10.0, 45), "mid": _lease(20.0, 15), "newer": _lease(60.0, 5)}
    released = []

    def release(sid, _reason, *, require_checkpoint):
        assert require_checkpoint
        released.append(sid)
        tokens = sessions[sid].handle.cached_len
        sessions[sid].handle = None
        cm.used -= tokens
        cm.prefix_cache.full_evictable_size += tokens
        return True

    obj = _shrink_obj(cm, sessions, release)
    Scheduler._release_leases_older_than_prefix(obj, 64, 64, 0)
    # need = 80 - 64 = 16 pages; the oldest lease alone hands back 45
    assert released == ["old"]


def test_idle_shrink_never_releases_a_lease_newer_than_the_prefix():
    cm = _ShrinkCM(committed=128, used=60, occupied=80, newest_leaf_ns=int(5e9))
    sessions = {"a": _lease(10.0, 45), "b": _lease(20.0, 15)}
    released = []
    obj = _shrink_obj(cm, sessions, lambda sid, *_a, **_k: released.append(sid) or True)
    Scheduler._release_leases_older_than_prefix(obj, 64, 64, 0)
    assert released == []


def test_idle_shrink_falls_back_when_the_checkpoint_fails():
    cm = _ShrinkCM(committed=128, used=60, occupied=80, newest_leaf_ns=int(50e9))
    sessions = {"old": _lease(10.0, 45), "mid": _lease(20.0, 15)}
    attempts = []
    obj = _shrink_obj(cm, sessions, lambda sid, *_a, **_k: attempts.append(sid) or False)
    Scheduler._release_leases_older_than_prefix(obj, 64, 64, 0)
    assert attempts == ["old"]                       # stops at the first refusal
    assert obj._shrink_lease_release_fallbacks == 1


def test_idle_shrink_does_nothing_when_no_shrink_is_possible():
    cm = _ShrinkCM(committed=128, used=100, occupied=110, newest_leaf_ns=int(50e9))
    released = []
    obj = _shrink_obj(cm, {"old": _lease(10.0, 45)},
                      lambda sid, *_a, **_k: released.append(sid) or True)
    Scheduler._release_leases_older_than_prefix(obj, 64, 64, 0)
    assert released == []                            # target 128 == committed


# ---------------------------------------------------------------- Anthropic billing header
def _anthropic_req(system):
    from freetoken.server.anthropic_api import AnthropicMessagesRequest

    return AnthropicMessagesRequest(
        model="m", max_tokens=8, system=system,
        messages=[{"role": "user", "content": "hi"}],
    )


def test_billing_header_block_is_stripped_by_default():
    from freetoken.server.anthropic_api import convert_anthropic_prompt

    system = [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.282.f5d; "
                                 "cc_entrypoint=cli;"},
        {"type": "text", "text": "You are a coding agent."},
    ]
    a = convert_anthropic_prompt(_anthropic_req(system))[0]
    system[0]["text"] = system[0]["text"].replace("f5d", "80e")
    b = convert_anthropic_prompt(_anthropic_req(system))[0]
    assert a[0] == {"role": "system", "content": "You are a coding agent."}
    assert a == b                                    # two conversations, one prefix
    kept = convert_anthropic_prompt(_anthropic_req(system), strip_billing_header=False)[0]
    assert kept[0]["content"].startswith("x-anthropic-billing-header:")


def test_billing_header_switch_defaults_on():
    from freetoken.server.args import ServerArgs

    assert ServerArgs.anthropic_strip_billing_header is True
    assert ServerArgs.anthropic_system_in_place is True


# ---------------------------------------------------------------- in-place system turns
class _FakeTok:
    """Char-level tokenizer with a tiny chat template: ``strict`` rejects a system turn
    anywhere but first (Ornith 1.5 / Qwen3.5), else it renders it in place (Nemotron 3.5)."""

    chat_template = "fake"
    name_or_path = ""

    def __init__(self, strict: bool):
        self.strict = strict

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            tools=None, **_kw):
        import json

        out = []
        for i, m in enumerate(messages):
            if m["role"] == "system" and i > 0 and self.strict:
                raise ValueError("System message must be at the beginning.")
            c = m.get("content") or ""
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c)
            out.append(f"<|{m['role']}|>{c}")
            if i == 0 and m["role"] == "system" and tools:
                out.append("<tools>" + json.dumps(tools) + "</tools>")
            out.append("<|end|>")
        if add_generation_prompt:
            out.append("<|assistant|>")
        return "".join(out)

    def encode(self, text, return_tensors=None, add_special_tokens=True):
        return torch.tensor([[ord(ch) for ch in text]], dtype=torch.int64)


_CC_TOOLS = [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}]


def _cc_body(turn: int) -> dict:
    """Claude Code 2.1.282's request shape: the environment block arrives as a system message
    AFTER the first user message, and every later user turn is followed by a system reminder."""
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                                     {"type": "text", "text": "first turn"}]},
        {"role": "system", "content": "# Environment\ncwd: /repo"},
    ]
    if turn >= 2:
        msgs += [
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "second turn"},
            {"role": "system", "content": "<total_tokens>1000</total_tokens>"},
        ]
    return {"model": "m", "max_tokens": 8, "system": [{"type": "text", "text": "You are CC."}],
            "tools": _CC_TOOLS, "messages": msgs}


def _cc_ids(tm, turn, *, in_place=True):
    from freetoken.message import TokenizeMsg
    from freetoken.server.anthropic_api import AnthropicMessagesRequest, convert_anthropic_prompt

    m, tools, _, ctk = convert_anthropic_prompt(
        AnthropicMessagesRequest(**_cc_body(turn)), system_in_place=in_place)
    msg = TokenizeMsg(uid=0, text=m, sampling_params=SamplingParams(), tools=tools,
                      chat_template_kwargs=ctk)
    return tm.tokenize([msg])[0], m


def test_adapter_keeps_only_leading_system_messages_in_the_system_text():
    from freetoken.server.anthropic_api import AnthropicMessagesRequest, convert_anthropic_prompt

    body = _cc_body(2)
    body["messages"].insert(0, {"role": "system", "content": "leading"})
    m = convert_anthropic_prompt(AnthropicMessagesRequest(**body))[0]
    assert m[0] == {"role": "system", "content": "You are CC.\n\nleading"}
    assert [(x["role"], bool(x.get("freetoken_in_place"))) for x in m[1:]] == [
        ("user", False), ("system", True), ("assistant", False), ("user", False),
        ("system", True)]
    hoisted = convert_anthropic_prompt(AnthropicMessagesRequest(**body), system_in_place=False)[0]
    assert [x["role"] for x in hoisted] == ["system", "user", "assistant", "user"]
    assert "# Environment" in hoisted[0]["content"]


def test_turn_n_prompt_is_a_token_prefix_of_turn_n_plus_1():
    from freetoken.tokenizer.tokenize import TokenizeManager

    for strict in (False, True):
        tm = TokenizeManager(_FakeTok(strict))
        assert tm.mid_system_supported() is (not strict)
        a, _ = _cc_ids(tm, 1)
        b, _ = _cc_ids(tm, 2)
        assert torch.equal(b.input_ids[: len(a.input_ids)], a.input_ids), strict
        # hoisting (switch off) rewrites the prompt ahead of the tool list on every turn
        tm2 = TokenizeManager(_FakeTok(strict))
        a2, _ = _cc_ids(tm2, 1, in_place=False)
        b2, _ = _cc_ids(tm2, 2, in_place=False)
        assert not torch.equal(b2.input_ids[: len(a2.input_ids)], a2.input_ids)


def test_strict_template_folds_the_system_turn_into_the_user_turn():
    from freetoken.message import TokenizeMsg
    from freetoken.tokenizer.tokenize import TokenizeManager

    tm = TokenizeManager(_FakeTok(strict=True))
    _, m = _cc_ids(tm, 2)
    text = tm.render_prompt(TokenizeMsg(uid=0, text=m, sampling_params=SamplingParams()))
    assert "<|user|>second turn\n\n<system-reminder>\n<total_tokens>1000</total_tokens>\n" \
           "</system-reminder><|end|>" in text
    assert text.count("<|system|>") == 1
    # a system turn after an assistant turn becomes a user turn of its own
    placed = tm.place_system_messages([
        {"role": "system", "content": "s"}, {"role": "assistant", "content": "a"},
        {"role": "system", "content": "r", "freetoken_in_place": True}])
    assert placed[-1] == {"role": "user", "content": "<system-reminder>\nr\n</system-reminder>"}
    # permissive template: kept as a system turn, marker stripped
    tm = TokenizeManager(_FakeTok(strict=False))
    placed = tm.place_system_messages([{"role": "user", "content": "u"},
                                       {"role": "system", "content": "r", "freetoken_in_place": True}])
    assert placed[-1] == {"role": "system", "content": "r"}


def test_segment_boundary_is_the_system_and_tools_head():
    from freetoken.tokenizer.tokenize import TokenizeManager

    tm = TokenizeManager(_FakeTok(strict=False))
    a, m = _cc_ids(tm, 1)
    head = "<|system|>You are CC.<tools>"
    text = "".join(chr(int(c)) for c in a.input_ids)
    b = a.prefix_boundary
    assert b is not None and text[:b].startswith(head) and text[:b].endswith("</tools><|end|>")
    other, _ = _cc_ids(tm, 2)
    assert other.prefix_boundary == b                 # same head, cached ids
    # no leading system message -> no segment
    from freetoken.message import TokenizeMsg
    u = tm.tokenize([TokenizeMsg(uid=0, text=[{"role": "user", "content": "x"},
                                              {"role": "user", "content": "y"}],
                                 sampling_params=SamplingParams())])[0]
    assert u.prefix_boundary is None


def test_real_template_keeps_turn_prefix():
    """Nemotron 3.5's own template, when the checkpoint is on this host: turn N's prompt
    minus its generation-prompt tail is a prefix of turn N+1's; the segment ends before."""
    import os

    import pytest

    path = os.path.expanduser("~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    if not os.path.isdir(path):
        pytest.skip("model not on this host")
    from transformers import AutoTokenizer

    from freetoken.tokenizer.tokenize import TokenizeManager

    tm = TokenizeManager(AutoTokenizer.from_pretrained(path))
    assert tm.mid_system_supported()
    a, _ = _cc_ids(tm, 1)
    b, _ = _cc_ids(tm, 2)
    x, y = a.input_ids.tolist(), b.input_ids.tolist()
    shared = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), len(x))
    assert shared >= len(x) - 2                        # only "<think>\n" of the gen prompt differs
    assert a.prefix_boundary is not None and a.prefix_boundary < shared


# ---------------------------------------------------------------- segment snapshot
def test_prefill_cuts_the_chunk_at_the_segment_and_snapshots_it():
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(512, 1, pt, "hybrid_radix", linear_state_pool=pool)
    cm.chunk_snapshots = True
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(1, 301, dtype=torch.int32),
                         SamplingParams(max_tokens=1), prefix_boundary=150)

    def step(req):
        pt[req.table_idx, req.cached_len : len(req.input_ids)] = torch.arange(
            req.cached_len, len(req.input_ids), dtype=torch.int32) + 1000
        req.cached_len = len(req.input_ids)           # the forward ran
        pending.chunked_req = req

    r1 = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm) \
        .try_add_one(pending)
    assert isinstance(r1, ChunkedReq) and r1.extend_len == 100
    step(r1)
    r2 = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm) \
        .try_add_one(pending)
    assert r2.extend_len == 50                        # cut at the segment boundary
    assert r2.mamba_boundary_copy is None
    step(r2)
    r3 = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm) \
        .try_add_one(pending)
    assert r3.cached_len == 150 and r3.extend_len == 100
    dst = r3.mamba_boundary_copy
    assert dst is not None and dst != r3.linear_slot_idx
    assert r3.cache_handle.cached_len == 150 and r3.cache_handle.node.snapshot_kind == "segment"
    # another conversation with the same system+tools resumes at the segment
    hit = cm.match_req(_pend(list(range(1, 151)) + [999, 998]))
    assert hit.cuda_handle.cached_len == 150 and hit.mamba_value == dst

    # engine stream: the segment copy runs before any restore of the same batch
    pool.recurrent_states[:, r3.linear_slot_idx] = 3
    fresh = SimpleNamespace(mamba_restore_src=dst, linear_slot_idx=pool.alloc(1)[0],
                            mamba_boundary_copy=None)
    sched = SimpleNamespace(engine=SimpleNamespace(linear_state_pool=pool))
    Scheduler._restore_linear_states(sched, SimpleNamespace(is_prefill=True, reqs=[fresh, r3]))
    assert r3.mamba_boundary_copy is None
    assert bool((pool.recurrent_states[:, dst] == 3).all())
    assert bool((pool.recurrent_states[:, fresh.linear_slot_idx] == 3).all())


def test_segment_commit_dedups_against_an_existing_snapshot():
    pool = _pool()
    cm = _cm(pool)
    cm.chunk_snapshots = True
    ids = list(range(1, 13))
    # both conversations were admitted cold, before either reached the segment
    pairs = []
    for uid in (0, 1):
        prev, _ = _chunked_req(cm, pool, ids, cached_len=6, boundary=None, table_idx=uid, uid=uid)
        cont = Req(input_ids=torch.tensor(ids[:10], dtype=torch.int32), table_idx=uid,
                   cached_len=6, output_len=1, uid=uid, sampling_params=SamplingParams(),
                   cache_handle=prev.cache_handle)
        cont.linear_slot_idx = prev.linear_slot_idx
        pairs.append((prev, cont))
    for uid, (prev, cont) in enumerate(pairs):
        free = pool.num_free_slots
        cm.commit_prefix_boundary(prev, cont, 6)
        assert cont.cache_handle.cached_len == 6 and prev.cache_handle is cont.cache_handle
        if uid == 0:
            assert cont.mamba_boundary_copy is not None and pool.num_free_slots == free - 1
        else:                                          # the tree already holds [0, 6)
            assert cont.mamba_boundary_copy is None and pool.num_free_slots == free
            # the second conversation's row now names the tree's pages
            assert pt_row(cm, 1, 6) == pt_row(cm, 0, 6)


def pt_row(cm, row, n):
    return cm.page_table[row, :n].tolist()


def test_segment_cut_is_off_without_chunk_snapshots():
    from freetoken.scheduler.prefill import PrefillAdder

    cm = SimpleNamespace(is_hybrid=True, chunk_snapshots=False, page_size=1)
    pend = SimpleNamespace(prefix_boundary=150, hidden_states=None, no_prefix_cache=False,
                           input_len=300)
    assert PrefillAdder._segment_cut(SimpleNamespace(cache_manager=cm), pend) is None
    cm.chunk_snapshots = True
    assert PrefillAdder._segment_cut(SimpleNamespace(cache_manager=cm), pend) == 150
    pend.no_prefix_cache = True
    assert PrefillAdder._segment_cut(SimpleNamespace(cache_manager=cm), pend) is None
