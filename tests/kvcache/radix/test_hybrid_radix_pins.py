"""Prefix auto-pin on the hybrid radix cache (``CacheManager.pin_prefix`` / ``unpin_all``).

A pin is nothing the tree does not already know: an extra ``inc_lock`` on the pinned node
(full ref node..root, mamba ref on the node) and on every snapshot-bearing ancestor, held by
the cache manager instead of a request. So what is pinned here is exactly what the existing
lock invariants protect -- ``evict_full`` cannot take a locked leaf, ``evict_mamba`` cannot
tombstone a locked snapshot -- and a release is the matching ``dec_lock``s. The tests drive
the tree through the manager with a real ``LinearStatePool`` on CPU (page_size 1), the way
``tests/scheduler/test_hybrid_cache_manager.py`` does, and check the tree's own
evictable/protected ledgers plus the ``/v1/stats`` pin gauges.

Trigger: ``note_prompt_admitted`` fires on the first chunk of an admitted prompt with the
request's session key. A node remembers the keys that matched through it; the deepest node
on the path that TWO DIFFERENT sessions matched through, ending at
``>= pin_prefix_min_tokens``, is pinned. A session re-matching its own history never pins.

Budgets: every pinned snapshot holds one GDN state slot, so pins are budgeted in slots
(``pin_prefix_max_slots``, auto = the pool's snapshot-cache slots minus 2) as well as KV
tokens; over either, the least-recently-matched pin is released first.

Scope (S13, ``--pin-prefix-scope``): everything above is ``shared``, the default. Under
``session`` a request with a session key also pins the prefix it leaves in the tree at its
finish (``_cache_req_hybrid`` -> ``_pin_session_prefix``), holding the whole path's KV but
only the deepest ``max(1, pin_slot_budget // 2)`` snapshots; a hit refreshes the pins it
shares a path with. Those tests drive whole requests through ``match_req`` / ``cache_req``
the way the scheduler does (``_serve``), including the production geometry of the
single-lane profile (13 state slots, 8192-token prefill chunks, Mamba-2 track 128).
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler import cache as cache_mod
from freetoken.scheduler.cache import (PIN_WORKING_SET_SLOTS_PER_REQUEST, CacheManager,
                                       pin_prefix_scope_error)


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def _cm(pool, min_tokens=4, max_tokens=0, max_slots=-1, working_set=0, num_pages=64,
        scope="shared", rows=4):
    page_table = torch.zeros(rows, num_pages, dtype=torch.int32)
    return CacheManager(num_pages, 1, page_table, "hybrid_radix", linear_state_pool=pool,
                        pin_prefix_min_tokens=min_tokens, pin_prefix_max_tokens=max_tokens,
                        pin_prefix_max_slots=max_slots, pin_working_set_slots=working_set,
                        pin_prefix_scope=scope)


def _insert(cm, pool, ids):
    """Donate ``ids`` with a fresh snapshot, the way a finished request would: the pages
    move from the free-list to the tree (so ``check_integrity`` balances)."""
    n = len(ids)
    pages, cm.free_slots = cm.free_slots[:n].clone(), cm.free_slots[n:]
    slot = pool.alloc(1)[0]
    matched, exists = cm.prefix_cache.insert(torch.tensor(ids, dtype=torch.int32), pages, slot)
    assert not exists
    # The deduped prefix pages go back to the free-list, as cache_req's ``_free`` does.
    cm.free_slots = torch.cat([cm.free_slots, pages[:matched]])
    return slot


def _admit(cm, ids, session="a"):
    """A request of ``session`` matching ``ids``: what the admission loop reports at the
    PromptAdmittedMsg point. Locks and unlocks like a request that came and went."""
    mr = cm.match_req(_pend(ids + [999]))                    # the last token is never matched
    cm.lock(mr.cuda_handle)
    cm.note_prompt_admitted(mr.cuda_handle, len(ids) + 1, session_key=session)
    cm.unlock(mr.cuda_handle)
    return mr.cuda_handle


def _node(cm, ids):
    return cm.prefix_cache.match_prefix(torch.tensor(ids, dtype=torch.int32)).node


def _tree(cm):
    return cm.prefix_cache


def _ledger(cm):
    c = cm.prefix_counters
    return (c.pinned_prefixes, c.pinned_tokens, c.pinned_slots)


# --------------------------------------------------------------------------- trigger
def test_a_second_session_pins_and_the_same_session_does_not():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])

    _admit(cm, [1, 2, 3, 4, 5], session="a")               # the first matcher records itself
    assert _ledger(cm) == (0, 0, 0) and cm._pins == {}
    assert _node(cm, [1, 2, 3, 4, 5]).pin_sessions == {"a"}
    _admit(cm, [1, 2, 3, 4, 5], session="a")               # its own next turn: no pin
    assert _ledger(cm) == (0, 0, 0)

    _admit(cm, [1, 2, 3, 4, 5], session="b")               # a different session: shared
    assert _ledger(cm) == (1, 5, 1)
    assert cm.prefix_counters.as_dict()["hits"] == 3
    node = _node(cm, [1, 2, 3, 4, 5])
    assert node.ref_count >= 1 and node.mamba_ref_count == 1  # the pin outlives the requests
    assert node.pin_sessions == {"a", "b"}


def test_a_shared_prefix_below_the_threshold_does_not_pin():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3])
    _admit(cm, [1, 2, 3], session="a")
    _admit(cm, [1, 2, 3], session="b")
    assert _ledger(cm) == (0, 0, 0)


def test_a_request_without_a_session_key_neither_records_nor_pins():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _admit(cm, [1, 2, 3, 4, 5], session=None)
    _admit(cm, [1, 2, 3, 4, 5], session=None)
    assert _ledger(cm) == (0, 0, 0)
    assert _node(cm, [1, 2, 3, 4, 5]).pin_sessions == set()
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session=None)             # both keys must be present
    assert _ledger(cm) == (0, 0, 0)


def test_the_deepest_shared_node_is_pinned_not_the_private_tail():
    """Session a's whole history is [1..6]; b shares only [1..4]. The pin lands on the shared
    snapshot node, a's private [5,6] stays evictable -- until b matches through it too."""
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4])
    _insert(cm, pool, [1, 2, 3, 4, 5, 6])
    _admit(cm, [1, 2, 3, 4, 5, 6], session="a")           # records a on [1..4] and [5,6]
    _admit(cm, [1, 2, 3, 4], session="b")
    assert _ledger(cm) == (1, 4, 1)
    assert _node(cm, [1, 2, 3, 4]).mamba_ref_count == 1
    assert _node(cm, [1, 2, 3, 4, 5, 6]).mamba_ref_count == 0

    _admit(cm, [1, 2, 3, 4, 5, 6], session="a")           # a again: [5,6] still a-only
    assert _ledger(cm) == (1, 4, 1)
    _admit(cm, [1, 2, 3, 4, 5, 6], session="b")           # now [5,6] is shared as well
    assert _ledger(cm) == (2, 6, 2)
    assert _node(cm, [1, 2, 3, 4, 5, 6]).mamba_ref_count == 1
    assert _tree(cm).full_protected == 6 and _tree(cm).mamba_protected == 2


def test_pinning_is_off_at_min_tokens_zero():
    pool = _pool()
    cm = _cm(pool, min_tokens=0)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")
    assert _ledger(cm) == (0, 0, 0)
    assert _tree(cm).full_protected == 0 and _tree(cm).mamba_protected == 0


# --------------------------------------------------------------------------- eviction
def test_a_pinned_path_survives_evict_full():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [7, 8, 9, 10, 11])                 # the unpinned victim
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")

    victim = _tree(cm).match_prefix(torch.tensor([7, 8, 9, 10, 11], dtype=torch.int32))
    er = _tree(cm).evict_full(10)                             # ask for everything
    assert sorted(er.kv_indices.tolist()) == sorted(victim.kv_indices.tolist())
    assert _tree(cm).full_protected == 5 and _tree(cm).full_evictable == 0
    m = _tree(cm).match_prefix(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32))
    assert m.cached_len == 5 and m.mamba_value is not None


def test_a_pinned_snapshot_survives_evict_mamba_and_is_not_tombstoned():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    s_pinned = _insert(cm, pool, [1, 2, 3, 4, 5])
    s_victim = _insert(cm, pool, [7, 8, 9, 10, 11])
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")

    er = _tree(cm).evict_mamba(2)
    assert er.mamba_slots == [s_victim]
    assert _tree(cm).mamba_protected == 1 and _tree(cm).mamba_evictable == 0
    m = _tree(cm).match_prefix(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32))
    assert m.mamba_value == s_pinned


def test_ensure_mamba_slots_cannot_reach_a_pin_and_fails_as_today():
    """A short pool is refused, not unpinned: the pin is invisible to reserve_mamba_slots
    exactly like a session lease, and the one-time warning names the pins. (The explicit
    slot budget lets the pin exist in a pool this small.)"""
    pool = _pool(num_slots=3)                                 # padding + 2 usable
    cm = _cm(pool, min_tokens=4, max_slots=1)
    _insert(cm, pool, [1, 2, 3, 4, 5])                        # one usable slot in the tree
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")
    assert pool.num_free_slots == 1
    assert cm._pin_starvation_logged is False
    assert cm.reserve_mamba_slots(2) is False
    assert _ledger(cm) == (1, 5, 1)                           # still pinned
    assert cm._pin_starvation_logged is True                  # logged once; never auto-unpinned
    assert cm.reserve_mamba_slots(2) is False


# --------------------------------------------------------------------------- slot budget
def test_the_auto_slot_budget_is_the_snapshot_cache_minus_two():
    pool = _pool(num_slots=16)
    # 16 slots = 4 x 2 working set + 7 cache + 1 padding: 7 - 2 = 5 for pins.
    assert _cm(pool, working_set=8).pin_slot_budget == 5
    assert _cm(pool, working_set=0).pin_slot_budget == 13     # no known working set
    assert _cm(pool, working_set=16).pin_slot_budget == 0     # never negative
    assert _cm(pool, working_set=8, max_slots=9).pin_slot_budget == 9   # explicit wins
    assert _cm(pool, working_set=8, max_slots=0).pin_slot_budget == 0


def test_a_pin_over_the_slot_budget_releases_the_least_recently_matched_pin():
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_slots=1)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [7, 8, 9, 10, 11])
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")               # pin 1: the only slot
    assert _ledger(cm) == (1, 5, 1)
    _admit(cm, [7, 8, 9, 10, 11], session="a")
    _admit(cm, [7, 8, 9, 10, 11], session="b")             # pin 2 needs the slot back
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pinned_tokens"], c["pinned_slots"]) == (1, 5, 1)
    assert (c["pin_evictions"], c["pin_budget_refusals"]) == (1, 0)
    assert _node(cm, [1, 2, 3, 4, 5]).mamba_ref_count == 0    # released...
    assert _node(cm, [1, 2, 3, 4, 5]).ref_count == 0
    assert _node(cm, [7, 8, 9, 10, 11]).mamba_ref_count == 1  # ...for the newer one
    assert _tree(cm).full_protected == 5 and _tree(cm).mamba_protected == 1
    cm.check_integrity()


def test_release_order_is_last_match_not_creation():
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_slots=2)
    for ids in ([1, 2, 3, 4, 5], [7, 8, 9, 10, 11], [13, 14, 15, 16, 17]):
        _insert(cm, pool, ids)
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")               # pin X
    _admit(cm, [7, 8, 9, 10, 11], session="a")
    _admit(cm, [7, 8, 9, 10, 11], session="b")             # pin Y
    _admit(cm, [1, 2, 3, 4, 5], session="c")               # X re-matched: now the newer
    assert _ledger(cm) == (2, 10, 2)
    _admit(cm, [13, 14, 15, 16, 17], session="a")
    _admit(cm, [13, 14, 15, 16, 17], session="b")          # pin Z evicts Y, not X
    assert _ledger(cm) == (2, 10, 2)
    assert _node(cm, [7, 8, 9, 10, 11]).mamba_ref_count == 0
    assert _node(cm, [1, 2, 3, 4, 5]).mamba_ref_count == 1
    assert _node(cm, [13, 14, 15, 16, 17]).mamba_ref_count == 1
    assert cm.prefix_counters.pin_evictions == 1


def test_a_pin_that_fits_no_budget_is_refused_and_counted():
    """Slot budget 0: a snapshot pin cannot fit even an empty ledger, so nothing is
    released and nothing is partially applied."""
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_slots=0)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pin_evictions"], c["pin_budget_refusals"]) == (0, 0, 1)
    assert c["hits"] == 2                                     # the hit itself still counts
    node = _node(cm, [1, 2, 3, 4, 5])
    assert node.ref_count == 0 and node.mamba_ref_count == 0
    assert _tree(cm).full_protected == 0


def test_the_token_budget_releases_lru_first_and_refuses_what_never_fits():
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_tokens=8)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [7, 8, 9, 10, 11])
    _insert(cm, pool, [20, 21, 22, 23, 24, 25, 26, 27, 28])
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")               # 5 of 8
    _admit(cm, [7, 8, 9, 10, 11], session="a")
    _admit(cm, [7, 8, 9, 10, 11], session="b")             # 5 more: releases the first
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pinned_tokens"], c["pin_evictions"]) == (1, 5, 1)
    assert _node(cm, [7, 8, 9, 10, 11]).mamba_ref_count == 1
    _admit(cm, [20, 21, 22, 23, 24, 25, 26, 27, 28], session="a")
    _admit(cm, [20, 21, 22, 23, 24, 25, 26, 27, 28], session="b")   # 9 > 8: never fits
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pinned_tokens"]) == (1, 5)   # the 5-token pin stays
    assert (c["pin_evictions"], c["pin_budget_refusals"]) == (1, 1)


# --------------------------------------------------------------------------- ledger
def test_re_matching_a_pinned_prefix_refreshes_it_without_a_second_lock():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [1, 2, 3, 4, 5, 6, 7])              # extends the same path by 2
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")
    node = _node(cm, [1, 2, 3, 4, 5])
    stamp = cm._pins[node].last_match
    _admit(cm, [1, 2, 3, 4, 5], session="c")               # the same prefix again
    assert _ledger(cm) == (1, 5, 1)
    assert node.mamba_ref_count == 1                          # pinned once, not twice
    assert cm._pins[node].last_match >= stamp

    _admit(cm, [1, 2, 3, 4, 5, 6, 7], session="a")
    _admit(cm, [1, 2, 3, 4, 5, 6, 7], session="b")         # the longer path: +2 tokens
    assert _ledger(cm) == (2, 7, 2)
    assert _tree(cm).full_protected == 7


def test_releasing_a_shallow_pin_keeps_the_locks_a_deeper_pin_needs():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4])
    _insert(cm, pool, [1, 2, 3, 4, 5, 6])
    _admit(cm, [1, 2, 3, 4], session="a")
    _admit(cm, [1, 2, 3, 4], session="b")                  # pin P = [1..4]
    _admit(cm, [1, 2, 3, 4, 5, 6], session="a")
    _admit(cm, [1, 2, 3, 4, 5, 6], session="b")            # pin C = [5,6], P its ancestor
    assert _ledger(cm) == (2, 6, 2)
    upper, lower = _node(cm, [1, 2, 3, 4]), _node(cm, [1, 2, 3, 4, 5, 6])
    assert upper.mamba_ref_count == 1 and lower.mamba_ref_count == 1

    cm._release_pin(upper)
    assert _ledger(cm) == (1, 6, 2)                           # C still needs P's snapshot
    assert upper.mamba_ref_count == 1 and upper.ref_count == 2  # P's own lock + C's path
    cm._release_pin(lower)
    assert _ledger(cm) == (0, 0, 0)
    assert upper.mamba_ref_count == 0 and lower.mamba_ref_count == 0
    assert _tree(cm).full_protected == 0 and _tree(cm).mamba_protected == 0


def test_a_split_below_a_pinned_node_keeps_the_pin_and_the_ledger_exact():
    """``split_at`` copies ref_count (and the recorded sessions) to the new root-side half,
    so the pin follows the KV; a later pin through the split half must not count the
    already-pinned tokens twice."""
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5, 6])
    _admit(cm, [1, 2, 3, 4, 5, 6], session="a")
    _admit(cm, [1, 2, 3, 4, 5, 6], session="b")
    _insert(cm, pool, [1, 2, 3, 4, 8, 9])                 # splits [1..6] at 4
    assert _tree(cm).full_protected == 6                      # both halves still locked
    assert _node(cm, [1, 2, 3, 4, 5, 6]).parent.pin_sessions == {"a", "b"}
    _admit(cm, [1, 2, 3, 4, 8, 9], session="a")
    _admit(cm, [1, 2, 3, 4, 8, 9], session="b")
    assert _ledger(cm) == (2, 8, 2)
    assert _tree(cm).full_protected == 8
    cm.check_integrity()


# --------------------------------------------------------------------------- unpin
def test_unpin_all_releases_every_lock_and_reports_what_it_held():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [7, 8, 9, 10, 11])
    for ids in ([1, 2, 3, 4, 5], [7, 8, 9, 10, 11]):
        _admit(cm, ids, session="a")
        _admit(cm, ids, session="b")
    assert _tree(cm).full_protected == 10 and _tree(cm).mamba_protected == 2

    assert cm.unpin_all() == {"pinned_prefixes": 2, "pinned_tokens": 10}
    assert _tree(cm).full_protected == 0 and _tree(cm).mamba_protected == 0
    assert _tree(cm).full_evictable == 10 and _tree(cm).mamba_evictable == 2
    assert _ledger(cm) == (0, 0, 0)
    assert cm.unpin_all() == {"pinned_prefixes": 0, "pinned_tokens": 0}   # idempotent
    # everything is LRU-evictable again
    er = _tree(cm).evict_full(10)
    assert len(er.kv_indices) == 10
    cm._free(er.kv_indices)
    pool.free(er.mamba_slots)
    cm.check_integrity()


def test_rebuild_drops_the_pins_with_the_tree():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")
    cm.rebuild(64, torch.zeros(4, 64, dtype=torch.int32))
    assert cm._pins == {} and cm._pin_locked == {} and _ledger(cm) == (0, 0, 0)
    assert cm.prefix_counters.hits == 2                       # the reuse counters are cumulative


# =========================================================================== S13: session scope
_uids = itertools.count(1000)


@pytest.fixture
def pin_clock(monkeypatch):
    """Strictly increasing ``time.monotonic`` for the pin ledger's ``last_match`` (the tree's
    own ``monotonic_ns`` stamps are already deterministic, conftest.py), so LRU-release
    assertions cannot tie."""
    monkeypatch.setattr(cache_mod, "time", SimpleNamespace(monotonic=itertools.count(1).__next__))


def _serve(cm, pool, ids, *, session=None, chunk=8, track=4, row=0):
    """One request through the manager the way the scheduler drives it:

    * admission (``prefill.py _try_allocate_one``): ``match_req``, ``lock``, reserve the 3-slot
      working set (live + 2 ping-pong), and the ``note_prompt_admitted`` the batch reports;
    * one prefill forward per ``chunk`` tokens, each tracking the deepest ``track`` boundary
      strictly inside its extend (``attention/linear.py build_fla_metadata``), then the
      per-chunk ``cache_req(finished=False)`` commit (``scheduler.py`` prefill branch);
    * the finish ``cache_req(finished=True)`` donating the live slot at ``len(ids)``.

    Returns the finished Req. page_size is 1, so page indices are token indices."""
    mr = cm.match_req(_pend(ids))
    h = mr.cuda_handle
    cm.lock(h)
    assert cm.reserve_mamba_slots(3), "the working set must seat"
    cm.note_prompt_admitted(h, len(ids), session_key=session)
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=row,
              cached_len=h.cached_len, output_len=1, uid=next(_uids),
              sampling_params=SamplingParams(), cache_handle=h, session_id=session)
    if h.cached_len:
        cm.page_table[row, :h.cached_len] = h.get_matched_indices()
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 0
    n = len(ids)
    while req.cached_len < n:
        s = req.cached_len
        e = min(n, s + chunk)
        cm.page_table[row, s:e] = cm._allocate(e - s)
        c = (e - s - 1) // track
        if c >= 1:
            req.mamba_last_track_seqlen = s + c * track
            req.mamba_next_track_idx = 1 - req.mamba_next_track_idx
        req.cached_len = e
        cm.cache_req(req, finished=False)
    cm.cache_req(req, finished=True)
    return req


def _path_nodes(cm, ids):
    """Every node on the root path the tree holds for ``ids`` (deepest first), tombstones too."""
    node, matched = cm.prefix_cache._walk(torch.tensor(ids, dtype=torch.int32))
    assert matched == len(ids)
    out = []
    while not node.is_root():
        out.append(node)
        node = node.parent
    return out


def _ends(cm, nodes):
    return sorted(cm.prefix_cache._path_len(n) for n in nodes)


# --------------------------------------------------------------------------- scope + refusal
def test_pin_prefix_scope_error_refuses_session_without_a_finite_token_budget():
    assert pin_prefix_scope_error("shared", 0) is None            # today's default stays legal
    assert pin_prefix_scope_error("shared", 65536) is None
    assert pin_prefix_scope_error("session", 262144) is None
    assert "finite --pin-prefix-max-tokens" in pin_prefix_scope_error("session", 0)
    assert "must be one of" in pin_prefix_scope_error("private", 262144)
    pool = _pool()
    assert _cm(pool).pin_prefix_scope == "shared"                  # the constructor default
    with pytest.raises(ValueError, match="finite"):
        _cm(pool, scope="session", max_tokens=0)
    with pytest.raises(ValueError, match="one of"):
        _cm(pool, scope="bogus", max_tokens=8)


# --------------------------------------------------------------------------- the producer's pin
def test_session_scope_pins_the_producers_prefix_at_finish(pin_clock):
    """20 tokens in 8-token chunks, track 4: chunk commits at 4 and 12 (the 4-token last
    chunk has no boundary strictly inside it), the finish donate at 20. The request's own
    lock is gone, the pin holds the whole path and its three snapshots."""
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_tokens=64, scope="session", num_pages=256)
    ids = list(range(1, 21))
    _serve(cm, pool, ids, session="a")
    path = _path_nodes(cm, ids)
    assert _ends(cm, [n for n in path if n.mamba_value is not None]) == [4, 12, 20]
    assert _ledger(cm) == (1, 20, 3)
    assert list(cm._pins) == [path[0]]                             # the finish node
    assert all(n.mamba_ref_count == 1 for n in path)
    assert _tree(cm).full_protected == 20 and _tree(cm).full_evictable == 0
    cm.check_integrity()

    # The growable-KV shrink (_evict_growable_prefix_pages -> evict_full) cannot take it.
    assert _tree(cm).evict_full(10 ** 6).kv_indices.numel() == 0
    # The session's next question hits the whole haystack.
    h = _admit(cm, ids, session="a")
    assert h.cached_len == 20


def test_shared_scope_ignores_the_producers_finish():
    """The default: the same request pins nothing, exactly as before S13."""
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_tokens=64, num_pages=256)
    ids = list(range(1, 21))
    _serve(cm, pool, ids, session="a")
    assert _ledger(cm) == (0, 0, 0) and cm._pins == {}
    assert _tree(cm).full_protected == 0 and _tree(cm).mamba_protected == 0
    assert _tree(cm).evict_full(10 ** 6).kv_indices.numel() == 20   # all of it evictable
    c = cm.prefix_counters.as_dict()
    assert (c["pin_evictions"], c["pin_budget_refusals"]) == (0, 0)


def test_session_scope_needs_a_session_key_and_the_min_tokens():
    pool = _pool()
    cm = _cm(pool, min_tokens=16, max_tokens=64, scope="session", num_pages=256)
    _serve(cm, pool, list(range(1, 21)), session=None)             # no key: a probe, a one-shot
    assert _ledger(cm) == (0, 0, 0)
    _serve(cm, pool, list(range(101, 113)), session="a")           # 12 < 16 tokens
    assert _ledger(cm) == (0, 0, 0)
    _serve(cm, pool, list(range(201, 221)), session="a")
    assert _ledger(cm) == (1, 20, 3)
    cm.check_integrity()


def test_session_scope_is_off_at_min_tokens_zero():
    pool = _pool()
    cm = _cm(pool, min_tokens=0, max_tokens=64, scope="session", num_pages=256)
    _serve(cm, pool, list(range(1, 21)), session="a")
    assert _ledger(cm) == (0, 0, 0)


# --------------------------------------------------------------------------- KV-only remainder
def test_a_session_pin_holds_the_whole_kv_path_but_only_the_deepest_snapshots():
    """Budget 6 (13 slots, one lane), cap 3: a 48-token path has commits at 4..44 and the
    finish at 48 -- but the pool's snapshot cache keeps only some of them. The pin holds the
    three deepest; the others are KV-only: ``evict_mamba`` tombstones them, ``evict_full``
    still cannot take their KV, and a question diverging inside the pinned tail still
    restores from a pinned snapshot."""
    pool = _pool(num_slots=13)
    cm = _cm(pool, min_tokens=4, max_tokens=64, scope="session", num_pages=256,
             working_set=PIN_WORKING_SET_SLOTS_PER_REQUEST * 1)
    assert cm.pin_slot_budget == 6 and cm._session_pin_slot_cap() == 3
    ids = list(range(1, 49))
    _serve(cm, pool, ids, session="a")
    path = _path_nodes(cm, ids)
    snaps = [n for n in path if n.mamba_value is not None]
    assert len(snaps) > 3
    assert _ledger(cm) == (1, 48, 3)
    locked = [n for n in path if n.mamba_ref_count]
    assert _ends(cm, locked) == [36, 44, 48]                       # the deepest three
    cm.ensure_mamba_slots(pool.num_slots)                          # all the pressure there is
    assert _ends(cm, [n for n in path if n.mamba_value is not None]) == [36, 44, 48]
    assert _tree(cm).evict_full(10 ** 6).kv_indices.numel() == 0
    assert _tree(cm).full_protected == 48
    h = _admit(cm, ids[:46] + [900, 901], session="a")
    assert h.cached_len == 44                                      # >= 48 - one 8-token chunk
    cm.check_integrity()


def test_a_session_pin_at_slot_budget_zero_is_refused_not_made_kv_only():
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_tokens=64, max_slots=0, scope="session", num_pages=256)
    _serve(cm, pool, list(range(1, 21)), session="a")
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pin_budget_refusals"]) == (0, 1)
    assert _tree(cm).full_protected == 0 and _tree(cm).mamba_protected == 0


def test_releasing_a_session_pin_keeps_exactly_what_the_remaining_pins_chose(pin_clock):
    """R1 = H+a (snapshots 4, 12, a20), R2 = H+b restores at 12 and commits 16 on the trunk
    (snapshots b20, 16, 12, 4). Cap 3: R2 holds b20, 16, 12 -- not 4. Releasing R1 must drop
    4 (a shared-scope pin would have kept every snapshot ancestor of R2) and keep 12."""
    pool = _pool(num_slots=13)
    cm = _cm(pool, min_tokens=4, max_tokens=64, scope="session", num_pages=256,
             working_set=PIN_WORKING_SET_SLOTS_PER_REQUEST * 1)
    trunk = list(range(1, 17))
    r1, r2 = trunk + [31, 32, 33, 34], trunk + [41, 42, 43, 44]
    _serve(cm, pool, r1, session="a")
    _serve(cm, pool, r2, session="a")
    n1, n2 = _path_nodes(cm, r1)[0], _path_nodes(cm, r2)[0]
    assert set(cm._pins) == {n1, n2}
    assert _ends(cm, cm._pins[n2].snapshots) == [12, 16, 20]
    assert _ledger(cm) == (2, 24, 5)                               # 4, 12, a20 + 16, b20
    cm._release_pin(n1)
    assert _ledger(cm) == (1, 20, 3)
    by_end = {cm.prefix_cache._path_len(n): n for n in _path_nodes(cm, r2)}
    assert by_end[12].mamba_ref_count == 1 and by_end[16].mamba_ref_count == 1
    assert by_end[4].mamba_ref_count == 0                          # KV-only now
    assert n1.mamba_ref_count == 0 and n1.ref_count == 0
    cm._release_pin(n2)
    assert _ledger(cm) == (0, 0, 0)
    assert _tree(cm).full_protected == 0 and _tree(cm).mamba_protected == 0
    cm.check_integrity()


# --------------------------------------------------------------------------- LRU + handoff
def test_session_pins_release_lru_under_pin_prefix_max_tokens(pin_clock):
    """max 45 tokens: X and Y (20 each) fit, Z does not. A hit on X refreshes it, so Z's
    pin releases Y -- the pin least recently MATCHED, not the oldest."""
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_tokens=45, scope="session", num_pages=256)
    x, y, z = list(range(1, 21)), list(range(101, 121)), list(range(201, 221))
    _serve(cm, pool, x, session="a")
    _serve(cm, pool, y, session="b")
    assert _ledger(cm)[:2] == (2, 40)
    _admit(cm, x, session="a")                                     # a re-reads its haystack
    _serve(cm, pool, z, session="c")
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pinned_tokens"]) == (2, 40)
    assert (c["pin_evictions"], c["pin_budget_refusals"]) == (1, 0)
    assert set(cm._pins) == {_path_nodes(cm, x)[0], _path_nodes(cm, z)[0]}
    assert all(n.ref_count == 0 for n in _path_nodes(cm, y))       # Y is evictable again
    # A path over the whole budget is refused, and releases nothing on the way.
    _serve(cm, pool, list(range(301, 351)), session="d")
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pin_evictions"], c["pin_budget_refusals"]) == (2, 1, 1)
    cm.check_integrity()


def test_handoff_session_b_admitted_while_a_pin_is_held(pin_clock):
    """The single-lane handoff at the production slot geometry (13 slots, one lane: budget
    6, cap 3). A pins its 40-token haystack; B is admitted and served while A's pin is
    held -- its working set seats and no snapshot donation is skipped; B's pin fits beside
    A's. A's fourth question still hits. C then needs room: the LRU release takes B, not
    A, because A's question refreshed A's pin."""
    pool = _pool(num_slots=13)
    cm = _cm(pool, min_tokens=16, max_tokens=64, scope="session", num_pages=256,
             working_set=PIN_WORKING_SET_SLOTS_PER_REQUEST * 1)
    hay = list(range(1, 41))
    _serve(cm, pool, hay, session="A")
    assert _ledger(cm) == (1, 40, 3)

    b = list(range(500, 524))
    _serve(cm, pool, b, session="B")
    assert cm._mamba_donation_skips == 0
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pinned_tokens"], c["pinned_slots"]) == (2, 64, 6)
    assert (c["pin_evictions"], c["pin_budget_refusals"]) == (0, 0)

    cm.ensure_mamba_slots(pool.num_slots)                          # every unpinned snapshot goes
    h = _admit(cm, hay[:38] + [777, 778], session="A")             # A's fourth question
    assert h.cached_len == 36                                      # >= 40 - one 8-token chunk
    assert _tree(cm).full_protected == 64

    _serve(cm, pool, list(range(600, 624)), session="C")
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pinned_tokens"], c["pinned_slots"]) == (2, 64, 6)
    assert (c["pin_evictions"], c["pin_budget_refusals"]) == (1, 0)
    assert _path_nodes(cm, hay)[0] in cm._pins                     # A survived the handoff
    assert _path_nodes(cm, b)[0] not in cm._pins
    assert _admit(cm, hay[:38] + [779], session="A").cached_len == 36
    cm.check_integrity()


# --------------------------------------------------------------------------- consequence (b)
def test_a_120k_haystack_carries_more_snapshots_than_the_single_lane_pin_budget():
    """The production numbers (scripts/serve-default.sh): --linear-state-slots 13,
    --max-running-requests 1, --max-prefill-length 8192, Nemotron-H Mamba-2 track 128,
    page_size 1. A 120,000-token prompt runs 15 prefill chunks (14 x 8192 + 5312), each
    committing one snapshot (at +8064, the last at 114688 + 5248 = 119936), and the finish
    donates one more at 120000: 16 nodes on the path. The pool (12 usable slots, 3 held by
    the request) keeps at most 9 of the chunk snapshots, so 10 snapshot-bearing nodes remain
    -- against a pin slot budget of 13 - 4 - 1 - 2 = 6. A shared-scope pin of that path is
    refused whole; the session-scope pin holds all 120,000 tokens of KV and the three
    deepest snapshots, and a question diverging at the haystack's end restores within one
    chunk of it."""
    tokens, chunk, track = 120_000, 8192, 128
    # The 25% clamp (PIN_BUDGET_MAX_FRACTION) takes 262,144 down to 121,024 here, still
    # above the path; production's 1M-token pool leaves 262,144 unclamped.
    num_pages = 4 * tokens + 4096
    ids = list(range(1, tokens + 1))

    def run(scope):
        pool = _pool(num_slots=13)
        cm = _cm(pool, min_tokens=1024, max_tokens=262_144, scope=scope,
                 num_pages=num_pages, rows=1,
                 working_set=PIN_WORKING_SET_SLOTS_PER_REQUEST * 1)
        _serve(cm, pool, ids, session="A", chunk=chunk, track=track)
        return pool, cm

    pool, cm = run("shared")
    assert cm.pin_slot_budget == 6
    path = _path_nodes(cm, ids)
    snaps = [n for n in path if n.mamba_value is not None]
    assert len(path) == 16 and len(snaps) == 10
    assert cm._mamba_donation_skips == 0
    assert cm.pin_prefix(path[0]) is False                         # 10 slots > budget 6
    assert cm.prefix_counters.pin_budget_refusals == 1 and _ledger(cm) == (0, 0, 0)

    pool, cm = run("session")
    path = _path_nodes(cm, ids)
    assert _ledger(cm) == (1, tokens, 3)
    assert cm.prefix_counters.pin_budget_refusals == 0
    assert _ends(cm, [n for n in path if n.mamba_ref_count]) == [114_560, 119_936, 120_000]
    cm.ensure_mamba_slots(pool.num_slots)
    assert _tree(cm).evict_full(10 ** 9).kv_indices.numel() == 0
    h = _admit(cm, ids[:119_990] + [0], session="A")
    assert h.cached_len == 119_936 and h.cached_len >= tokens - chunk
    cm.check_integrity()
