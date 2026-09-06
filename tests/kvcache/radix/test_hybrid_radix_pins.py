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
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


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


def _cm(pool, min_tokens=4, max_tokens=0, max_slots=-1, working_set=0, num_pages=64):
    page_table = torch.zeros(4, num_pages, dtype=torch.int32)
    return CacheManager(num_pages, 1, page_table, "hybrid_radix", linear_state_pool=pool,
                        pin_prefix_min_tokens=min_tokens, pin_prefix_max_tokens=max_tokens,
                        pin_prefix_max_slots=max_slots, pin_working_set_slots=working_set)


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


def test_enforce_pin_budget_releases_pins_a_smaller_pool_cannot_carry():
    """An elastic shrink re-derives the working set and enforces against the target pool."""
    pool = _pool(num_slots=16)
    cm = _cm(pool, min_tokens=4, working_set=4)              # budget 16 - 4 - 3 = 9
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [7, 8, 9, 10, 11])
    _admit(cm, [1, 2, 3, 4, 5], session="a")
    _admit(cm, [1, 2, 3, 4, 5], session="b")
    _admit(cm, [7, 8, 9, 10, 11], session="a")
    _admit(cm, [7, 8, 9, 10, 11], session="b")
    assert _ledger(cm) == (2, 10, 2)
    assert cm.enforce_pin_budget() == 0                        # within the live budget
    # Target tier: 1 request (4 slots) + 4 cache + padding = 9 slots -> budget 4 - 2 = 2.
    assert cm.enforce_pin_budget(pool_slots=9) == 0
    # Target 8 slots (cache 3) -> budget 1: the older pin goes.
    assert cm.enforce_pin_budget(pool_slots=8) == 1
    assert _ledger(cm) == (1, 5, 1) and cm.prefix_counters.pin_evictions == 1
    assert _node(cm, [7, 8, 9, 10, 11]).mamba_ref_count == 1
    cm.check_integrity()


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
