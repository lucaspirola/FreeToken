"""Prefix auto-pin on the hybrid radix cache (``CacheManager.pin_prefix`` / ``unpin_all``).

A pin is nothing the tree does not already know: an extra ``inc_lock`` on the matched node
(full ref node..root, mamba ref on the node) and on every snapshot-bearing ancestor, held by
the cache manager instead of a request. So what is pinned here is exactly what the existing
lock invariants protect -- ``evict_full`` cannot take a locked leaf, ``evict_mamba`` cannot
tombstone a locked snapshot -- and ``unpin_all`` is the matching ``dec_lock``s. The tests
drive the tree through the manager with a real ``LinearStatePool`` on CPU (page_size 1),
the way ``tests/scheduler/test_hybrid_cache_manager.py`` does, and check the tree's own
evictable/protected ledgers plus the ``/v1/stats`` pin gauges.

Trigger: ``note_prompt_admitted`` fires on the first chunk of an admitted prompt, where a
``cached_len > 0`` is by construction a *second* request on the path (the first one donated
it); ``cached_len >= pin_prefix_min_tokens`` pins.
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


def _cm(pool, min_tokens=4, max_tokens=0, num_pages=64):
    page_table = torch.zeros(4, num_pages, dtype=torch.int32)
    return CacheManager(num_pages, 1, page_table, "hybrid_radix", linear_state_pool=pool,
                        pin_prefix_min_tokens=min_tokens, pin_prefix_max_tokens=max_tokens)


def _insert(cm, pool, ids):
    """Donate ``ids`` with a fresh snapshot, the way a finished request would: the pages
    move from the free-list to the tree (so ``check_integrity`` balances)."""
    n = len(ids)
    pages, cm.free_slots = cm.free_slots[:n].clone(), cm.free_slots[n:]
    slot = pool.alloc(1)[0]
    _, exists = cm.prefix_cache.insert(torch.tensor(ids, dtype=torch.int32), pages, slot)
    assert not exists
    return slot


def _admit(cm, ids):
    """A second request matching ``ids``: what the admission loop reports at the
    PromptAdmittedMsg point. Returns the handle (still locked, like a live request)."""
    mr = cm.match_req(_pend(ids + [999]))                    # the last token is never matched
    cm.lock(mr.cuda_handle)
    cm.note_prompt_admitted(mr.cuda_handle, len(ids) + 1)
    return mr.cuda_handle


def _tree(cm):
    return cm.prefix_cache


# --------------------------------------------------------------------------- trigger
def test_a_second_request_at_or_above_the_threshold_pins_and_a_shorter_one_does_not():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3])                          # 3 tokens: below threshold
    _insert(cm, pool, [1, 2, 3, 4, 5])                    # 5 tokens

    h = _admit(cm, [1, 2, 3])
    assert cm.prefix_counters.pinned_prefixes == 0 and cm._pin_locks == []
    cm.unlock(h)

    h = _admit(cm, [1, 2, 3, 4, 5])
    assert cm.prefix_counters.pinned_prefixes == 1
    assert cm.prefix_counters.pinned_tokens == 5
    assert cm.prefix_counters.as_dict()["hits"] == 2
    cm.unlock(h)                                              # the request goes away...
    node = _tree(cm).match_prefix(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)).node
    assert node.ref_count >= 1 and node.mamba_ref_count == 1  # ...the pin stays
    # the ancestor [1,2,3] carries a snapshot too, so it is locked as a restore point
    anc = node.parent
    assert anc.mamba_value is not None and anc.mamba_ref_count == 1


def test_pinning_is_off_at_min_tokens_zero():
    pool = _pool()
    cm = _cm(pool, min_tokens=0)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    h = _admit(cm, [1, 2, 3, 4, 5])
    cm.unlock(h)
    assert cm.prefix_counters.pinned_prefixes == 0
    assert _tree(cm).full_protected == 0 and _tree(cm).mamba_protected == 0


# --------------------------------------------------------------------------- eviction
def test_a_pinned_path_survives_evict_full():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [7, 8, 9, 10, 11])                 # the unpinned victim
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5]))

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
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5]))

    er = _tree(cm).evict_mamba(2)
    assert er.mamba_slots == [s_victim]
    assert _tree(cm).mamba_protected == 1 and _tree(cm).mamba_evictable == 0
    m = _tree(cm).match_prefix(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32))
    assert m.mamba_value == s_pinned


def test_ensure_mamba_slots_cannot_reach_a_pin_and_fails_as_today():
    """A short pool is refused, not unpinned: the pin is invisible to reserve_mamba_slots
    exactly like a session lease, and the one-time warning names the pins."""
    pool = _pool(num_slots=3)                                 # padding + 2 usable
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])                        # one usable slot in the tree
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5]))
    assert pool.num_free_slots == 1
    assert cm._pin_starvation_logged is False
    assert cm.reserve_mamba_slots(2) is False
    assert cm.prefix_counters.pinned_prefixes == 1            # still pinned
    assert cm._pin_starvation_logged is True                  # logged once; never auto-unpinned
    assert cm.reserve_mamba_slots(2) is False


# --------------------------------------------------------------------------- ledger / budget
def test_re_entrant_pins_count_a_shared_prefix_once():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [1, 2, 3, 4, 5, 6, 7])              # extends the same path by 2
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5]))
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5]))                    # the same prefix again
    assert (cm.prefix_counters.pinned_prefixes, cm.prefix_counters.pinned_tokens) == (1, 5)
    node = _tree(cm).match_prefix(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)).node
    assert node.mamba_ref_count == 1                          # pinned once, not twice

    cm.unlock(_admit(cm, [1, 2, 3, 4, 5, 6, 7]))              # the longer path: +2 tokens
    assert (cm.prefix_counters.pinned_prefixes, cm.prefix_counters.pinned_tokens) == (2, 7)
    assert _tree(cm).full_protected == 7


def test_the_budget_refuses_a_pin_that_would_exceed_it_and_counts_the_refusal():
    pool = _pool()
    cm = _cm(pool, min_tokens=4, max_tokens=8)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [7, 8, 9, 10, 11])
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5]))                    # 5 of 8
    h = _admit(cm, [7, 8, 9, 10, 11])                         # 5 more would be 10 > 8
    cm.unlock(h)
    c = cm.prefix_counters.as_dict()
    assert (c["pinned_prefixes"], c["pinned_tokens"], c["pin_budget_refusals"]) == (1, 5, 1)
    assert c["hits"] == 2                                     # the hit itself still counts
    assert _tree(cm).full_protected == 5                      # nothing partially applied
    node = _tree(cm).match_prefix(torch.tensor([7, 8, 9, 10, 11], dtype=torch.int32)).node
    assert node.ref_count == 0 and node.mamba_ref_count == 0


def test_a_split_below_a_pinned_node_keeps_the_pin_and_the_ledger_exact():
    """``split_at`` copies ref_count to the new root-side half, so the pin follows the KV;
    a later pin through the split half must not count the already-pinned tokens twice."""
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5, 6])
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5, 6]))
    _insert(cm, pool, [1, 2, 3, 4, 8, 9])                 # splits [1..6] at 4
    assert _tree(cm).full_protected == 6                      # both halves still locked
    cm.unlock(_admit(cm, [1, 2, 3, 4, 8, 9]))
    assert (cm.prefix_counters.pinned_prefixes, cm.prefix_counters.pinned_tokens) == (2, 8)
    assert _tree(cm).full_protected == 8
    cm.check_integrity()


# --------------------------------------------------------------------------- unpin
def test_unpin_all_releases_every_lock_and_reports_what_it_held():
    pool = _pool()
    cm = _cm(pool, min_tokens=4)
    _insert(cm, pool, [1, 2, 3, 4, 5])
    _insert(cm, pool, [7, 8, 9, 10, 11])
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5]))
    cm.unlock(_admit(cm, [7, 8, 9, 10, 11]))
    assert _tree(cm).full_protected == 10 and _tree(cm).mamba_protected == 2

    assert cm.unpin_all() == {"pinned_prefixes": 2, "pinned_tokens": 10}
    assert _tree(cm).full_protected == 0 and _tree(cm).mamba_protected == 0
    assert _tree(cm).full_evictable == 10 and _tree(cm).mamba_evictable == 2
    assert cm.prefix_counters.pinned_prefixes == 0 and cm.prefix_counters.pinned_tokens == 0
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
    cm.unlock(_admit(cm, [1, 2, 3, 4, 5]))
    cm.rebuild(64, torch.zeros(4, 64, dtype=torch.int32))
    assert cm._pin_locks == [] and cm.prefix_counters.pinned_prefixes == 0
    assert cm.prefix_counters.hits == 1                       # the reuse counters are cumulative
