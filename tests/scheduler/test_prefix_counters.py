"""``/v1/stats["scheduler"]["prefix"]``: prefix-cache hit/miss accounting and the pin ledger.

The counters are noted at the one point the admission loop knows a prompt's final
``cached_len`` (where ``PromptAdmittedMsg`` is built), so these pin the arithmetic and
the wire document without a tree: ``PrefixCounters`` is torch-free, and
``build_scheduler_counters`` reads it off a duck-typed manager. The tree-level trigger,
eviction and unpin behaviour is in ``tests/kvcache/radix/test_hybrid_radix_pins.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from freetoken.scheduler.counters import PrefixCounters, build_scheduler_counters


def test_a_hit_is_cached_len_positive_and_a_miss_charges_the_whole_prompt():
    c = PrefixCounters()
    c.note_admitted(prompt_tokens=100, cached_len=64)     # hit: 64 reused, 36 forwarded
    c.note_admitted(prompt_tokens=50, cached_len=0)       # miss: all 50 forwarded
    c.note_admitted(prompt_tokens=7, cached_len=6)        # a re-run: input_len - 1 is a hit
    assert (c.hits, c.misses) == (2, 1)
    assert (c.hit_tokens, c.miss_tokens) == (70, 50)


def test_the_document_has_every_field_and_the_pin_gauges_start_at_zero():
    doc = PrefixCounters().as_dict()
    assert doc == {
        "hits": 0, "misses": 0, "hit_tokens": 0, "miss_tokens": 0,
        "pooled_hits": 0, "pooled_hit_tokens": 0,
        "pinned_prefixes": 0, "pinned_tokens": 0, "pinned_slots": 0,
        "pin_evictions": 0, "pin_budget_refusals": 0,
    }


def test_pooled_hits_are_a_subset_of_hits():
    """A pooled probe's hit counts in both ledgers; its miss and a plain hit in neither
    pooled field."""
    c = PrefixCounters()
    c.note_admitted(prompt_tokens=300, cached_len=256, pooled=True)
    c.note_admitted(prompt_tokens=300, cached_len=0, pooled=True)      # pooled miss
    c.note_admitted(prompt_tokens=100, cached_len=64)                  # plain hit
    assert (c.hits, c.hit_tokens) == (2, 320)
    assert (c.pooled_hits, c.pooled_hit_tokens) == (1, 256)
    assert (c.misses, c.miss_tokens) == (1, 300)
    doc = c.as_dict()
    assert doc["pooled_hits"] == 1 and doc["pooled_hit_tokens"] == 256


def test_build_scheduler_counters_reads_the_manager_and_distinguishes_absent():
    doc = build_scheduler_counters(None, None, None)
    assert doc["prefix"] is None                          # no manager: off, not idle

    counters = PrefixCounters(hits=3, hit_tokens=3000, pinned_prefixes=1, pinned_tokens=2048,
                              pinned_slots=1, pin_evictions=2)
    doc = build_scheduler_counters(cache_manager=SimpleNamespace(prefix_counters=counters))
    assert doc["prefix"]["hits"] == 3 and doc["prefix"]["pinned_tokens"] == 2048
    assert doc["prefix"]["pinned_slots"] == 1 and doc["prefix"]["pin_evictions"] == 2
    assert json.loads(json.dumps(doc)) == doc
