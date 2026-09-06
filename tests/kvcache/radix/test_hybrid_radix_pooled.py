"""Pooled hidden-state sums on the hybrid radix cache (``RadixTreeNode.pooled_sums``).

A pooled probe (``kv_transfer_params.pooling``) may resume from a GDN snapshot only if
the tree also holds the residual-stream sum over every position before that boundary:
``insert(..., pooled_sums=)`` attaches it, ``match_prefix(pooled=True)`` requires it,
and the sums live and die with the snapshot (eviction, tombstone, split, lock). Direct
tree tests at page_size 1 / chunk 4; the cache-manager wiring is in
``tests/server/test_hidden_states_pooled.py``.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache

LAYERS, HIDDEN = 3, 4


def ids(*tokens: int) -> torch.Tensor:
    return torch.tensor(tokens, dtype=torch.int32)


def pages(start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n, dtype=torch.int32)


def sums(fill: float) -> torch.Tensor:
    return torch.full((LAYERS, HIDDEN), fill, dtype=torch.float32)


@pytest.fixture
def tree() -> HybridRadixCache:
    return HybridRadixCache(torch.device("cpu"), page_size=1, track_chunk_size=4)


def test_insert_attaches_sums_and_only_a_pooled_match_needs_them(tree):
    matched, exist = tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7, pooled_sums=sums(1.0))
    assert (matched, exist) == (0, False)
    node = tree.match_prefix(ids(1, 2, 3, 4)).node
    assert node.mamba_value == 7 and node.pooled_count == 4
    torch.testing.assert_close(node.pooled_sums, sums(1.0))

    plain = tree.match_prefix(ids(1, 2, 3, 4, 5))
    pooled = tree.match_prefix(ids(1, 2, 3, 4, 5), pooled=True)
    assert (plain.cached_len, plain.mamba_value) == (4, 7)
    assert (pooled.cached_len, pooled.mamba_value, pooled.node) == (4, 7, node)
    tree.check_integrity()


def test_a_bare_snapshot_is_no_reuse_point_for_a_pooled_match(tree):
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7)                            # plain producer
    assert tree.match_prefix(ids(1, 2, 3, 4, 5)).cached_len == 4
    m = tree.match_prefix(ids(1, 2, 3, 4, 5), pooled=True)
    assert (m.cached_len, m.mamba_value, m.node.is_root()) == (0, None, True)


def test_pooled_match_truncates_to_the_deepest_node_with_sums(tree):
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7, pooled_sums=sums(1.0))
    tree.insert(ids(1, 2, 3, 4, 5, 6, 7, 8), pages(10, 8), 8)                 # deeper, bare
    assert tree.match_prefix(ids(1, 2, 3, 4, 5, 6, 7, 8, 9)).cached_len == 8
    m = tree.match_prefix(ids(1, 2, 3, 4, 5, 6, 7, 8, 9), pooled=True)
    assert (m.cached_len, m.mamba_value) == (4, 7)
    torch.testing.assert_close(m.node.pooled_sums, sums(1.0))
    tree.check_integrity()


def test_a_pooled_recompute_adds_sums_to_an_existing_bare_snapshot(tree):
    """Dedup keeps the tree's slot, but the recomputed sums make the node pooled-reusable."""
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7)
    matched, exist = tree.insert(ids(1, 2, 3, 4), pages(20, 4), 9, pooled_sums=sums(2.0))
    assert (matched, exist) == (4, True)                                     # caller frees 9
    node = tree.match_prefix(ids(1, 2, 3, 4), pooled=True).node
    assert node.mamba_value == 7 and node.pooled_count == 4
    torch.testing.assert_close(node.pooled_sums, sums(2.0))
    # Existing sums are kept, not overwritten (they describe the same positions).
    tree.insert(ids(1, 2, 3, 4), pages(30, 4), 11, pooled_sums=sums(3.0))
    torch.testing.assert_close(node.pooled_sums, sums(2.0))


def test_a_plain_producer_never_attaches_sums(tree):
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7, pooled_sums=None)
    node = tree.match_prefix(ids(1, 2, 3, 4)).node
    assert node.pooled_sums is None and node.pooled_count == 0


def test_tombstoning_the_snapshot_drops_the_sums(tree):
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7, pooled_sums=sums(1.0))
    tree.insert(ids(1, 2, 3, 4, 5, 6, 7, 8), pages(10, 8), 8, pooled_sums=sums(2.0))
    # LRU over unlocked snapshot nodes: the internal node (older) is tombstoned first.
    result = tree.evict_mamba(1)
    assert result.mamba_slots == [7] and result.kv_indices.numel() == 0
    parent = tree.match_prefix(ids(1, 2, 3, 4, 5, 6, 7, 8), pooled=True).node.parent
    assert parent.mamba_value is None and parent.pooled_sums is None and parent.pooled_count == 0
    assert tree.match_prefix(ids(1, 2, 3, 4, 9), pooled=True).cached_len == 0
    tree.check_integrity()
    # Refilling the tombstone with a bare snapshot does not resurrect the old sums.
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 12)
    assert parent.mamba_value == 12 and parent.pooled_sums is None


def test_evict_full_takes_the_sums_with_the_leaf(tree):
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7, pooled_sums=sums(1.0))
    node = tree.match_prefix(ids(1, 2, 3, 4)).node
    result = tree.evict_full(4)
    assert result.mamba_slots == [7] and result.kv_indices.tolist() == [10, 11, 12, 13]
    assert node.pooled_sums is None
    assert tree.match_prefix(ids(1, 2, 3, 4), pooled=True).cached_len == 0


def test_split_leaves_the_sums_on_the_suffix_half(tree):
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7, pooled_sums=sums(1.0))
    m = tree.match_prefix(ids(1, 2, 9), pooled=True)                        # splits at 2
    assert m.cached_len == 0
    suffix = tree.match_prefix(ids(1, 2, 3, 4), pooled=True).node
    assert suffix.length == 2 and suffix.pooled_count == 4
    torch.testing.assert_close(suffix.pooled_sums, sums(1.0))
    assert suffix.parent.pooled_sums is None and suffix.parent.mamba_value is None
    tree.check_integrity()


def test_a_locked_node_keeps_its_sums_under_snapshot_pressure(tree):
    """The pin path (``CacheManager.pin_prefix``) is ``inc_lock``: a pinned node's
    snapshot -- and so its sums -- survive ``evict_mamba``."""
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7, pooled_sums=sums(1.0))
    node = tree.match_prefix(ids(1, 2, 3, 4)).node
    tree.inc_lock(node)
    assert tree.evict_mamba(1).mamba_slots == []
    assert node.pooled_sums is not None
    tree.dec_lock(node)
    assert tree.evict_mamba(1).mamba_slots == [7]
    assert node.pooled_sums is None


def test_integrity_rejects_sums_without_a_snapshot_or_at_the_wrong_length(tree):
    tree.insert(ids(1, 2, 3, 4), pages(10, 4), 7, pooled_sums=sums(1.0))
    node = tree.match_prefix(ids(1, 2, 3, 4)).node
    node.pooled_count = 3
    with pytest.raises(AssertionError):
        tree.check_integrity()
    node.pooled_count = 4
    node.mamba_value = None
    with pytest.raises(AssertionError):
        tree.check_integrity()
