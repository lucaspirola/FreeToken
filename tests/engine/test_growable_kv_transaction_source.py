"""Offline fault injection for the growable KV/MoE ownership transaction.

The production module imports Torch and model kernels, so these tests extract only the
three methods under test and run them against tiny Python stubs.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ENGINE = Path(__file__).parents[2] / "python/freetoken/engine/engine.py"


def _methods(*names: str):
    tree = ast.parse(ENGINE.read_text())
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engine")
    selected = []
    for node in engine.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    from freetoken.engine.engine import _arena_chunk_boundaries

    ns = {
        "math": math,
        "mem_GB": lambda n: str(n),
        "logger": SimpleNamespace(info_rank0=lambda *a, **k: None, exception=lambda *a, **k: None),
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda *_: None,
                memory_allocated=lambda *_: 0,
                memory_reserved=lambda *_: 0,
            )
        ),
        # Module-level helper (not a class method, so the ClassDef-only extraction
        # above never sees it); _shrink_runtime_kv_arena calls it by bare name.
        "_arena_chunk_boundaries": _arena_chunk_boundaries,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ENGINE), "exec"), ns)
    return SimpleNamespace(**{name: ns[name] for name in names})


class _Pool:
    def __init__(self, pages: int, *, fail_commit: bool = False):
        self.committed_pages = pages
        self.fail_commit = fail_commit

    def mapped_bytes_for_pages(self, pages):
        return pages

    def commit_pages(self, pages):
        if self.fail_commit:
            self.fail_commit = False
            raise MemoryError("injected VMM commit failure")
        self.committed_pages = pages

    def decommit_pages(self, pages):
        self.committed_pages = pages


class _Moe:
    num_experts = 2
    # None off the FREETOKEN_EXPERT_ARENA gate -- the real OffloadMoeCache's
    # arena_layout/bank_row_bytes properties return None there too (see
    # offload_cache.py), which is exactly what routes grow_runtime_kv/
    # shrink_runtime_kv into the legacy rebuild path these tests exercise.
    arena_layout = None
    bank_row_bytes = None

    def __init__(self, size: int, *, fail_size: int | None = None):
        self.cache_size = size
        self.prefill_overlap = True
        self.fail_size = fail_size
        self.rebuilds = []
        self.usable_calls: list[int] = []
        self.cpu_layer_ids = frozenset({0, 4})
        self.cpu_executor = object()
        self.bank_sources = {"gate": [object(), object()]}

    def rebuild(self, size):
        self.rebuilds.append(size)
        self.cache_size = size  # model destructive rebuild's early geometry update
        if size == self.fail_size:
            self.fail_size = None
            raise MemoryError("injected expert allocation failure")

    def set_usable_slots(self, size):
        raise AssertionError(
            "legacy (non-arena) MoE cache must never take the set_usable_slots path"
        )


class _ArenaMoe(_Moe):
    """Same fault-injection surface as ``_Moe``, but with an expert arena active
    (``arena_layout``/``bank_row_bytes`` set), so grow_runtime_kv/shrink_runtime_kv
    take the set_usable_slots branch (design step 5) instead of rebuild."""

    def __init__(
        self,
        size: int,
        *,
        capacity: int,
        step: int,
        row_bytes: int = 2 * 1024 * 1024,
        fail_target: int | None = None,
    ):
        super().__init__(size)
        self.arena_layout = (capacity, step)
        self.bank_row_bytes = [row_bytes]
        self.fail_target = fail_target

    def set_usable_slots(self, size):
        self.usable_calls.append(size)
        if size == self.fail_target:
            self.fail_target = None
            raise MemoryError("injected arena resize failure")
        self.cache_size = size
        return 0

    def rebuild(self, size):  # pragma: no cover - must never be called
        raise AssertionError("expert-arena grow/shrink must never call rebuild")


def _engine(pool, moe):
    methods = _methods(
        "_rollback_growable_kv_transition",
        "_refuse_if_growable_transition_failed",
        "_grow_runtime_kv_arena",
        "_shrink_runtime_kv_arena",
        "grow_runtime_kv",
        "shrink_runtime_kv",
        "forward_batch",
    )
    obj = SimpleNamespace(
        kv_cache=pool,
        moe_offload_cache=moe,
        num_pages=32,
        device="cuda",
        config=SimpleNamespace(
            kv_grow_step_tokens=8,
            page_size=1,
            moe_cache_size=moe.cache_size,
            tp_info=SimpleNamespace(size=1),
        ),
        _growable_moe_prefill_overlap=True,
        _pending_graph_bs=None,
        graph_runner=SimpleNamespace(
            graph_bs_list=[1], destroy_cuda_graphs=lambda: None
        ),
        attn_backend=SimpleNamespace(reset_capture=lambda: None),
        _plan_growable_kv=lambda pages, **kw: (4 if pages > 8 else 8, pages),
        _growable_moe_bytes=lambda size: size,
        _sync_get_memory=lambda: (10**9, 10**9),
        ensure_decode_graphs=lambda: setattr(obj, "_pending_graph_bs", None),
    )
    obj._rollback_growable_kv_transition = methods._rollback_growable_kv_transition.__get__(obj)
    obj._refuse_if_growable_transition_failed = methods._refuse_if_growable_transition_failed.__get__(obj)
    obj._grow_runtime_kv_arena = methods._grow_runtime_kv_arena.__get__(obj)
    obj._shrink_runtime_kv_arena = methods._shrink_runtime_kv_arena.__get__(obj)
    return obj, methods


def test_growth_commit_failure_restores_experts_config_and_graph_state():
    obj, methods = _engine(_Pool(8, fail_commit=True), _Moe(8))
    executor = obj.moe_offload_cache.cpu_executor
    banks = obj.moe_offload_cache.bank_sources

    with pytest.raises(MemoryError, match="VMM commit"):
        methods.grow_runtime_kv(obj, 16)

    assert obj.kv_cache.committed_pages == 8
    assert obj.moe_offload_cache.cache_size == 8
    assert obj.config.moe_cache_size == 8
    assert obj._pending_graph_bs is None
    assert obj.moe_offload_cache.rebuilds == [4, 8]
    assert obj.moe_offload_cache.cpu_layer_ids == frozenset({0, 4})
    assert obj.moe_offload_cache.cpu_executor is executor
    assert obj.moe_offload_cache.bank_sources is banks


def test_shrink_expert_failure_recommits_kv_before_propagating():
    obj, methods = _engine(_Pool(16), _Moe(4, fail_size=8))

    with pytest.raises(MemoryError, match="expert allocation"):
        methods.shrink_runtime_kv(obj, 8)

    assert obj.kv_cache.committed_pages == 16
    assert obj.moe_offload_cache.cache_size == 4
    assert obj.config.moe_cache_size == 4
    assert obj._pending_graph_bs is None
    assert obj.moe_offload_cache.rebuilds == [8, 4]


def test_post_graph_planning_failure_restores_graph_readiness():
    obj, methods = _engine(_Pool(8), _Moe(8))
    graph_restores = []

    def fail_memory_probe():
        raise RuntimeError("injected post-teardown probe failure")

    obj._sync_get_memory = fail_memory_probe
    obj.ensure_decode_graphs = lambda: (
        graph_restores.append(tuple(obj._pending_graph_bs)),
        setattr(obj, "_pending_graph_bs", None),
    )
    with pytest.raises(RuntimeError, match="post-teardown probe"):
        methods.grow_runtime_kv(obj, 16)

    assert graph_restores == [(1,)]
    assert obj._pending_graph_bs is None
    assert obj.kv_cache.committed_pages == 8
    assert obj.moe_offload_cache.cache_size == 8


def test_shrink_partial_graph_teardown_failure_restores_graph_readiness():
    obj, methods = _engine(_Pool(16), _Moe(4))
    restores = []

    def fail_reset():
        raise RuntimeError("injected graph reset failure")

    obj.attn_backend.reset_capture = fail_reset
    obj.ensure_decode_graphs = lambda: (
        restores.append(tuple(obj._pending_graph_bs)),
        setattr(obj, "_pending_graph_bs", None),
    )
    with pytest.raises(RuntimeError, match="graph reset"):
        methods.shrink_runtime_kv(obj, 8)

    assert restores == [(1,)]
    assert obj._pending_graph_bs is None
    assert obj.kv_cache.committed_pages == 16
    assert obj.moe_offload_cache.cache_size == 4


def test_explicit_and_implicit_cpu_splits_resolve_without_changing_cache_ownership():
    tree = ast.parse(ENGINE.read_text())
    funcs = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_parse_cpu_layers_spec", "_resolve_cpu_layers"}
    ]
    ns = {"is_offload_moe_backend": lambda name: name in {"offload", "hybrid"}}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(ENGINE), "exec"), ns)

    explicit = ns["_resolve_cpu_layers"](
        SimpleNamespace(moe_backend="offload", moe_cpu_layers="2"), 6
    )
    implicit_auto = ns["_resolve_cpu_layers"](
        SimpleNamespace(moe_backend="offload", moe_cpu_layers=None), 6
    )
    assert explicit == frozenset({0, 3})
    # Pin-budget auto selection happens after this resolver and starts from the empty set.
    assert implicit_auto == frozenset()


def test_failed_rollback_poison_refuses_forward_before_model_execution():
    pool = _Pool(8, fail_commit=True)
    obj, methods = _engine(pool, _Moe(4))
    obj._growable_transition_failed = False

    with pytest.raises(MemoryError, match="VMM commit"):
        obj._rollback_growable_kv_transition(
            old_pages=16, old_moe=4, old_overlap=True, recapture_graphs=False
        )
    assert obj._growable_transition_failed is True

    called = []
    obj.model = SimpleNamespace(forward=lambda: called.append(True))
    with pytest.raises(RuntimeError, match="engine restart is required"):
        methods.forward_batch(obj, object(), object())
    assert called == []


MIB = 1024 * 1024


def test_arena_grow_funds_kv_via_set_usable_slots_never_rebuild():
    """Design step 5: with an expert arena active, grow_runtime_kv must resize the
    MoE cache via ``set_usable_slots`` (a chunk-boundary usable count) instead of
    ``rebuild``, and must never touch decode-graph recapture state."""
    capacity, step = 1024, 64
    moe = _ArenaMoe(capacity, capacity=capacity, step=step)
    pool = _Pool(8)
    pool.mapped_bytes_for_pages = lambda pages: pages * 2 * MIB
    obj, methods = _engine(pool, moe)
    obj._plan_growable_kv = lambda pages, **kw: (0, pages * 2 * MIB)

    def free_probe():
        # Free VRAM grows exactly as much as the arena releases, so the live-memory
        # guard observes a real before/after delta across set_usable_slots.
        freed = (capacity - moe.cache_size) * 2 * MIB
        return freed, freed

    obj._sync_get_memory = free_probe

    old_pages, new_pages = methods.grow_runtime_kv(obj, 16)

    assert (old_pages, new_pages) == (8, 16)
    assert obj.kv_cache.committed_pages == 16
    assert moe.rebuilds == []
    assert len(moe.usable_calls) == 1
    target = moe.usable_calls[0]
    assert target % step == 0  # a real arena chunk boundary
    assert target >= 2 * moe.num_experts  # the prefill-overlap floor
    assert target < capacity
    freed_bytes = (capacity - target) * 2 * MIB
    assert freed_bytes >= 256 * MIB  # covers at least the fixed VMM reserve
    assert obj._pending_graph_bs is None


def test_arena_shrink_regrows_experts_via_set_usable_slots_never_rebuild():
    """Design step 5: shrink_runtime_kv must regrow experts with
    ``set_usable_slots`` up to the largest boundary the released KV bytes fund,
    never ``rebuild``, and never touch decode-graph recapture state."""
    capacity, step = 1024, 64
    moe = _ArenaMoe(768, capacity=capacity, step=step)
    pool = _Pool(144)
    pool.mapped_bytes_for_pages = lambda pages: pages * 2 * MIB
    obj, methods = _engine(pool, moe)
    obj._plan_growable_kv = lambda pages, **kw: (0, pages * 2 * MIB)

    old_pages, new_pages = methods.shrink_runtime_kv(obj, 16)

    assert (old_pages, new_pages) == (144, 16)
    assert obj.kv_cache.committed_pages == 16
    assert moe.rebuilds == []
    assert len(moe.usable_calls) == 1
    target = moe.usable_calls[0]
    assert target % step == 0
    assert target > 768
    assert target <= capacity
    grown_bytes = (target - 768) * 2 * MIB
    released_bytes = (144 - 16) * 2 * MIB
    assert grown_bytes <= released_bytes
    assert obj._pending_graph_bs is None


def test_arena_grow_rollback_regrows_experts_on_failed_commit():
    """The transaction shape (shrink experts, commit KV, rollback on failure) is
    unchanged under the arena branch -- only the mechanism (set_usable_slots, not
    rebuild) differs."""
    capacity, step = 1024, 64
    moe = _ArenaMoe(capacity, capacity=capacity, step=step)
    pool = _Pool(8, fail_commit=True)
    pool.mapped_bytes_for_pages = lambda pages: pages * 2 * MIB
    obj, methods = _engine(pool, moe)
    obj._plan_growable_kv = lambda pages, **kw: (0, pages * 2 * MIB)

    def free_probe():
        freed = (capacity - moe.cache_size) * 2 * MIB
        return freed, freed

    obj._sync_get_memory = free_probe

    with pytest.raises(MemoryError, match="VMM commit"):
        methods.grow_runtime_kv(obj, 16)

    assert moe.rebuilds == []
    assert moe.usable_calls[-1] == capacity  # rolled back to the original usable count
    assert moe.cache_size == capacity
    assert obj.kv_cache.committed_pages == 8
    assert obj.config.moe_cache_size == capacity
    assert obj._pending_graph_bs is None
    assert getattr(obj, "_growable_transition_failed", False) is False
