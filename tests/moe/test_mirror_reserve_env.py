"""FREETOKEN_MIRROR_RESERVE_ROWS: the pool's reserve as a swept knob.

``default_reserve_rows`` (3 * num_experts) was chosen before retention
existed and is likely oversized; lever 4 makes it a knob so 3E / 2E / E can
be swept. The one way this knob can quietly corrupt a run is for the
capacity planner (``plan_capacity(reserve=...)``) and the pool itself
(``MirrorExpertPool(reserve_rows=...)``) to disagree about the resolved
value -- one arm's floor priced against a reserve the pool does not
actually withhold. ``resolve_reserve_rows`` is the single place the env var
is read; these tests pin its parsing contract and then prove, with real
numbers, what happens when a caller forgets to plumb its result to both
sites.
"""
from __future__ import annotations

import re
import types

import pytest
import torch

from freetoken.models.nemotron_h.weight import (
    NVFP4_EXPERT_SOURCE_SPEC as NEMOTRON_SPEC,
)
from freetoken.moe.mirror_pool import (
    MirrorExpertPool,
    default_reserve_rows,
    plan_capacity,
    resolve_reserve_rows,
)

from tests.moe._mirror_checkpoint import write_nvfp4_checkpoint

ENV = "FREETOKEN_MIRROR_RESERVE_ROWS"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mirror pool needs CUDA for the swap kernels"
)


# --------------------------------------------------------------------------
# resolve_reserve_rows: parsing contract
# --------------------------------------------------------------------------

def test_unset_matches_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert resolve_reserve_rows(128) == default_reserve_rows(128) == 3 * 128


def test_empty_string_matches_default(monkeypatch):
    monkeypatch.setenv(ENV, "")
    assert resolve_reserve_rows(128) == default_reserve_rows(128)


def test_whitespace_only_matches_default(monkeypatch):
    monkeypatch.setenv(ENV, "   ")
    assert resolve_reserve_rows(128) == default_reserve_rows(128)


def test_override_two_e(monkeypatch):
    e = 128
    monkeypatch.setenv(ENV, str(2 * e))
    assert resolve_reserve_rows(e) == 2 * e


def test_override_one_e(monkeypatch):
    e = 128
    monkeypatch.setenv(ENV, str(e))
    assert resolve_reserve_rows(e) == e


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "12.5"])
def test_invalid_values_raise_naming_the_variable(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match=ENV):
        resolve_reserve_rows(128)


def test_invalid_value_does_not_silently_fall_back(monkeypatch):
    # A typo must never resolve to the default: that would produce a
    # valid-looking arm sized against the WRONG reserve.
    monkeypatch.setenv(ENV, "not-a-number")
    with pytest.raises(ValueError):
        resolve_reserve_rows(128)


# --------------------------------------------------------------------------
# The resolved value must drive BOTH the planner and the pool identically
# --------------------------------------------------------------------------

LAYERS, EXPERTS, H, I = 6, 8, 32, 32
TOTAL = LAYERS * EXPERTS  # 48


def _pool(root, capacity, reserve_rows):
    return MirrorExpertPool(
        root, LAYERS, EXPERTS, capacity, hidden_size=H, intermediate_size=I,
        spec=NEMOTRON_SPEC,
        config=types.SimpleNamespace(moe_layer_ids=list(range(LAYERS))),
        reserve_rows=reserve_rows,
    )


@pytest.fixture
def checkpoint(tmp_path):
    write_nvfp4_checkpoint(str(tmp_path), LAYERS, EXPERTS, H, I)
    return str(tmp_path)


def test_planner_and_pool_capacity_changes_with_reserve(checkpoint):
    final_gpu_slots = 40  # coverage need = TOTAL - 40 = 8
    cap_3e = plan_capacity(LAYERS, EXPERTS, final_gpu_slots,
                           reserve=default_reserve_rows(EXPERTS))
    cap_2e = plan_capacity(LAYERS, EXPERTS, final_gpu_slots, reserve=2 * EXPERTS)
    cap_1e = plan_capacity(LAYERS, EXPERTS, final_gpu_slots, reserve=EXPERTS)
    assert cap_3e == 8 + 3 * EXPERTS == 32
    assert cap_2e == 8 + 2 * EXPERTS == 24
    assert cap_1e == 8 + EXPERTS == 16
    assert cap_3e > cap_2e > cap_1e

    for reserve, capacity in ((3 * EXPERTS, cap_3e), (2 * EXPERTS, cap_2e), (EXPERTS, cap_1e)):
        pool = _pool(checkpoint, capacity, reserve)
        try:
            assert pool.reserve_rows == reserve
            assert pool.capacity == capacity
        finally:
            pool.close()


def test_unset_env_end_to_end_matches_default_behaviour(checkpoint, monkeypatch):
    """Full resolve -> plan -> pool path, unset env: byte-identical to HEAD."""
    monkeypatch.delenv(ENV, raising=False)
    resolved = resolve_reserve_rows(EXPERTS)
    assert resolved == default_reserve_rows(EXPERTS)
    final_gpu_slots = 40
    capacity_with_default = plan_capacity(LAYERS, EXPERTS, final_gpu_slots)
    capacity_resolved = plan_capacity(LAYERS, EXPERTS, final_gpu_slots, reserve=resolved)
    assert capacity_with_default == capacity_resolved
    pool = _pool(checkpoint, capacity_resolved, resolved)
    try:
        assert pool.reserve_rows == default_reserve_rows(EXPERTS)
        assert pool.capacity == capacity_with_default
    finally:
        pool.close()


def test_mismatched_reserve_between_planner_and_pool_fails_loudly(checkpoint):
    """The bug this knob makes possible: override reaches the planner but not
    the pool (or vice-versa), so the two size against different reserves.

    Sizing here is deliberately picked so the mismatch is NOT a subtle
    coverage shortfall but an outright refusal to construct -- capacity
    planned against the override (E) does not exceed the reserve the pool
    would withhold if it fell back to the default (3E). Correct coupling
    (same resolved value on both sides) must construct cleanly; passing the
    override to only one side must not.
    """
    final_gpu_slots = 40  # coverage need = 8
    resolved = EXPERTS  # e.g. FREETOKEN_MIRROR_RESERVE_ROWS=8 with E=8
    capacity = plan_capacity(LAYERS, EXPERTS, final_gpu_slots, reserve=resolved)
    assert capacity == 16

    # Correct: pool receives the SAME resolved reserve the planner used.
    good = _pool(checkpoint, capacity, resolved)
    try:
        assert good.reserve_rows == resolved
    finally:
        good.close()

    # Broken: planner got the override, pool did not (fell back to the
    # unconfigured default of 3E=24). capacity(16) <= 24 leaves the pool
    # unable to cover a single expert, so construction must refuse loudly
    # rather than silently starve writebacks.
    with pytest.raises(ValueError, match="does not exceed"):
        _pool(checkpoint, capacity, default_reserve_rows(EXPERTS))


# --------------------------------------------------------------------------
# Guard against the coupling drifting apart in engine.py itself
# --------------------------------------------------------------------------

def test_engine_wires_one_resolved_value_to_both_call_sites():
    """Static guard on engine.py's mirror-pool construction block.

    The whole point of ``resolve_reserve_rows`` is that engine.py calls it
    ONCE and threads the same variable into both ``plan_capacity(reserve=)``
    and ``MirrorExpertPool(reserve_rows=)``. This does not re-derive
    behaviour (the tests above do that) -- it only makes a future edit that
    re-introduces two independent reads of the env var, or that passes the
    override to one call and not the other, fail CI instead of shipping
    silently.
    """
    import inspect
    from freetoken.engine.engine import Engine

    src = inspect.getsource(Engine._init_offload_moe_cache)

    assert "resolve_reserve_rows(" in src, (
        "engine.py must resolve FREETOKEN_MIRROR_RESERVE_ROWS via "
        "resolve_reserve_rows()"
    )
    m = re.search(r"(\w+)\s*=\s*resolve_reserve_rows\(", src)
    assert m, "expected `<name> = resolve_reserve_rows(...)` in engine.py"
    var = m.group(1)

    def _call_block(name: str) -> str:
        idx = src.index(f"{name}(")
        return src[idx:idx + 600]

    assert re.search(rf"reserve\s*=\s*{re.escape(var)}\b", _call_block("plan_capacity")), (
        f"plan_capacity(...) must receive reserve={var}"
    )
    assert re.search(rf"reserve_rows\s*=\s*{re.escape(var)}\b", _call_block("MirrorExpertPool")), (
        f"MirrorExpertPool(...) must receive reserve_rows={var}"
    )
