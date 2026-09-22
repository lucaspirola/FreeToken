"""The ExpertResidency seam (refactor plan S6): CPU checks, no model, no GPU.

``OffloadMoeCache`` reaches the host side of its expert cache only through
``moe/residency.py``. These pin the seam's contract: the default is the
whole model in host RAM, its hooks are no-ops, the whole-model path never
imports the bounded pool, and both implementations answer every hook the
cache calls.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.residency import (
    ExpertResidency,
    MirrorResidency,
    WholeModelResidency,
    build_residency,
)

_PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"

# Every hook OffloadMoeCache calls, per the Protocol (methods only).
_HOOKS = sorted(
    name for name, value in vars(ExpertResidency).items()
    if callable(value) and not name.startswith("_")
)


def _cache(**kwargs) -> OffloadMoeCache:
    defaults = dict(num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu"))
    defaults.update(kwargs)
    return OffloadMoeCache(**defaults)


def test_protocol_lists_the_plan_hooks():
    assert {"min_gpu_slots", "before_ensure", "before_buffer_fill", "prefetch_layer",
            "before_shrink", "fault_check"} <= set(_HOOKS)


@pytest.mark.parametrize("cls", [WholeModelResidency, MirrorResidency])
def test_both_residencies_implement_every_hook(cls):
    missing = [name for name in _HOOKS if not callable(getattr(cls, name, None))]
    assert not missing, f"{cls.__name__} lacks {missing}"
    assert cls.kind in ("whole", "mirror")
    assert isinstance(cls.bounded, bool)


def test_default_cache_residency_is_the_whole_model():
    cache = _cache()
    residency = cache.residency
    assert isinstance(residency, WholeModelResidency)
    assert residency.kind == "whole" and not residency.bounded
    # The S7 contact point: the whole model imposes no coverage floor.
    assert cache.residency.min_gpu_slots() == 0
    # Compatibility surface other modules still read.
    assert cache._mirror is None
    assert cache.mirror_stats() == {}
    cache.mirror_fault_check()  # no-op, must not raise


def test_whole_model_hooks_decline_every_step():
    residency = WholeModelResidency()
    assert residency.before_ensure(0) is None
    assert residency.before_buffer_fill(0) is False
    assert residency.prefetch_layer(0, 0) is False
    assert residency.before_shrink(2, 6) is None
    assert residency.fault_check() is None
    assert residency.init_prefill_buffers() is False
    assert residency.begin_prefill() is False
    assert residency.copy_missing() is False
    assert residency.after_reset() is None
    assert residency.stats() == {}


def test_attach_whole_residency_binds_without_touching_banks():
    cache = _cache()
    residency = WholeModelResidency()
    cache.attach_residency(residency)
    assert cache.residency is residency and residency.cache is cache
    assert not cache.bank_caches


def test_mirror_min_gpu_slots_is_the_pools_own_bound():
    pool = types.SimpleNamespace(min_gpu_slots=1234)
    residency = MirrorResidency(pool)
    assert residency.pool is pool
    assert residency.min_gpu_slots() == 1234
    assert residency.kind == "mirror" and residency.bounded
    # Nothing is allocated until a cache attaches it.
    assert residency.cache is None and residency._mirror is None


def test_mirror_fault_check_is_silent_before_attach():
    MirrorResidency(types.SimpleNamespace(min_gpu_slots=0)).fault_check()


def test_mirror_pool_rejects_a_non_arena_cache():
    """attach_mirror_pool keeps its guards; nothing is bound on refusal."""
    cache = _cache(quant_format="bf16")
    pool = types.SimpleNamespace(num_layers=2, num_experts=4, schema_order=())
    with pytest.raises(ValueError, match="native NVFP4"):
        cache.attach_mirror_pool(pool)
    assert isinstance(cache.residency, WholeModelResidency)


def test_build_residency_whole_needs_no_engine_state():
    config = types.SimpleNamespace(expert_residency="whole", moe_mirror_host_rows=0)
    assert isinstance(build_residency(config, None, None), WholeModelResidency)


def test_build_residency_rejects_unknown_kind():
    config = types.SimpleNamespace(expert_residency="disk", moe_mirror_host_rows=0)
    with pytest.raises(ValueError, match="expert_residency"):
        build_residency(config, None, None)


def test_whole_model_path_never_imports_the_mirror_pool():
    """The plan's S6 check, as a test: importing the cache (and building the
    default residency) must not load ``moe.mirror_pool`` or its kernels."""
    code = (
        "import sys, torch\n"
        "import freetoken.moe.offload_cache as oc\n"
        "from freetoken.moe.residency import build_residency\n"
        "import types\n"
        "oc.OffloadMoeCache(num_layers=1, num_experts=2, cache_size=2,\n"
        "                   device=torch.device('cpu'))\n"
        "build_residency(types.SimpleNamespace(expert_residency='whole',\n"
        "                moe_mirror_host_rows=0), None, None)\n"
        "bad = [m for m in ('freetoken.moe.mirror_pool', 'freetoken.moe.mirror_kernels')\n"
        "       if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_PYTHON_DIR))
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
