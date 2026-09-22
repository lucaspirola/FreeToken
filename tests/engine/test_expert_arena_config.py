"""The expert-arena gate as a config value (refactor step S7).

``offload_cache.py`` and ``offload_kernels.py`` used to read ``FREETOKEN_EXPERT_ARENA``
at import time. Now ``EngineConfig.expert_arena`` carries it: ``server/args.py`` resolves
``--expert-arena`` and its env alias (``scripts/serve-default.sh`` exports
``FREETOKEN_EXPERT_ARENA=1``), and the engine publishes it with ``set_expert_arena``
before any expert cache exists. CPU-only: no engine is built.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from freetoken.moe import offload_cache, offload_kernels
from freetoken.moe.offload_cache import OffloadMoeCache, set_expert_arena
from freetoken.server.args import parse_args


class _Config:
    def to_dict(self):
        return {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"}


def _parse(*extra):
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", "/models/unit-model", *extra])
    return args


@pytest.fixture
def _restore_gate():
    prev = (offload_cache.FREETOKEN_EXPERT_ARENA, offload_kernels.FREETOKEN_EXPERT_ARENA)
    try:
        yield
    finally:
        offload_cache.FREETOKEN_EXPERT_ARENA, offload_kernels.FREETOKEN_EXPERT_ARENA = prev


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("FREETOKEN_EXPERT_ARENA", raising=False)
    assert _parse().expert_arena is False


def test_serve_default_alias_turns_it_on(monkeypatch):
    monkeypatch.setenv("FREETOKEN_EXPERT_ARENA", "1")
    assert _parse().expert_arena is True


def test_alias_zero_is_off(monkeypatch):
    monkeypatch.setenv("FREETOKEN_EXPERT_ARENA", "0")
    assert _parse().expert_arena is False


def test_flag_beats_the_alias(monkeypatch):
    monkeypatch.setenv("FREETOKEN_EXPERT_ARENA", "1")
    assert _parse("--no-expert-arena").expert_arena is False
    monkeypatch.delenv("FREETOKEN_EXPERT_ARENA")
    assert _parse("--expert-arena").expert_arena is True


def test_import_does_not_read_the_env():
    # A plain module constant; only set_expert_arena changes it.
    import ast
    import inspect

    for mod in (offload_cache, offload_kernels):
        tree = ast.parse(inspect.getsource(mod))
        (value,) = [
            n.value for n in tree.body
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "FREETOKEN_EXPERT_ARENA" for t in n.targets)
        ]
        assert isinstance(value, ast.Constant) and value.value is False, mod.__name__


def test_set_expert_arena_publishes_to_cache_and_kernels(_restore_gate):
    for enabled in (True, False, True):
        set_expert_arena(enabled)
        assert offload_cache.FREETOKEN_EXPERT_ARENA is enabled
        assert offload_kernels.FREETOKEN_EXPERT_ARENA is enabled
        cache = OffloadMoeCache(
            num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu")
        )
        assert cache._expert_arena_enabled is enabled
