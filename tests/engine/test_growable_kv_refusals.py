"""Startup refusals for ``--kv-grow-step-tokens`` (refactor step S10).

The expert arena is the only growable-KV mechanism since S10 deleted the
rebuild-and-recapture fallback. ``_refuse_unsupported_growable_kv`` runs before the
expert cache is built, so an unsupported format or a missing arena fails at startup with
a message naming the fix, never at the first KV boundary of a live request.
CPU-only: meta tensors, no engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import _refuse_unsupported_growable_kv


def _banks(fmt: str, row_elems: list[int]):
    """One bank, one tensor per layer; a layer's row size is ``row_elems[layer]`` bytes."""
    return SimpleNamespace(
        quant_format=fmt,
        sources={
            "w": [torch.empty((4, n), dtype=torch.uint8, device="meta") for n in row_elems]
        },
    )


def _config(expert_arena: bool = True):
    return SimpleNamespace(expert_arena=expert_arena)


def test_arena_nvfp4_is_accepted():
    _refuse_unsupported_growable_kv(_config(), _banks("nvfp4", [64, 64]))


def test_uniform_gguf_on_the_arena_is_accepted():
    _refuse_unsupported_growable_kv(_config(), _banks("gguf", [96, 96, 96]))


@pytest.mark.parametrize("fmt", ["nvfp4_marlin", "nvfp4_b12x"])
def test_tiled_nvfp4_formats_are_refused(fmt):
    with pytest.raises(ValueError, match="growable KV unsupported for this format") as exc:
        _refuse_unsupported_growable_kv(_config(), _banks(fmt, [64]))
    assert fmt in str(exc.value)


def test_mixed_size_class_gguf_off_the_arena_is_refused():
    with pytest.raises(ValueError, match="growable KV unsupported for this format") as exc:
        _refuse_unsupported_growable_kv(
            _config(expert_arena=False), _banks("gguf", [96, 128, 96])
        )
    assert "2 size classes" in str(exc.value)


def test_mixed_size_class_gguf_passes_the_startup_refusal_gate_with_arena_on():
    # S12b: the arena builds one arena per size class, so this startup-time
    # format/arena refusal check no longer fires for mixed-GGUF once
    # --expert-arena is on. This ONLY exercises _refuse_unsupported_growable_kv
    # (the config-time gate); it does not drive the engine or the runtime
    # grow/shrink transaction -- see test_growable_kv_arena_engine.py's
    # class-arena cases for that.
    _refuse_unsupported_growable_kv(_config(expert_arena=True), _banks("gguf", [96, 128, 96]))


def test_format_refusal_wins_over_the_missing_arena():
    # The operator is told the format cannot grow at all, not to turn on an arena
    # that would refuse the format anyway.
    with pytest.raises(ValueError, match="unsupported for this format"):
        _refuse_unsupported_growable_kv(_config(expert_arena=False), _banks("nvfp4_b12x", [64]))


def test_arena_off_is_refused_and_names_the_switch():
    with pytest.raises(ValueError, match="requires the expert arena") as exc:
        _refuse_unsupported_growable_kv(_config(expert_arena=False), _banks("nvfp4", [64]))
    assert "--expert-arena" in str(exc.value)
    assert "FREETOKEN_EXPERT_ARENA=1" in str(exc.value)
