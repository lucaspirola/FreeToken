"""``--expert-residency`` and its environment aliases (refactor plan S6).

``server/args.py`` is the only place ``FREETOKEN_MIRROR_EXPERT_RAM`` and
``FREETOKEN_MIRROR_HOST_ROWS`` are read. ``tasks/exclusive-expert-ram/measure.sh``
drives both GPU checkpoints through exactly those two names (it writes
``=0``/``=0`` for a whole-model arm and ``=1``/``=<rows>`` for a mirror arm), so
the resolution below must keep answering the way the engine's own reading did
before S6. CPU-only: nothing here builds an engine.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from freetoken.server.args import parse_args


class _Config:
    def to_dict(self):
        return {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MIRROR_EXPERT_RAM", raising=False)
    monkeypatch.delenv("FREETOKEN_MIRROR_HOST_ROWS", raising=False)


def _parse(*extra):
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", "/models/unit-model", *extra])
    return args


def test_default_is_the_whole_model():
    args = _parse()
    assert (args.expert_residency, args.moe_mirror_host_rows) == ("whole", 0)


def test_measure_sh_whole_model_arm(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_EXPERT_RAM", "0")
    monkeypatch.setenv("FREETOKEN_MIRROR_HOST_ROWS", "0")
    args = _parse()
    assert (args.expert_residency, args.moe_mirror_host_rows) == ("whole", 0)


def test_measure_sh_mirror_arm(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_EXPERT_RAM", "1")
    monkeypatch.setenv("FREETOKEN_MIRROR_HOST_ROWS", "1700")
    args = _parse()
    assert (args.expert_residency, args.moe_mirror_host_rows) == ("mirror", 1700)


def test_plain_on_switch_auto_sizes(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_EXPERT_RAM", "1")
    args = _parse()
    assert (args.expert_residency, args.moe_mirror_host_rows) == ("mirror", 0)


def test_env_rows_alone_turn_the_mirror_on(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_HOST_ROWS", "-1")
    args = _parse()
    assert (args.expert_residency, args.moe_mirror_host_rows) == ("mirror", -1)


def test_flag_rows_turn_the_mirror_on_and_beat_env_rows(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_HOST_ROWS", "1700")
    args = _parse("--moe-mirror-host-rows", "1200")
    assert (args.expert_residency, args.moe_mirror_host_rows) == ("mirror", 1200)


def test_explicit_mirror_without_rows_auto_sizes():
    args = _parse("--expert-residency", "mirror")
    assert (args.expert_residency, args.moe_mirror_host_rows) == ("mirror", 0)


def test_explicit_whole_beats_the_env_aliases(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MIRROR_EXPERT_RAM", "1")
    monkeypatch.setenv("FREETOKEN_MIRROR_HOST_ROWS", "1700")
    assert _parse("--expert-residency", "whole").expert_residency == "whole"


def test_explicit_whole_with_explicit_rows_is_refused():
    with pytest.raises(SystemExit):
        _parse("--expert-residency", "whole", "--moe-mirror-host-rows", "1700")
