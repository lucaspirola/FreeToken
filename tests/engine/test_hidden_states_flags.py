"""``--hidden-states-dir`` / ``--hidden-states-max-tokens`` resolution.

The directory is canonicalized once at startup, because every per-request path is later
checked for containment in it -- a relative or symlinked root would make that check
meaningless. CPU-only: nothing here builds an engine.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from freetoken.hidden_states import DEFAULT_MAX_TOKENS
from freetoken.server.args import parse_args


class _Config:
    """The minimum an unreachable checkpoint has to answer for parser auto-selection."""

    def to_dict(self):
        return {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"}


def _parse(*extra):
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", "/models/unit-model", *extra])
    return args


def test_disabled_by_default():
    args = _parse()
    assert args.hidden_states_dir is None
    assert args.hidden_states_max_tokens == DEFAULT_MAX_TOKENS == 4096


def test_directory_is_canonicalized(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert _parse("--hidden-states-dir", str(link)).hidden_states_dir == str(
        os.path.realpath(real)
    )
    assert _parse(
        "--hidden-states-dir", str(real / ".." / "real")
    ).hidden_states_dir == str(os.path.realpath(real))


def test_missing_directory_is_a_startup_error(tmp_path):
    with pytest.raises(SystemExit):
        _parse("--hidden-states-dir", str(tmp_path / "nope"))


def test_max_tokens_must_be_positive(tmp_path):
    assert _parse("--hidden-states-max-tokens", "512").hidden_states_max_tokens == 512
    with pytest.raises(SystemExit):
        _parse("--hidden-states-max-tokens", "0")
