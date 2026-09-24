"""engine._ensure_expandable_segments: expandable segments on every host, WSL included,
unless the user configured the allocator."""
from __future__ import annotations

import pytest

from freetoken.engine import engine as eng


@pytest.fixture
def calls(monkeypatch):
    got = []
    monkeypatch.setattr(eng.torch.cuda.memory, "_set_allocator_settings", got.append)
    for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "WSL_DISTRO_NAME"):
        monkeypatch.delenv(k, raising=False)
    return got


@pytest.mark.parametrize("host", ["linux", "wsl"])
def test_expandable_on_every_host(monkeypatch, calls, host):
    if host == "wsl":
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    eng._ensure_expandable_segments()
    assert calls == ["expandable_segments:True"]


@pytest.mark.parametrize("var", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
def test_user_allocator_config_is_left_alone(monkeypatch, calls, var):
    monkeypatch.setenv(var, "expandable_segments:False")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    eng._ensure_expandable_segments()
    assert calls == []
