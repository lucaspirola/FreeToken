"""The VMM extension's JIT build must not hang on a killed build's lock (2026-09-24)."""

from __future__ import annotations

import os
import time

from freetoken.kernel import vmm


def _lock_path(tmp_path, monkeypatch, name):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    from torch.utils.cpp_extension import _get_build_directory

    build = tmp_path / os.path.basename(_get_build_directory(name, verbose=False))
    build.mkdir(parents=True, exist_ok=True)
    return build / "lock"


def test_a_stale_lock_is_removed(tmp_path, monkeypatch):
    name = vmm._extension_name()
    lock = _lock_path(tmp_path, monkeypatch, name)
    lock.touch()
    old = time.time() - vmm._STALE_LOCK_SECONDS - 60
    os.utime(lock, (old, old))
    vmm._clear_stale_lock(name)
    assert not lock.exists()


def test_a_fresh_lock_is_kept(tmp_path, monkeypatch):
    name = vmm._extension_name()
    lock = _lock_path(tmp_path, monkeypatch, name)
    lock.touch()
    vmm._clear_stale_lock(name)
    assert lock.exists()


def test_the_name_follows_source_path_and_content(tmp_path, monkeypatch):
    a = tmp_path / "a" / "vmm_tensor.cpp"
    b = tmp_path / "b" / "vmm_tensor.cpp"
    for p in (a, b):
        p.parent.mkdir()
        p.write_text("int x;")
    monkeypatch.setattr(vmm, "_CSRC", a)
    first = vmm._extension_name()
    assert vmm._extension_name() == first
    monkeypatch.setattr(vmm, "_CSRC", b)
    assert vmm._extension_name() != first
    b.write_text("int y;")
    changed = vmm._extension_name()
    a.write_text("int y;")
    monkeypatch.setattr(vmm, "_CSRC", a)
    assert vmm._extension_name() not in (first, changed)
