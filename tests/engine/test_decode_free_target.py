"""The decode-level free target: 128 MiB for every model unless the override
FREETOKEN_DECODE_FREE_TARGET_MB is set."""
from freetoken.engine.growable_kv import (
    DECODE_FREE_TARGET_BYTES,
    GrowableKvController,
    growable_headroom_bytes,
)


def test_default_is_128_mib(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DECODE_FREE_TARGET_MB", raising=False)
    assert DECODE_FREE_TARGET_BYTES == 128 * 1024 * 1024
    assert GrowableKvController.decode_free_target_bytes() == 128 * 1024 * 1024


def test_default_is_below_every_prefill_level(monkeypatch):
    """Decode keeps less than the bare VMM cushion, so the dynamic headroom
    (prefill level > decode level) is on even for a zero prefill transient."""
    monkeypatch.delenv("FREETOKEN_DECODE_FREE_TARGET_MB", raising=False)
    assert GrowableKvController.decode_free_target_bytes() < growable_headroom_bytes(0)


def test_empty_override_keeps_the_default(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DECODE_FREE_TARGET_MB", "  ")
    assert GrowableKvController.decode_free_target_bytes() == DECODE_FREE_TARGET_BYTES


def test_measurement_override(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DECODE_FREE_TARGET_MB", "160")
    assert GrowableKvController.decode_free_target_bytes() == 160 * 1024 * 1024


def test_probe_is_inert_without_a_window():
    ctl = GrowableKvController.__new__(GrowableKvController)
    ctl._sample_decode_memory()                      # no window opened: nothing to read
    ctl._log_decode_memory_window(closing=True)
