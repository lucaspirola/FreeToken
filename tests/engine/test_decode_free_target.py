"""The decode-level free target: 0.375 GiB (VMM cushion 0.25 + margin 0.125) unless the
measurement override FREETOKEN_DECODE_FREE_TARGET_MB is set."""
from freetoken.engine.growable_kv import GrowableKvController


def test_default_is_cushion_plus_margin(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DECODE_FREE_TARGET_MB", raising=False)
    assert GrowableKvController.decode_free_target_bytes() == (256 + 128) * 1024 * 1024


def test_measurement_override(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DECODE_FREE_TARGET_MB", "160")
    assert GrowableKvController.decode_free_target_bytes() == 160 * 1024 * 1024


def test_probe_is_inert_without_a_window():
    ctl = GrowableKvController.__new__(GrowableKvController)
    ctl._sample_decode_memory()                      # no window opened: nothing to read
    ctl._log_decode_memory_window(closing=True)
