import pytest


@pytest.fixture(autouse=True)
def _synchronous_checkpoint_writes(monkeypatch):
    # Tests that inspect checkpoint files right after a spill need them written inline;
    # the background-write tests raise the limit again themselves.
    import freetoken.scheduler.session_spill as spill_mod

    monkeypatch.setattr(spill_mod, "BACKGROUND_WRITE_MAX_BYTES", 0)
